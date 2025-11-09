import torch
import numpy as np
import os
import sys
import copy
import re

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from datasets.dataset import DataSet, DataSetVelocity, DataSetVelocityOnly, DataSetVelocityRM, DataSetVelocityOnlyRM
from datasets.dataset import add_noise
from utils.skeleton import Skeleton
from utils.thops import nn2vis_dformat, vis2nn_dformat
from utils.thops import forward_kinematics, fkl_torch
from utils.thops import rotmat2euler, rotmat2euler, expmap2rotmat, expmap2rotmat_torch, rotmat2expmap_torch
from datasets.AMASS.ang2joint import ang2joint


class DatasetH36MBase(DataSet, DataSetVelocity, DataSetVelocityOnly, DataSetVelocityRM, DataSetVelocityOnlyRM):
    Human36M = True
    
    ActionTypeList = (
        'directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 
        'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether', 'walking' 
    )
    action_to_idx = {action: idx for idx, action in enumerate(ActionTypeList)}
    
    @classmethod
    def get_action_types(cls):
        return cls.ActionTypeList
    
    @classmethod
    def get_action_nums(cls):
        return len(cls.ActionTypeList)

    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, *args, **kwargs):
        self.cali_subject = None if "cali_subject" not in kwargs else kwargs["cali_subject"] 
        self.noise_std = 0.0 if "noisy_input" not in kwargs else kwargs["noisy_input"]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]) 

        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate)
        self.current_subject = None     
        return


    def process_data(self):
        data_o = np.load(self.data_path, allow_pickle=True)[self.data_type].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if 'all' not in self.actions:
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: any([a == re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", x[0]) for a in self.actions]), \
                                          data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)

        data_n = {}
        for subject, data_s in data_f.items():
            data_n[subject] = {}
            for action, data_a in data_s.items():
                seq = data_a
                if hasattr(self, 'Human36MexpMap'):
                    seq = np.reshape(seq, (seq.shape[0], -1, 3))[:, 1:, :] 
                
                elif hasattr(self, 'Human36MPos'):
                    if self.local_coord:
                        seq[:, 1:] -= seq[:, :1]    # 将其余关节点的全局坐标变换到以hip(0)关节点为原点的局部坐标系
                    if self.hip_zeros:
                        seq[:, :1] = 0.0    # 将hip(0)关节点的坐标设置为局部坐标系的原点                
                
                if hasattr(self, 'set_zero_joints'):
                    seq[:, self.set_zero_joints, :] = 0.0   
                
                seq = seq[:, self.kept_joints, :]             
                seq = seq[0::self.frame_rate, ...]
                
                if hasattr(self, 'copy_joints'):
                    for cjs in self.copy_joints:
                        target, source = cjs[0], cjs[1]
                        seq[:, target, :] = seq[:, source, :]
                
                seq = self.posprocessing(seq) 
                data_n[subject][action] = seq
                
        self.data = data_n
        if self.cali_subject and len(data_n) > 1:
            self.cali_data = {}
            key = 'S'+str(self.cali_subject)
            self.cali_data[key] = self.data[key]
            self.subjects.remove(key)
            self.data.pop(key)
        else:
            self.cali_data = None
            
        self.bones_length = self._get_bones_length()
        return


    def prepare_data(self):
        raise NotImplementedError

    def _get_bones_length(self):
        raise NotImplementedError


    def sample(self):
        #从self.subject中随机采样一个
        subject = np.random.choice(self.subjects)
        # self.current_subject = subject     # only meaningful when sample to plot gif.
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0]-self.t_total+1)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]     
        
        if self.noise_std > 0 and self.mode == 'train':
            traj = add_noise(traj=traj, noise_std=self.noise_std)
                        
        return traj[None, ...], re.sub(r'[0-9]+', '', action[:-2] if '_m' in action else action)
    
    
    def sampling_generator(self, num_samples=1000, batch_size=8):
        for _ in range(num_samples // batch_size):
            sample, action = [], []
            for _ in range(batch_size):
                sample_i, act_i = self.sample()
                sample.append(sample_i)
                action.append(act_i)
            sample = np.concatenate(sample, axis=0)    
            sample = np.swapaxes(sample, -1, -2)
            sample = np.squeeze(sample)
            yield sample, action   

 
    def iter_generator(self, step=0, batch_size=1):
        step = step if step>0 else self.t_his        
        for subject, data_s in self.data.items():
            
            if batch_size == 1:
                self.current_subject = subject
                
            traj_list, action_list = [], []
            for key, seq in data_s.items():
                if self.mode == 'test' and '_m' in key:
                    continue
                
                seq_len = seq.shape[0]
                for i in range(0, seq_len-self.t_total, step):
                    traj = seq[None, i: i+self.t_total]
                    traj_list.append(traj)
                    action_list.append(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key))
                    
                    if len(traj_list)>=batch_size:
                        _traj, _action = np.concatenate(traj_list, 0), copy.deepcopy(action_list)
                        if self.noise_std > 0 and self.mode == 'train':
                            _traj = add_noise(traj=_traj, noise_std=self.noise_std)
                        _traj = np.swapaxes(_traj, -1, -2)
                        _traj = np.squeeze(_traj)
                        traj_list.clear()
                        action_list.clear()
                        yield _traj, _action
                        
            if len(traj_list) > 0:
                _traj, _action = np.concatenate(traj_list, 0), copy.deepcopy(action_list)
                if self.noise_std > 0 and self.mode == 'train':
                    _traj = add_noise(traj=_traj, noise_std=self.noise_std)
                _traj = np.swapaxes(_traj, -1, -2)
                _traj = np.squeeze(_traj)
                traj_list.clear()
                action_list.clear()
                yield _traj, _action    
    

    def convert_to_position_mode(self, data, option=0, **kwargs):
        raise NotImplementedError


    def get_skeleton(self, plt=False):
        raise NotImplementedError


    @property
    def zero_joints(self):
        raise NotImplementedError


    def posprocessing(self, seq):
        if hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocity':
            return DataSetVelocity.posprocessing(self, seq)
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityOnly':
            return DataSetVelocityOnly.posprocessing(self, seq)
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityRM':
            return DataSetVelocityRM.posprocessing(self, seq)
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityOnlyRM':
            return DataSetVelocityOnlyRM.posprocessing(self, seq)
        else:
            return DataSet.posprocessing(self, seq)
    
    
    def get_cali_data(self, step=0):
        if not self.cali_data:
            return None 
        
        if hasattr(self, "cali_seq_list") and hasattr(self, "copula_seq_list"):
            return self.cali_seq_list, self.copula_seq_list
        
        cali_d, copula_d = [], []
        for action in self.ActionTypeList:
            for _, data_s in self.cali_data.items():
                data = dict(filter(lambda x: action == re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", x[0]), data_s.items()))
                data = list(data.values())
                for idx, sub_data in enumerate(data):
                    if idx % 2 == 0:
                        cali_d.append(sub_data)
                    else:
                        copula_d.append(sub_data)                

        step = step if step > 0 else self.t_his
        def generate(data):
            seq_list = []
            for idx, seq in enumerate(data):
                num_frames = seq.shape[0]
                fs = np.arange(0, num_frames-self.t_total+1, step)
                fs_sel = fs
                for i in np.arange(self.t_total-1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = seq[fs_sel, :]
                seq_list.append(seq_sel)
            seq_list = np.concatenate(seq_list, axis=0)
            return seq_list
        self.cali_seq_list, self.copula_seq_list = generate(cali_d), generate(copula_d)     # output_size: (B, L, N, C)
        return self.cali_seq_list, self.copula_seq_list


    def get_index_label(self, label):
        label = np.array(label)
        indices = np.vectorize(self.action_to_idx.get)(label) + 1   # 为无类别提供一个占位
        indices = indices.astype(np.int32)
        return indices
    
    
    def get_index_labels_without_empty(self, labels):
        indices = [(self.get_index_label(label)).reshape(1,) for label in labels]
        indices = np.concatenate(indices, axis=0) - 1
        return indices
        


class DatasetH36MexpMap(DatasetH36MBase):
    Human36MexpMap = True
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, *args, **kwargs):
        self.data_type = 'expmap_3d'
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    
    
    def _get_bones_length(self):
        self.origin_bones_length = np.load(self.data_path, allow_pickle=True)['bone_length'].item()
        self.bones_length = {}
        for subject, bone_length in self.origin_bones_length.items():
            self.bones_length[subject] = bone_length
        return self.bones_length

    
    def _fill_joints(self, data):
        new_shape = data.shape[:-2] + (32,) + data.shape[-1:]
        new_data = np.zeros(shape=new_shape)
        new_data[..., self.kept_joints, :] = data
        return new_data


    def get_skeleton(self, plt=False):
        if not plt:
            return self.kept_skeleton
        else:
            return self.skeleton


    def convert_to_position_mode(self, data, option=0, **kwargs):
        if isinstance(data, torch.Tensor):
            device = data.device
            _data = nn2vis_dformat(data.to('cpu'))
        else:
            device = torch.device('cpu')
            _data = data
        _data = self._fill_joints(_data)
         
        position = forward_kinematics(_data, parents=self.get_skeleton(True).parents(), offsets=self.bones_length[self.current_subject])        
        if option == 0 or option == 1:
            position = position
        elif option == 2:
            position = self._get_eval_joints(position)
        
        if isinstance(data, torch.Tensor):
            position = vis2nn_dformat(position).to(device)
        return position
    
    
    def _get_eval_joints(self, data):
        raise NotImplementedError 


    def expmap2eular(self, data):
        if isinstance(data, torch.Tensor):
            device = data.device
            _data = nn2vis_dformat(data.to('cpu'))
        else:
            _data = data
        _data = rotmat2euler(expmap2rotmat(_data))    
        if isinstance(data, torch.Tensor):
            _data = vis2nn_dformat(_data).to(device)
        return _data 


       
class DatasetH36MPos(DatasetH36MBase):
    Human36MPos = True
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, 
                local_coord=True, hip_zeros=True, *args, **kwargs):
        self.local_coord = local_coord
        self.hip_zeros = hip_zeros
        self.data_type = 'positions_3d'
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return


    def _get_bones_length(self):             
        return


    def _fill_joints(self, data):
        raise NotImplementedError 

    def get_skeleton(self, plt=False):
        raise NotImplementedError
        
       
    def convert_to_position_mode(self, data, option=0, **kwargs):
        if isinstance(data, torch.Tensor):
            device = data.device
            _data = nn2vis_dformat(data.to('cpu'))
        else:
            _data = data
            
        _data = self._fill_joints(_data)
        if option == 0 or option == 1:
            position = _data
        elif option == 2:
            position = self._get_eval_joints(_data)
            
        if isinstance(data, torch.Tensor):
            position = vis2nn_dformat(position).to(device)
        return position


    def _get_eval_joints(self, data):
        raise NotImplementedError 




class StochasticPredictionInterface(object):
    def set_configs(self, mode, cali_subject):
        self.subjects_split = {'train': [1, 5, 6, 7, 8], 'test': [9, 11], 'val': [6]}
        cur_subjects = self.subjects_split[mode]
        if cali_subject and cali_subject not in cur_subjects:
            cur_subjects.append(cali_subject)
        self.subjects = ['S%d' % x for x in cur_subjects]   
             
        self.used_joints = np.array([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27])
        self.ignore_joints = np.array([])
        self.equal_joints = np.array([])
        self.constant_joints = np.array([])
        self.constant_joints_values = np.array([])
        return



class DeterministicPredictionInterface(object):
    def set_configs(self, mode, cali_subject):
        self.subjects_split = {'train': [1, 6, 7, 8, 9, 11], 'test': [5]}
        cur_subjects = self.subjects_split[mode]
        if cali_subject and cali_subject not in cur_subjects:
            cur_subjects.append(cali_subject)
        self.subjects = ['S%d' % x for x in cur_subjects]     
        
        self.used_joints = np.array([ x for x in range(32) if x not in [0, 1, 6, 11, 16, 20, 23, 24, 28, 31] ])
        self.ignore_joints = np.array([16, 20, 23, 24, 28, 31])
        self.equal_joints = np.array([13, 19, 22, 13, 27, 30])
        self.constant_joints = np.array([0, 1, 6, 11])
        self.constant_joints_values = np.array([[0, 0, 0], [-0.13295, 0, 0], [0.13295, 0, 0], [0, 0.0001, 0]])
        return


    def sample_256(self):
        """
        Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
        which originaly from
        In order to find the same action indices as in SRNN.
        https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """
        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        def find_indices_256(seq_len):
            SEED = 1234567890
            rng = np.random.RandomState(SEED)
            T = seq_len - 150
            idx_o = []
            for _ in np.arange(0, 128):
                idx_ran = rng.randint(16, T)
                idx_o.append(idx_ran)
            return idx_o
            
        for _, dict_s in self.data.items():
            for key, seq in dict_s.items():
                if self.mode == 'test' and '_m' in key:
                    continue
                subseq_list, action_list = [], []
                idx_o = find_indices_256(seq.shape[0])
                for idx in idx_o:
                    subseq = seq[None, idx:idx+self.t_total]
                    subseq = np.swapaxes(subseq, -1, -2)
                    subseq_list.append(subseq)
                    action_list.append(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key))
                _subseqs, _action = np.concatenate(subseq_list, 0), action_list
                yield _subseqs, _action


    def sample_8(self):
        def find_indices_8(seq_len):
            SEED = 1234567890
            rng = np.random.RandomState(SEED)
            T = seq_len - 150
            idx_o = []
            for _ in np.arange(0, 4):
                idx_ran = rng.randint(16, T)
                idx_o.append(idx_ran)
            return idx_o
            
        for _, dict_s in self.data.items():
            for key, seq in dict_s.items():
                if self.mode == 'test' and '_m' in key:
                    continue
                subseq_list, action_list = [], []
                idx_o = find_indices_8(seq.shape[0])
                for idx in idx_o:
                    subseq = seq[None, idx:idx+self.t_total]
                    subseq = np.swapaxes(subseq, -1, -2)
                    subseq_list.append(subseq)
                    action_list.append(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key))
                _subseqs, _action = np.concatenate(subseq_list, 0), action_list
                yield _subseqs, _action




class StochasticDatasetH36MexpMap(DatasetH36MexpMap, StochasticPredictionInterface):
    def prepare_data(self):
        self.removed_joints = [4, 5, 9, 10, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31]
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        
        self.kept_skeleton = copy.deepcopy(self.skeleton)
        self.kept_skeleton.remove_joints(self.removed_joints)

        self.set_zero_joints = [3, 8, 15] # before remove joints.
        self.zero_joints_after_remove = [3, 6, 11]   # after remove joints.  before [3, 8, 15]  
                     
        self.set_configs(self.mode, self.cali_subject)   
        self.process_data()
        
        plt_removed_joints = [4, 5, 9, 10, 20, 21, 22, 23, 28, 29, 30, 31]
        new_bones_length = {}
        for subject, bone_length in self.bones_length.items():
            bone_length[plt_removed_joints, :] = 0.0
            new_bones_length[subject] = bone_length
        self.bones_length = new_bones_length
        return

    def _get_eval_joints(self, data):
        return data[..., self.used_joints, :]

    @property
    def zero_joints(self):
        return self.zero_joints_after_remove


class StandardStochasticDatasetH36MexpMap(StochasticDatasetH36MexpMap):
    def prepare_data(self):
        self.removed_joints = [4, 5, 8, 9, 10, 15, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31]
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        
        self.kept_skeleton = copy.deepcopy(self.skeleton)
        self.kept_skeleton.remove_joints(self.removed_joints)

        self.set_zero_joints = [3]
        self.zero_joints_after_remove = [3]  
                     
        self.set_configs(self.mode, self.cali_subject)   
        self.process_data()
        
        plt_removed_joints = [4, 5, 9, 10, 20, 21, 22, 23, 28, 29, 30, 31]
        new_bones_length = {}
        for subject, bone_length in self.bones_length.items():
            bone_length[plt_removed_joints, :] = 0.0
            new_bones_length[subject] = bone_length
        self.bones_length = new_bones_length
        return
    


class StochasticDatasetH36MPos(DatasetH36MPos, StochasticPredictionInterface):
    def prepare_data(self):
        self.removed_joints = {4, 5, 9, 10, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        
        self.kept_skeleton = copy.deepcopy(self.skeleton)
        self.kept_skeleton.remove_joints(self.removed_joints)
        self.kept_skeleton.adjust_connection_manually(([12, 9], [15, 9]))
        
        self.set_configs(self.mode, self.cali_subject)          
        self.process_data()
        return
    
    def _get_eval_joints(self, data):
        new_shape = data.shape[:-2] + (32,) + data.shape[-1:]
        new_data = np.zeros(shape=new_shape)
        new_data[..., self.kept_joints, :] = data
        new_data = new_data[..., self.used_joints, :]
        return new_data
    
    def _fill_joints(self, data):
        return data

    def get_skeleton(self, plt=False):
        return self.kept_skeleton

    @property
    def zero_joints(self):
        return [0] if self.hip_zeros else []   



class DeterministicDatasetH36MexpMap(DatasetH36MexpMap, DeterministicPredictionInterface):
    s5_offset = np.array([0.000000, 0.000000, 0.000000,
                        -132.948591, 0.000000, 0.000000,
                        0.000000, -442.894612, 0.000000,
                        0.000000, -454.206447, 0.000000,
                        0.000000, 0.000000, 162.767078,
                        0.000000, 0.000000, 74.999437,
                        132.948826, 0.000000, 0.000000,
                        0.000000, -442.894413, 0.000000,
                        0.000000, -454.206590, 0.000000,
                        0.000000, 0.000000, 162.767426,
                        0.000000, 0.000000, 74.999948,
                        0.000000, 0.100000, 0.000000,
                        0.000000, 233.383263, 0.000000,
                        0.000000, 257.077681, 0.000000,
                        0.000000, 121.134938, 0.000000,
                        0.000000, 115.002227, 0.000000,
                        0.000000, 257.077681, 0.000000,
                        0.000000, 151.034226, 0.000000,
                        0.000000, 278.882773, 0.000000,
                        0.000000, 251.733451, 0.000000,
                        0.000000, 0.000000, 0.000000,
                        0.000000, 0.000000, 99.999627,
                        0.000000, 100.000188, 0.000000,
                        0.000000, 0.000000, 0.000000,
                        0.000000, 257.077681, 0.000000,
                        0.000000, 151.031437, 0.000000,
                        0.000000, 278.892924, 0.000000,
                        0.000000, 251.728680, 0.000000,
                        0.000000, 0.000000, 0.000000,
                        0.000000, 0.000000, 99.999888,
                        0.000000, 137.499922, 0.000000,
                        0.000000, 0.000000, 0.000000]).reshape((32, 3)) * 0.001

    
    def prepare_data(self):
        self.removed_joints = [5, 10, 20, 21, 22, 23, 28, 29]
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        
        self.kept_skeleton = copy.deepcopy(self.skeleton)
        self.kept_skeleton.remove_joints(self.removed_joints)
        self.kept_skeleton.adjust_connection_manually(([[22, 13]]))
        self.kept_skeleton.adjust_connection_manually(([[23, 22]]))
        
        self.set_configs(self.mode, self.cali_subject)               
        self.process_data()
        
        self.current_subject = 'S5'
        self.bones_length['S5'] = self.s5_offset
        return

    def _get_eval_joints(self, data):
        position = np.zeros_like(data)
        position[..., self.used_joints, :] = data[..., self.used_joints, :]
        position[..., self.ignore_joints, :] = data[..., self.equal_joints, :]
        position[..., self.constant_joints, :] = self.constant_joints_values   
        return position

    @property
    def zero_joints(self):
        return [22, 23]



class DeterministicDatasetH36MPos(DatasetH36MPos, DeterministicPredictionInterface):
    def prepare_data(self):
        if hasattr(self, '22N'):
            self.removed_joints = {0, 1, 6, 11, 16, 20, 23, 24, 28, 31}
            self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
            
            self.kept_skeleton = copy.deepcopy(self.skeleton)
            self.kept_skeleton.remove_joints(self.removed_joints)
            self.kept_skeleton.adjust_connection_manually(([17, 9], [12, 9]))
            
        elif hasattr(self, '23N'):
            self.removed_joints = {1, 6, 11, 16, 20, 23, 24, 28, 31}
            self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
            
            self.kept_skeleton = copy.deepcopy(self.skeleton)
            self.kept_skeleton.remove_joints(self.removed_joints)
            self.kept_skeleton.adjust_connection_manually(([17, 9], [12, 9]))
            
        else:
            self.removed_joints = {1, 6, 11, 16, 20, 23, 24, 28}
            self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
            
            self.kept_skeleton = copy.deepcopy(self.skeleton)
            self.kept_skeleton.remove_joints(self.removed_joints)
            self.copy_joints = [[23, 12]]  # copy 12 to 23  # after remove_joints
            self.kept_skeleton.adjust_connection_manually((self.copy_joints))
        
        self.current_subject = 'S5'
        
        self.set_configs(self.mode, self.cali_subject)          
        self.process_data()
        return
    
    def _get_eval_joints(self, data):
        return data

    def _fill_joints(self, data):
        new_shape = data.shape[:-2] + (32,) + data.shape[-1:]
        new_data = np.zeros(shape=new_shape)
        new_data[..., self.kept_joints, :] = data
        new_data[..., self.ignore_joints, :] = new_data[..., self.equal_joints, :]
        new_data[..., self.constant_joints, :] = self.constant_joints_values
        return new_data

    def get_skeleton(self, plt=False):
        if not plt:
            return self.kept_skeleton
        else:
            return self.skeleton

    @property
    def zero_joints(self):
        return [0] if self.hip_zeros else []




class H36MtorchDatasetBase(torch.utils.data.Dataset):
    DataLoader = True    
    def generate_segments(self, step):
        if step < 0:
            step = 1 if self.mode == 'train' else self.t_his
        else:
            step = step
        self.segments = []
        for subject, data_s in self.data.items():
            for action, seq in data_s.items():
                if self.mode == 'test' and '_m' in action:
                    continue
                for i in range(0, seq.shape[0]-self.t_total, step):
                    self.segments.append([subject, action, i])
        return

    def __getitem__(self, idx):
        subject, action, seq_st = self.segments[idx]
        traj = self.data[subject][action][seq_st:seq_st+self.t_total]
        traj = np.swapaxes(traj, -2, -1)
        action = re.sub(r'[0-9]+', '', action[:-2] if '_m' in action else action)
        return traj, action    

    def __len__(self):
        return len(self.segments)    



class StochasticTorchDatasetH36MexpMap(H36MtorchDatasetBase, StochasticDatasetH36MexpMap):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, *args, **kwargs):
        StochasticDatasetH36MexpMap.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return

    def __getitem__(self, idx):
        subject, action, seq_st = self.segments[idx]
        self.current_subject = subject
        traj = self.data[subject][action][seq_st:seq_st+self.t_total]
        traj = np.swapaxes(traj, -2, -1)
        action = re.sub(r'[0-9]+', '', action[:-2] if '_m' in action else action)
        return traj, action 



class StochasticTorchDatasetH36MPos(H36MtorchDatasetBase, StochasticDatasetH36MPos):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, 
                local_coord=True, hip_zeros=True, *args, **kwargs):
        StochasticDatasetH36MPos.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, 
                                        local_coord, hip_zeros, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return



class DeterministicTorchDatasetH36MexpMap(H36MtorchDatasetBase, DeterministicDatasetH36MexpMap):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, *args, **kwargs):
        DeterministicDatasetH36MexpMap.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return



class DeterministicTorchDatasetH36MPos(H36MtorchDatasetBase, DeterministicDatasetH36MPos):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, 
                local_coord=True, hip_zeros=True, *args, **kwargs):
        DeterministicDatasetH36MPos.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, 
                                            local_coord, hip_zeros, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return