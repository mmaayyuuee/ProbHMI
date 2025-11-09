"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_humaneva.py
"""

import torch
import numpy as np
import os
import sys
import copy
import re

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from datasets.dataset import DataSetPosition, DataSetRotation
from datasets.dataset import DataSetVelocity, DataSetVelocityOnly, DataSet
from utils.skeleton import Skeleton
from utils.thops import nn2vis_dformat, vis2nn_dformat, expmap2rotmat, rotmat2expmap



class DatasetHumanEvaBase(DataSet):
    HumanEva = True
    
    ActionTypeList = (
        'Box', 'Gestures', 'Jog', 'ThrowCatch', 'Walking'
    )
    
    @classmethod
    def get_action_types(cls):
        return cls.ActionTypeList
    
    @classmethod
    def get_action_nums(cls):
        return len(cls.ActionTypeList)
    
    
    def __init__(self, data_path, mode, t_his=15, t_pred=60, actions='all', frame_rate=1, *args, **kwargs):
        self.cali_subject = None if "cali_subject" not in kwargs else kwargs["cali_subject"] 
        self.cali_params = {} if "cali_params" not in kwargs else kwargs["cali_params"] 
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate)
        return


    def prepare_data(self):
        self.subjects_split = {'train': ['Train/S1', 'Train/S2', 'Train/S3'],
                               'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']}
        self.subjects = [x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                 joints_left=[2, 3, 4, 8, 9, 10],
                                 joints_right=[5, 6, 7, 11, 12, 13])
        self.kept_joints = np.arange(15)

        if hasattr(self, '16N'):
            self.placehold_joints = [14]   ## parents_list: 15连14
            self.plt_joints = [15]
        elif hasattr(self, '18N'):
            self.placehold_joints = [14, 0, 16]
            self.plt_joints = [15, 16, 17]
        else:
            self.placehold_joints = []
            self.plt_joints = []
        
        self._zero_joints = copy.deepcopy(self.plt_joints)
        if (hasattr(self, 'HumanEvaPos') and self.hip_zeros) or hasattr(self, 'HumanEvaExpMap'):
            self._zero_joints.append(0)
            
        self.kept_skeleton = copy.deepcopy(self.skeleton)
        self.kept_skeleton.add_new_nodes_manually(self.placehold_joints)
        
        self.process_data()
        return
    
    
    def process_data(self):
        data_o = np.load(self.data_path, allow_pickle=True)[self.data_type].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        
        # these takes have wrong head position, excluded from training and testing
        if self.mode == 'train':
            if hasattr(self, "HumanEvaPos") and self.HumanEvaPos:
                data_f['Train/S3'].pop('Walking 1 chunk0')
                data_f['Train/S3'].pop('Walking 1 chunk2')
        else:
            if hasattr(self, "HumanEvaPos") and self.HumanEvaPos:
                data_f['Validate/S3'].pop('Walking 1 chunk4')    

        if 'all' not in self.actions:
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: any([a in x[0] for a in self.actions]) and x[1].shape[0] >= self.t_total, 
                                                    data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        else:
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: x[1].shape[0] >= self.t_total, data_f[key].items()))  
                if len(data_f[key]) == 0:
                    data_f.pop(key) 

        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                seq = seq[0::self.frame_rate, ...]
                
                if hasattr(self, 'HumanEvaPos'):
                    if self.local_coord:
                        seq[:, 1:] -= seq[:, :1]    # 将其余关节点的全局坐标变换到以hip(0)关节点为原点的局部坐标系
                    if self.hip_zeros:
                        seq[:, :1] = 0.0    # 将hip(0)关节点的坐标设置为局部坐标系的原点
                
                _seq = np.zeros(shape=(seq.shape[0], seq.shape[1]+len(self.placehold_joints), seq.shape[2]), dtype=np.float32)
                _seq[:, :seq.shape[1], :] = seq
                
                _seq = self.posprocessing(_seq)
                data_s[action] = _seq

        self._get_bones_length()
        self.data = data_f

        if self.cali_subject:
            self.generate_cali_data()
        return

    
    def get_skeleton(self, plt=False):
        if not plt:
            return self.kept_skeleton
        else:
            return self.skeleton
    
    
    @property
    def zero_joints(self):
        return self._zero_joints
    
    
    def _get_bones_length(self):
        raise NotImplementedError
    
    
    def _construct_uniform_sampling_sets(self):
        _uniform_sets = []
        for _, dict_s in self.data.items():
            for _, seq in dict_s.items():
                for idx in range(0, seq.shape[0] - self.t_total):
                    _uniform_sets.append(seq[idx: idx+self.t_total])
        self.uniform_sets = np.array(_uniform_sets)
        return


    def sample(self):
        #从self.subject中随机采样一个
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0]-self.t_total+1)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        return traj[None, ...], action.split()[0]


    def sampling_generator(self, num_samples=1000, batch_size=8):    
        for _ in range(num_samples // batch_size):
            sample, action = [], []
            for _ in range(batch_size):
                sample_i, action_i = self.sample()
                sample.append(sample_i), action.append(action_i)
    
            sample = np.concatenate(sample, axis=0)
            sample = np.swapaxes(sample, -1, -2)
            sample = np.squeeze(sample)
            yield sample, action
    

    def sampling_every_data(self):
        for _, data_s in self.data.items():
            for key, seq in data_s.items():
                yield seq, key

 
    def iter_generator(self, step=0, batch_size=1):
        step = step if step>0 else self.t_his
        for data_s in self.data.values():
            traj_list, action_list = [], []
            for key, seq in data_s.items():
                _key = key.split()[0]
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    traj_list.append(traj)
                    action_list.append(_key)
                    if len(traj_list) >= batch_size:
                        _traj, _action = np.concatenate(traj_list, 0), copy.deepcopy(action_list)
                        _traj = np.swapaxes(_traj, -1, -2)
                        _traj = np.squeeze(_traj)
                        traj_list.clear()
                        action_list.clear()
                        yield _traj, _action
                        
            if len(traj_list) > 0:
                _traj, _action = np.concatenate(traj_list, 0), copy.deepcopy(action_list)
                _traj = np.swapaxes(_traj, -1, -2)
                _traj = np.squeeze(_traj)
                traj_list.clear()
                action_list.clear()
                yield _traj, _action  


    def generate_cali_data(self):
        cali_subjects = [x for x in self.subjects_split['train']]
        data_o = np.load(self.data_path, allow_pickle=True)[self.data_type].item()
        data_f = dict(filter(lambda x: x[0] in cali_subjects, data_o.items()))
        if hasattr(self, "HumanEvaPos") and self.HumanEvaPos:
            data_f['Train/S3'].pop('Walking 1 chunk0')
            data_f['Train/S3'].pop('Walking 1 chunk2') 
            
        t_total = self.cali_params["t_total"]    

        if 'all' not in self.actions:
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: any([a in x[0] for a in self.actions]) and x[1].shape[0] >= self.t_total, 
                                                    data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        else:
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: x[1].shape[0] >= t_total, data_f[key].items()))  
                if len(data_f[key]) == 0:
                    data_f.pop(key) 

        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                seq = seq[0::self.frame_rate, ...]
                
                if hasattr(self, 'HumanEvaPos'):
                    if self.local_coord:
                        seq[:, 1:] -= seq[:, :1]    # 将其余关节点的全局坐标变换到以hip(0)关节点为原点的局部坐标系
                    if self.hip_zeros:
                        seq[:, :1] = 0.0    # 将hip(0)关节点的坐标设置为局部坐标系的原点
                
                _seq = np.zeros(shape=(seq.shape[0], seq.shape[1]+len(self.placehold_joints), seq.shape[2]), dtype=np.float32)
                _seq[:, :seq.shape[1], :] = seq
                
                _seq = self.posprocessing(_seq)
                data_s[action] = _seq
        
        train_parts, cali_parts, copula_parts = {}, {}, {}
        for subject, data_s in data_f.items():
            train_parts[subject], cali_parts[subject], copula_parts[subject] = {}, {}, {}
            
            for action, seq in data_s.items():
                split1 = int((seq.shape[0]-t_total+1) * 0.8)
                split2 = int((seq.shape[0]-t_total+1) * 0.9)
                
                train_parts[subject][action] = seq[:split1+t_total-1]
                cali_parts[subject][action] = seq[split1:split2+t_total-1]
                copula_parts[subject][action] = seq[split2:]
                
        if self.mode == 'train':
            self.data = train_parts
        self.cali_data, self.copula_data = cali_parts, copula_parts
        return
    

    def get_cali_data(self, step=0):
        if not hasattr(self, "cali_data"):
            return None 
        
        if hasattr(self, "cali_seq_list") and hasattr(self, "copula_seq_list"):
            return self.cali_seq_list, self.copula_seq_list
               
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
        self.cali_seq_list, self.copula_seq_list = generate(self.cali_data), generate(self.copula_data)     # output_size: (B, L, N, C)
        return self.cali_seq_list, self.copula_seq_list



class DatasetHumanEvaExpMap(DatasetHumanEvaBase):
    HumanEvaExpMap = True

    def __init__(self, data_path, mode, t_his=15, t_pred=60, actions='all', frame_rate=1, *args, **kwargs):
        self.data_type = 'expmap_3d'
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        
    
    def convert_to_position_mode(self, data, option=0, **kwargs):
        if isinstance(data, torch.Tensor):
            device = data.device
            _data = nn2vis_dformat(data.to('cpu'))
        else:
            _data = data
        _data = self.remove_unused_data(_data)
        postion = self.forward_kinematics(_data, parents=self.get_skeleton(True).parents(), offsets=self.bones_length[self.current_subject])
        if option == 0 or option == 1:
            postion = postion[..., 0:15, :]
        else:
            postion = postion[..., 1:15, :]

        if isinstance(data, torch.Tensor):
            postion = vis2nn_dformat(postion).to(device)
        return postion


    def forward_kinematics(self, expmaps, parents, offsets):
        origin_shape = expmaps.shape
        if len(origin_shape) == 2:
            expmaps = np.expand_dims(expmaps, [0,1])
        elif len(origin_shape) == 3:
            expmaps = np.expand_dims(expmaps, 0)
        expmaps = np.reshape(expmaps, (-1, expmaps.shape[-2], expmaps.shape[-1]))
        
        njoints = expmaps.shape[-2]
        xyzStruct = [dict() for _ in range(njoints)]
        
        for i in range(njoints):
            r = expmaps[:, i:i+1, :]
            rotation = expmap2rotmat(r)
            
            if parents[i] == -1:
                xyzStruct[i]['rotation'] = rotation
                xyzStruct[i]['xyz'] = np.broadcast_to(offsets[i], (expmaps.shape[0], offsets[i].shape[0]))
            else:
                xyzStruct[i]['rotation'] = np.matmul(rotation, xyzStruct[parents[i]]['rotation'])
                xyzStruct[i]['xyz'] = np.matmul(xyzStruct[i]['rotation'], offsets[i]) + xyzStruct[parents[i]]['xyz']
                
        xyz = [np.expand_dims(xyzStruct[i]['xyz'], 1) for i in range(njoints)]
        xyz = np.concatenate(xyz, 1)
        xyz = np.reshape(xyz, origin_shape)
        return xyz
    
    
    def _get_bones_length(self):
        self.origin_bones_length = np.load(self.data_path, allow_pickle=True)['bone_length'].item()
        self.bones_length = {}
        for subject, bone_length in self.origin_bones_length.items():
            self.bones_length[subject] = bone_length[self.kept_joints, :]
        return


    def remove_unused_data(self, data):
        return self._remove_plt_joints(data)

    
    def _remove_plt_joints(self, data):
        new_data = np.delete(data, obj=self.plt_joints, axis=-2)
        return new_data    



class DatasetHumanEvaPos(DatasetHumanEvaBase):
    HumanEvaPos = True

    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, 
                local_coord=True, hip_zeros=True, *args, **kwargs):
        self.local_coord = local_coord
        self.hip_zeros = hip_zeros
        self.data_type = 'positions_3d'
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    

    def convert_to_position_mode(self, data, option=0, **kwargs):
        if isinstance(data, torch.Tensor):
            device = data.device
            _data = nn2vis_dformat(data.to('cpu'))
        else:
            _data = data
        
        if option == 0 or option == 1:
            new_data = _data[..., 0:15, :]
        else:
            new_data = _data[..., 1:15, :]
            
        if isinstance(data, torch.Tensor):
            new_data = vis2nn_dformat(new_data).to(device)
        return new_data
    
    
    def _get_bones_length(self):
        return




class StochasticDatasetHumanEvaExpMap(DatasetHumanEvaExpMap, DataSetVelocity, DataSetVelocityOnly):
    def posprocessing(self, seq):
        if hasattr(self, 'velocity'):
            return DataSetVelocity.posprocessing(self, seq)
        elif hasattr(self, 'velocityOnly'):
            return DataSetVelocityOnly.posprocessing(self, seq)
        else:
            return DatasetHumanEvaExpMap.posprocessing(self, seq)   
    
    

class StochasticDatasetHumanEvaPos(DatasetHumanEvaPos, DataSetVelocity, DataSetVelocityOnly):
    def posprocessing(self, seq):
        if hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocity':
            return DataSetVelocity.posprocessing(self, seq)
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityOnly':
            return DataSetVelocityOnly.posprocessing(self, seq)
        else:
            return DatasetHumanEvaPos.posprocessing(self, seq)   
    



class HumanEvaTorchDatasetBase(torch.utils.data.Dataset):
    DataLoader = True    
    def generate_segments(self, step):
        step = 1 if step < 0 else step
        self.segments = []
        for subject, data_s in self.data.items():
            for action, seq in data_s.items():
                for i in range(0, seq.shape[0]-self.t_total, step):
                    self.segments.append([subject, action, i])
        return

    def __getitem__(self, idx):
        subject, action, seq_st = self.segments[idx]
        traj = self.data[subject][action][seq_st:seq_st+self.t_total]
        traj = np.swapaxes(traj, -2, -1)
        return traj, action.split()[0]    

    def __len__(self):
        return len(self.segments)    



class StochasticTorchDatasetHumanEvaExpMap(HumanEvaTorchDatasetBase, StochasticDatasetHumanEvaExpMap):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, *args, **kwargs):
        StochasticDatasetHumanEvaExpMap.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return



class StochasticTorchDatasetHumanEvaPos(HumanEvaTorchDatasetBase, StochasticDatasetHumanEvaPos):
    def __init__(self, data_path, mode, t_his=25, t_pred=100, actions='all', frame_rate=1, 
                local_coord=True, hip_zeros=True, *args, **kwargs):
        StochasticDatasetHumanEvaPos.__init__(self, data_path, mode, t_his, t_pred, actions, frame_rate, 
                                              local_coord, hip_zeros, *args, **kwargs)
        return
    
    def process_data(self):
        super().process_data()
        return