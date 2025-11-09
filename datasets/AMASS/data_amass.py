import torch
import torch.utils
import numpy as np
import torch.utils.data
import zarr
import pandas as pd
import copy
import re
from tqdm import tqdm
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from utils.skeleton import Skeleton
from utils.thops import nn2vis_dformat, vis2nn_dformat
from utils.thops import expmap2rotmat, rotmat_to_sixd, expmap2rotmat_torch_V2, rotmat2euler_torch, euler2rotmat_torch, rotmat2expmap_torch
from datasets.AMASS import ang2joint
from datasets.dataset import DataSetVelocity, DataSetVelocityOnly



def gender_mapper(input):
    if isinstance(input, str):
        if input == 'male':
            return 1
        elif input == 'female':
            return 0
        return 1
    
    elif isinstance(input, int) or isinstance(input, np.ndarray):
        input = int(input)
        if input == 1:
            return 'male'
        elif input == 0:
            return 'female'
        return 'male'



class DataSetAMASSBase(object):
    AMASS = True

    ActionTypeList = (
        'ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'EKUT', 'EyesJapanDataset', 'KIT', 
        'PosePrior', 'TCDHands', 'TotalCapture', 'HumanEva', 'HDM05', 'SFU', 'MoSh', 'DFaust', 
        'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions'
    )
    action_to_idx = {action: idx for idx, action in enumerate(ActionTypeList)}
    
    def __init__(self, data_path, mode, t_his, t_pred, stride=1, *args, **kwargs):
        self.data_path = data_path
        self.mode = mode
        self.t_his, self.t_pred = t_his, t_pred
        self.t_total = t_his + t_pred
        self.stride = 1 if self.mode == "test" else stride
        self.skeleton = None
        self.data, self.data_idx, self.gender = {}, {}, {}
        
        self.cali_subject = 0.0 if "cali_subject" not in kwargs else kwargs["cali_subject"]
        self.norm_type = None if "norm_type" not in kwargs else kwargs["norm_type"]
    
        self.amass_splits = {            
            'train': ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'EKUT', 'EyesJapanDataset', 'KIT', 'PosePrior', 
                    'TCDHands', 'TotalCapture'],
            'validation': ['HumanEva', 'HDM05', 'SFU', 'MoSh'],
            'test': ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions'],
            'all': ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'EKUT', 'EyesJapanDataset', 'KIT', 'PosePrior', 
                    'TCDHands', 'TotalCapture', 'HumanEva', 'HDM05', 'SFU', 'MoSh', 'DFaust', 'DanceDB', 'GRAB', 
                    'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        }
        self.prepare_data()   
        super().__init__()
        return
    
    def prepare_data():
        raise NotImplementedError

    @classmethod
    def get_action_types(cls):
        return cls.ActionTypeList
    
    @classmethod
    def get_action_nums(cls):
        return len(cls.ActionTypeList)

    def get_index_label(self, label, *args, **kwargs):
        label = np.array(label)
        indices = np.vectorize(self.action_to_idx.get)(label) + 1   # 为无类别提供一个占位
        indices = indices.astype(np.int32)
        return indices
    
    def get_index_labels_without_empty(self, labels):
        indices = [(self.get_index_label(label)).reshape(1,) for label in labels]
        indices = np.concatenate(indices, axis=0) - 1
        return indices
    

    def posprocessing(self, seq):
        if hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocity':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0    
            seq = np.concatenate((seq, resdiuals), axis=-1)
            return seq   
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityOnly':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0    
            return resdiuals
        else:
            return seq



class DataSetAMASSexpMap(DataSetAMASSBase):
    def __init__(self, data_path, mode, t_his, t_pred, stride=1, *args, **kwargs): 
        mean_std = np.load(os.path.join(os.getcwd(), "data/auxiliary/amass_expmap_mean_std_train.npz"))
        self.amass_mean, self.amass_std = mean_std['means'].astype(np.float32), mean_std['stds'].astype(np.float32)       
        
        super().__init__(data_path, mode, t_his, t_pred, stride=1, *args, **kwargs)
        return
    
        
    def prepare_data(self):
        self.skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 22],
                                 joints_left=[1, 4, 7, 10, 14, 17, 19, 21],
                                 joints_right=[2, 5, 8, 11, 13, 16, 18, 20])  
        self.plt_skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                                     joints_left=[1, 4, 7, 10, 14, 17, 19, 21],
                                     joints_right=[2, 5, 8, 11, 13, 16, 18, 20]) 
        self.zero_pose_joints = [10, 11, 22, 23]
        self.kept_joints = np.array([x for x in range(24)])
        
        self.estimated_joints = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
                
        self.t_poses = {}
        t_poses = np.load(os.path.join(os.getcwd(), "data/auxiliary/amass_t_poses.npz"))
        self.t_poses['male'] = torch.from_numpy(t_poses['male'].astype(np.float32))[None]
        self.t_poses['female'] = torch.from_numpy(t_poses['female'].astype(np.float32))[None]

        self.annotations, self.meta_anns_all, self.dict_indices = self._read_annotations(self.mode)
        self.segments = self._prepare_segments(self.annotations)
        return


    def _read_annotations(self, mode):
        """
        convert data dic to annotation list: []
        """
        anns_all = []
        meta_anns_all = []
        dict_indices = {}          # dic[dataset] := dic[file_idx] := index
        counter = 0

        for dataset in self.amass_splits[mode]:
            dict_indices[dataset] = {}

            z_poses = zarr.open(os.path.join(self.data_path, dataset, 'poses.zarr'), mode='r')
            z_index = zarr.open(os.path.join(self.data_path, dataset, 'poses_index.zarr'), mode='r')

            max_length = int(z_index.shape[0] * (1.0 - self.cali_subject)) if mode == 'train' else z_index.shape[0]
            # max_length = int(z_index.shape[0] - 2) if mode == 'train' else z_index.shape[0]
            for file_idx in range(max_length):
                dict_indices[dataset][file_idx] = counter
                i0, i = z_index[file_idx]
                seq = z_poses[i0:i]
                seq[:, self.zero_pose_joints, :] = 0.0
                
                if hasattr(self, "6D_representation"):
                    seq_ori_shape = seq.shape
                    seq = np.reshape(seq, newshape=(-1, seq.shape[-2], seq.shape[-1]))
                    seq = expmap2rotmat(seq)
                    seq = rotmat_to_sixd(seq)
                    seq = np.reshape(seq, newshape=[*seq_ori_shape[:-1], 6]).astype(np.float32)
                    
                seq = self.posprocessing(seq)
                dict_indices[dataset][file_idx] = counter
                counter += 1
                anns_all.append(seq)
                meta_anns_all.append((dataset, file_idx))
        return anns_all, meta_anns_all, dict_indices 


    def _prepare_segments(self, annotations):
        if self.mode == 'test':
            segments = self._load_annotations_and_segments(os.path.join(os.getcwd(), "data/auxiliary/segments_test.csv"))
        else:
            segments = self._generate_segments(annotations)
        return segments
            
            
    def _generate_segments(self, annotations):
        segments = []
        for idx in range(len(annotations)): 
            for init in range(0, annotations[idx].shape[0]-self.t_total+1):
                segments.append([idx, init, init+self.t_total-1])
        return segments


    def _load_annotations_and_segments(self, segments_path):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
                
        segments = [(self.dict_indices[row["dataset"]][row["file_idx"]], 
                    row["pred_init"] - self.t_his, 
                    row["pred_init"] + self.t_pred - 1) for _, row in df.iterrows()]
        return segments


    def _get_segment(self, i, init, end):
        traj = self.annotations[i][init:end+1]  # (T, N, C)
        traj = traj[None, ...]
        label = self.meta_anns_all[i]
        return traj, label[0]


    def get_cali_data(self, step):
        if self.cali_subject < 0.0 or self.cali_subject == 0.0:
            return None 
        
        if hasattr(self, "cali_seq_list") and hasattr(self, "copula_seq_list"):
            return self.cali_seq_list, self.copula_seq_list        
        
        cali_anns_full, cali_meta_anns_full = [], []
        copula_anns_full, copula_meta_anns_full = [], []
        
        for dataset in self.amass_splits['train']:
            z_poses = zarr.open(os.path.join(self.data_path, dataset, 'poses.zarr'), mode='r')
            z_index = zarr.open(os.path.join(self.data_path, dataset, 'poses_index.zarr'), mode='r')

            beg = int(z_index.shape[0] * (1.0 - self.cali_subject))
            mid = int(z_index.shape[0] * (1.0 - self.cali_subject / 2))
            end = z_index.shape[0]
            # beg = int(z_index.shape[0] - 2)
            # mid = int(z_index.shape[0] - 1)
            # end = z_index.shape[0]
            
            def split_dataset(st, ed): 
                anns, meta_anns = [], []  
                counter = 0        
                for file_idx in range(st, ed):
                    i0, i = z_index[file_idx]
                    seq = z_poses[i0:i]
                    seq[:, self.zero_joints, :] = 0.0
                    seq = self.posprocessing(seq)
                    counter += 1
                    anns.append(seq)
                    meta_anns.append((dataset, file_idx))
                return anns, meta_anns
            
            cali_anns, cali_meta_anns = split_dataset(beg, mid)
            copula_anns, copula_meta_anns = split_dataset(mid, end)
            
            cali_anns_full += cali_anns
            cali_meta_anns_full += cali_meta_anns
            copula_anns_full += copula_anns 
            copula_meta_anns_full += copula_meta_anns
            
        cali_segments = self._generate_segments(cali_anns_full)
        copula_segments = self._generate_segments(copula_anns_full)
        
        step = step if step > 0 else self.t_his
        def generate(segments, anns, meta_anns):
            length = len(segments) // step
            # length = len(segments)
            traj_list, label_list = [], []
            for idx in range(0, length, 1):
                # (i, init, end) = segments[idx]
                (i, init, end) = segments[idx*step]
                
                traj = anns[i][init:end+1]  # (T, N, C)
                traj = traj[None, ...]
                label = meta_anns[i]

                traj_list.append(traj)
                label_list.append(label)
                
            seq_list = np.concatenate(traj_list, axis=0)    
            return seq_list, label_list
        
        self.cali_seq_list, _ = generate(cali_segments, cali_anns_full, cali_meta_anns_full) # output_size: (B, L, N, C)
        self.copula_seq_list, _ = generate(copula_segments, copula_anns_full, copula_meta_anns_full)
        return self.cali_seq_list, self.copula_seq_list        
        
        
    @property
    def zero_joints(self):
        return [0, 10, 11, 22, 23]

    def get_skeleton(self, plt=False):
        return self.plt_skeleton if plt else self.skeleton  

    def remove_unused_data(self, data):
        return data
    
    def sampling_generator(self, num_samples=1000, batch_size=8):
        max_length = len(self.segments) // self.stride
        for _ in range(num_samples // batch_size):
            indices = np.random.randint(0, max_length, size=(batch_size,))
            
            sample, labels = [], []
            for index in indices:
                (i, init, end) = self.segments[index*self.stride]
                traj, label = self._get_segment(i, init, end)
                if traj.shape[1] < self.t_total:    # 先打个临时补丁
                    continue
                sample.append(traj)
                labels.append(label)
                    
            sample = np.concatenate(sample, axis=0)
            sample = np.swapaxes(sample, -1, -2)
            yield sample, labels
        
    
    def iter_generator(self, step=1, batch_size=1, complete_data=False):
        step = step if step > 0 else self.t_his
        max_length = len(self.segments) // self.stride
        
        traj_list, label_list = [], []
        for idx in range(0, max_length, 1):
            (i, init, end) = self.segments[idx*self.stride]
            traj, label = self._get_segment(i, init, end)
            if traj.shape[1] < self.t_total:    # 先打个临时补丁
                continue
            traj_list.append(traj)
            label_list.append(label)
            
            if len(traj_list) >= batch_size:
                _traj = np.concatenate(traj_list, 0)
                _label = copy.deepcopy(label_list)
                if not complete_data:
                    _traj = self.remove_unused_data(_traj)
                _traj = np.swapaxes(_traj, -1, -2)
                traj_list.clear()
                label_list.clear()
                yield _traj, _label      
            
        if len(traj_list) > 0:      
            _traj = np.concatenate(traj_list, 0)
            _label = copy.deepcopy(label_list)
            if not complete_data:
                _traj = self.remove_unused_data(_traj)
            _traj = np.swapaxes(_traj, -1, -2)
            traj_list.clear()
            label_list.clear()
            yield _traj, _label  


    def convert_to_position_mode(self, data, **kwargs):        
        if isinstance(data, np.ndarray):
            data_t = torch.from_numpy(data)
            data_t = torch.swapaxes(data_t, -2, -1)
        else:
            data_t = data.to('cpu')
            data_t = torch.swapaxes(data_t, -2, -1)
                
        if 'gender' not in kwargs:
            gender = 'female'
            if data_t.size(-1) != self.t_poses[gender].size(-1):
                data_t = torch.swapaxes(data_t, -2, -1)
            t_pose = self.t_poses[gender].expand_as(data_t)
        else:
            genders = kwargs['gender']
            t_pose = [self.t_poses[gender] for gender in genders]
            t_pose = torch.concat(t_pose, dim=0)
            
            # (batch_size, length, :, :)
            if data_t.ndim == 4:
                t_pose = t_pose[:, None, :, :].expand_as(data_t)
                static_pos = static_pos[:, None, :, :].repeat(1, data_t.shape[1], 1, 1)
            # (samples, batch_size, length, :, :)
            elif data_t.ndim == 5:
                t_pose = t_pose[None, :, None, :, :].expand_as(data_t)
                static_pos = static_pos[None, :, None, :, :].repeat(data_t.shape[0], 1, data_t.shape[2], 1, 1)
        
        data_shape_ori = data_t.shape
        data_t, t_pose = [d.reshape((-1, d.shape[-2], d.shape[-1])) for d in [data_t, t_pose]]
        
        f'''
        expmap_a = data_t
        rot_a = expmap2rotmat_torch_V2(data_t)
        euler_a = rotmat2euler_torch(rot_a)
        rot_b = euler2rotmat_torch(euler_a)
        expmap_b = rotmat2expmap_torch(rot_b)
        
        expmap_c = rotmat2expmap_torch(rot_a)
        
        s1 = expmap_a - expmap_b
        s2 = rot_a - rot_b
        s3 = expmap_a - expmap_c
        '''
        
        f'''
        data_e = rotmat2euler_torch(expmap2rotmat_torch_V2(data_t))
        data_e[..., 0, (0,1,2)] = 0.0
        data_t = rotmat2expmap_torch(euler2rotmat_torch(data_e))
        '''      
          
        if hasattr(self, "6D_representation"):
            position = ang2joint.ang2joint_6d(t_pose, data_t, self.get_skeleton(False).parents())
        else:
            position = ang2joint.ang2joint(t_pose, data_t, self.get_skeleton(False).parents())
        position[..., 1:, :] = position[..., 1:, :] - position[..., :1, :]
        position[..., :1, :] = 0.0
        
        position = position.reshape(data_shape_ori)
        
        option = 0 if 'option' not in kwargs else kwargs['option']
        if option == 0 or option == 1:
            position = position[..., :22, :]
        elif option == 2:
            position = position[..., self.estimated_joints, :]
        
        if isinstance(data, np.ndarray):
            position = position.numpy()
        else:
            position =  torch.swapaxes(position, -2, -1)
        return position

        
    def compute_mean_std_normalize(self):        
        data_generator = self.iter_generator(batch_size=128)
        old_n, n = 0, 0
        for data, label in tqdm(data_generator):    # data_shape: (batch_size, length, channel, node_n)
            _, _, channel, node_n = data.shape
            
            data_ = np.reshape(data, newshape=(-1, channel, node_n))
            chunk_size = data_.shape[0]
            
            old_n = n
            n += chunk_size
            
            # 一次性计算块的统计量
            chunk_sum = np.sum(data_, axis=0)
            chunk_sum_sq = np.sum(data_**2, axis=0)
            
            # 更新全局均值
            mean = (old_n*mean + chunk_sum) / n if old_n > 0 else chunk_sum / n
            # 更新 M2 (使用更稳定的公式)
            if old_n > 0:
                # 当 n > 0 时的更新公式
                M2 += chunk_sum_sq - 2 * chunk_sum * mean + chunk_size * mean ** 2
            else:
                # 第一个块的特殊处理
                M2 = chunk_sum_sq - chunk_size * mean ** 2

        # 计算标准差
        if n <= 0:
            std_dev = float('nan')
        else:
            variance = M2 / n
            std_dev = np.sqrt(variance)
        
        save_dir = os.path.join(os.getcwd(), 'data/auxiliary', 'amass_expmap_mean_std_'+str(self.mode)+'.npz')
        np.savez(save_dir, means=mean, stds=std_dev)
        return 
    
    
    def norm_(self, x, mean, std):
        mean, std = np.swapaxes(mean, 0, 1), np.swapaxes(std, 0, 1)
        if self.norm_type == 'gaussian_root_norm':
            x[..., 0, :] =  (x[..., 0, :] - mean[..., 0, :]) / (std[..., 0, :] + 1e-6)
        elif self.norm_type == 'gaussian_norm':
            x =  (x - mean) / (self.std + 1e-6)
        elif self.norm_type == 'mean_root_norm':
            x[..., 0, :] = x[..., 0, :] - mean[..., 0, :]
        elif self.norm_type == 'mean_norm':
            x = x - mean
        return x
    
    @property
    def dataset_mean_std(self):
        mean, std = np.zeros_like(self.amass_mean), np.ones_like(self.amass_std)
        if self.norm_type == 'gaussian_root_norm':
            mean[..., 0], std[..., 0] = self.amass_mean[..., 0], self.amass_std[..., 0]
        elif self.norm_type == 'gaussian_norm':
            mean, std = self.amass_mean, self.amass_std
        elif self.norm_type == 'mean_root_norm':
            mean[..., 0], std = self.amass_mean[..., 0], None
        elif self.norm_type == 'mean_norm':
            mean, std = self.amass_mean, None
        return mean, std        
        

    def posprocessing(self, seq):
        if hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocity':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0    
            seq = np.concatenate((seq, resdiuals), axis=-1)
            if self.norm_type is not None:
                seq = self.norm_(seq, self.amass_mean, self.amass_std)
            return seq   

        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityRoot':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0    
            seq[..., 0, :] = resdiuals[..., 0, :]
            seq = np.concatenate((seq, resdiuals), axis=-1)
            return seq        
        
        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityOnly':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0   
            if self.norm_type is not None:
                resdiuals = self.norm_(resdiuals, self.amass_mean[..., 3:, :], self.amass_std[..., 3:, :]) 
            return resdiuals

        elif hasattr(self, 'velocityFlag') and self.velocityFlag == 'velocityRootOnly':
            resdiuals = seq - np.roll(seq, shift=1, axis=-3)
            resdiuals[..., 0, :, :] = 0.0   
            seq[..., 0, :] = resdiuals[..., 0, :]
            return seq   
        
        else:
            if self.norm_type is not None:
                seq = self.norm_(seq, self.amass_mean[..., :3, :], self.amass_std[..., :3, :])
            return seq



class DataSetAMASSexpMapDataLoader(torch.utils.data.Dataset, DataSetAMASSexpMap):
    DataLoader = True

    def __getitem__(self, index):
        segment_idx = int(self.stride * index)
        (i, init, end) = self.segments[segment_idx]
        traj, label = self._get_segment(i, init, end)
        traj = traj.swapaxes(-2, -1)
        return traj[0], label  

    def __len__(self):
        return len(self.segments) // self.stride 

    def generate_segments(self, *args, **kwargs):
        return 



class DataSetAMASSPos(DataSetAMASSBase):
    def __init__(self, data_path, mode, t_his, t_pred, stride=1, *args, **kwargs):
        super().__init__(data_path, mode, t_his, t_pred, stride=1, *args, **kwargs)
        return    

    def prepare_data(self):
        self.skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 22],
                                 joints_left=[1, 4, 7, 10, 14, 17, 19, 21],
                                 joints_right=[2, 5, 8, 11, 13, 16, 18, 20])  
        self.plt_skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                                     joints_left=[1, 4, 7, 10, 14, 17, 19, 21],
                                     joints_right=[2, 5, 8, 11, 13, 16, 18, 20]) 
        self.zero_pose_joints = [0, 22, 23]
        self.kept_joints = np.array([x for x in range(24)])
        
        self.estimated_joints = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        
        self.process_data()
        return

    def process_data(self):
        if self.mode == 'train':
            data_file = self.data_path + '.npz'
        elif self.mode == 'test':
            data_file = self.data_path + '_test.npz'
            
        seq_ = np.load(data_file, allow_pickle=True)['arr_0']
        seq_[..., :1, :] = 0.0
        
        seq = np.zeros(shape=(seq_.shape[0], seq_.shape[1], 24, seq_.shape[3]), dtype=np.float32)
        seq[:, :, :22, :] = seq_
        seq = self.posprocessing(seq)
        
        self.data = seq
        self.data_size = self.data.shape[0]
        return

    @property
    def zero_joints(self):
        return [0, 22, 23]

    def get_skeleton(self, plt=False):
        return self.plt_skeleton if plt else self.skeleton  

    def remove_unused_data(self, data):
        return data


    def sample(self):
        idx = np.random.randint(0, self.data_size)
        traj = self.data[idx]
        return traj[None, ...]
    
    def sampling_generator(self, num_samples=1000, batch_size=8):
        max_length = self.data_size // self.stride
        for _ in range(num_samples // batch_size):
            indices = np.random.randint(0, max_length, size=(batch_size,))
            sample = []
            for index in indices:
                traj = self.data[index*self.stride]
                sample.append(traj[None, ...])     
            sample = np.concatenate(sample, axis=0)
            sample = np.swapaxes(sample, -1, -2)
            yield sample, None
        
    
    def iter_generator(self, step=1, batch_size=1, complete_data=False):
        step = step if step > 0 else self.t_his
        max_length = self.data_size // self.stride
        
        traj_list, label_list = [], []
        for idx in range(0, max_length, 1):
            traj = self.data[idx*self.stride]
            traj_list.append(traj[None])
            
            if len(traj_list) >= batch_size:
                _traj = np.concatenate(traj_list, 0)
                if not complete_data:
                    _traj = self.remove_unused_data(_traj)
                _traj = np.swapaxes(_traj, -1, -2)
                traj_list.clear()
                yield _traj, None      
            
        if len(traj_list) > 0:      
            _traj = np.concatenate(traj_list, 0)
            if not complete_data:
                _traj = self.remove_unused_data(_traj)
            _traj = np.swapaxes(_traj, -1, -2)
            traj_list.clear()
            yield _traj, None 


    def convert_to_position_mode(self, data, option=0, **kwargs):
        if isinstance(data, torch.Tensor):
            device = data.device
            position = nn2vis_dformat(data.to('cpu'))
        else:
            position = data
            
        if option == 0 or option == 1:
            position = position[..., :22, :]
        elif option == 2:
            position = position[..., self.estimated_joints, :]
            
        if isinstance(data, torch.Tensor):
            position = vis2nn_dformat(position).to(device)
        return position
    


class DataSetAMASSPosDataLoader(torch.utils.data.Dataset, DataSetAMASSPos):
    DataLoader = True
    def __getitem__(self, index):
        segment_idx = int(self.stride * index)
        traj = self.data[segment_idx]
        traj = traj.swapaxes(-2, -1)
        return traj, 'None'  

    def __len__(self):
        return self.data_size // self.stride 

    def generate_segments(self, *args, **kwargs):
        return 