import numpy as np
import torch
import copy
import re
from utils.thops import nn2vis_dformat, vis2nn_dformat, expmap2rotmat, rotmat2expmap, rotmat2euler, rotmat2quat


class DataSet(object):
    def __init__(self, data_path, mode, t_his, t_pred, actions='all', frame_rate=1, *args, **kwargs):
        self.data_path = data_path
        self.mode = mode
        self.t_his, self.t_pred = t_his, t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.frame_rate = frame_rate
        
        # self.skeleton, self.data = None, None
        self.prepare_data()
        
        if not hasattr(self, 'CMU'):            
            self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
            self.data_channels = list(list(self.data.values())[0].values())[0].shape[-1]

        super().__init__()
        return

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
            
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / (self.std+1e-9)
                # data_s[action][:, 0] = 0.0
        return
    
    def posprocessing(self, seq):
        return seq


     
class DataSetPosition(DataSet):
    def __init__(self, data_path, mode, t_his, t_pred, actions='all', frame_rate=1, *args, **kwargs):
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return

    def sample(self):
        raise NotImplementedError
    
    
    def sampling_generator(self, num_samples=1000, batch_size=8, augmentation=0):
        for _ in range(num_samples // batch_size):
            sample, action = [], []
            for _ in range(batch_size):
                sample_i, action_i = self.sample()
                sample.append(sample_i)
                action.append(action_i)
                
            sample = np.concatenate(sample, axis=0)
            sample = np.swapaxes(sample, -1, -2)
            sample = np.squeeze(sample)
            
            samples = [sample]
            for _ in range(augmentation): 
                angle = np.deg2rad(np.random.uniform(0, 360))
                if sample.ndim == 2:
                    rotate_data = self.rotate_data(sample[:, 1:], angle)
                    rotate_data = np.concatenate((sample[:, :1], rotate_data), -1)
                elif sample.ndim == 3:
                    rotate_data = self.rotate_data(sample[:, :, 1:], angle)
                    rotate_data = np.concatenate((sample[:, :, :1], rotate_data), -1)
                samples.append(rotate_data)
            
            for i in range(augmentation+1):
                yield samples[i], action
    

    def sampling_every_data(self):
        for _, data_s in self.data.items():
            for key, seq in data_s.items():
                if self.mode == 'test' and '_m' in key:
                    continue
                # yield seq, self.act2vec.act2vec(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key))
                yield seq, re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key)

 
    def iter_generator(self, step=0, batch_size=1):
        step = step if step>0 else self.t_his
        for data_s in self.data.values():
            traj_list, action_list = [], []
            for key, seq in data_s.items():
                if self.mode == 'test' and '_m' in key:
                    continue
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    traj_list.append(traj)
                    # action_list.append(self.act2vec.act2vec(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key)))
                    action_list.append(re.sub(r'[0-9]+', '', key[:-2] if '_m' in key else key))
                    if len(traj_list) >= batch_size:
                        # _traj, _action = np.concatenate(traj_list, 0), np.concatenate(action_list, 0)
                        _traj, _action = np.concatenate(traj_list, 0), copy.deepcopy(action_list)
                        _traj = np.swapaxes(_traj, -1, -2)
                        _traj = np.squeeze(_traj)
                        traj_list.clear()
                        action_list.clear()
                        yield _traj, _action


    def get_skeleton(self, plt=False):
        return self.skeleton

    def convert_to_position_mode(self, data, **kwargs):
        return data 



class DataSetRotation(DataSet):
    def __init__(self, data_path, mode, t_his, t_pred, actions='all', frame_rate=1, *args, **kwargs):  
        super().__init__(data_path, mode, t_his, t_pred, actions, frame_rate, *args, **kwargs)
        return
    
    def convert_to_position_mode(self, data):
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
     


class DataSetVelocity(object):
    '''
    def __init__(self, *args, **kwargs) -> None:
        self.data_channels = int(self.data_channels / 2)
    '''
    def posprocessing(self, seq):
        resdiuals = seq - np.roll(seq, shift=1, axis=0)
        resdiuals[0, ...] = 0.0    
        seq = np.concatenate((seq, resdiuals), axis=-1)
        return seq 


class DataSetVelocityOnly(object):
    def posprocessing(self, seq):
        resdiuals = seq - np.roll(seq, shift=1, axis=0)
        resdiuals[0, ...] = 0.0    
        return resdiuals


class DataSetVelocityRM(object):
    def posprocessing(self, seq):        
        # compute the residual and add it to postion/rotation
        posf, pref = expmap2rotmat(seq), expmap2rotmat(np.roll(seq, shift=1, axis=0))
        resdiuals = posf @ np.transpose(pref, (0, 1, 3, 2))
        # resdiuals = np.transpose(pref, (0, 1, 3, 2)) @ posf
        resdiuals = rotmat2expmap(resdiuals)
        resdiuals[0, ...] = 0.0
        resdiuals = resdiuals.astype(np.float32)    
        seq = np.concatenate((seq, resdiuals), axis=-1) 
        return seq    


class DataSetVelocityOnlyRM(object):
    def posprocessing(self, seq):
        # compute the residual and add it to postion/rotation
        posf, pref = expmap2rotmat(seq), expmap2rotmat(np.roll(seq, shift=1, axis=0))
        resdiuals = posf @ np.transpose(pref, (0, 1, 3, 2))
        # resdiuals = np.transpose(pref, (0, 1, 3, 2)) @ posf
        resdiuals = rotmat2expmap(resdiuals)
        resdiuals[0, ...] = 0.0
        resdiuals = resdiuals.astype(np.float32)            
        return resdiuals    


    
'''
    Semantics Base Object
'''
class SkeletonSemantics(object):
    def __init__(self, semantics_dict:dict):
        self.semantics_dict = semantics_dict
      
     
    def get_semantic_nodes(self, levels:list, root_level=1):
        self.check_level_dims(root_level)
        
        root = self.semantics_dict
        for idx in range(root_level):
            root = root[levels[idx]]
        
        subs = []
        for _, key in enumerate(levels[root_level:]):
            tmp = root[key]
            if isinstance(tmp, list) is True:
                subs += tmp
            else:
                subs.append(tmp)
            
        univ = []
        def get_all_nodes(container):
            nonlocal univ
            if isinstance(container, dict) is False:
                if isinstance(container, list):
                    univ += container
                else:
                    univ.append(container)
            else:
                for _, value in container.items():
                    get_all_nodes(value)
            return
        get_all_nodes(root)
        
        comp = []
        for x in univ:
            if x not in subs:
                comp.append(x)
        subs_mask = [1 if x in subs else 0 for x in list(np.arange(len(univ)))]
        comp_mask = [1 if x in comp else 0 for x in list(np.arange(len(univ)))]
        
        return subs, comp, subs_mask, comp_mask
    
    
    def get(self, level):
        level = str(level)
        if level not in self.semantics_dict:
            whole_body = []
            for _, val in self.semantics_dict['0'].items():
                whole_body = whole_body + val
            # whole_body = [whole_body]
            return {'whole body': whole_body}
        else:
            return self.semantics_dict[level]


    def next_level(self, level):
        level = int(level)
        return level - 1
      
    def check_level_dims(self, dims):
        raise NotImplementedError    



'''
class ActionDict(object):    
    def __init__(self, actions, mapping_func=None) -> None:
        self.actions = actions
        if mapping_func is None:
            self.mapping_func = lambda action: np.array([0.0])
        else:
            if mapping_func == 'default':
                self._mean = np.random.normal(loc=0.0, scale=1.0, size=len(self.actions))
                self.mapping_func = lambda action: self._mean[self.actions.index(action)], np.array([1.0])
            else:                
                self.mapping_func = mapping_func
        return
    
    def act2vec(self, action):
        return self.mapping_func(action)
    
    @property
    def mean(self):
        return self._mean if hasattr(self, '_mean') else np.array([0.0])
    
    @property
    def stds(self):
        return self._stds if hasattr(self, '_stds') else np.array([1.0]) 
'''


def add_noise(traj, noise_std):
    pos, vel = traj[..., :3], traj[..., 3:]
    noise = np.random.normal(loc=0.0, scale=noise_std, size=pos.shape).astype(np.float32)
    noise_abs = noise * pos
    pos = pos + noise_abs
    if vel.shape[-1] > 0:
        noise_vel = noise_abs - np.roll(noise_abs, shift=1, axis=0)
        noise_vel[0] = noise_abs[0]
        vel = vel + noise_vel
    traj = np.concatenate((pos, vel), axis=-1)
    return traj