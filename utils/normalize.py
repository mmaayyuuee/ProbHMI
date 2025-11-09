import numpy as np
import torch

import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from models import flow_encoder


#############################

        #Deprecated

#############################

class Normalize(object):
    def __init__(self, dataset=None, flow=None, gaussian_path=None, max_min_path=None, temporal_path=None, type="max_min", device='cpu') -> None:
        self.dataset = dataset
        self.flow = flow
        self.gaussian_path = gaussian_path
        self.max_min_path = max_min_path
        self.temporal_path = temporal_path
        self.type = type
        self.device = device
        return
    
    
    def normalize(self, data):
        if self.type == "max_min":
            return self.normalize_using_max_min(data)
        elif self.type == "gaussian":
            return self.normalize_using_gaussian(data)
        elif self.type == "temporal":
            return self.temporal_normalize(data)
        elif self.type == "temporal_over_dataset":
            return self.temporal_norm_over_dataset(data)
    
    
    def unnormalize(self, data):
        if self.type == "max_min":
            return self.unnormalize_using_max_min(data)
        elif self.type == "gaussian":
            return self.unnormalize_using_gaussian(data)
        elif self.type == "temporal":
            return self.temporal_unnormalize(data)
        elif self.type == "temporal_over_dataset":
            return self.temporal_unnorm_over_dataset(data)
    
    
    def normalize_using_max_min(self, data):
        if os.path.exists(self.max_min_path) is False:      
            data_generator = self.dataset.sampling_every_data()
            with torch.no_grad():
                MAX, MIN = None, None
                for data in data_generator:
                    data = torch.tensor(data, device=self.device)
                    data = torch.swapaxes(data, 1, 2)
                    emd, _ = flow_encoder(self.flow, data, None)
                    tmp_max, _ = torch.max(emd, 0) 
                    tmp_min, _ = torch.min(emd, 0)
                    MAX = tmp_max if MAX is None else torch.maximum(MAX, tmp_max)
                    MIN = tmp_min if MIN is None else torch.minimum(MIN, tmp_min)
                self.max = MAX
                self.min = MIN
            np.savez(file=self.max_min_path, max=MAX.to('cpu').numpy(), min=MIN.to('cpu').numpy())
            
        if hasattr(self, 'max') is False or hasattr(self, 'min') is False:
            dptr = np.load(self.max_min_path)
            self.max = torch.tensor(dptr['max'], device=self.device, requires_grad=False)
            self.min = torch.tensor(dptr['min'], device=self.device, requires_grad=False)
        
        return torch.divide( torch.sub(data, self.min), torch.sub(self.max, self.min) )
    
    
    def unnormalize_using_max_min(self, data):
        try:
            if hasattr(self, 'max') is False or hasattr(self, 'min') is False:
                dptr = np.load(self.max_min_path)
                self.max = torch.tensor(dptr['max'], device=self.device, requires_grad=False)
                self.min = torch.tensor(dptr['min'], device=self.device, requires_grad=False)
            return torch.mul(data, self.max-self.min) + self.min
        
        except OSError:
            print("ERROR: 所需文件无法打开！")
            return data
    
    
    def normalize_using_gaussian(self, data):
        if os.path.exists(self.gaussian_path) is False:      
            data_generator = self.dataset.sampling_every_data()
            with torch.no_grad():
                data_list = []
                for data in data_generator:
                    data = torch.tensor(data, device=self.device)
                    data = torch.swapaxes(data, 1, 2)
                    emd, _ = flow_encoder(self.flow, data, None)
                    data_list.append(emd)
                total_data = torch.cat(data_list, 0)
                std, mean = torch.std_mean(total_data, 0)
                np.savez(file=self.gaussian_path, std=std.to('cpu').numpy(), mean=mean.to('cpu').numpy())
            
        if hasattr(self, 'mean') is False or hasattr(self, 'std') is False:
            dptr = np.load(self.gaussian_path)
            self.mean = torch.tensor(dptr['mean'], device=self.device, requires_grad=False)
            self.std = torch.tensor(dptr['std'], device=self.device, requires_grad=False)
        return torch.divide((data-self.mean), self.std)
    
    
    def unnormalize_using_gaussian(self, data):
        try:
            if hasattr(self, 'mean') is False or hasattr(self, 'std') is False:
                dptr = np.load(self.gaussian_path)
                self.mean = torch.tensor(dptr['mean'], device=self.device, requires_grad=False)
                self.std = torch.tensor(dptr['std'], device=self.device, requires_grad=False)
            return torch.mul(data, self.std) + self.mean
        
        except OSError:
            print("ERROR: 所需文件无法打开！")
            return data
    
    
    def temporal_normalize(self, x):
        self.x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - self.x_mean
        self.x_std=torch.std(x, dim=1, keepdim=True)+ 1e-5
        x = x / self.x_std
        return x
    
    
    def temporal_unnormalize(self, x):
        x = x * self.x_std + self.x_mean
        return x
    
    
    def temporal_norm_over_dataset(self, x):
        if not os.path.exists(self.temporal_path):      
            data_generator = self.dataset.sampling_every_data()
            with torch.no_grad():
                data_list = []
                for data in data_generator:
                    data = torch.tensor(data, device=self.device)
                    data = data[None, ...]
                    data = torch.swapaxes(data, 1, 2)
                    data_list.append(data)
                total_data = torch.cat(data_list, 0)
                std, mean = torch.std_mean(total_data, 0)
                np.savez(file=self.temporal_path, std=std.to('cpu').numpy(), mean=mean.to('cpu').numpy())
            
        if hasattr(self, 'mean') is False or hasattr(self, 'std') is False:
            dptr = np.load(self.temporal_path)
            self.mean = torch.tensor(dptr['mean'], device=self.device, requires_grad=False)
            self.std = torch.tensor(dptr['std'], device=self.device, requires_grad=False)
        return torch.divide((x-self.mean), self.std)
    
    
    def temporal_unnorm_over_dataset(self, x):
        try:
            if hasattr(self, 'mean') is False or hasattr(self, 'std') is False:
                dptr = np.load(self.temporal_path)
                self.mean = torch.tensor(dptr['mean'], device=self.device, requires_grad=False)
                self.std = torch.tensor(dptr['std'], device=self.device, requires_grad=False)
            return torch.mul(x, self.std) + self.mean
        except OSError:
            print("ERROR: 所需文件无法打开！")
            return x



if __name__ == "__main__":
    from datasets.CMU_Mocap.dataset_cmu import CMUMocapPos
    
    cmup = CMUMocapPos(data_path="data/data_3d_CMU.npz", mode="train", t_his=0, t_pred=1)
    norm = Normalize(cmup, type="temporal_over_dataset", temporal_path=os.path.join(os.getcwd(), 'data/CMU_Mocap/CMU_Temporal_Norm.npz'))
    a = norm.normalize(data=torch.zeros((100, 3, 25)))
    b = norm.unnormalize(data=a)
    print()