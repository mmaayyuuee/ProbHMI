import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor as Tensor
# from geomstats.geometry.special_orthogonal import SpecialOrthogonal
# import geomstats.backend as gs  # type: ignore
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


class FlowPathBase(object):
    def __init__(self, schedule, noise_std, hz=50, **kwargs):
        self.schedule, self.noise_std = schedule, noise_std
        self.hz = hz
        return
    
    def sample(self, x_0:Tensor, x_1:Tensor, t:Tensor):
        raise NotImplementedError
    
    def get_hz(self):
        return self.hz


class VelocityPath(FlowPathBase):
    pass

class MixVelocityPath(FlowPathBase):
    pass

class LatentVelocityPath(FlowPathBase):
    pass



'''  #### 弃用版本
class VelocityPath(FlowPathBase):
    velocity_dict = {
        "linear_euclidean": lambda self,x_seq,t,unit_vel: self.linear_euclidean_velocity(x_seq, t, unit_vel),
        "linear_euclidean_v2": lambda self,x_seq,t,unit_vel: self.linear_euclidean_velocity_v2(x_seq, t, unit_vel),
        "continous_linear_euclidean": lambda self,x_seq,t,unit_vel,random_flag: self.continous_linear_euclidean_velocity(x_seq, t, unit_vel, random_flag),
        "linear_SO3": lambda self,x_seq,t,unit_vel: self.linear_SO3_velocity(x_seq, t, unit_vel),
        "linear_SO3_v2": lambda self,x_seq,t,unit_vel: self.linear_SO3_velocity_v2(x_seq, t, unit_vel),
        "progressive_linear_euclidean": lambda self,x_seq,t,unit_vel: self.progressive_linear_euclidean_velocity(x_seq, t, unit_vel),
        "linear_euclidean_w_vel": lambda self,x_seq,t,unit_vel: self.linear_euclidean_velocity_with_vel(x_seq, t, unit_vel),
    }
    SO3 = SpecialOrthogonal(n=3, point_type="vector", equip=True)
            
    def sample(self, x_0:Tensor, x_1:Tensor, t:Tensor, unit_vel=True, random_flag=True, **kwargs):
        dx_t = self.velocity_dict[self.schedule](self, x_1, t, unit_vel, random_flag)
        return dx_t
    
    
    def linear_euclidean_velocity(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        dx_t = x_t_plus - x_t
        if unit_vel:
            dx_t = dx_t * self.hz
        # epsilon = torch.normal(mean=torch.zeros_like(dx_t), std=self.noise_std)
        # return x_t, dx_t+epsilon
        epsilon = torch.normal(mean=torch.zeros_like(x_t), std=self.noise_std)
        return x_t+epsilon, dx_t
    
    
    def linear_euclidean_velocity_v2(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        dx_t = x_t_plus - x_t
        epsilon = torch.normal(mean=torch.zeros_like(dx_t), std=self.noise_std)
        dx_t_ep = dx_t + epsilon
        if unit_vel:
            dx_t = dx_t * self.hz
        return x_t, dx_t_ep


    def continous_linear_euclidean_velocity(self, x_seq:Tensor, t:Tensor, unit_vel=True, random_flag=True):
        x_t_loc = t.floor().type(torch.int64)
        x_t, x_t_plus = x_seq[torch.arange(x_seq.size(0)), x_t_loc], x_seq[torch.arange(x_seq.size(0)), x_t_loc+1]
        x_t, x_t_plus = x_t[:, None], x_t_plus[:, None]
        dx_t = x_t_plus - x_t
        
        resi_t = (t - x_t_loc)[:, None, None, None].expand_as(dx_t)
        x_t = resi_t * dx_t + x_t
        
        if unit_vel:
            dx_t = dx_t * self.hz
        
        if random_flag:
            epsilon = torch.normal(mean=torch.zeros_like(x_t), std=self.noise_std)
            return x_t+epsilon, dx_t
        else:
            return x_t, dx_t
    
    
    def linear_SO3_velocity(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        x_t, x_t_plus = x_t.permute(0, 1, 3, 2), x_t_plus.permute(0, 1, 3, 2)
                
        size = x_t.shape
        f_t, f_t_plus = torch.reshape(x_t, shape=(-1, size[-1])), torch.reshape(x_t_plus, shape=(-1, size[-1]))
        df_t = VelocityPath.SO3.compose(f_t_plus.to('cpu'), (-f_t).to('cpu'))
        if unit_vel:
            df_t = df_t * self.hz
        
        epsilon = torch.normal(mean=torch.zeros_like(df_t), std=self.noise_std) 
        epsilon_so3 = VelocityPath.SO3.exp(epsilon)
        dx_t = VelocityPath.SO3.compose(df_t, epsilon_so3)
        dx_t = torch.reshape(dx_t, shape=size).type(torch.float32).to(x_t.device)
        x_t, dx_t = x_t.permute(0, 1, 3, 2), dx_t.permute(0, 1, 3, 2)
        return x_t, dx_t


    def linear_SO3_velocity_v2(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        x_t, x_t_plus = x_t.permute(0, 1, 3, 2), x_t_plus.permute(0, 1, 3, 2)
                
        size = x_t.shape
        f_t, f_t_plus = torch.reshape(x_t, shape=(-1, size[-1])), torch.reshape(x_t_plus, shape=(-1, size[-1]))

        epsilon = torch.normal(mean=torch.zeros_like(f_t_plus), std=self.noise_std) 
        epsilon_so3 = VelocityPath.SO3.exp(epsilon)
        f_t_plus = VelocityPath.SO3.compose(f_t_plus.to('cpu'), epsilon_so3.to('cpu'))        
        
        dx_t = VelocityPath.SO3.compose(f_t_plus.to('cpu'), (-f_t).to('cpu'))
        if unit_vel:
            dx_t = dx_t * self.hz
            
        dx_t = torch.reshape(dx_t, shape=size).type(torch.float32).to(x_t.device)
        x_t, dx_t = x_t.permute(0, 1, 3, 2), dx_t.permute(0, 1, 3, 2)
        return x_t, dx_t


    def progressive_linear_euclidean_velocity(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        dx_t = x_t_plus - x_t
        if unit_vel:
            dx_t = dx_t * self.hz
        ### 临时措施 !!!!!!!!!
        std = (self.noise_std * (1.0 - t/100))[:, None, None, None].expand(size=(-1, dx_t.shape[1], dx_t.shape[2], dx_t.shape[3]))
        epsilon = torch.normal(mean=torch.zeros_like(dx_t), std=std)
        return x_t, dx_t+epsilon


    def _retrieval_sample(self, x_seq:Tensor, t:Tensor):
        x_t_loc = t.floor().type(torch.int64)
        x_t, x_t_plus = x_seq[torch.arange(x_seq.size(0)), x_t_loc], x_seq[torch.arange(x_seq.size(0)), x_t_loc+1]
        x_t, x_t_plus = x_t[:, None], x_t_plus[:, None]
        return x_t, x_t_plus     
    
    
    def linear_euclidean_velocity_with_vel(self, x_seq:Tensor, t:Tensor, unit_vel=True):
        x_t_loc = t.floor().type(torch.int64)
        
        x_t_minus = x_seq[torch.arange(x_seq.size(0)), x_t_loc]
        x_t = x_seq[torch.arange(x_seq.size(0)), x_t_loc+1]
        x_t_plus = x_seq[torch.arange(x_seq.size(0)), x_t_loc+2]
        
        x_t_minus, x_t, x_t_plus = x_t_minus[:, None], x_t[:, None], x_t_plus[:, None]
        
        dx_t_minus, dx_t = x_t-x_t_minus, x_t_plus-x_t
        if unit_vel:
            dx_t = dx_t * self.hz
            dx_t_minus = dx_t_minus * self.hz
            
        epsilon = torch.normal(mean=torch.zeros_like(dx_t), std=self.noise_std)
        return x_t, dx_t+epsilon, dx_t_minus  



class LatentVelocityPath(FlowPathBase):
    velocity_dict = {
        "rectified": lambda self,x_0,x_1,t: self.rectified(x_0, x_1, t),
    }

    def sample(self, x_0:Tensor, x_1:Tensor, t:Tensor, **kwargs):
        ## x_0=None / x_1:(B, T, C, N)
        dx_t = self.velocity_dict[self.schedule](self, x_0, x_1, t)
        return dx_t

    def rectified(self, x_0:Tensor, x_1:Tensor, t:Tensor):
        t = t.view(-1, 1, 1, 1).expand_as(x_1)
        x_t = (1.0-t)*x_0 + t*x_1
        dx_t = x_1 - x_0
        epsilon = torch.normal(mean=torch.zeros_like(dx_t), std=self.noise_std)
        return x_t+epsilon, dx_t
    


class MixVelocityPath(FlowPathBase):
    velocity_dict = {
        "linear": lambda self,x_seq,t: self.linear_velocity(x_seq, t),
    }    

    def sample(self, x_0:Tensor, x_1:Tensor, t:Tensor):
        ## x_0=None / x_1:(B, T, C, N)
        dx_t = self.velocity_dict[self.schedule](self, x_1, t)
        return dx_t

    def linear_velocity(self, x_seq:Tensor, t:Tensor):
        x_t, x_t_plus = self._retrieval_sample(x_seq, t)
        dx_t = x_t_plus - x_t
        dx_t = dx_t * self.hz
        epsilon = torch.normal(mean=torch.zeros_like(x_t), std=self.noise_std)
        return x_t+epsilon, dx_t

    def _retrieval_sample(self, x_seq:Tensor, t:Tensor):
        x_t_loc = t.floor().type(torch.int64)
        x_t, x_t_plus = x_seq[torch.arange(x_seq.size(0)), x_t_loc], x_seq[torch.arange(x_seq.size(0)), x_t_loc+1]
        x_t, x_t_plus = x_t[:, None], x_t_plus[:, None]
        return x_t, x_t_plus     
'''    
                


if __name__ == "__main__":
    path = VelocityPath(schedule="linear_euclidean")
    x_1 = torch.pow(torch.arange(0, 10).reshape((1, 10, 1, 1)), 2)
    print(x_1)
    t = torch.tensor(0.25)
    dx_t = path.sample(x_0=None, x_1=x_1, t=t)
    print()

