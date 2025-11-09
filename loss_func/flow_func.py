import torch
import numpy as np
import math
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from loss_func.base_func import BaseLoss


def maximum_likelihood_estimation(nll):
    return torch.mean(nll)



def gaussian_distrib_sampling_RC2_wrapper(func="mse"):
    def gaussian_distrib_sampling_RC2(model, x, z, global_step, cur_step, sample_nums=1):
        std_candidates = (1e-1, 5e-1, 1, 1, 1)
        std_idx = math.floor((cur_step*1.0 / global_step) * len(std_candidates))
        std = std_candidates[std_idx]

        recon_loss = 0
        for _ in range(0, sample_nums):
            _z = torch.reshape(z, shape=(-1, z.shape[-2], z.shape[-1]))
            _z = _z + std*torch.normal(mean=torch.zeros_like(z), std=torch.ones_like(z))
            _x = model(input=_z, reverse=True)
            if func == "mse":
                tmp = torch.mean(torch.sum(torch.pow(torch.subtract(x, _x), 2), dim=[1, 2]))
            elif func == "mae":
                tmp = torch.mean(torch.sum(torch.abs(torch.subtract(x, _x)), dim=[1, 2]))
            recon_loss += tmp
        recon_loss = recon_loss / sample_nums
        return recon_loss
    return gaussian_distrib_sampling_RC2

gaussian_distrib_sampling_RC2_mse = gaussian_distrib_sampling_RC2_wrapper(func="mse")
gaussian_distrib_sampling_RC2_mae = gaussian_distrib_sampling_RC2_wrapper(func="mae")
        

def linear_path_wrapper(func="mse"):
    def linear_path(z, indice):
        if indice is None:
            return 0.0
        
        _indice = indice - indice[0]
        coef2 = torch.reshape(_indice, shape=(-1, 1, 1, 1)).broadcast_to(z.shape) * (1.0/_indice[-1])
        coef1 = torch.flip(coef2, dims=[0])
        
        _z = coef1*z[0] + coef2*z[-1]
        if func == "mse":
            lp_loss = torch.sum(torch.pow(torch.subtract(z, _z), 2), dim=(0, 2, 3))
        elif func == "mae":
            lp_loss = torch.sum(torch.abs(torch.subtract(z, _z)), dim=(0, 2, 3))
        lp_loss = torch.mean(lp_loss)
        return lp_loss
    return linear_path

linear_path_mse = linear_path_wrapper(func="mse")
linear_path_mae = linear_path_wrapper(func="mae")



'''
以下函数仅用在包含速度项的情况
'''
def latent_consistency_wrapper(func="mse"):
    # z_shape = (length, batch_size, channels, nodes)
    def latent_consistency(z):
        z_p, z_v = torch.split(z, int(z.shape[-2]/2), dim=-2)
        z_v_sl = torch.roll(z_v, shifts=-1, dims=0)
        _z_p = z_p + z_v_sl
        if func == "mse":
            z_loss = torch.mean(torch.pow(torch.subtract(z_p, _z_p), 2))
        elif func == "mae":
            z_loss = torch.mean(torch.abs(torch.subtract(z_p, _z_p)))
        return z_loss
    return latent_consistency

latent_consistency_mse = latent_consistency_wrapper(func="mse")
latent_consistency_mae = latent_consistency_wrapper(func="mae")


def position_consistency_wrapper(func="mse"):
    def position_consistency(model, x, z):
        z_p, z_v = torch.split(z, int(z.shape[-2]/2), dim=-2)
        z_v_sl = torch.roll(z_v, shifts=-1, dims=0)
        _z_p = z_p + z_v_sl
        _z = torch.concat((_z_p, z_v), dim=-2)
        _x = model(input=_z, reverse=True)
        if func == "mse":
            x_loss = torch.mean(torch.pow(torch.subtract(x, _x), 2))
        elif func == "mae":
            x_loss = torch.mean(torch.abs(torch.subtract(x, _x)))
        return x_loss   
    return position_consistency

position_consistency_mse = position_consistency_wrapper(func="mse")
position_consistency_mae = position_consistency_wrapper(func="mae")



def expmap_mse_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.sum(torch.pow(diff, 2), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def expmap_mae_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.sum(torch.abs(diff), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss



class TripletLossMatrix(object):
    def __init__(self, margin=1.0, p=2):
        super(TripletLossMatrix, self).__init__()
        self.margin, self.p = margin, p
        return 
    
    def triplet_loss(self, embeddings, labels):
        """
        参数:
            embeddings: 所有样本的嵌入向量 [batch_size, channel, node_n]
            labels: 每个样本的标签 [batch_size]
        """
        
        if embeddings.ndim == 4:
            embeddings = embeddings[0]
        embeddings = torch.reshape(embeddings, shape=(embeddings.shape[0], -1))
        labels = torch.from_numpy(labels).to(embeddings.device)
        
        # 计算所有样本之间的成对距离矩阵
        pairwise_dist = torch.cdist(embeddings, embeddings, p=self.p)
        
        # 创建掩码矩阵，标识哪些样本对是同一类
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_negative = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        # 对于每个锚点，找到最难的正样本和负样本
        hardest_positive = (pairwise_dist * mask_positive.float()).max(dim=1)[0]
        hardest_negative = (pairwise_dist + 1e6 * (~mask_negative).float()).min(dim=1)[0]
        
        # 计算triplet loss
        loss = torch.clamp(hardest_positive - hardest_negative + self.margin, min=0.0)
        loss = loss.sum(dim=-1).mean()
        return loss

        


class FlowLoss(BaseLoss):
    loss_func_dict = {
        "likelihood_space":{
            "MLE": maximum_likelihood_estimation         
        },
        "euler_space":{
            "mse": expmap_mse_loss,
            "mae": expmap_mae_loss
        },
        "pose_space":{
            "gaussian_distrib_sampling_RC2_mse": gaussian_distrib_sampling_RC2_mse,
            "gaussian_distrib_sampling_RC2_mae": gaussian_distrib_sampling_RC2_mae
        },
        "latent_linear_space":{
            "linear_path_mse": linear_path_mse,
            "linear_path_mae": linear_path_mae  
        },
        "velocity_latent_sapce":{
            "latent_consistency_mse": latent_consistency_mse,
            "latent_consistency_mae": latent_consistency_mae
        },
        "velocity_position_sapce":{
            "position_consistency_mse": position_consistency_mse,
            "position_consistency_mae": position_consistency_mae
        },
        "contrastive_based_loss":{
            "triplet_loss": None
        }
    }
    
    def __init__(self, cmds:dict, **kwargs) -> None:
        if "contrastive_based_loss" in cmds:
            self.contrastive_based_loss = TripletLossMatrix(**kwargs)
            self.loss_func_dict["contrastive_based_loss"]["triplet_loss"] = self.contrastive_based_loss.triplet_loss
        super(FlowLoss, self).__init__(cmds)
        return