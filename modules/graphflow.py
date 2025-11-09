import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from layers import ActNorm2d, ActNorm2d_plus, ActNorm2d_channel, GraphSplitMatrix 
from layers import GraphConvolution, AdaGraphConv, GraphLinear, Graph1x1Conv, RdnAdjGraphConv
from layers import SplitSqueezeConfig, Squeeze, UnSqueeze
from modules.stgcn import STGCN, DctNet, DctSTGCN, FowardSTGCN
# from modules.trfm_dynamics import TrFmEncoder_V4
from utils.distributions import GaussianDiag, StudentT, GaussianMixture, LearnableGaussianMixture, ClusteringGaussianMixture
from utils.dct import get_dct_matrix, Dct



NetworkModelList = {
    "GraphConvolution": GraphConvolution,
    "AdaGraphConvolution": AdaGraphConv,
    "RdnAdjGraphConvolution": RdnAdjGraphConv,
    "GraphLinear": GraphLinear,
    "Graph1x1Convolution": Graph1x1Conv
}


class CouplingModule(nn.Module):
    def __init__(self, in_channels, node_n, channels_list, network_model):
        super().__init__()
        
        self.in_channels = in_channels
        self.node_n = node_n
        self.channels_list = channels_list
        self.network_model = network_model
        self._build_module()   
        return
    
    def forward(self, input, adj_matrix, d2_z1_adj_matrix):
        z = input
        for i, layer in enumerate(self.layers):
            if (layer.__class__.__name__.find("Graph") >= 0):
                if i == len(self.layers)-1:
                    z = layer(z, adj_matrix)
                    continue
                z = layer(z, d2_z1_adj_matrix)
            else:
                z = layer(z)
        return z

    def _build_module(self):
        self.layers = nn.ModuleList()
        
        _in_channels = self.in_channels
        for channels in self.channels_list:
            _out_channels = channels
            self.layers.append(
                NetworkModelList[self.network_model](_in_channels, _out_channels, self.node_n, bias=False)
            )
            self.layers.append(nn.LeakyReLU())
            _in_channels = _out_channels
        self.layers.append(
            NetworkModelList[self.network_model](_in_channels, _out_channels, self.node_n, bias=False)
        )
        return
   
            
# 一层图卷积后接1*1卷积
class CouplingModule_V2(nn.Module):
    ActivationDict = {
        "LeakyReLU": nn.LeakyReLU(),
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU(),
        "Linear": nn.Identity(),
    } 
    
    def __init__(self, in_channels, out_channels, node_n, channels_list, network_model, 
                 spectral_norm=None, activation='LeakyReLU', last_activation=False, **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_n = node_n
        self.channels_list = channels_list
        self.network_model = network_model
        
        self.spectral_norm = spectral_norm
        self.act_func = self.ActivationDict[activation]
        self.last_activation = last_activation
         
        self.layers = self._build_module()       
        return
    
    def forward(self, input, adj_matrix):
        z = input
        for _, layer in enumerate(self.layers):
            if (layer.__class__.__name__.find("Graph") >= 0):
                z = layer(z, adj_matrix)
            else:
                if input.ndim >= 4:
                    old_shape = input.shape
                    z = torch.reshape(z, shape=(-1, z.shape[-2], z.shape[-1]))
                    z = layer(z)
                    z = torch.reshape(z, shape=list(old_shape[:-2])+list(z.shape[-2:]) )
                else:
                    z = layer(z)
        return z
    
    def _build_module(self):
        layers = nn.ModuleList()
        _in_channels = self.in_channels
        for idx, channels in enumerate(self.channels_list):
            _out_channels = channels
            if idx == 0:
                layer = NetworkModelList[self.network_model](_in_channels, _out_channels, self.node_n, bias=False)
                layer = layer if not self.spectral_norm else torch.nn.utils.parametrizations.spectral_norm(layer)
                layers.append(layer)
            else:
                layer = torch.nn.Conv1d(_in_channels, _out_channels, 1, 1, bias=False)
                layer = layer if not self.spectral_norm else torch.nn.utils.parametrizations.spectral_norm(layer)
                layers.append(layer)
            # self.layers.append(nn.LeakyReLU())
            layers.append(self.act_func)
            _in_channels = _out_channels

        layer = torch.nn.Conv1d(_in_channels, self.out_channels, 1, 1, bias=False)
        layer = layer if not self.spectral_norm else torch.nn.utils.parametrizations.spectral_norm(layer)
        layers.append(layer)
        if self.last_activation:
            layers.append(self.act_func)
        return layers



class CouplingModule_V3(CouplingModule_V2):
    def __init__(self, in_channels, out_channels, node_n, channels_list, network_model, 
                 spectral_norm=None, activation='LeakyReLU', last_activation=False, 
                 expert_nums=1, top_k=1, router_dim=1.0, **kwargs):
        super(CouplingModule_V3, self).__init__(in_channels, out_channels, node_n, channels_list, network_model, \
                                                spectral_norm, activation, last_activation, **kwargs) 
        self.expert_nums = expert_nums   
        self.top_k = top_k 
        self.router_dim = router_dim
        self._build_model()
        return
    
    
    def _build_model(self):
        input_dim = self.in_channels*self.node_n
        self.router = nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.GELU(),
                        nn.Linear(input_dim, self.expert_nums)
                    )
        self.experts = nn.ModuleList([self._build_module() for _ in range(self.expert_nums)])
        return


    def forward(self, input, adj_matrix):
        z = input

        z_gate = torch.reshape(z, shape=(z.shape[0], -1))
        router_outputs = F.softmax(self.router(z_gate), dim=-1)
        top_k_gate_values, top_k_indices = torch.topk(router_outputs, self.top_k, dim=1)
        top_k_gate_values = top_k_gate_values / top_k_gate_values.sum(dim=1, keepdim=True)

        output = torch.zeros(size=(z.shape[0], self.out_channels, self.node_n), device=z.device)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=1)
            if mask.any():
                expert_input = z[mask]  
                expert_output = self.forward_once(expert_input, adj_matrix, expert)           
                expert_output = expert_output.reshape(expert_output.shape[0], self.out_channels, self.node_n)
                
                gate_values = top_k_gate_values[mask]
                expert_indices_in_top_k = (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]
                selected_gate_values = gate_values.gather(1, expert_indices_in_top_k.unsqueeze(1))
                selected_gate_values = (selected_gate_values[:, :, None]).expand(size=(-1, self.out_channels, self.node_n))   
                
                output[mask] += selected_gate_values * expert_output 
        return output
        
    
    def forward_once(self, input, adj_matrix, layers):
        z = input
        for _, layer in enumerate(layers):
            if (layer.__class__.__name__.find("Graph") >= 0):
                z = layer(z, adj_matrix)
            else:
                if input.ndim >= 4:
                    old_shape = input.shape
                    z = torch.reshape(z, shape=(-1, z.shape[-2], z.shape[-1]))
                    z = layer(z)
                    z = torch.reshape(z, shape=list(old_shape[:-2])+list(z.shape[-2:]) )
                else:
                    z = layer(z)
        return z



class CouplingModule_V4(CouplingModule_V3):
    def _build_model(self):
        input_dim = self.in_channels*self.node_n
        if self.router_dim > 0.0:
            self.router = nn.Sequential(
                            nn.Linear(input_dim, int(self.router_dim*input_dim)),
                            nn.GELU(),
                            nn.Linear(int(self.router_dim*input_dim), self.expert_nums)
                        )
        else:
            self.router = nn.Sequential(
                            nn.Linear(int(input_dim), self.expert_nums)
                        )
            
        self.experts = nn.ModuleList()
        _in_channels, _out_channels = self.in_channels, self.channels_list[0]
        for _ in range(self.expert_nums):
            temp_expert = nn.Sequential(
                NetworkModelList[self.network_model](_in_channels, _out_channels, self.node_n, bias=False),
                self.act_func
            )
            self.experts.append(temp_expert)
            
        self.layers = nn.ModuleList()
        _in_channels = self.channels_list[0]
        for _, channels in enumerate(self.channels_list[1:]):
            _out_channels = channels
            layer = torch.nn.Conv1d(_in_channels, _out_channels, 1, 1, bias=False)
            layer = layer if not self.spectral_norm else torch.nn.utils.parametrizations.spectral_norm(layer)
            self.layers.append(layer)
            self.layers.append(self.act_func)
            _in_channels = _out_channels

        layer = torch.nn.Conv1d(_in_channels, self.out_channels, 1, 1, bias=False)
        layer = layer if not self.spectral_norm else torch.nn.utils.parametrizations.spectral_norm(layer)
        self.layers.append(layer)
        if self.last_activation:
            self.layers.append(self.act_func)
        return


    def forward(self, input, adj_matrix):
        z = input

        z_gate = torch.reshape(z, shape=(z.shape[0], -1))
        router_outputs = F.softmax(self.router(z_gate), dim=-1)
        top_k_gate_values, top_k_indices = torch.topk(router_outputs, self.top_k, dim=1)
        top_k_gate_values = top_k_gate_values / top_k_gate_values.sum(dim=1, keepdim=True)

        output = torch.zeros(size=(z.shape[0], self.channels_list[0], self.node_n), device=z.device)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=1)
            if mask.any():
                expert_input = z[mask]  
                expert_output = self.forward_once(expert_input, adj_matrix, expert)         
                  
                # 动态获取输出维度
                gate_values = top_k_gate_values[mask]
                expert_indices = (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]
                selected_gate_values = gate_values.gather(1, expert_indices.unsqueeze(1))
                
                # 自动扩展门控值到正确维度
                expand_dims = [1] * (expert_output.dim() - 1)
                selected_gate_values = selected_gate_values.view(-1, *expand_dims)
                
                output[mask] += selected_gate_values * expert_output 
        output = self.forward_once(output, adj_matrix, self.layers)
        return output



class CouplingModule_V5(CouplingModule_V4):     
    def forward(self, input, adj_matrix):
        output_ = super(CouplingModule_V5, self).forward(input, adj_matrix)
        input_ = torch.matmul(input, adj_matrix)
        output = output_ + input_
        return output



class GraphFlowStepBase(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, split_matrix,
                actnorm_scale = 1.0,
                flow_coupling = "additive",
                network_model = "GraphConvolution",
                intermediate_channels = [32, 64, 32],
                spectral_norm = False,
                couple_activation = 'LeakyReLU', couple_last_activation = False,
                actnorm_activations = ('Default', 'Default'), actnorm_type = 'Default',
                f_frozen = False,
                device = "cpu",
                **kwarg):
        super().__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.adj_matrix = adj_matrix
        self.node_n = adj_matrix.shape[0]
        self.split_matrix = split_matrix
        self.actnorm_scale = actnorm_scale
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        self.intermediate_channels = intermediate_channels
        self.spectral_norm = spectral_norm
        
        self.couple_activation = couple_activation
        self.couple_last_activation = couple_last_activation
        self.actnorm_activations = actnorm_activations
        self.actnorm_type = actnorm_type
        
        self.device = device
        
        self._build_flow_step()
        
        self.f_frozen = f_frozen
        if f_frozen:
            self._freeze_f_function()
        return
    
    def _build_flow_step(self):
        raise NotImplementedError

    def _freeze_f_function(self):
        for param in self.f.parameters():
            param.requires_grad = False
        return
    
    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)
    
    def normal_flow(self, input, logdet):
        z, cond_input = input
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False) if self.actnorm is not None \
                else (z, logdet)
        if self.flow_coupling == "additive" or \
           self.flow_coupling == "gcn_moe_additive" or \
           self.flow_coupling == "res_gcn_moe_additive":
            _z = torch.concat((z, cond_input), dim=-2) if cond_input is not None else z
            z = z + self.f(_z, self.z2_adj_matrix)
        elif self.flow_coupling == "affine":
            _z = torch.concat((z, cond_input), dim=-2) if cond_input is not None else z
            h = self.f(_z, self.z2_adj_matrix)
            shift, scale = torch.split(h, self.out_channels, dim=-2)
            '''
            scale_mask = torch.where(scale == 0, 0, 1)
            scale = torch.sigmoid(scale) + 0.5
            scale = scale * scale_mask + torch.ones_like(scale_mask) * (1 - scale_mask)
            '''
            scale = torch.where(scale == 0, 1.0, torch.sigmoid(scale).double()+0.5).float()
            z = scale * z + shift
            logdet = torch.sum(torch.log(scale), dim=[-2, -1]) + logdet
        elif self.flow_coupling == "actnorm_only":
            z = z
        return z, logdet

    def reverse_flow(self, input, logdet):
        z, cond_input = input
        if self.flow_coupling == "additive" or \
           self.flow_coupling == "gcn_moe_additive" or \
           self.flow_coupling == "res_gcn_moe_additive":
            _z = torch.concat((z, cond_input), dim=-2) if cond_input is not None else z
            z = z - self.f(_z, self.z2_adj_matrix)
        elif self.flow_coupling == "affine":
            _z = torch.concat((z, cond_input), dim=-2) if cond_input is not None else z
            h = self.f(_z, self.z2_adj_matrix)
            shift, scale = torch.split(h, self.out_channels, dim=-2)
            '''
            scale_mask = torch.where(scale == 0, 0, 1)
            scale = torch.sigmoid(scale) + 0.5
            scale = scale * scale_mask + torch.ones_like(scale_mask) * (1 - scale_mask)
            '''
            scale = torch.where(scale == 0, 1.0, torch.sigmoid(scale).double()+0.5).float()     
            z = (z - shift) / scale
            logdet = torch.sum(torch.log(scale), dim=[-2, -1]) + logdet
        elif self.flow_coupling == "actnorm_only":
            z = z
            
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True) if self.actnorm is not None \
                else (z, logdet)
        # if torch.any(torch.isnan(z)):
        #     print("ERRORS: Nan appears in GraphFlowStep_V2.reverse_flow !!!")  
        return z, logdet

 
@DeprecationWarning       
class GraphFlowStep_V1(GraphFlowStepBase):    
    def _build_flow_step(self):
        z1_mask, z1_m, z2_mask, z2_m = self.split_matrix
        self.z2_adj_matrix = torch.multiply(z2_mask, self.adj_matrix).to(device)
        
        with torch.no_grad():
            # 邻接矩阵小于3*3时，不存在距离等于2的通路，需要从图卷积退化为1*1卷积（全连接）
            # z1_m本不是作此用，权宜之计
            if self.node_n <= 2:
                self.d2_z1_adj_matrix = z1_m.to(device)
            else:
                self.d2_z1_adj_matrix = torch.multiply(z1_mask, self.adj_matrix).to(device)
                d2_adj_matrix = torch.matmul(self.adj_matrix, self.adj_matrix)
                d2_adj_matrix = torch.where(d2_adj_matrix>=1, 1, 0)
                self.d2_z1_adj_matrix = torch.multiply(z1_mask, d2_adj_matrix).to(device)
        
        self.f = CouplingModule(self.in_channels, self.node_n, self.intermediate_channels, self.network_model)
        self.actnorm = ActNorm2d(self.in_channels, self.actnorm_scale) if self.actnorm_scale > 0.0 else None
        return    
    

class GraphFlowStep_V2(GraphFlowStepBase):    
    def _build_flow_step(self):
        z1_mask, z1_m, z2_mask, z2_m = self.split_matrix
        self.z2_adj_matrix = torch.multiply(z2_mask, self.adj_matrix).to(self.device)

        if self.flow_coupling == "additive":
            self.f = CouplingModule_V2(self.in_channels, self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       spectral_norm = self.spectral_norm,
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation)
        elif self.flow_coupling == "affine":
            self.f = CouplingModule_V2(self.in_channels, 2*self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       spectral_norm = self.spectral_norm,
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation) 
        elif self.flow_coupling == "actnorm_only":
            self.f = None  
        else:
            raise NotImplementedError

        if self.actnorm_scale > 0.0:
            if self.actnorm_type == 'Default':
                self.actnorm = ActNorm2d(self.node_n, self.actnorm_scale, self.actnorm_activations)
            elif self.actnorm_type == 'Channel':
                self.actnorm = ActNorm2d_channel(self.in_channels, self.actnorm_scale, self.actnorm_activations)
            elif self.actnorm_type == 'Plus':
                self.actnorm = ActNorm2d_plus([self.in_channels, self.node_n], self.actnorm_scale, self.actnorm_activations)
        else: 
            self.actnorm = None
        return


class GraphFlowStep_V3(GraphFlowStepBase):    
    def __init__(self, in_channels, out_channels, adj_matrix, split_matrix,
                actnorm_scale = 1.0,
                flow_coupling = "additive",
                network_model = "GraphConvolution",
                intermediate_channels = [32, 64, 32],
                spectral_norm = False,
                couple_activation = 'LeakyReLU', couple_last_activation = False,
                actnorm_activations = ('Default', 'Default'), actnorm_type = 'Default',
                f_frozen = False,
                device = "cpu",
                expert_nums = 1, top_k = 1, router_dim=1.0, **kwargs):
        self.expert_nums, self.top_k, self.router_dim = expert_nums, top_k, router_dim
        super(GraphFlowStep_V3, self).__init__(in_channels, out_channels, adj_matrix, split_matrix, \
                                        actnorm_scale, flow_coupling, network_model, intermediate_channels, \
                                        spectral_norm, couple_activation, couple_last_activation, \
                                        actnorm_activations, actnorm_type, f_frozen, device)
        return

    def _build_flow_step(self):
        z1_mask, z1_m, z2_mask, z2_m = self.split_matrix
        self.z2_adj_matrix = torch.multiply(z2_mask, self.adj_matrix).to(self.device)

        if self.flow_coupling == "additive":
            self.f = CouplingModule_V3(self.in_channels, self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       spectral_norm = self.spectral_norm,
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation,
                                       expert_nums = self.expert_nums, 
                                       top_k = self.top_k,
                                       router_dim = self.router_dim) 
        elif self.flow_coupling == "gcn_moe_additive":
            self.f = CouplingModule_V4(self.in_channels, self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       spectral_norm = self.spectral_norm,
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation,
                                       expert_nums = self.expert_nums, 
                                       top_k = self.top_k,
                                       router_dim = self.router_dim)  
        elif self.flow_coupling == "res_gcn_moe_additive":    
            self.f = CouplingModule_V5(self.in_channels, self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       spectral_norm = self.spectral_norm,
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation,
                                       expert_nums = self.expert_nums, 
                                       top_k = self.top_k,
                                       router_dim = self.router_dim)       
        elif self.flow_coupling == "actnorm_only":
            self.f = None  
        else:
            raise NotImplementedError
        
        if self.actnorm_scale > 0.0:
            if self.actnorm_type == 'Default':
                self.actnorm = ActNorm2d(self.node_n, self.actnorm_scale, self.actnorm_activations)
            elif self.actnorm_type == 'Channel':
                self.actnorm = ActNorm2d_channel(self.in_channels, self.actnorm_scale, self.actnorm_activations)
            elif self.actnorm_type == 'Plus':
                self.actnorm = ActNorm2d_plus([self.in_channels, self.node_n], self.actnorm_scale, self.actnorm_activations)
        else: 
            self.actnorm = None
        return



class ChannelFlowStepBase(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, keep_upper,
                actnorm_scale = 1.0,
                flow_coupling = "additive",
                network_model = "GraphConvolution",
                intermediate_channels = [32, 64, 32],
                couple_activation = 'LeakyReLU', couple_last_activation = False,
                actnorm_activations = ('Default', 'Default'), actnorm_type = 'Channel',
                f_frozen = False,
                device = "cpu",
                **kwarg):
        super().__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.adj_matrix = adj_matrix.to(device) * 1.0
        self.keep_upper = keep_upper
        self.node_n = adj_matrix.shape[0]
        self.actnorm_scale = actnorm_scale
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        self.intermediate_channels = intermediate_channels
        
        self.couple_activation = couple_activation
        self.couple_last_activation = couple_last_activation
        self.actnorm_activations = actnorm_activations
        self.actnorm_type = actnorm_type
        
        self.device = device
        
        self._build_flow_step()
        
        self.f_frozen = f_frozen
        if f_frozen:
            self._freeze_f_function()
        return
    
    def _build_flow_step(self):
        raise NotImplementedError

    def _freeze_f_function(self):
        for param in self.f.parameters():
            param.requires_grad = False
        return
    
    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)
    
    
    def normal_flow(self, input, logdet):
        z, cond_input = input
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False) if self.actnorm is not None \
                else (z, logdet)
                
        if self.flow_coupling == "additive" or \
           self.flow_coupling == "gcn_moe_additive" or \
           self.flow_coupling == "res_gcn_moe_additive":
            z_1, z_2 = torch.split(z, math.floor(z.shape[-2]/2), dim=-2)
            if self.keep_upper:
                z_c = torch.concat((z_1, cond_input), dim=-2) if cond_input is not None else z_1
                _z_2 = z_2 + self.f(z_c, self.adj_matrix)
                z = torch.concat((z_1, _z_2), dim=-2)
            else:
                z_c = torch.concat((z_2, cond_input), dim=-2) if cond_input is not None else z_2
                _z_1 = z_1 + self.f(z_c, self.adj_matrix)
                z = torch.concat((_z_1, z_2), dim=-2)
            
        elif self.flow_coupling == "affine":
            z_1, z_2 = torch.split(z, math.floor(z.shape[-2]/2), dim=-2)
            if self.keep_upper:
                z_c = torch.concat((z_1, cond_input), dim=-2) if cond_input is not None else z_1
            else:
                z_c = torch.concat((z_2, cond_input), dim=-2) if cond_input is not None else z_2
                
            h = self.f(z_c, self.adj_matrix)
            shift, scale = torch.split(h, self.out_channels, dim=-2)
            scale = torch.where(scale == 0, 1.0, torch.sigmoid(scale).double()+0.5).float()
            
            if self.keep_upper:
                _z_2 = scale * z_2 + shift
                z = torch.concat((z_1, _z_2), dim=-2)
            else:   
                _z_1 = scale * z_1 + shift
                z = torch.concat((_z_1, z_2), dim=-2)
            logdet = torch.sum(torch.log(scale), dim=[-2, -1]) + logdet
            
        elif self.flow_coupling == "actnorm_only":
            z = z
        return z, logdet


    def reverse_flow(self, input, logdet):
        z, cond_input = input
        
        if self.flow_coupling == "additive" or \
           self.flow_coupling == "gcn_moe_additive" or \
           self.flow_coupling == "res_gcn_moe_additive":
            z_1, z_2 = torch.split(z, math.floor(z.shape[-2]/2), dim=-2)
            if self.keep_upper:
                z_c = torch.concat((z_1, cond_input), dim=-2) if cond_input is not None else z_1
                _z_2 = z_2 - self.f(z_c, self.adj_matrix)
                z = torch.concat((z_1, _z_2), dim=-2)
            else:
                z_c = torch.concat((z_2, cond_input), dim=-2) if cond_input is not None else z_2
                _z_1 = z_1 - self.f(z_c, self.adj_matrix)
                z = torch.concat((_z_1, z_2), dim=-2)
            
        elif self.flow_coupling == "affine":
            z_1, z_2 = torch.split(z, math.floor(z.shape[-2]/2), dim=-2)
            if self.keep_upper:
                z_c = torch.concat((z_1, cond_input), dim=-2) if cond_input is not None else z_1
            else:
                z_c = torch.concat((z_2, cond_input), dim=-2) if cond_input is not None else z_2
                
            h = self.f(z_c, self.adj_matrix)
            shift, scale = torch.split(h, self.out_channels, dim=-2)
            scale = torch.where(scale == 0, 1.0, torch.sigmoid(scale).double()+0.5).float()
            
            if self.keep_upper:
                _z_2 = (z_2 - shift) / scale
                z = torch.concat((z_1, _z_2), dim=-2)
            else:   
                _z_1 = (z_1 - shift) / scale
                z = torch.concat((_z_1, z_2), dim=-2)
            logdet = torch.sum(torch.log(scale), dim=[-2, -1]) + logdet
                        
        elif self.flow_coupling == "actnorm_only":
            z = z
            
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True) if self.actnorm is not None \
                else (z, logdet)
        return z, logdet


class ChannelFlowStep_V2(ChannelFlowStepBase):
    def _build_flow_step(self):
        if self.flow_coupling == "additive":
            self.f = CouplingModule_V2(self.in_channels, self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation)
        elif self.flow_coupling == "affine":
            self.f = CouplingModule_V2(self.in_channels, 2*self.out_channels, self.node_n, self.intermediate_channels, self.network_model, 
                                       activation = self.couple_activation,
                                       last_activation = self.couple_last_activation) 
        elif self.flow_coupling == "actnorm_only":
            self.f = None  
        else:
            raise NotImplementedError

        if self.actnorm_scale > 0.0:
            if self.actnorm_type == 'Default':
                self.actnorm = ActNorm2d(self.node_n, self.actnorm_scale)
            elif self.actnorm_type == 'Channel':
                self.actnorm = ActNorm2d_channel(self.in_channels*2, self.actnorm_scale)
            elif self.actnorm_type == 'Plus':
                self.actnorm = ActNorm2d_plus([self.in_channels*2, self.node_n], self.actnorm_scale)
            else:
                raise NotImplementedError
        else: 
            self.actnorm = None
        return



class FlowBaseNet(nn.Module):
    GraphFlowStepList = {
        "v1": GraphFlowStep_V1,
        "v2": GraphFlowStep_V2,
        "v3": GraphFlowStep_V3
    }
    ChannelFlowStepList = {
        "v2": ChannelFlowStep_V2,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError 
    
    def set_actnorm_init(self):
        raise NotImplementedError 
    
    def get_layer_params_size(self):
        raise NotImplementedError
    
    def load_distribution_parameters(self):
        raise NotImplementedError 
    
    def data_preprocessing(self):
        raise NotImplementedError 
    
    def data_posprocessing(self):
        raise NotImplementedError
    
    '''
    def data_pre_preprocessing(self, x):
        if hasattr(self, 'cond_model'):
            # x: (length, group_size, batch_size, channel, node_n) or (length/group_size, batch_size, channel, node_n)      
            if x.ndim == 4:
                x = x[None, ...]
                
            cond_his, cond_pred = self.get_condition_length()           
            input, cond_input = x[:, cond_his:], x[:, :cond_his]
                
            if 2 * self.get_channels() == x.shape[-2]:
                _, input = torch.split(input, int(input.shape[-2]/2), dim=-2)
                cond_input, _ = torch.split(cond_input, int(cond_input.shape[-2]/2), dim=-2) if min(cond_input.shape) > 0 \
                                                                                            else (cond_input, _)
                                                                                                                             
            input = self.data_preprocessing(input)
            return (input, cond_input)
        else:
            input = self.data_preprocessing(x)
            return input 
    ''' 



class ConditionFlowInterface(nn.Module):
    ConditionModelList = {
        'stgcn': STGCN,
        'dct': DctNet,
        'dct_stgcn': DctSTGCN,
        'foward_stgcn': FowardSTGCN,
        'None' : None,
    }
    
    def __init__(self, condition_model_type, condition_t_his=0, condition_t_pred=0, *args, **kwargs):
        super().__init__()
        self.condition_model_type = condition_model_type
        self.condition_t_his, self.condition_t_pred = condition_t_his, condition_t_pred
        if self.ConditionModelList[condition_model_type] != None:
            self.cond_model = self.ConditionModelList[condition_model_type](**kwargs)
        return
    
    def forward(self, x):
        x = self.cond_model(x) if hasattr(self, 'cond_model') else x
        return x
    
    def get_condition_length(self):
        return self.condition_t_his, self.condition_t_pred
    
    def get_condition_channels(self):
        if hasattr(self, 'cond_model'):
            if self.condition_model_type == 'dct':
                return self.cond_model.n_pre * self.cond_model.in_channel
            else:
                return self.cond_model.channels[-1][-1]
        else:
            return 0
        


class FlowNet(ConditionFlowInterface, FlowBaseNet):   
    def __init__(self, in_channels, K, depth, imc_list,
                actnorm_scale = 1.0,
                flow_type = "graph",
                flow_step = "v1",
                flow_coupling = "additive",
                flow_coupling_params = {},
                mixture_expert_params = {},
                network_model = "GraphConvolution",
                distribution = "normal",
                device = "cpu",
                t_his = 0,
                t_pred = 1,
                condition_model_type = 'None',
                condition_model_params = {},
                *args,  
                **kwargs):        
        super(FlowNet, self).__init__(condition_model_type, t_his, t_pred, **condition_model_params)
        
        self.layers = nn.ModuleList()
        self.output_shapes = []
        
        self.K = K
        if isinstance(depth, int) is True:
            self.depth = [depth] * self.K
        else:
            self.depth = depth
        
        self.imc_list = imc_list
        self.in_channels = in_channels
        self.device = device
        self.condition_channels = self.get_condition_channels()
        self.t_his, self.t_pred = t_his, t_pred

        if flow_type == "graph" or flow_type == "channel":
            assert kwargs["node_n"] is not None, "numbers of node should not be None." 
            assert kwargs["split"] is not None, "the split type should not be None."
            assert kwargs["squeeze"] is not None, "the squeeze type should not be None."
            assert kwargs["adj_matrix"] is not None, "the adjacency matrix should not be None."
            assert kwargs["restricted"] is not None, "the flag of restriction should not be None."
            
            self.node_n = kwargs["node_n"]
            self.split_squeeze_config = SplitSqueezeConfig(kwargs["split"], kwargs["squeeze"])
            self.adj_matrix = kwargs["adj_matrix"]
            self.restricted = kwargs["restricted"]
            
            if flow_type == "graph":        
                self.flow_step = FlowBaseNet.GraphFlowStepList[flow_step]
            else:
                self.flow_step = FlowBaseNet.ChannelFlowStepList[flow_step]        
        else:
            raise KeyError("flow_type is ERROR, only \'graph\' has been implemented.")
        
        if distribution == "normal":
            self.distribution = GaussianDiag()
        elif distribution == "studentT":
            self.distribution = StudentT()
        elif distribution == "mixture_normal":
            self.distribution = GaussianMixture(component = kwargs["mixture_actions"], 
                                                config = kwargs['mixture_config'], 
                                                initial_coef = kwargs['mixture_initialize_coef'],
                                                channel = self.in_channels,
                                                node_n = self.node_n,
                                                device = self.device) 
        elif distribution == "learnable_mixture_normal":
            self.distribution = LearnableGaussianMixture(component = kwargs["mixture_actions"], 
                                                         config = kwargs['mixture_config'],
                                                         initial_coef = kwargs['mixture_initialize_coef'],
                                                         channel = self.in_channels,
                                                         node_n = self.node_n,
                                                         device = self.device) 
        elif distribution == "clustering_mixture_normal":
            self.distribution = ClusteringGaussianMixture(component = kwargs["mixture_actions"], 
                                                         config = kwargs['mixture_config'],
                                                         initial_coef = kwargs['mixture_initialize_coef'],
                                                         channel = self.in_channels,
                                                         node_n = self.node_n,
                                                         device = self.device) 
        else:
            raise NotImplementedError
                
        for i in range(self.K):
            if flow_type == "graph":
                if self.depth[i] <= 0:
                    continue
                init_invar_nidx, squeeze_list = self.split_squeeze_config.get(i)
                gsm = GraphSplitMatrix(node_n = self.node_n if squeeze_list is None else len(squeeze_list), \
                                       init_invar_nidx = init_invar_nidx)
                
                channels = self.in_channels + self.condition_channels
                channels = channels if squeeze_list is None \
                                       else int(channels * len(squeeze_list[0]))    
                out_channels = self.in_channels if squeeze_list is None \
                                       else int(self.in_channels * len(squeeze_list[0]))
                                                        
                adj_m = self.split_squeeze_config.compute_new_adjacency_matrix_after_squeeze(i, self.adj_matrix, self.restricted)
                
                self.layers.append(
                    Squeeze(squeeze_list, self.adj_matrix, self.device)
                )
                for _ in range(self.depth[i]):
                    self.layers.append(
                        self.flow_step(in_channels = channels,
                                       out_channels = out_channels,
                                       adj_matrix = adj_m,
                                       split_matrix = gsm.get_split_matrix(),
                                       actnorm_scale = actnorm_scale,
                                       flow_coupling = flow_coupling,
                                       network_model = network_model,
                                       intermediate_channels = self.imc_list[i],
                                       device = device,
                                       **flow_coupling_params,
                                       **mixture_expert_params)
                                )
                self.layers.append(
                    UnSqueeze(squeeze_list, self.adj_matrix, self.device)
                )
                
            elif flow_type == "channel":
                if self.depth[i] <= 0:
                    continue

                _, squeeze_list = self.split_squeeze_config.get(i)
                
                in_channels = math.floor(self.in_channels / 2)
                channels = in_channels + self.condition_channels
                channels = channels if squeeze_list is None else int(channels * len(squeeze_list[0]))    
                out_channels = in_channels if squeeze_list is None else int(in_channels * len(squeeze_list[0]))
                                  
                adj_m = self.split_squeeze_config.compute_new_adjacency_matrix_after_squeeze(i, self.adj_matrix, self.restricted)

                self.layers.append(
                    Squeeze(squeeze_list, self.adj_matrix, self.device)
                )
                for jdx in range(self.depth[i]):
                    keep_upper = True if jdx % 2 == 0 else False
                    self.layers.append(
                        self.flow_step(in_channels = channels,
                                       out_channels = out_channels,
                                       adj_matrix = adj_m,
                                       keep_upper = keep_upper,
                                       actnorm_scale = actnorm_scale,
                                       flow_coupling = flow_coupling,
                                       network_model = network_model,
                                       intermediate_channels = self.imc_list[i],
                                       device = device,
                                       **flow_coupling_params,
                                       **mixture_expert_params)
                                )
                self.layers.append(
                    UnSqueeze(squeeze_list, self.adj_matrix, self.device)
                )
                                
            else:
                raise KeyError("flow_type is ERROR, only \'graph\' or \'channel\'  has been implemented.")
        return


    def forward(self, input, logdet=0.0, eps_std=1.0, reverse=False, label=None, z_alpha=1.0):
        '''
        Args:
            input/cond_input: (length, group_size, batch_size, channel, node_n) or
                              (length, batch_size, channel, node_n)
        '''
        if hasattr(self, 'cond_model'):
            input, cond_input = input
            clen, cbs = cond_input.shape[0], cond_input.shape[2]
            
            cond_input = torch.swapaxes(cond_input, 1, 2)
            cond_input = torch.reshape(cond_input, shape=(-1, *(list(cond_input.shape[-3:]))))
            cond_input = ConditionFlowInterface.forward(self, cond_input)
            cond_input = torch.reshape(cond_input, shape=(clen, cbs, *(list(cond_input.shape[-3:]))))
            
            if hasattr(self.cond_model, 'batch_first_order'):
                cond_input = torch.swapaxes(cond_input, 1, 2)
                
            _input = (input, cond_input)
        else:
            _input = (input, None)
            
        if not reverse:
            return self.normal_flow(_input, 0.0, label, z_alpha)
        else:
            return self.reverse_flow(_input, logdet, label, eps_std)
    
    
    def normal_flow(self, x, logdet, label=None, z_alpha=1.0):
        z, cond_input = x
        if z.ndim == 5:
            length, group_size, batch_size, channel, node_n = z.shape
        elif z.ndim == 4:
            length, batch_size, channel, node_n = z.shape
        elif z.ndim == 3:
            length, batch_size, channel, node_n = 1, z.shape
        
        z = torch.reshape(z, (-1, channel, node_n))
        cond_input = torch.reshape(cond_input, (-1, cond_input.shape[-2], cond_input.shape[-1])) \
                    if cond_input is not None else cond_input
        for layer in self.layers:
            output, logdet = layer((z, cond_input), logdet, reverse=False)
            if isinstance(output, (list, tuple)):
                z, cond_input = output
            else:
                z = output
        z = torch.reshape(z, (length, batch_size, channel, node_n))

        logp = self.distribution.logp(z, label)
        if isinstance(logdet, float):
            nll = z_alpha * logp + logdet
        else:        
            nll = z_alpha * logp + logdet.reshape(length, batch_size) if logdet.size() != torch.Size([]) \
                else z_alpha * logp + logdet
        nll = nll / (z.shape[-2] * z.shape[-1])
        nll = -nll
        return z, nll
    
    
    def reverse_flow(self, z, logdet, label=None, eps_std=1.0):
        z, cond_input = z
        if z is None:
            z = self.distribution.sample(self.z_shape, eps_std)

        if z.ndim == 5:
            length, group_size, batch_size, channel, node_n = z.shape
        elif z.ndim == 4:
            length, batch_size, channel, node_n = z.shape
        elif z.ndim == 3:
            length, batch_size, channel, node_n = [1, *list(z.shape)]
            
        x = torch.reshape(z, (-1, channel, node_n))
        cond_input = torch.reshape(cond_input, (-1, cond_input.shape[-2], cond_input.shape[-1])) \
                    if cond_input is not None else cond_input
        
        if isinstance(logdet, float) and logdet == 0.0:             
            nll = self.distribution.logp(x, label=label)
        else:
            nll = 0.0
        
        for _, layer in enumerate(reversed(self.layers)):
            output, logdet = layer((x, cond_input), logdet=logdet, reverse=True)
            if isinstance(output, (list, tuple)):
                x, cond_input = output
            else:
                x = output
        x = torch.reshape(x, (length, batch_size, channel, node_n))
        
        nll = nll + logdet
        nll = nll / (z.shape[-2] * z.shape[-1])
        nll = -nll        
        return x, nll
    
    
    def set_actnorm_init(self, inited=True):
        for _, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited
    
     
    def get_layer_params_size(self, prefix=''):
        named_size = {}
        for name, params in self.named_parameters():
            named_size[prefix+'flow.'+name] = params.size()
        total_params = sum(p.numel() for p in self.parameters())
        return named_size, total_params
    
    
    @staticmethod
    def loss_generative(nll):
        return torch.mean(nll)
    
    
    def data_preprocessing(self, x, keep_channel=False):
        return x
    
    def data_posprocessing(self, x, keep_channel=False):
        return x


    def get_channels(self):
        return self.in_channels
    
    def get_seq_length(self):
        return self.t_his, self.t_pred
    
    
    def delete_model(self):
        self.layers = None


    @property
    def distribution_parameters(self):
        return self.distribution.distrib_parameters

    def load_distribution_parameters(self, parameters_dict):
        return self.distribution.load_distribution_parameters(parameters_dict)



class DctFlowNet(FlowNet):
    dct_flag = True
    
    def __init__(self, in_channels, K, depth, imc_list,
                actnorm_scale = 1.0,
                flow_type = "graph",
                flow_step = "v1",
                flow_coupling = "additive",
                flow_coupling_params = {},
                mixture_expert_params = {},
                network_model = "GraphConvolution",
                distribution = "normal",
                n_len = 125,
                n_pre = 10, 
                device = "cpu",
                t_his = 0,
                t_pred = 1,
                condition_model_type = 'None',
                condition_model_params = {},
                *args,    
                **kwargs):         
        self.n_len = n_len
        self.n_pre = n_pre  # len(n_pre)==1/2 -> time_step/(time_step, channels)
        
        if isinstance(n_pre, int):
            in_channels = in_channels * n_pre
        else:
            in_channels = n_pre[0] * n_pre[1]
                          
        super(DctFlowNet, self).__init__(in_channels, K, depth, imc_list, actnorm_scale,
                flow_type, flow_step, flow_coupling, flow_coupling_params, mixture_expert_params,
                network_model, distribution, device, t_his, t_pred, condition_model_type, 
                condition_model_params, *args, **kwargs)
        
        self.dct = Dct(n_len=self.n_len, n_pre=self.n_pre, device=self.device)  
        return


    def data_preprocessing(self, x, keep_channel=False):
        if x.ndim == 4:
            x = x[None, ...]
        ph, length, batch_size, channel, node_n = x.shape
        
        x_slices = []
        n_len = self.n_len if isinstance(self.n_len, int) else self.n_len[0]
        if length < n_len:
            # idx_pad = list(range(length)) + [length-1]*(self.n_len-length)
            idx_pad = [0]*(n_len-length) + list(range(length))
            x = x[:, idx_pad]
            x_slices.append(x)
            
        elif length > n_len:
            times = math.ceil(1.0 * length / n_len)
            reman = times * n_len - length
            idx_pad = [0]*reman + list(range(length))
            x = x[:, idx_pad]
            for i in range(times):
                x_slices.append(x[:, i*n_len:(i+1)*n_len])
        
        else:
            x_slices.append(x)
        
        x_dcts = []
        for x_s in x_slices:            
            x_dct = self.dct(x_s, 1, False) if isinstance(self.n_pre, int) else self.dct(x_s, (1, 3), False)
            x_dct = torch.swapaxes(x_dct, 1, 2)
            if keep_channel:
                x_dct = torch.reshape(x_dct, (ph, 1, batch_size, -1, channel, node_n))
            else:
                x_dct = torch.reshape(x_dct, (ph, 1, batch_size, -1, node_n))
            x_dcts.append(x_dct)
        y = torch.concat(x_dcts, dim=1)    
        return y
    
    
    def data_posprocessing(self, x, keep_channel=False):
        if keep_channel:
            shape = list(x.shape)
            new_shape = shape[:-3] + [shape[-3] * shape[-2]] + shape[-1:]
            x = torch.reshape(x, new_shape)
        
        if x.ndim == 4:
            samples, batch_size, channel, node_n = x.shape
        elif x.ndim == 3:
            samples, batch_size, channel, node_n = 1, x.shape
            
        if isinstance(self.n_pre, int):
            channel = int(channel / self.n_pre)
            x = torch.reshape(x, shape=(samples, batch_size, self.n_pre, channel, node_n))
            x = self.dct(x, 2, True)
            x = torch.swapaxes(x, 1, 2)
        else:    
            x = torch.reshape(x, shape=(samples, batch_size, self.n_pre[0], self.n_pre[1], node_n))
            x = self.dct(x, (2, 3), True)
            x = torch.swapaxes(x, 1, 2)
             
        x = x if x.ndim == 5 else x[None, ...]
        return x      
    

    
class PairFlowContainer(FlowBaseNet):
    def __init__(self, flows:tuple=(), device="cpu",  same_backward=True, **kwargs):
        super().__init__()
        
        flow_nums = len(flows)
        if flow_nums == 0:
            return None
        elif flow_nums == 1:
            self.flow = flows[0]
            if same_backward:
                self.reverse_flow = self.flow
            else:
                self.reverse_flow = copy.deepcopy(self.flow)
                # print(self.flow == self.reverse_flow)
        elif flow_nums >= 2:
            self.flow = flows[0]
            self.reverse_flow = flows[1]
        
        self.flow_nums = flow_nums
        self.same_backward = same_backward
        self.device = device
        return        
    
    def forward(self, input, logdet=0.0, reverse=False):
        if not reverse:            
            output = self.flow(input, logdet=logdet, reverse=False)
            return output
        else:
            output = self.reverse_flow(input, logdet=logdet, reverse=True)
            return output    

    def data_preprocessing(self, x):
        return self.flow.data_preprocessing(x)
    
    def data_posprocessing(self, x):
        return self.reverse_flow.data_posprocessing(x)

    def get_layer_params_size(self):
        if self.flow_nums == 0:
            return None, None
        
        named_size, total_params = self.flow.get_layer_params_size()     
        if self.flow_nums >= 2 or not self.same_backward:
            named_size_o, total_params_o = self.reverse_flow.get_layer_params_size('reverse_')
            named_size = dict(**named_size, **named_size_o)
            total_params = total_params + total_params_o
            
        return named_size, total_params

    def set_actnorm_init(self, inited):
        self.flow.set_actnorm_init(inited)
        if self.flow_nums >= 2 or self.same_backward:
            self.reverse_flow.set_actnorm_init(inited)
            
            
            

if __name__ == "__main__":
    import argparse    
    import interface
    from utils.config import JsonConfig
    from utils.sl import load
    
    #'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_path', default='hparams/PRED/Human36M/auto_likelihood_N18')
    parser.add_argument('--hparams_name', default='Lmix_dct_likelihood_PoseSO3_V8_2.json')
    args = parser.parse_args()

    hparams_path = os.path.join(os.getcwd(), args.hparams_path, args.hparams_name)    
    assert os.path.exists(hparams_path), (
        "Failed to find hparams josn `{}`".format(hparams_path))
    hparams = JsonConfig(hparams_path)
    
    dataset_name = hparams.Dataset
    
    device = torch.device('cpu')
    
    train_dataset = interface.dataset_interface(dataset_name, "train", hparams)
    model = interface.flow_model_builder(train_dataset, device, hparams)
    
    #(length, batch_size, channel, node_n)
    seq = torch.rand((25, 10, 3, 18))
    z, nll = interface.flow_interface(model, seq, logdet=0.0, reverse=False)
    x, _ = interface.flow_interface(model, z, logdet=0.0, reverse=True)
    res = x - model.data_preprocessing(seq)
    print()
    #'''
    
    f'''
    from interface import flow_interface

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default="20240725_1628")
    # parser.add_argument('--date', default="20240705_0018")
    parser.add_argument('--hparams_path', default='results/likelihood/H36Mso3N18VelOnly')
    parser.add_argument('--hparams_name', default='likelihood_V4_Velonly_S1-1.json')
    parser.add_argument('--epoch', default='best')
    args = parser.parse_args()

    date         = args.date
    hparams_path = args.hparams_path
    hparams_name = args.hparams_name
    epoch        = args.epoch if args.epoch == 'best' else int(args.epoch)

    hparams = os.path.join(os.getcwd(), hparams_path, "trained_"+date, hparams_name)
    hparams = JsonConfig(hparams)
    
    dataset_name = hparams.Dataset
    
    device = torch.device('cpu')
    
    train_dataset = interface.dataset_interface(dataset_name, "train", hparams)
    model = interface.network_interface(train_dataset, device, hparams)

    load(
        epoch_or_path = epoch,
        model = model,
        optim = None,
        schedule = None,
        pkg_dir = os.path.join(hparams.Dir.trained_model_root, dataset_name, "trained_"+date)
    )
    
    # seq = torch.zeros((1, 11, 1, 3, 18))
    # input, cond_input = seq[:, -1:], seq[:, :-1]
    # z, nll = model((input, cond_input), logdet=0.0, reverse=False)
    # x, _ = model((z[None, ...], cond_input), logdet=0.0, reverse=True)
    # res = x - input
    # print()
    
    x = torch.zeros((1, 1, 3, 18))
    z, nll = flow_interface(model, x, 0.0, False, None)
    _x, _log = flow_interface(model, z, 0.0, True, None)
    print()
    f'''