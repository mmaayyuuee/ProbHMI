import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from utils.flow_matching import ODESolver
from utils.wrapper import this_is_wrapper
from layers import DiT_V3, LightningDiT



class FLowMatchingModelBase(nn.Module):
    pass


class FlowMatchingModel_V1(FLowMatchingModelBase):
    def __init__(self, length, node_n, in_channels=3, hidden_size=64, depth=6, num_heads=8, mlp_ratio=4.0, num_classes=16, **kwargs):
        super().__init__()
        self.model = DiT_V3(length=length, node_n=node_n, in_channels=in_channels, hidden_size=hidden_size, 
                            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes)
        return
    
    def forward(self, x, t, label=None, **kwargs):
        y = self.model(x=x, t=t, label=label)
        return y 

    def get_layer_params_size(self, prefix=''):
        named_size = {}
        for name, params in self.named_parameters():
            named_size[prefix+'FlowMatchingModel_V1.'+name] = params.size()
        total_params = sum(p.numel() for p in self.parameters())
        return named_size, total_params


class FlowMatchingModel_V2(FLowMatchingModelBase):
    def __init__(self, length, node_n, in_channels=3, cond_in_channels=3, hidden_size=64, depth=6, num_heads=8, mlp_ratio=4.0, num_classes=16, **kwargs):
        super().__init__()
        if hasattr(self, "cond_as_input"):
            model_net = this_is_wrapper({"cond_as_input": True})(DiT_V3)
        elif hasattr(self, "cond_everywhere"):
            model_net = this_is_wrapper({"cond_everywhere": True})(DiT_V3)
        else:
            model_net = DiT_V3
        self.model = model_net(length=length, node_n=node_n, in_channels=in_channels, hidden_size=hidden_size, 
                               depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes,
                               cond_in_channels=cond_in_channels)
        return

    def forward(self, x, t, x_cond=None, label=None):
        y = self.model(x=x, t=t, x_cond=x_cond, label=label)
        return y 
    
    def get_layer_params_size(self, prefix=''):
        named_size = {}
        for name, params in self.named_parameters():
            named_size[prefix+'FlowMatchingModel_V2.'+name] = params.size()
        total_params = sum(p.numel() for p in self.parameters())
        return named_size, total_params


class FlowMatchingModel_V3(FLowMatchingModelBase):
    def __init__(self, length, node_n, in_channels=3, hidden_size=64, depth=6, num_heads=8, mlp_ratio=4.0, num_classes=16,
                use_qknorm=False, use_swiglu=False, use_rope=False, use_rmsnorm=False, wo_shift=False, context_fusion_type='adaLN',
                use_graph_pos_embed=False, use_checkpoint=False, expert_nums=1, top_k=1, expert_noise_epsilon=1e-2, expert_capacity_factor=1.25, 
                expert_dropout_rate=0.0, cond_in_channels=3, cond_expert_nums=1, cond_top_k=1, cond_expert_noise_epsilon=1e-2, 
                cond_expert_capacity_factor=1.25, cond_expert_dropout_rate=0.0, cond_embedding_type='Linear', residual_embedding=False, 
                cond_embedding_params={}, **kwargs):
        super().__init__()
        if hasattr(self, "cond_as_input"):
            model_net = this_is_wrapper({"cond_as_input": True})(LightningDiT)
        elif hasattr(self, "cond_everywhere"):
            model_net = this_is_wrapper({"cond_everywhere": True})(LightningDiT)
        else:
            model_net = LightningDiT
        self.model = model_net(length=length, node_n=node_n, in_channels=in_channels, hidden_size=hidden_size, 
                               depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes,
                               use_qknorm=use_qknorm, use_swiglu=use_swiglu, use_rope=use_rope, use_rmsnorm=use_rmsnorm, 
                               wo_shift=wo_shift, use_graph_pos_embed=use_graph_pos_embed, use_checkpoint=use_checkpoint, 
                               context_fusion_type=context_fusion_type, expert_nums=expert_nums, top_k=top_k, 
                               expert_noise_epsilon=expert_noise_epsilon, expert_capacity_factor=expert_capacity_factor, 
                               expert_dropout_rate=expert_dropout_rate, cond_in_channels=cond_in_channels, cond_expert_nums=cond_expert_nums, 
                               cond_top_k=cond_top_k, cond_expert_noise_epsilon=cond_expert_noise_epsilon, cond_expert_capacity_factor=cond_expert_capacity_factor, 
                               cond_expert_dropout_rate=cond_expert_dropout_rate, cond_embedding_type=cond_embedding_type, 
                               residual_embedding=residual_embedding, cond_embedding_params=cond_embedding_params,
                               adj_matrix = kwargs['adj_matrix'])
        return

    def forward(self, x, t, x_cond=None, label=None):
        y = self.model(x=x, t=t, x_cond=x_cond, label=label)
        return y 
    
    def get_layer_params_size(self, prefix=''):
        named_size = {}
        for name, params in self.named_parameters():
            named_size[prefix+'FlowMatchingModel_V3.'+name] = params.size()
        total_params = sum(p.numel() for p in self.parameters())
        return named_size, total_params



class VelocityPath(object):
    velocity_dict = {
        "linear_euclidean": lambda self,x0,x1,t: self.linear_euclidean_velocity(x0, x1, t),
        "linear_euclidean_v2": lambda self,x0,x1,t: self.linear_euclidean_velocity_v2(x0, x1, t),
    }
    def __init__(self, schedule, noise_std, **kwargs):
        self.schedule, self.noise_std = schedule, noise_std
        return
                
    def sample(self, x_0, x_1, t, **kwargs):
        ''' x_0/ x_1:(T, B, C, N) '''
        dx_t = self.velocity_dict[self.schedule](self, x_0, x_1, t)
        return dx_t

    def linear_euclidean_velocity(self, x_0, x_1, t):
        t = t[None, :, None, None].expand(x_0.shape[0], -1, x_0.shape[-2], x_0.shape[-1])
        x_t = x_0 + t * (x_1 - x_0)
        dx_t = x_1 - x_0
        epsilon = torch.normal(mean=torch.zeros_like(x_t), std=self.noise_std)
        return x_t+epsilon, dx_t
    
    def linear_euclidean_velocity_v2(self, x_0, x_1, t):
        t = t[None, :, None, None].expand(x_0.shape[0], -1, x_0.shape[-2], x_0.shape[-1])
        x_t = x_0 + t * (x_1 - x_0)
        dx_t = x_1 - x_0
        epsilon = torch.normal(mean=torch.zeros_like(x_t), std=self.noise_std) * (1 - t)
        return x_t+epsilon, dx_t



class FlowMatchingDynamicsBase(nn.Module):
    flow_model_dict = {
        1: FlowMatchingModel_V1,
        2: FlowMatchingModel_V2,
        3: [this_is_wrapper({"cond_as_input": True}), FlowMatchingModel_V2],
        4: FlowMatchingModel_V3,
        5: [this_is_wrapper({"cond_as_input": True}), FlowMatchingModel_V3],
        6: [this_is_wrapper({"cond_everywhere": True}), FlowMatchingModel_V3],
    }
    velocity_path_dict = {
        1: VelocityPath
    }
    
    def __init__(self, flow_model_params, flow_path_params, **kwargs) -> None:
        super().__init__()
        self.flow_model = self._build_flow_model(flow_model_params['version'], flow_model_params['hparams'], **kwargs)
        self.flow_path = self._build_flow_path(flow_path_params['version'], flow_path_params['hparams'])
        self.ode_solver = ODESolver(self.flow_model)
        return
    
    def _build_flow_model(self, version, hparams, **kwargs):
        flow_model = self.flow_model_dict[version]
        if not isinstance(flow_model, list):
            flow_model = flow_model(**hparams, adj_matrix=kwargs['adj_matrix'])
        else:
            flow_model = flow_model[0](flow_model[1])(**hparams, adj_matrix=kwargs['adj_matrix'])
        return flow_model
    
    def _build_flow_path(self, version, hparams, **kwargs):
        flow_path = self.velocity_path_dict[version](**hparams)
        return flow_path
    
    def forward(self, x_0, x_1=None, t=None, eval=False, x_cond=None, label=None, **kwargs):
        ''' x_0/x_1 : (T, B, C, N) '''
        if eval:
            step_size = 0.01 if "step_size" not in kwargs else kwargs["step_size"]
            method = "euler" if "method" not in kwargs else kwargs["method"]
            x_1 = self.forward_eval(x_0=x_0, time_grid=t, method=method, step_size=step_size, x_cond=x_cond, label=label)
            return x_1
        else:
            output = self.forward_train(x_0=x_0, x_1=x_1, t=t, x_cond=x_cond, label=label)
            return output   # dx_t, dx_t_pred / dx_t, dx_t_pred, aux_loss
    
    def forward_train(self, x_0, x_1, t, x_cond=None, label=None):
        raise NotImplementedError
    
    def forward_eval(self, x_0, time_grid, method, step_size, x_cond=None, label=None):
        raise NotImplementedError

    def get_layer_params_size(self):
        named_size, total_params = self.flow_model.get_layer_params_size()
        return named_size, total_params



class FlowMatchingDynamics_V1(FlowMatchingDynamicsBase):
    def forward_train(self, x_0, x_1, t, x_cond=None, label=None):
        ''' x_0/x_1 : (T, B, C, N) '''
        ''' t belongs to [0, 1] '''
        x_t, dx_t = self.flow_path.sample(x_0=x_0, x_1=x_1, t=t)
        dx_t_pred = self.flow_model(x=x_t, t=t, x_cond=x_cond, label=label)
        if isinstance(dx_t_pred, tuple) or isinstance(dx_t_pred, list):
            dx_t_pred, aux_loss = dx_t_pred
            return dx_t, dx_t_pred, aux_loss
        else:
            return dx_t, dx_t_pred
           
    def forward_eval(self, x_0, time_grid, method, step_size, x_cond=None, label=None):
        ''' x_0 : (T, B, C, N) '''
        x_1 = self.ode_solver.sample(time_grid=time_grid, x_init=x_0, method=method, \
                                     step_size=step_size, return_intermediates=False, \
                                     x_cond=x_cond, label=label)
        return x_1

    def compute_likelihood(self, x_1, log_p0, time_grid, method, step_size, x_cond=None, label=None):       
        ll = self.ode_solver.compute_likelihood(x_1, log_p0, time_grid=time_grid, method=method,
                                                step_size=step_size, x_cond=x_cond, label=label)
        return ll


