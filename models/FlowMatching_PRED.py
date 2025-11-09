import torch
import torch.nn as nn
from torch.distributions import LogNormal
from torch.func import functional_call
import numpy as np
import math

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from modules import FlowBaseNet
from modules import FlowMatchingDynamicsBase, FlowMatchingDynamics_V1
from modules.cfm_dynamics import VelocityPath, FLowMatchingModelBase
from models.PRED import flow_encoder_new, flow_decoder_new
from utils.flow_matching import ode_solver



class FlowMatchingPredictBase(nn.Module):
    def __init__(self, flow:FlowBaseNet, flow_dynamics:FlowMatchingDynamicsBase, **kwargs):
        super(FlowMatchingPredictBase, self).__init__()
        self.flow, self.flow_dynamics = flow, flow_dynamics

        drop_coef = None if "drop_coef" not in kwargs else kwargs["drop_coef"]
        if drop_coef is not None:
            self.dropout = nn.Dropout(p=drop_coef)
        else:
            self.dropout = None
        return
        
    def get_layer_params_size(self):
        named_size_f, total_params_f = self.flow.get_layer_params_size()
        named_size_r, total_params_r = self.flow_dynamics.get_layer_params_size()
        named_size = dict(**named_size_f, **named_size_r)
        total_params = total_params_f + total_params_r
        return named_size, total_params

    def set_actnorm_init(self, inited):
        self.flow.set_actnorm_init(inited)
        return

    @property
    def distribution_parameters(self):
        return self.flow.distribution_parameters

    def load_distribution_parameters(self, parameters_dict):
        return self.flow.load_distribution_parameters(parameters_dict)

    def __getitem__(self, item):
        if item =='0' or str.upper(item) == "FLOW":
            return self.flow
        elif item =='1' or str.upper(item) == "FLOW_MATCHING" or item == 1:
            return self.flow_dynamics

    def data_preprocessing(self, data, **kwargs):
        if hasattr(self.flow, "dct_flag"):
            return self.flow.data_preprocessing(data, **kwargs)
        else:
            return data if data.ndim == 5 else data[None, ...]
    
    def data_posprocessing(self, data, **kwargs):
        if hasattr(self.flow, "dct_flag"):
            return self.flow.data_posprocessing(data, **kwargs)
        else:
            return data if data.ndim == 5 else data[:, None, ...]


    def encode(self, x, label=None, padding=True, **kwargs):
        tlen, bs, channels, nodes = x.shape
        if hasattr(self.flow, 'cond_model'):
            c_his, c_pred = self.flow.get_condition_length()

            times = math.ceil(1.0 * tlen / c_pred)
            remain = times * c_pred - tlen 
            idx_pad = [0]*remain + list(range(tlen))
            x = x[idx_pad]
            tlen = len(idx_pad)

            if padding:                
                _x = torch.zeros((c_his+tlen, bs, channels, nodes), device=x.device)
                _x[c_his:] = x
            else:
                _x = x
            cond_list = []
            for idx in range(0, _x.shape[0]-c_his-c_pred+1, c_pred):
                cond_list.append(_x[idx:idx+c_his+c_pred][None, ...])
            conds = torch.cat(cond_list, dim=0)
            x_en, nll = flow_encoder_new(self.flow, conds, label, **kwargs)
            return x_en, nll
        else:
            _, t_pred = self.flow.get_seq_length()

            times = math.ceil(1.0 * tlen / t_pred)
            remain = times * t_pred - tlen 
            idx_pad = [0]*remain + list(range(tlen))
            x = x[idx_pad]
            tlen = len(idx_pad)
            
            group_list = []
            for idx in range(0, x.shape[0]-t_pred+1, t_pred):
                group_list.append(x[idx:idx+t_pred][None, ...])
            groups = torch.cat(group_list, dim=0)
            
            x_en, nll = flow_encoder_new(self.flow, groups, label, **kwargs)
            return x_en, nll


    def decode(self, x, label=None, logdet=0.0):
        x_de, nll = flow_decoder_new(self.flow, x, label, logdet=logdet)
        return x_de, nll



class FlowMatchingPredict_V1(FlowMatchingPredictBase):
    def __init__(self, flow:FlowBaseNet, flow_dynamics:FlowMatchingDynamicsBase, **kwargs):
        super(FlowMatchingPredict_V1, self).__init__(flow, flow_dynamics, **kwargs)
        return

    def forward(self, x, t_pred=100, t_his=0, step_size=0.1, eval=True, bias=None, label=None, **kwargs):
        _x_pos, _x_vel, _x_en = x
        if _x_vel.ndim == 5:
            S, T, B, C, N = _x_pos.shape 
        else: 
            S, T, B, C, N =[1, *list(_x_vel.shape)]
            _x_pos, _x_vel, _x_en = _x_pos[None, ...], _x_vel[None, ...], _x_en[None, ...]

        _, f_pred = self.flow.get_seq_length()
        _, c_pred = self.flow.get_condition_length()
        F_C = self.flow.get_channels()
        group_t_his, group_t_pred = math.ceil(t_his/f_pred), math.ceil(t_pred/f_pred)
        F_T = group_t_his + group_t_pred
        
        x_pos, x_vel = self.data_preprocessing(_x_pos), self.data_preprocessing(_x_vel)
        x_pos, x_vel, x_en = [torch.swapaxes(item, 0, 1).reshape((F_T, S*B, F_C, N)) for item in [x_pos, x_vel, _x_en]]
        x_pos_his = x_pos[:t_his]
    
        x_0, x_1 = self._build_model_input_target_pair((x_pos, x_vel, x_en), group_t_his)
        
        if eval:
            x_1_pred = self.flow_dynamics(x_0, x_1, t=torch.tensor([0.0, 1.0]).to(x_pos.device), \
                                          eval=True, step_size=step_size, label=label)
            if hasattr(self, 'copy_last_frame') and bias is not None:
                bias = torch.swapaxes(bias, 0, 1)
                bias = bias[..., None, None].expand(-1, -1, F_C, N)[:F_T]
                x_1_pred = x_1_pred + bias
            
            y_1_pred, nll = self.decode(x_1_pred, label=label, logdet=0.0)
            y_1_pred = self.data_posprocessing(y_1_pred[group_t_his:])
            # y_1_pred = y_1_pred.reshape(S, -1, B, C, N)
            y_1_pred = y_1_pred.reshape(-1, S*B, C, N).reshape(-1, S, B, C, N).swapaxes(0, 1)
            y_pos_pred = self.add_velocity(_x_pos[:, t_his-1:t_his], y_1_pred)
            y_pos_pred = y_pos_pred.swapaxes(1, 2)
            return y_pos_pred
        else:
            t = torch.rand(size=(B,)).to(x_pos.device).type(torch.float32)
            dx_t, dx_t_pred = self.flow_dynamics(x_0, x_1, t=t, eval=False, label=label)
            return dx_t, dx_t_pred
        
        
    def _build_model_input(self, x_his, x_input):
        t_his, t_total = x_his.shape[0], x_input.shape[0]   # (T, B, C, N)
        if hasattr(self, 'copy_last_frame'):
            x_input[:t_his] = x_his
            x_input[t_his:t_total] = x_his[-1]
        elif hasattr(self, 'noise_copy_last_frame'):
            x_input[:t_his] = x_his
            x_input[t_his:t_total] = x_input[t_his:t_total] + x_his[-1]
        else:
            x_input[:t_his] = x_his
        return x_input 
    
    
    def _build_model_input_target_pair(self, x, t_his):
        x_pos, x_vel, x_en = x  # (T, B, C, N)
        if hasattr(self, 'input_pose'):
            raise NotImplementedError
        else:
            x_input = torch.rand_like(x_en)
            x_input = self._build_model_input(x_en[:t_his], x_input)
            x_0, x_1 = x_input, x_en
        return x_0, x_1


    def add_velocity(self, pos, vel):
        if hasattr(self, "output_pose"):
            new_pos = vel
        else:
            new_pos = torch.zeros_like(vel)
            for i in range(vel.shape[1]):
                new_pos[:, i:i+1] = pos + vel[:, i:i+1] if i == 0 else new_pos[:, i-1:i] + vel[:, i:i+1]
        return new_pos



class FlowMatchingPredict_V2(FlowMatchingPredictBase):
    def __init__(self, flow:FlowBaseNet, flow_dynamics:FlowMatchingDynamicsBase, **kwargs):
        super(FlowMatchingPredict_V2, self).__init__(flow, flow_dynamics, **kwargs)
        self.mean_std = None if 'mean_std' not in kwargs else kwargs['mean_std']
        if self.mean_std is not None:
            self.mean = torch.from_numpy(self.mean_std[0]).to(flow.device)
            self.std  = torch.from_numpy(self.mean_std[1]).to(flow.device) if self.mean_std[1] is not None \
                        else self.mean_std[1]
        
        self.adj_matrix = kwargs["adj_matrix"] * 1.0
        self.anisotropic_coef = 0.0 if "anisotropic_coef" not in kwargs else kwargs["anisotropic_coef"]
        self.Lambda, self.Sigma, self.U = self._compute_covariance_matrices()
        
        self.sampling_method = "Uniform" if "sampling_method" not in kwargs else kwargs["sampling_method"]        
        self.sampling_coef = 1.0 if "sampling_coef" not in kwargs else kwargs["sampling_coef"]
        self.random_drop = -1 if "random_drop" not in kwargs else kwargs["random_drop"]
        self.temporal_correlation = 1.0 if "temporal_correlation" not in kwargs else kwargs["temporal_correlation"]
                        
        if hasattr(self, "without_latent_space"):
            self.flow.delete_model()
        return


    def add_velocity(self, pos, vel):
        if hasattr(self, "output_pose"):
            new_pos = vel
            
        elif hasattr(self, "output_pose_velocity"):
            pred_pos, pred_vel = torch.split(vel, int(vel.shape[-2]/2), dim=-2)
            new_pos = torch.zeros_like(pred_pos)
            for i in range(pred_vel.shape[1]):
                new_pos[:, i:i+1] = pos + pred_vel[:, i:i+1] if i == 0 else new_pos[:, i-1:i] + pred_vel[:, i:i+1]
            new_pos = (new_pos + pred_pos) / 2
            
        elif hasattr(self, "output_pose_execpt_root"):
            new_pos = torch.zeros_like(vel)
            new_pos[..., 1:] = vel[..., 1:]
            for i in range(vel.shape[1]):
                new_pos[:, i:i+1, ..., 0] = pos[..., 0] + vel[:, i:i+1, ..., 0] if i == 0 \
                                        else new_pos[:, i-1:i, ..., 0] + vel[:, i:i+1, ..., 0]
        else:
            new_pos = torch.zeros_like(vel)
            for i in range(vel.shape[1]):
                new_pos[:, i:i+1] = pos + vel[:, i:i+1] if i == 0 else new_pos[:, i-1:i] + vel[:, i:i+1]
        return new_pos


    def forward(self, x, t_pred=100, t_his=0, step_size=0.1, ode_method='euler', eval=True, bias=None, label=None, fm_nll=False, 
                **kwargs):
        _x_pos, _x_vel = x
        if _x_vel.ndim == 5:
            S, T, B, C, N = _x_pos.shape 
        else: 
            S, T, B, C, N =[1, *list(_x_vel.shape)]
            _x_pos, _x_vel = _x_pos[None, ...], _x_vel[None, ...]   
        x_pos, x_vel = [torch.swapaxes(item, 0, 1).reshape((T, S*B, C, N)) for item in [_x_pos, _x_vel]]

        _, f_pred = self.flow.get_seq_length()
        F_t_H = math.ceil(t_his / f_pred)
        # F_T = math.ceil((t_his + t_pred) / f_pred) if not hasattr(self, "only_conditional_history") else \
        #       math.ceil(t_pred / f_pred)
        F_T = math.ceil((t_his + t_pred) / f_pred)
        
        if hasattr(self, "output_pose") or hasattr(self, "output_pose_execpt_root"):
            x_0, x_1, x_0_nll, x_1_nll = self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval) if "flow_model_params" not in kwargs \
                                        else self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval, model_params=kwargs["flow_model_params"])
        elif hasattr(self, "output_pose_velocity"):
            x_0, x_1, x_0_nll, x_1_nll = self._build_model_input_target_pair(torch.concat((x_pos, x_vel), dim=-2), t_his, F_t_H, eval=eval) if "flow_model_params" not in kwargs \
                                        else self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval, model_params=kwargs["flow_model_params"])
            C = 2* C
        else:
            x_0, x_1, x_0_nll, x_1_nll = self._build_model_input_target_pair(x_vel, t_his, F_t_H, eval=eval) if "flow_model_params" not in kwargs \
                                        else self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval, model_params=kwargs["flow_model_params"])
        
        x_0_full, x_1_full = [item.reshape((F_T, S*B, -1, C, N)) for item in [x_0, x_1]]
        if hasattr(self, "only_conditional_history"):
            x_0, x_1 = x_0_full[F_t_H:], x_1_full[F_t_H:]
            F_T = F_T - F_t_H
        else:
            x_0, x_1 = x_0_full, x_1_full   
        # if not hasattr(self, "embedding_cond"):
        #     x_cond = torch.swapaxes(torch.clone(x_pos[:t_his]), 0, 1)[None, ...] if not hasattr(self, "output_pose_velocity") \
        #         else torch.swapaxes(torch.clone(torch.concat((x_pos, x_vel), dim=-2)[:t_his]), 0, 1)[None, ...] 
        # else:
        #     x_cond = x_1[:F_t_H]
        
        if hasattr(self, "embedding_cond"):          
            x_cond = x_1_full[:F_t_H]
        elif hasattr(self, "preprocessing_cond"):   
            x_cond = self.data_preprocessing(x_pos[:t_his]) 
        elif hasattr(self, "double_cond"):
            x_cond_1 = x_1_full[:F_t_H]
            x_cond_2 = self.data_preprocessing(x_pos[:t_his]).reshape(x_cond_1.shape) 
            x_cond = torch.concat((x_cond_1, x_cond_2), dim=-3)
        else:
            x_cond = torch.swapaxes(torch.clone(x_pos[:t_his]), 0, 1)[None, ...] if not hasattr(self, "output_pose_velocity") \
                else torch.swapaxes(torch.clone(torch.concat((x_pos, x_vel), dim=-2)[:t_his]), 0, 1)[None, ...] 
        
        if self.dropout:
            x_cond = self.dropout(x_cond)
        
        if hasattr(self, "level") and self.level == "channel":
            x_0, x_1 = [item.reshape((F_T, S*B, -1, C*N)) for item in [x_0, x_1]]
            x_cond = x_cond.reshape(1, S*B, -1, C*N)
        else:
            x_0, x_1 = [item.reshape((F_T, S*B, -1, N)) for item in [x_0, x_1]] 
            x_cond = x_cond.reshape(1, S*B, -1, N)
        x_0, x_1 = [item.permute(3, 1, 2, 0) for item in [x_0, x_1]]  
        x_cond = x_cond.permute(3, 1, 2, 0)
                 
        if eval:
            x_1_pred = self.flow_dynamics(x_0, x_1, t=torch.tensor([0.0, 1.0]).to(x_pos.device), \
                                          eval=True, step_size=step_size, method=ode_method, \
                                          x_cond=x_cond, label=label)
            x_1_pred_copy = x_1_pred.clone()
            x_1_pred = x_1_pred.permute(3, 1, 2, 0)
            if hasattr(self, "level") and self.level == "channel":
                x_1_pred = x_1_pred.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                
            if not (hasattr(self, 'noise_copy') or hasattr(self, 'noise_copy_without_root') or \
                    hasattr(self, 'noise_only') or hasattr(self, 'noise_full_copy') or \
                    hasattr(self, 'noise_full_only') or hasattr(self, 'noise_true_only') or \
                    hasattr(self, 'noise_copy_only_eval') or hasattr(self, 'without_noise')) \
                and bias is not None:
                bias = torch.swapaxes(bias, 0, 1)
                bias = bias[..., None, None].expand(-1, -1, x_1_pred.shape[-2], x_1_pred.shape[-1])[:F_T]
                x_1_pred = x_1_pred + bias
            
            if hasattr(self, "without_latent_space"):
                y_1_pred = x_1_pred
                _, ll = self.flow_dynamics.compute_likelihood(x_1_pred_copy, lambda x: torch.sum(-0.5*(((x)**2) + float(np.log(2*np.pi))), dim=(0, 2, 3)), 
                                                              time_grid=torch.tensor([1.0, 0.0]).to(x_1_pred.device),
                                                              step_size=step_size, method=ode_method, x_cond=x_cond, 
                                                              label=label)
                nll_fm = -ll / (x_1_pred.shape[-2] * x_1_pred.shape[-1])
                nll = nll_fm
                nll = nll.reshape(S, B, 1)
            else:    
                y_1_pred, nll = self.decode(x_1_pred, label=label, logdet=0.0)
                if fm_nll:
                    ll_2 = -nll
                    ll_2 = ll_2 * (x_1_pred.shape[-2] * x_1_pred.shape[-1])
                    logp = self.flow.distribution.logp(x_1_pred, label)
                    ll_2 = ll_2 - logp                
                    _, ll_1 = self.flow_dynamics.compute_likelihood(x_1_pred_copy, lambda x: torch.sum(-0.5*(((x)**2) + float(np.log(2*np.pi))), dim=(0, 2, 3)), 
                                                                    time_grid=torch.tensor([1.0, 0.0]).to(x_1_pred.device),
                                                                    step_size=step_size, method=ode_method, x_cond=x_cond, 
                                                                    label=label)
                    ll = ll_1 + ll_2
                    nll_fm = -ll / (x_1_pred.shape[-2] * x_1_pred.shape[-1])
                    nll = nll_fm
                nll = nll.reshape(S, B, 1)
                
            y_1_pred = self.data_posprocessing(y_1_pred)
            y_1_pred = y_1_pred.reshape(-1, S*B, C, N).reshape(-1, S, B, C, N).swapaxes(0, 1)

            y_1_pred = y_1_pred[:, -t_pred:]
            y_pos_pred = self.add_velocity(_x_pos[:, t_his-1:t_his], y_1_pred)
            y_pos_pred = y_pos_pred.swapaxes(1, 2)
            return y_pos_pred, nll
        else:
            if hasattr(self, "lognorm_sampling"):
                log_normal = LogNormal(0.0, 1.0)
                samples = log_normal.sample(sample_shape=(B,))
                samples = 1 / (1 + torch.exp(-samples))
                t = samples.to(x_pos.device).type(torch.float32)
            else:
                t = torch.rand(size=(B,)).to(x_pos.device).type(torch.float32)
            
            if "matching_model_params" not in kwargs:    
                output = self.flow_dynamics(x_0, x_1, t=t, eval=False, x_cond=x_cond, label=label)
            else: 
                output = functional_call(self.flow_dynamics, kwargs["matching_model_params"], args=(x_0, x_1), \
                                        kwargs={'t':t, 'eval':False, 'x_cond':x_cond, 'label':label})
                
            if len(output) == 2:
                dx_t, dx_t_pred = output
                aux_loss = 0.0
            else:
                dx_t, dx_t_pred, aux_loss = output
            
            dx_t, dx_t_pred = dx_t.permute(3, 1, 2, 0), dx_t_pred.permute(3, 1, 2, 0)
            if hasattr(self, "level") and self.level == "channel":
                dx_t = dx_t.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                dx_t_pred = dx_t_pred.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                x_0 = x_0.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                x_1 = x_1.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N) 
            return dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss

    
    def _build_model_input_target_pair(self, x, t_his=25, F_t_H=1, label=None, eval=False, **kwargs):
        x = x  # (T, B, C, N)
        x_input = torch.clone(x)
        x_input[t_his:] = x_input[t_his-1]
        
        if hasattr(self, "without_latent_space"):
            x_en = self.data_preprocessing(x)
            x_input_en = self.data_preprocessing(x_input)
            x_nll, x_input_nll = torch.tensor([0.0]), torch.tensor([0.0])
        else:
            if hasattr(self, "only_conditional_history"):
                x_en, x_nll = self.encode(x[t_his:], label)
                x_input_en, x_input_nll = self.encode(x_input[t_his:], label)
                raise NotImplementedError
            else:
                x_en, x_nll = self.encode(x, label)
                x_input_en, x_input_nll = self.encode(x_input, label)
            
        if hasattr(self, 'noise_copy'):
            x_input_noise = self._sample_func(x_input_en)
            x_input_en = x_input_en + x_input_noise            
        elif hasattr(self, 'noise_copy_without_root'):
            x_input_noise = self._sample_func(x_input_en)
            x_input_noise[..., 0] = 0.0
            x_input_en = x_input_en + x_input_noise
        elif hasattr(self, 'noise_only'):    
            x_input_en = self._sample_func(x_input_en)   
        elif hasattr(self, 'noise_copy_only_eval'):
            if eval:
                x_input_noise = self._sample_func(x_input_en)
                x_input_en = x_input_en + x_input_noise
            else:
                x_input_en = x_input_en   
        elif hasattr(self, 'without_noise'):
            x_input_en = x_input_en 
                    
        x_0, x_1 = x_input_en, x_en
        return x_0, x_1, x_input_nll, x_nll
            
    
    def unnormalize_data(self, x):
        if self.mean_std is None:
            return x
        
        if self.std is None:
            if hasattr(self, "output_pose"):
                x = x + self.mean[..., :3, :]
            elif hasattr(self, "output_pose_velocity"):
                x = x + self.mean
            elif hasattr(self, "output_pose_execpt_root"):
                raise NotImplementedError
            else:
                x = x + self.mean[..., 3:, :]
        else:
            if hasattr(self, "output_pose"):
                x = x * self.std[..., :3, :] + self.mean[..., :3, :]
            elif hasattr(self, "output_pose_velocity"):
                x = x * self.std + self.mean
            elif hasattr(self, "output_pose_execpt_root"):
                raise NotImplementedError
            else:
                x = x * self.std[..., 3:, :] + self.mean[..., 3:, :]
        return x


    def _sample_func(self, x, label=None):
        def generate_temporal_noise(method="Uniform"):
            noise = torch.zeros_like(x)
            T = x.shape[0]
            for i in range(T):
                noise[i] = torch.rand_like(noise[i]) * self.sampling_coef if method == "Uniform" else \
                           torch.normal(mean=torch.zeros_like(noise[i]), std=self.sampling_coef)                
                if i != 0:
                    noise[i] = self.temporal_correlation * noise[i] + (1 - self.temporal_correlation) * noise[i-1]
            return noise
        
        if self.sampling_method == "Uniform":
            return torch.rand_like(x) * self.sampling_coef
        elif self.sampling_method == "Gaussian":
            return torch.normal(mean=torch.zeros_like(x), std=self.sampling_coef)
 
        if self.sampling_method == "Mixed_Uniform":
            samples = self.flow.distribution.sampling(x, None, self.sampling_coef, method="Uniform")
            return samples
        elif self.sampling_method == "Min_Mixed_Uniform":  
            samples = self.flow.distribution.sampling(x, torch.zeros_like(x), self.sampling_coef, method="Uniform")   ### 触发机制
            return samples        
        if self.sampling_method == "Mixed_Gaussian":
            samples = self.flow.distribution.sampling(x, None, self.sampling_coef, method="Gaussian")
            return samples
        elif self.sampling_method == "Min_Mixed_Gaussian":  
            samples = self.flow.distribution.sampling(x, torch.zeros_like(x), self.sampling_coef, method="Gaussian")   ### 触发机制
            return samples
        
        elif self.sampling_method == "Equal_Uniform":
            samples = torch.rand_like(x) * self.sampling_coef
            samples[1:] = samples[:1]
            return samples
        elif self.sampling_method == "Equal_Gaussian":
            samples = torch.normal(mean=torch.zeros_like(x), std=self.sampling_coef)
            samples[1:] = samples[:1]
            return samples
        
        elif self.sampling_method == "Anisotropic_Uniform":
            samples = torch.rand_like(x) * self.sampling_coef
            samples = samples * (self.anisotropic_coef * self.Lambda + (1 - self.anisotropic_coef))
            return samples
        elif self.sampling_method == "Anisotropic_Gaussian":
            samples = torch.normal(mean=torch.zeros_like(x), std=self.sampling_coef)
            samples = samples * (self.anisotropic_coef * self.Lambda + (1 - self.anisotropic_coef))
            return samples

        elif self.sampling_method == "Anisotropic_Graph_Uniform":
            samples = torch.rand_like(x) * self.sampling_coef
            Sigma = self.U @ torch.diag_embed((self.anisotropic_coef * self.Lambda + (1 - self.anisotropic_coef)))
            samples = torch.einsum("nn,tbcn->tbcn", Sigma, samples)
            return samples
        elif self.sampling_method == "Anisotropic_Graph_Gaussian":
            samples = torch.normal(mean=torch.zeros_like(x), std=self.sampling_coef)
            Sigma = self.U @ torch.diag_embed((self.anisotropic_coef * self.Lambda + (1 - self.anisotropic_coef)))
            samples = torch.einsum("nn,tbcn->tbcn", Sigma, samples)
            return samples
        
        elif self.sampling_method == "Temporal_Uniform":
            return generate_temporal_noise("Uniform")
        elif self.sampling_method == "Temporal_Gaussian":
            return generate_temporal_noise("Gaussian")
        
        else:
            raise NotImplementedError


    def _compute_covariance_matrices(self):
        eigenvalues = torch.linalg.eigvals(self.adj_matrix)
        eigenvalues = torch.real(eigenvalues)
        min_eig = eigenvalues.min()
        pos_def_matrix = self.adj_matrix + torch.eye(self.adj_matrix.shape[0], device=self.adj_matrix.device)*(- min_eig + 1e-6)
        Sigma = pos_def_matrix
        
        Lambda, U = torch.linalg.eigh(Sigma, UPLO='L')
        relative_scale_factor = Lambda.max() 
        Lambda = Lambda / relative_scale_factor
        Sigma = Sigma / relative_scale_factor
        return Lambda, Sigma, U
        


class FlowMatchingPredict_V3(FlowMatchingPredict_V2):
    def __init__(self, flow_encoder:FlowBaseNet, flow_decoder:FlowBaseNet, flow_dynamics:FlowMatchingDynamicsBase, **kwargs):
        super(FlowMatchingPredict_V3, self).__init__(flow_encoder, flow_dynamics, **kwargs)
        if hasattr(self, "without_latent_space"):
            self.flow.delete_model()
        self.flow_decoder = flow_decoder
        return


    def _build_model_input_target_pair(self, x, t_his=25, F_t_H=1, label=None):
        x = x  # (T, B, C, N)
        x_input = torch.clone(x)
        x_input[t_his:] = x_input[t_his-1]
        
        if hasattr(self, "without_latent_space"):
            x_en, x_nll = self.encode(x, self.flow_decoder, label)
            x_input_en = self.data_preprocessing(x_input)
            x_input_nll = torch.tensor([0.0])
        else:
            if hasattr(self, "only_conditional_history"):
                x_en, x_nll = self.encode(x[t_his:], self.flow_decoder, label)
                x_input_en, x_input_nll = self.encode(x_input[t_his:], self.flow, label)
                raise NotImplementedError
            else:
                x_en, x_nll = self.encode(x, self.flow_decoder, label)
                x_input_en, x_input_nll = self.encode(x_input, self.flow, label)
            
        if hasattr(self, 'noise_copy'):
            x_input_noise = torch.rand_like(x_input_en)
            x_input_en = x_input_en + x_input_noise
        elif hasattr(self, 'noise_copy_without_root'):
            x_input_noise = torch.rand_like(x_input_en)
            x_input_noise[..., 0] = 0.0
            x_input_en = x_input_en + x_input_noise
        elif hasattr(self, 'noise_only'):
            x_input_en = torch.rand_like(x_input_en)              
            
        x_0, x_1 = x_input_en, x_en
        return x_0, x_1, x_input_nll, x_nll


    def encode(self, x, flow, label=None, padding=True, **kwargs):
        tlen, bs, channels, nodes = x.shape
        if hasattr(flow, 'cond_model'):
            c_his, c_pred = flow.get_condition_length()

            times = math.ceil(1.0 * tlen / c_pred)
            remain = times * c_pred - tlen 
            idx_pad = [0]*remain + list(range(tlen))
            x = x[idx_pad]
            tlen = len(idx_pad)

            if padding:                
                _x = torch.zeros((c_his+tlen, bs, channels, nodes), device=x.device)
                _x[c_his:] = x
            else:
                _x = x
            cond_list = []
            for idx in range(0, _x.shape[0]-c_his-c_pred+1, c_pred):
                cond_list.append(_x[idx:idx+c_his+c_pred][None, ...])
            conds = torch.cat(cond_list, dim=0)
            x_en, nll = flow_encoder_new(flow, conds, label, **kwargs)
            return x_en, nll
        else:
            _, t_pred = flow.get_seq_length()

            times = math.ceil(1.0 * tlen / t_pred)
            remain = times * t_pred - tlen 
            idx_pad = [0]*remain + list(range(tlen))
            x = x[idx_pad]
            tlen = len(idx_pad)
            
            group_list = []
            for idx in range(0, x.shape[0]-t_pred+1, t_pred):
                group_list.append(x[idx:idx+t_pred][None, ...])
            groups = torch.cat(group_list, dim=0)
            
            x_en, nll = flow_encoder_new(flow, groups, label, **kwargs)
            return x_en, nll


    def decode(self, x, label=None, logdet=0.0):
        x_de, nll = flow_decoder_new(self.flow_decoder, x, label, logdet=logdet)
        return x_de, nll


    def __getitem__(self, item):
        if item =='0' or str.upper(item) == "FLOW":
            return self.flow
        elif item =='1' or str.upper(item) == "FLOW_MATCHING":
            return self.flow_dynamics
        elif item == '2' or str.upper(item) == "FLOW_DECODER":
            return self.flow_decoder
        elif item == '3' or str.upper(item) == "FLOW_AND_FLOW_MATCHING":
            return [self.flow, self.flow_dynamics]


    def get_layer_params_size(self):
        named_size_f, total_params_f = self.flow.get_layer_params_size()
        named_size_d, total_params_d = self.flow_decoder.get_layer_params_size(prefix='decoder')
        named_size_r, total_params_r = self.flow_dynamics.get_layer_params_size()
        named_size = dict(**named_size_f, **named_size_d, **named_size_r)
        total_params = total_params_f + total_params_d + total_params_r
        return named_size, total_params



class FlowMatchingPredict_V4(FlowMatchingPredict_V2):
    def _build_model_input_target_pair(self, x, t_his=25, F_t_H=1, label=None, **kwargs):
        x = x  # (T, B, C, N)

        if hasattr(self, "without_latent_space"):
            x_en = self.data_preprocessing(x)   # (S, F_T, B, F_C, N)
            x_input_en = torch.clone(x_en)
            x_input_en[:, F_t_H:] = x_en[:, :F_t_H]
            x_input_en = x_input_en.swapaxes(0, 1)
            x_input_en = x_input_en.reshape(x_input_en.shape[0], -1, x_input_en.shape[-2], x_input_en.shape[-1])
            x_nll = torch.tensor([0.0])
        else:
            # if hasattr(self, "only_conditional_history"):
            #     x_en, x_nll = self.encode(x[t_his:], label)
            #     x_input_en = torch.clone(x_en)
            # else:
            #     if "model_params" in kwargs:
            #         x_en, x_nll = self.encode(x, label, model_params=kwargs["model_params"])
            #     else:
            #         x_en, x_nll = self.encode(x, label)
            #     x_input_en = torch.clone(x_en)
            #     # x_input_en[:, F_t_H:] = x_en[:, :F_t_H]
            #     x_input_en[F_t_H:] = x_en[:F_t_H]
            if hasattr(self, "only_conditional_history"):
                x_en_full, x_nll = self.encode(x, label)
                x_en = torch.clone(x_en_full)
                x_en[:F_t_H] =  x_en[F_t_H:]
                x_input_en = torch.clone(x_en_full)
                x_input_en[F_t_H:] =  x_input_en[:F_t_H]
            else:            
                if "model_params" in kwargs:
                    x_en, x_nll = self.encode(x, label, model_params=kwargs["model_params"])
                else:
                    x_en, x_nll = self.encode(x, label)
                x_input_en = torch.clone(x_en)            
                x_input_en[F_t_H:] = x_en[:F_t_H]

        def random_drop_func(x_in):
            if not kwargs["eval"] and self.random_drop > 0:
                drop_idx = torch.randint(0, x_in.shape[1], size=(self.random_drop,))
                x_out = torch.clone(x_in)
                x_out[:, drop_idx] = x_en[:, drop_idx]
            else:
                x_out =  x_in
            return x_out
            
        if hasattr(self, 'noise_full_copy'):
            # x_input_noise = torch.rand_like(x_input_en)
            x_input_noise = self._sample_func(x_input_en, label)
            x_input_en = x_input_en + x_input_noise
            x_input_en = random_drop_func(x_input_en)
            
        elif hasattr(self, 'noise_copy'):
            ## x_input_noise = torch.rand_like(x_input_en[:, F_t_H:])
            # x_input_noise = self._sample_func(x_input_en[:, F_t_H:])
            # x_input_en[:, F_t_H:] = x_input_en[:, F_t_H:] + x_input_noise 
            x_input_noise = self._sample_func(x_input_en[F_t_H:], label)
            x_input_en[F_t_H:] = x_input_en[F_t_H:] + x_input_noise 
            x_input_en = random_drop_func(x_input_en)
            
        elif hasattr(self, 'noise_full_only'):
            # x_input_en = torch.rand_like(x_input_en)
            x_input_en = self._sample_func(x_input_en, label)
            x_input_en = random_drop_func(x_input_en)
            
        elif hasattr(self, 'noise_true_only'):       
            ## x_input_en[:, F_t_H:] = torch.rand_like(x_input_en[:, F_t_H:])
            # x_input_en[:, F_t_H:] = self._sample_func(x_input_en[:, F_t_H:])
            x_input_en[F_t_H:] = self._sample_func(x_input_en[F_t_H:], label)
            x_input_en = random_drop_func(x_input_en)
                
        elif hasattr(self, 'noise_only'):
            if self.random_drop <= 0:
                if kwargs["eval"]:
                    x_input_en = self._sample_func(x_input_en, label)
                else:
                    x_input_en = torch.clone(x_en)
                    x_input_en[:, F_t_H:] = self._sample_func(x_input_en[:, F_t_H:], label)
            else:
                x_input_en = self._sample_func(x_input_en, label)
                x_input_en = random_drop_func(x_input_en)

        elif hasattr(self, 'without_noise'):
            x_input_en = x_input_en 
            
        x_0, x_1 = x_input_en, x_en
        return x_0, x_1, torch.zeros_like(x_nll), x_nll 



class FlowMatchingPredict_V5(FlowMatchingPredictBase):
    def __init__(self, flow:FlowBaseNet, flow_dynamics:FlowMatchingDynamicsBase, **kwargs):
        super(FlowMatchingPredict_V5, self).__init__(flow, flow_dynamics, **kwargs)

        self.sampling_method = "Uniform" if "sampling_method" not in kwargs else kwargs["sampling_method"]        
        self.sampling_coef = 1.0 if "sampling_coef" not in kwargs else kwargs["sampling_coef"]                        
        if hasattr(self, "without_latent_space"):
            self.flow.delete_model()
        return

    def forward(self, x, t_pred=100, t_his=0, step_size=0.1, ode_method='euler', eval=True, bias=None, label=None, fm_nll=False, 
                **kwargs):
        _x_pos, _x_vel = x
        if _x_vel.ndim == 5:
            S, T, B, C, N = _x_pos.shape 
        else: 
            S, T, B, C, N =[1, *list(_x_vel.shape)]
            _x_pos, _x_vel = _x_pos[None, ...], _x_vel[None, ...]   
        x_pos, x_vel = [torch.swapaxes(item, 0, 1).reshape((T, S*B, C, N)) for item in [_x_pos, _x_vel]]

        _, f_pred = self.flow.get_seq_length()
        F_t_H = math.ceil(t_his / f_pred)
        F_T = math.ceil((t_his + t_pred) / f_pred)
        
        if hasattr(self, "output_pose") or hasattr(self, "output_pose_execpt_root"):
            x_0, x_1, x_0_nll, x_1_nll = self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval) if "flow_model_params" not in kwargs \
                                        else self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval, model_params=kwargs["flow_model_params"])
        else:
            x_0, x_1, x_0_nll, x_1_nll = self._build_model_input_target_pair(x_vel, t_his, F_t_H, eval=eval) if "flow_model_params" not in kwargs \
                                        else self._build_model_input_target_pair(x_pos, t_his, F_t_H, eval=eval, model_params=kwargs["flow_model_params"])
        
        x_0_full, x_1_full = [item.reshape((F_T, S*B, -1, C, N)) for item in [x_0, x_1]]
        if hasattr(self, "only_conditional_history"):
            raise NotImplementedError
        else:
            x_0, x_1 = x_0_full, x_1_full   
        
        if hasattr(self, "embedding_cond"):          
            x_cond = x_1_full[:F_t_H]
        elif hasattr(self, "preprocessing_cond"):   
            x_cond = self.data_preprocessing(x_pos[:t_his]) 
        elif hasattr(self, "double_cond"):
            x_cond_1 = x_1_full[:F_t_H]
            x_cond_2 = self.data_preprocessing(x_pos[:t_his]).reshape(x_cond_1.shape) 
            x_cond = torch.concat((x_cond_1, x_cond_2), dim=-3)
        else:
            x_cond = torch.swapaxes(torch.clone(x_pos[:t_his]), 0, 1)[None, ...] if not hasattr(self, "output_pose_velocity") \
                else torch.swapaxes(torch.clone(torch.concat((x_pos, x_vel), dim=-2)[:t_his]), 0, 1)[None, ...] 
        
        if self.dropout:
            x_cond = self.dropout(x_cond)
        
        if hasattr(self, "level") and self.level == "channel":            
            x_0, x_1 = [item.reshape((F_T, S*B, -1, C*N)) for item in [x_0, x_1]]
            x_cond = x_cond.reshape(1, S*B, -1, C*N)
        else:
            x_0, x_1 = [item.reshape((F_T, S*B, -1, N)) for item in [x_0, x_1]] 
            x_cond = x_cond.reshape(1, S*B, -1, N)
        x_0, x_1 = [item.permute(2, 1, 3, 0) for item in [x_0, x_1]]  
        x_cond = x_cond.permute(2, 1, 3, 0)
                 
        if eval:
            x_1_pred = self.flow_dynamics(x_0, x_1, t=torch.tensor([0.0, 1.0]).to(x_pos.device), \
                                          eval=True, step_size=step_size, method=ode_method, \
                                          x_cond=x_cond, label=label)
            x_1_pred_copy = x_1_pred.clone()
            x_1_pred = x_1_pred.permute(3, 1, 0, 2)
            if hasattr(self, "level") and self.level == "channel":
                x_1_pred = x_1_pred.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                            
            if hasattr(self, "without_latent_space"):
                y_1_pred = x_1_pred
                nll = 0.0
            else:    
                y_1_pred, nll = self.decode(x_1_pred, label=label, logdet=0.0)
                if fm_nll:
                    ll_2 = -nll
                    ll_2 = ll_2 * (x_1_pred.shape[-2] * x_1_pred.shape[-1])
                    logp = self.flow.distribution.logp(x_1_pred, label)
                    ll_2 = ll_2 - logp                
                    _, ll_1 = self.flow_dynamics.compute_likelihood(x_1_pred_copy, lambda x: torch.sum(-0.5*(((x)**2) + float(np.log(2*np.pi))), dim=(0, 2, 3)), 
                                                                    time_grid=torch.tensor([1.0, 0.0]).to(x_1_pred.device),
                                                                    step_size=step_size, method=ode_method, x_cond=x_cond, 
                                                                    label=label)
                    ll = ll_1 + ll_2
                    nll_fm = -ll / (x_1_pred.shape[-2] * x_1_pred.shape[-1])
                    nll = nll_fm
                nll = nll.reshape(S, B, 1)
                
            y_1_pred = self.data_posprocessing(y_1_pred)
            y_1_pred = y_1_pred.reshape(-1, S*B, C, N).reshape(-1, S, B, C, N).swapaxes(0, 1)

            y_1_pred = y_1_pred[:, -t_pred:]
            y_pos_pred = self.add_velocity(_x_pos[:, t_his-1:t_his], y_1_pred)
            y_pos_pred = y_pos_pred.swapaxes(1, 2)
            return y_pos_pred, nll
        else:
            if hasattr(self, "lognorm_sampling"):
                log_normal = LogNormal(0.0, 1.0)
                samples = log_normal.sample(sample_shape=(B,))
                samples = 1 / (1 + torch.exp(-samples))
                t = samples.to(x_pos.device).type(torch.float32)
            else:
                t = torch.rand(size=(B,)).to(x_pos.device).type(torch.float32)
            
            output = self.flow_dynamics(x_0, x_1, t=t, eval=False, x_cond=x_cond, label=label)
                
            if len(output) == 2:
                dx_t, dx_t_pred = output
                aux_loss = 0.0
            else:
                dx_t, dx_t_pred, aux_loss = output
            
            dx_t, dx_t_pred = dx_t.permute(3, 1, 0, 2), dx_t_pred.permute(3, 1, 0, 2)
            if hasattr(self, "level") and self.level == "channel":
                dx_t = dx_t.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                dx_t_pred = dx_t_pred.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                x_0 = x_0.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N)
                x_1 = x_1.reshape(F_T, S*B, -1, C, N).reshape(F_T, S*B, -1, N) 
            return dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss
        
        
    def _build_model_input_target_pair(self, x, t_his=25, F_t_H=1, label=None, eval=False, **kwargs):
        x = x  # (T, B, C, N)
        x_input = torch.clone(x)
        x_input[t_his:] = x_input[t_his-1]
        
        if hasattr(self, "without_latent_space"):
            x_en = self.data_preprocessing(x)
            x_input_en = self.data_preprocessing(x_input)
            x_nll, x_input_nll = torch.tensor([0.0]), torch.tensor([0.0])
        else:
            x_en, x_nll = self.encode(x, label)
            x_input_en, x_input_nll = self.encode(x_input, label)
            
        if hasattr(self, 'noise_copy'):
            x_input_noise = self._sample_func(x_input_en)
            x_input_en = x_input_en + x_input_noise            
        elif hasattr(self, 'noise_only'):    
            x_input_en = self._sample_func(x_input_en)   
        elif hasattr(self, 'noise_copy_only_eval'):
            if eval:
                x_input_noise = self._sample_func(x_input_en)
                x_input_en = x_input_en + x_input_noise
            else:
                x_input_en = x_input_en   
        elif hasattr(self, 'without_noise'):
            x_input_en = x_input_en 
                    
        x_0, x_1 = x_input_en, x_en
        return x_0, x_1, x_input_nll, x_nll
    

    def _sample_func(self, x, label=None):        
        if self.sampling_method == "Uniform":
            return torch.rand_like(x) * self.sampling_coef
        elif self.sampling_method == "Gaussian":
            return torch.normal(mean=torch.zeros_like(x), std=self.sampling_coef)        
        else:
            raise NotImplementedError


    def unnormalize_data(self, x):
        return x


    def add_velocity(self, pos, vel):
        if hasattr(self, "output_pose"):
            new_pos = vel            
        else:
            new_pos = torch.zeros_like(vel)
            for i in range(vel.shape[1]):
                new_pos[:, i:i+1] = pos + vel[:, i:i+1] if i == 0 else new_pos[:, i-1:i] + vel[:, i:i+1]
        return new_pos