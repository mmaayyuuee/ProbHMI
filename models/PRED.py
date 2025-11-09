import torch
import torch.nn as nn
import numpy as np
import math

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from modules import FlowBaseNet
from utils.motion_matching_database import MotionMatchingDatabase
import interface


MAX_BATCHSIZE = 1600

def flow_encoder_new(flow, x, label, **kwargs):
    if x.ndim == 5:
        length, group_size, batch_size, channel, node_n = x.shape
    elif x.ndim == 4:
        length, batch_size, channel, node_n = x.shape
    
    length_per_round = max(math.floor(MAX_BATCHSIZE / batch_size), 1)
    begin_idx = 0
    x_en, nll = [], []
    while(begin_idx < length):
        end_idx = begin_idx + length_per_round
        if end_idx > length:
            end_idx = length
        _x_en, _nll = interface.flow_interface(model=flow, x=x[begin_idx:end_idx], 
                                               logdet=0.0,
                                               reverse=False,
                                               label=label,
                                               in_batch_order=False,
                                               out_batch_order=False,
                                               **kwargs)
        x_en.append(_x_en), nll.append(_nll)        
        begin_idx = end_idx

    x_en, nll = torch.cat(x_en, 0), torch.cat(nll, 0)
    return x_en, nll 


def flow_decoder_new(flow, x, label, **kwargs):
    logdet = 0.0 if 'logdet' not in kwargs else kwargs['logdet']
    
    if isinstance(x, list) or isinstance(x, tuple):
        x, x_cond = x
    else:
        x, x_cond = x, None
        
    if x.ndim == 5:
        length, group_size, batch_size, channel, node_n = x.shape
    elif x.ndim == 4:
        length, batch_size, channel, node_n = x.shape
    
    length_per_round = max(math.floor(MAX_BATCHSIZE / batch_size), 1)
    begin_idx = 0
    x_de = []
    while(begin_idx < length):
        end_idx = begin_idx + length_per_round
        if end_idx > length:
            end_idx = length
        
        if x_cond is None:
            input = x[begin_idx:end_idx]
        else:
            input = (x[begin_idx:end_idx], x_cond[begin_idx:end_idx])
        
        _x_de, nll = interface.flow_interface(model=flow, x=input, 
                                              logdet=logdet,
                                              reverse=True,
                                              label=label,
                                              in_batch_order=False,
                                              out_batch_order=False)
        x_de.append(_x_de)  
        begin_idx = end_idx

    x_de = torch.cat(x_de, 0)
    return x_de, nll

      
    
class PredictBase(nn.Module):
    def __init__(self, flow:FlowBaseNet, rnn, norm=None, **kwargs) -> None:
        super(PredictBase, self).__init__()
        self.flow = flow
        self.rnn = rnn
        self.norm = norm
    
    def forward(self, x, t_pred, t_his=0, eval=True, step_by_step=None, **kwargs):
        raise NotImplementedError    
    
    def forward_train(self, x, t_pred, t_his):
        raise NotImplementedError

    def forward_eval(self, x, t_pred, t_his):
        raise NotImplementedError
    
    def forward_probability(self, x, t_pred, t_his, stds):
        raise NotImplementedError

    def forward_onestep(self, x, hncn, imbd):
        raise NotImplementedError
    
    def forward_onestep_probability(self, x, hncn, imbd, stds):
        raise NotImplementedError
    
    def reparameterize(self, mean, std=1.0, bias=None):
        if bias is not None:
            return mean + bias
        else:
            if std is not None:
                return mean + std*torch.rand_like(mean)
            else:
                return mean
    
    def truncated_sampling(self, mu, sigma, lt, ht):
        samp = torch.nn.init._no_grad_trunc_normal_(tensor=torch.empty_like(mu), mean=mu, std=sigma, a=lt, b=ht)
        return samp
    
    '''
    def forward_onestep(self, x, hncn):
        return self.predict(x, hncn)
    
    def forward_onestep_probability(self, x, hncn, std):
        if hncn is None:
            return self.forward_onestep(x, hncn)
        else:
            _x = x + torch.normal(mean=x, std=std)
            return self.forward_onestep(_x, hncn) 
    '''    
    
    # x: (L, B, C, N)
    def encode(self, x, label=None, padding=True):
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
            x_en, nll = flow_encoder_new(self.flow, conds, label)
            '''
            if padding:
                x_en = torch.reshape(x_en, (tlen, bs, -1, nodes))
            else:
                x_en = torch.reshape(x_en, (tlen-c_his, bs, -1, nodes))
            '''
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
            
            x_en, nll = flow_encoder_new(self.flow, groups, label)
            return x_en, nll
        
        
    def predict(self, x_his, hncn):
        x_pred, hncn = self.rnn(x_his, None, hncn)
        return x_pred, hncn
    
    
    # x: (L, B, C, N)
    def decode(self, x, label=None, padding=False):
        ### 好像没什么用了
        if x.ndim == 4:
            x = x[None, ...]
        samples, tlen, bs, channels, nodes = x.shape    
            
        if hasattr(self.flow, 'cond_model'):
            c_his, c_pred = self.flow.get_condition_length()
            if padding:
                _x = torch.zeros((samples, c_his+tlen, bs, channels, nodes), device=x.device)
                _x[:, c_his:] = x
            else:
                _x = x
            cond_list = []
            for idx in range(samples):
                for jdx in range(_x.shape[1]-c_his-c_pred+1):
                    cond_list.append(_x[idx:idx+1, jdx:jdx+c_his+c_pred])
            conds = torch.cat(cond_list, dim=0)
            x_de, _ = flow_decoder_new(self.flow, conds, label)
            if padding:
                x_de = torch.reshape(x_de, (samples, tlen, bs, channels, nodes))
            else:
                x_de = torch.reshape(x_de, (samples, tlen-c_his, bs, channels, nodes))
            return x_de
        else:
            _x = torch.reshape(x, (samples*tlen, bs, channels, nodes))
            _x = _x if self.norm is None else self.norm.unnormalize(_x)
            x_de, _ = flow_decoder_new(self.flow, _x, label)
            x_de = torch.reshape(x_de, x.shape)
            return x_de
    
    def set_actnorm_init(self, inited):
        self.flow.set_actnorm_init(inited)
    
    def get_layer_params_size(self):
        named_size_f, total_params_f = self.flow.get_layer_params_size()
        named_size_r, total_params_r = self.rnn.get_layer_params_size()
        named_size = dict(**named_size_f, **named_size_r)
        total_params = total_params_f + total_params_r
        return named_size, total_params


    @property
    def distribution_parameters(self):
        return self.flow.distribution_parameters

    def load_distribution_parameters(self, parameters_dict):
        return self.flow.load_distribution_parameters(parameters_dict)

    
    def __getitem__(self, item):
        if item =='0' or str.upper(item) == "FLOW":
            return self.flow
        elif item =='1' or str.upper(item) == "RNN":
            return self.rnn
    


class Seq2SeqPredictVelonlyDistribNet(PredictBase):
    def __init__(self, flow:FlowBaseNet, rnn, norm=None, **kwargs) -> None:
        super(Seq2SeqPredictVelonlyDistribNet, self).__init__(flow, rnn, norm, **kwargs)
                
        if "var_temp" in kwargs and kwargs["var_temp"]:
            self.var_temp = kwargs["var_temp"]
        else:
            self.var_temp = 0.9
        self.softplus = torch.nn.Softplus()        
        
        drop_coef = None if "drop_coef" not in kwargs else kwargs["drop_coef"]
        if hasattr(self, 'dropout_coef'):
            self.dropout = nn.Dropout(p=drop_coef) if drop_coef else nn.Dropout(p=self.dropout_coef)
        else:
            self.dropout = nn.Dropout(p=drop_coef) if drop_coef else None
        return
       
        
    def forward(self, x, t_pred=100, t_his=0, eval=True, step_by_step=False, sample=False, bias=None, **kwargs):
        zero_joints = kwargs['zero_joints'] if 'zero_joints' in kwargs else None
        mask        = kwargs['mask']        if 'mask'        in kwargs else None
        unsample_cond = kwargs['unsample_cond'] if 'unsample_cond' in kwargs else False
        CNF_only = kwargs['CNF_only'] if 'CNF_only' in kwargs else False
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
        copula_type   = kwargs['copula_type']   if 'copula_type'   in kwargs else None
        label = kwargs['label'] if 'label' in kwargs else None

        _x_pos, _x_vel, _x_vel_en = x
        hncn, iemb = None, None
        
        if _x_vel.ndim == 5:
            S, L, B, C, N = _x_vel.shape 
        else: 
            S, L, B, C, N =[1, *list(_x_vel.shape)]
            _x_pos, _x_vel, _x_vel_en = _x_pos[None, ...], _x_vel[None, ...], _x_vel_en[None, ...]
        
        x_pos, x_vel = [torch.swapaxes(item, 0, 1).reshape((L, S*B, C, N)) for item in [_x_pos, _x_vel]]
        x_pos_his, x_vel_his = x_pos[:t_his], x_vel[:t_his]

        y_pos, y_vel = torch.zeros(size=(S, L, B, C, N), device=x_pos.device), torch.zeros(size=(S, L, B, C, N), device=x_pos.device)
        y_pos[:, :t_his], y_vel[:, :t_his] = _x_pos[:, :t_his], _x_vel[:, :t_his]
        
        f_his, f_pred = self.flow.get_seq_length()
        f_in_channel = self.flow.get_channels()
        sub_group_num_his, sub_group_num_pred = math.ceil(t_his/f_pred), math.ceil(t_pred/f_pred)

        x_vel_en = torch.swapaxes(_x_vel_en, 0, 1).reshape((sub_group_num_his+sub_group_num_pred, S*B, f_in_channel, N))
        x_vel_en_his = x_vel_en[:sub_group_num_his][None, ...]
        y_vel_en, mus, sigmas = [torch.zeros((S, sub_group_num_pred, B, f_in_channel, N), device=x_pos.device) for _ in range(3)]
                    
        c_his, c_pred = self.flow.get_condition_length()
        if c_his > 0:
            if unsample_cond and eval:
                x_cond = y_pos.clone() if hasattr(self, 'pose_cond') else y_vel.clone()
            else:
                x_cond = y_pos if hasattr(self, 'pose_cond') else y_vel
        
        x_pos_his, x_vel_his = self.data_preprocessing(x_pos_his), self.data_preprocessing(x_vel_his)                
        input = torch.concat((x_pos_his, x_vel_his), dim=-2) if hasattr(self, 'decoder_only') else \
                torch.concat((x_pos_his, x_vel_en_his), dim=-2)
        if input.ndim == 5:
            input = torch.swapaxes(input, 0, 1).reshape((sub_group_num_his, S*B, 2*f_in_channel, N))
        
        nll_list = []
        for idx in range(sub_group_num_pred):
            if not eval:
                output, hncn, iemb = self.forward_onestep(input, hncn, iemb, std=False, bias=None, \
                                                        copula_radius=None, copula_type=None, CNF_only=CNF_only)
            else:
                _bias = bias[..., idx]
                if copula_radius is not None:
                    if copula_radius.ndim == 3:
                        _copula_radius = copula_radius[idx]
                    elif copula_radius.ndim == 4:
                        _copula_radius = copula_radius[:, idx][None]
                        _copula_radius = _copula_radius.expand((int(_bias.shape[0] / _copula_radius.shape[1]), -1, -1, -1))
                        _copula_radius = _copula_radius.reshape(-1, _copula_radius.shape[-2], _copula_radius.shape[-1])
                else:
                    _copula_radius = None
                if mask is not None:
                    output, hncn, iemb = self.forward_onestep(input, hncn, iemb, sample, _bias, mask, \
                                                            copula_radius=_copula_radius, copula_type=copula_type, CNF_only=CNF_only)
                else:
                    output, hncn, iemb = self.forward_onestep(input, hncn, iemb, sample, _bias, \
                                                            copula_radius=_copula_radius, copula_type=copula_type, CNF_only=CNF_only)
            y_vel_en_t, mus[:, idx:idx+1], sigmas[:, idx:idx+1] = [item.reshape((1, S, B, f_in_channel, N)).swapaxes(0, 1) for item in output]
            
            if eval:
                distrib = torch.distributions.Normal(loc=mus[:, idx:idx+1], scale=sigmas[:, idx:idx+1])
                log_prob = distrib.log_prob(y_vel_en_t)
            else:
                log_prob = 0.0
                                   
            if c_his > 0:
                # y_cond_t = self.data_preprocessing(x_cond[:, idx*c_pred+t_his-c_his:idx*c_pred+t_his])
                y_cond_t = x_cond[:, idx*c_pred+t_his-c_his:idx*c_pred+t_his]
                # y_vel_de = self.decode(torch.cat((y_cond_t, y_vel_en_t), dim=1))
                y_vel_de, nll = self.decode((y_vel_en_t, y_cond_t.clone()), label=label, logdet=log_prob)
                if unsample_cond and eval:
                    # temp = self.decode(torch.cat((y_cond_t, mus[:, idx:idx+1]), dim=1))
                    temp, _ = self.decode((mus[:, idx:idx+1], y_cond_t), label=label, logdet=log_prob)
                    x_cond[:, idx*c_pred+t_his:idx*(c_pred+1)+t_his] = \
                            self.add_velocity(x_cond[:, idx*c_pred+t_his-1:idx*c_pred+t_his], temp)  \
                                    if hasattr(self, 'pose_cond') else temp
            else:
                y_vel_de, nll = self.decode(y_vel_en_t, label=label, logdet=log_prob)

            if eval:
                nll = torch.mean(nll, dim=(-2, -1)) 
                nll_list.append(nll)
                
            y_vel_de = self.data_posprocessing(y_vel_de)
            
            if zero_joints is not None and hasattr(self, 'set_zeros'):
                y_vel_de[..., zero_joints] = 0.0
                
            # y_vel[:, t_his+idx:t_his+idx+1] = y_vel_de
            y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = y_vel_de
            # y_pos[:, t_his+idx:t_his+idx+1] = self.add_velocity(y_pos[:, t_his+idx-1:t_his+idx], y_vel_de)
            y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = self.add_velocity(y_pos[:, t_his+idx*c_pred-1:t_his+idx*c_pred], y_vel_de)
            
            y_vel_en[:, idx:idx+1] = y_vel_en_t

            y_pos_in = self.data_preprocessing(y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred])
            y_vel_in = self.data_preprocessing(y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred]) \
                            if hasattr(self, 'decoder_only') else y_vel_en[:, idx:idx+1]
            input = torch.concat((y_pos_in, y_vel_in), dim=-2)    
            input = torch.swapaxes(input, 0, 1).reshape((1, S*B, 2*f_in_channel, N))

            if self.dropout:
                input = self.dropout(input)
        
        if eval:        
            nll_seq = torch.cat(nll_list, dim=1)
            ll_seq = -1 * nll_seq
        else:
            ll_seq = 0.0
                        
        y_en = y_vel_en
        y = torch.concat((y_pos, y_vel), dim=-2)[:, t_his:]
        return y, y_en, (mus, sigmas), ll_seq


    def forward_onestep(self, x, hncn, iemb, std=None, bias=None, mask=None, \
                        copula_radius=None, copula_type=None, CNF_only=False):            
        x = x if not isinstance(x, tuple) else x[0]
        
        if CNF_only:
            length, batch_size, channels, node_n = x.shape
            pred_shape = (1, batch_size, self.flow.in_channels, node_n)
            mu, sigma = torch.zeros(size=pred_shape, device=x.device), torch.ones(size=pred_shape, device=x.device)        
        else:
            if hncn is None:
                _, hncn = self.predict(x, None)
                iemb = torch.cat(hncn, dim=-1) if len(hncn)>1 else hncn[0]
                iemb = torch.reshape(iemb, (iemb.shape[0], iemb.shape[1], -1, x.shape[-1]))
                rnn_in = x[-1]
            else:
                rnn_in = x
            
            rnn_in = torch.unsqueeze(rnn_in, 0) if rnn_in.dim() == 3 else rnn_in
            rnn_in = torch.cat((rnn_in, iemb), dim=-2)
            rnn_out, hncn = self.predict(rnn_in, hncn)
            mu, log_sigma = torch.split(rnn_out, int(rnn_out.shape[-2]/2), dim=-2)
            sigma = 0.1 + self.var_temp * self.softplus(log_sigma)
        
        if not copula_type:
            samp = self.reparameterize(mean=mu, std=sigma, bias=bias, mask=mask)
        else:
            if copula_type == 'mean':
                bias = bias[..., None, None].expand(-1, copula_radius.shape[-2], copula_radius.shape[-1])
                bias = bias * copula_radius
                samp = self.reparameterize(mean=mu, std=1.0, bias=bias, mask=mask)
            elif copula_type == 'stds':
                sigma = sigma * copula_radius
                samp = self.reparameterize(mean=mu, std=sigma, bias=bias, mask=mask)
        return (samp, mu, sigma), hncn, iemb


    def reparameterize(self, mean, std=1.0, bias=None, mask=None):        
        if bias is not None:
            indicator = torch.where(mean>=0, 1.0, -1.0)
            if bias.ndim <= 1:
                bias = bias[..., None, None].expand(-1, mean.shape[-2], mean.shape[-1])
            if mask is not None:
                return mean + (indicator * bias) * std * mask
            else:
                return mean + (indicator * bias) * std
        else:
            return mean


    def add_velocity(self, pos, vel):
        new_pos = torch.zeros_like(vel)
        for i in range(vel.shape[1]):
            new_pos[:, i:i+1] = pos + vel[:, i:i+1] if i == 0 else new_pos[:, i-1:i] + vel[:, i:i+1]
        return new_pos
    
    
    def data_preprocessing(self, data):
        if hasattr(self, 'dct_group'):
            return self.flow.data_preprocessing(data)
        else:
            return data if data.ndim == 5 else data[None, ...]
    
    def data_posprocessing(self, data):
        if hasattr(self, 'dct_group'):
            return self.flow.data_posprocessing(data)
        else:
            return data if data.ndim == 5 else data[:, None, ...]
    
    
    def decode(self, x, label=None, padding=False, logdet=0.0):
        '''
        if isinstance(x, list) or isinstance(x, tuple):
            x_de, nll = flow_decoder_new(self.flow, x, label, logdet=logdet)
            return x_de, nll
        else:
            return super().decode(x, label, padding)
        '''
        x_de, nll = flow_decoder_new(self.flow, x, label, logdet=logdet)
        return x_de, nll


    def non_autoregressive_forward(self, x, t_pred=100, t_his=0, eval=True, step_by_step=False, sample=False, bias=None, nums=1, **kwargs):
        zero_joints = kwargs['zero_joints'] if 'zero_joints' in kwargs else None
        mask = kwargs['mask'] if 'mask' in kwargs else None
        deterministic_bias = kwargs['deterministic_bias']
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
                
        y_d, _, (mus, sigmas), _ = self(x, t_pred, t_his, eval, step_by_step, sample, bias=deterministic_bias, **kwargs)
        y_d = y_d.expand((nums, -1, -1, -1, -1))
        
        mus, sigmas = mus.expand((nums, -1, -1, -1, -1)), sigmas.expand((nums, -1, -1, -1, -1))
        y_vel_en = torch.zeros_like(mus)
        
        mus = torch.swapaxes(mus, 0, 1).reshape((mus.shape[1], mus.shape[0]*mus.shape[2], mus.shape[3], mus.shape[4]))
        sigmas = torch.swapaxes(sigmas, 0, 1).reshape((sigmas.shape[1], sigmas.shape[0]*sigmas.shape[2], sigmas.shape[3], sigmas.shape[4]))
                
        _x_pos, _x_vel, _ = x
        if _x_vel.ndim == 5:
            _, L, B, C, N = _x_vel.shape 
        else: 
            L, B, C, N = _x_vel.shape 
                
        y_pos = torch.zeros(size=(nums, L, B, C, N), device=_x_pos.device) 
        y_vel = torch.zeros(size=(nums, L, B, C, N), device=_x_vel.device)
        y_pos[:, :t_his], y_vel[:, :t_his] = _x_pos[:, :t_his], _x_vel[:, :t_his]
        
        _, f_pred = self.flow.get_seq_length()
        f_in_channel = self.flow.get_channels()
        sub_group_num_pred = math.ceil(t_pred/f_pred)
                    
        c_his, c_pred = self.flow.get_condition_length()
        if c_his > 0:
            x_cond = torch.concat((y_pos[:, :t_his], y_d[..., :3, :]), dim=1) if hasattr(self, 'pose_cond') \
                    else torch.concat((y_vel[:, :t_his], y_d[..., 3:, :]), dim=1)
        
        nll_list = []    
        for idx in range(sub_group_num_pred):
            _bias = bias[..., idx]
            if copula_radius is not None:
                _bias = _bias[..., None, None].expand(-1, copula_radius[idx].shape[-2], copula_radius[idx].shape[-1])
                _bias = _bias * copula_radius[idx]
                y_vel_en_t = self.reparameterize(mean=mus[idx], std=1.0, bias=_bias, mask=mask)
            else:
                y_vel_en_t = self.reparameterize(mean=mus[idx], std=sigmas[idx], bias=_bias, mask=mask)
            
            y_vel_en_t = y_vel_en_t.reshape((1, nums, B, f_in_channel, N)).swapaxes(0, 1)
            y_vel_en[:, idx:idx+1] = y_vel_en_t

            if eval:
                distrib = torch.distributions.Normal(loc=mus[:, idx:idx+1], scale=sigmas[:, idx:idx+1])
                log_prob = distrib.log_prob(y_vel_en_t)
            else:
                log_prob = 0.0
            
            if c_his > 0:
                y_cond_t = x_cond[:, idx*c_pred+t_his-c_his:idx*c_pred+t_his]
                y_vel_de, nll = self.decode((y_vel_en_t, y_cond_t), logdet=log_prob)
            else:
                y_vel_de, nll = self.decode(y_vel_en_t, logdet=log_prob)
            
            if eval:
                nll = torch.mean(nll, dim=(-2, -1)) 
                nll_list.append(nll)
                
            y_vel_de = self.data_posprocessing(y_vel_de)  
            if zero_joints is not None and hasattr(self, 'set_zeros'):
                y_vel_de[..., zero_joints] = 0.0

            y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = y_vel_de
            y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = self.add_velocity(y_pos[:, t_his+idx*c_pred-1:t_his+idx*c_pred], y_vel_de) 

        if eval:
            nll_seq = torch.cat(nll_list, dim=1)
            ll_seq = -1 * nll_seq
        else:
            ll_seq = 0.0
        
        y_en = y_vel_en
        y = torch.concat((y_pos, y_vel), dim=-2)[:, t_his:]   
        return y, y_en, (mus, sigmas), ll_seq


    def motion_matching_single_forward(self, x, t_pred=100, t_his=0, sample_nums=1, bias=None, \
                                database:MotionMatchingDatabase=None, matching_metric=None, **kwargs):
        mask        = kwargs['mask']        if 'mask'        in kwargs else None
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
        copula_type   = kwargs['copula_type']   if 'copula_type'   in kwargs else None

        _x_pos, _x_vel, _x_vel_en = x
        hncn, iemb = None, None
        
        S, L, B, C, N = [sample_nums, *list(_x_vel.shape)]
        
        x_pos, x_vel = _x_pos, _x_vel
        x_pos_his, x_vel_his = x_pos[:t_his], x_vel[:t_his]
        
        y_pos, y_vel = torch.zeros(size=(S, L, B, C, N), device=x_pos.device), torch.zeros(size=(S, L, B, C, N), device=x_pos.device)
        y_pos[:, :t_his], y_vel[:, :t_his] = x_pos_his, x_vel_his

        f_his, f_pred = self.flow.get_seq_length()
        f_in_channel = self.flow.get_channels()
        sub_group_num_his, sub_group_num_pred = math.ceil(t_his/f_pred), math.ceil(t_pred/f_pred)   
        
        c_his, c_pred = self.flow.get_condition_length()     

        x_vel_en_his = _x_vel_en[:sub_group_num_his]
        y_vel_en = torch.zeros((S, sub_group_num_pred, B, f_in_channel, N), device=x_pos.device)
        mus, sigmas = [torch.zeros((sub_group_num_pred, B, f_in_channel, N), device=x_pos.device) for _ in range(2)]
                
        x_pos_his, x_vel_his = self.data_preprocessing(x_pos_his), self.data_preprocessing(x_vel_his)                
        input = torch.concat((x_pos_his, x_vel_his), dim=-2) if hasattr(self, 'decoder_only') else \
                torch.concat((x_pos_his, x_vel_en_his[None]), dim=-2)                
        if input.ndim == 5:
            input = torch.swapaxes(input, 0, 1).reshape((sub_group_num_his, 1*B, 2*f_in_channel, N))
        
        nll_list = []        
        for idx in range(sub_group_num_pred):
            bias_i = bias[..., idx] if bias is not None else None
            # copula_radius_idx = copula_radius[idx] if copula_radius is not None else None
            output, hncn, iemb = self.forward_onestep(input, hncn, iemb, bias=bias_i, mask=mask)           
            _, mu, sigma = output
            mu, sigma = mu.swapaxes(0, 1), sigma.swapaxes(0, 1)
            
            # y_de_t / y_vel_ent_t : (B, S, C, N) ; nll_t : (B, S)
            y_de_t, y_vel_en_t, nll_t = database.search(y=mu, y_sigma=sigma, topk=sample_nums, metric=matching_metric)
            y_de_t, y_vel_en_t = [d.swapaxes(0, 1)[:, None] for d in [y_de_t, y_vel_en_t]]
            
            # _, y_vel_de_t = torch.split(y_de_t, int(y_de_t.shape[-2]/2), dim=-2)
            y_vel_de_t = y_de_t
                
            y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = y_vel_de_t
            y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = self.add_velocity(y_pos[:, t_his+idx*c_pred-1:t_his+idx*c_pred], y_vel_de_t)    
                            
            y_vel_en[:, idx:idx+1] = y_vel_en_t
            nll = nll_t[None].permute(2, 1, 0)
            nll_list.append(nll)

            y_pos_in = self.data_preprocessing(y_pos[0:1, t_his+idx*c_pred:t_his+(idx+1)*c_pred])
            y_vel_in = self.data_preprocessing(y_vel[0:1, t_his+idx*c_pred:t_his+(idx+1)*c_pred]) \
                            if hasattr(self, 'decoder_only') else y_vel_en[0:1, idx:idx+1]
            input = torch.concat((y_pos_in, y_vel_in), dim=-2)    
            input = torch.swapaxes(input, 0, 1).reshape((1, 1*B, 2*f_in_channel, N))

            if self.dropout:
                input = self.dropout(input)
        
        nll_seq = torch.cat(nll_list, dim=-1)
        ll_seq = -1 * nll_seq
                                
        y_en = y_vel_en
        y = torch.concat((y_pos, y_vel), dim=-2)[:, t_his:]
        return y, y_en, (mus, sigmas), ll_seq


    def motion_matching_forward(self, x, t_pred=100, t_his=0, sample_nums=1, bias=None, \
                                database:MotionMatchingDatabase=None, matching_metric=None, **kwargs):
        mask        = kwargs['mask']        if 'mask'        in kwargs else None
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
        copula_type   = kwargs['copula_type']   if 'copula_type'   in kwargs else None

        _x_pos, _x_vel, _x_vel_en = x
        hncn, iemb = None, None
        
        if _x_vel.ndim == 5:
            S, L, B, C, N = _x_vel.shape 
        else: 
            S, L, B, C, N =[1, *list(_x_vel.shape)]
            _x_pos, _x_vel, _x_vel_en = _x_pos[None, ...], _x_vel[None, ...], _x_vel_en[None, ...]
        
        x_pos, x_vel = [torch.swapaxes(item, 0, 1).reshape((L, S*B, C, N)) for item in [_x_pos, _x_vel]]
        x_pos_his, x_vel_his = x_pos[:t_his], x_vel[:t_his]

        y_pos, y_vel = torch.zeros(size=(S, L, B, C, N), device=x_pos.device), torch.zeros(size=(S, L, B, C, N), device=x_pos.device)
        y_pos[:, :t_his], y_vel[:, :t_his] = _x_pos[:, :t_his], _x_vel[:, :t_his]
        
        f_his, f_pred = self.flow.get_seq_length()
        f_in_channel = self.flow.get_channels()
        sub_group_num_his, sub_group_num_pred = math.ceil(t_his/f_pred), math.ceil(t_pred/f_pred)

        c_his, c_pred = self.flow.get_condition_length() 

        x_vel_en = torch.swapaxes(_x_vel_en, 0, 1).reshape((sub_group_num_his+sub_group_num_pred, S*B, f_in_channel, N))
        x_vel_en_his = x_vel_en[:sub_group_num_his][None, ...]
        y_vel_en, mus, sigmas = [torch.zeros((S, sub_group_num_pred, B, f_in_channel, N), device=x_pos.device) for _ in range(3)]
        # y_vel_en = torch.zeros((S, sub_group_num_pred, B, f_in_channel, N), device=x_pos.device)
                            
        x_pos_his, x_vel_his = self.data_preprocessing(x_pos_his), self.data_preprocessing(x_vel_his)                
        input = torch.concat((x_pos_his, x_vel_his), dim=-2) if hasattr(self, 'decoder_only') else \
                torch.concat((x_pos_his, x_vel_en_his), dim=-2)
        if input.ndim == 5:
            input = torch.swapaxes(input, 0, 1).reshape((sub_group_num_his, S*B, 2*f_in_channel, N))
        
        nll_list = []        
        for idx in range(sub_group_num_pred):
            bias_i = bias[..., idx] if bias is not None else None
            # copula_radius_idx = copula_radius[idx] if copula_radius is not None else None
            output, hncn, iemb = self.forward_onestep(input, hncn, iemb, bias=bias_i, mask=mask)   
            # preds/mu/sigma : (1, S*B, C, N)        
            preds, mu, sigma = output
            preds, mu, sigma = preds.swapaxes(0, 1), mu.swapaxes(0, 1), sigma.swapaxes(0, 1)
            
            # y_de_t / y_vel_ent_t : (B, S, C, N) ; nll_t : (B, S)
            y_de_t, y_vel_en_t, nll_t = database.search(y=preds, y_sigma=sigma, topk=1, metric=matching_metric)
            y_de_t, y_vel_en_t = y_de_t.reshape((1, S, B, C, N)).swapaxes(0, 1), y_vel_en_t.reshape((1, S, B, f_in_channel, N)).swapaxes(0, 1)
                        
            y_vel_de_t = y_de_t
                
            y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = y_vel_de_t
            y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred] = self.add_velocity(y_pos[:, t_his+idx*c_pred-1:t_his+idx*c_pred], y_vel_de_t)    
                            
            y_vel_en[:, idx:idx+1] = y_vel_en_t
            nll = nll_t.reshape((S, B, -1))
            nll_list.append(nll)

            y_pos_in = self.data_preprocessing(y_pos[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred])
            y_vel_in = self.data_preprocessing(y_vel[:, t_his+idx*c_pred:t_his+(idx+1)*c_pred]) \
                            if hasattr(self, 'decoder_only') else y_vel_en[:, idx:idx+1]
            input = torch.concat((y_pos_in, y_vel_in), dim=-2)    
            input = torch.swapaxes(input, 0, 1).reshape((1, S*B, 2*f_in_channel, N))

            if self.dropout:
                input = self.dropout(input)
        
        nll_seq = torch.cat(nll_list, dim=-1)
        ll_seq = -1 * nll_seq
                                
        y_en = y_vel_en
        y = torch.concat((y_pos, y_vel), dim=-2)[:, t_his:]
        return y, y_en, (mus, sigmas), ll_seq
        