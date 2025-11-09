import torch
import torch.nn as nn
import numpy as np
import math

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from layers import SpatialTemporalGraphConvBlock, ForwardSpatialTemporalGraphConvLayer
from utils.dct import get_dct_matrix, Dct


class STGCN(nn.Module):
    batch_first_order = True
    
    def __init__(self, in_channel, channels, adjs, tks, wms, gcns, norms, dps,
                 spatial_act_list = ['ReLU'],
                 temporal_act_list = ['ReLU'], 
                 block_nums = 1,
                 *args, **kwargs) -> None:
        
        self.in_channel, self.channels = in_channel, channels
        self.adjs, self.wms = adjs, wms
        self.tks = tks
        self.gcns = gcns
        self.norms, self.dps = norms, dps
        self.spatial_act_list, self.temporal_act_list = spatial_act_list, temporal_act_list
        self.block_nums = block_nums
        super(STGCN, self).__init__(*args, **kwargs)
        
        self.model = nn.ModuleList()
        for idx in range(self.block_nums):
            st_block = SpatialTemporalGraphConvBlock(
                in_channel = self.in_channel if idx == 0 else channels[idx-1][-1],
                channels = self.channels[idx],
                adj_matrix = self.adjs[idx],
                temporal_kernel_size = self.tks[idx],
                weight_mode = self.wms[idx],
                gcn = self.gcns[idx],
                spatial_activation = self.spatial_act_list[idx],
                temporal_activation = self.temporal_act_list[idx],
                norm = self.norms[idx],
                dropout = self.dps[idx]
            )
            self.model.append(st_block)
        return
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def get_layer_params_size(self):   
        named_size, total_params = {}, 0 
        for idx, layer in enumerate(self.model):
            params = layer.named_parameters()
            for name, param in params:
                named_size['stgcn.block.' + str(idx) + '.' + name] = param.size()
            total_params = total_params + sum(p.numel() for p in layer.parameters())    
        return named_size, total_params  



class FowardSTGCN(nn.Module):
    batch_first_order = True

    NormDict = {
        "BatchNorm": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "None": None   
    }
    ActivationDict = {
        "LeakyReLU": nn.LeakyReLU(),
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU(),
        "Linear": None,
    }
    
    def __init__(self, t_in_channel, t_out_channels, s_in_channel, s_out_channels, node_n, \
                norms=["None"], dps=[0.0], acts=['ReLU'], adjs=None, *args, **kwargs) -> None:
        self.t_in_channel, self.t_out_channels = t_in_channel, t_out_channels
        self.s_in_channel, self.s_out_channels = s_in_channel, s_out_channels
        self.node_n = node_n
        self.norms, self.dps, self.acts = norms, dps, acts
        super(FowardSTGCN, self).__init__(*args, **kwargs)
        self.build_model()
        return
    
    def build_model(self):
        self.model = nn.ModuleList()
        t_channel_list = [self.t_in_channel] + self.t_out_channels
        s_channel_list = [self.s_in_channel] + self.s_out_channels
        for idx, out_channels in enumerate(zip(self.t_out_channels, self.s_out_channels)):
            t_out_channel, s_out_channel = out_channels
            layer = ForwardSpatialTemporalGraphConvLayer(t_in_channel = t_channel_list[idx], 
                                                         t_out_channel = t_out_channel,
                                                         s_in_channel = s_channel_list[idx]*self.node_n,
                                                         s_out_channel = s_out_channel*self.node_n,
                                                         bias = True)
            self.model.append(layer)
            
            norm = self.NormDict[self.norms[idx]]
            if norm is not None:
                norm_layer = norm(t_out_channel, self.channel*self.node_n)
                self.model.append(norm_layer)
            
            act = self.ActivationDict[self.acts[idx]]
            if act is not None:
                self.model.append(act)

            if self.dps[idx] > 0.0:
                self.model.append(nn.Dropout(p=self.dps[idx]))
        return
        
    def forward(self, x):
        ''' input : (B, T, C, N)'''
        _x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))   # shape: (B, T, C*N)
        for layer in self.model:
            _x = layer(_x)
        _y = torch.reshape(_x, shape=(_x.shape[0], _x.shape[1], -1, self.node_n))
        y = _y
        return y

    def get_layer_params_size(self):   
        named_size, total_params = {}, 0 
        for idx, layer in enumerate(self.model):
            params = layer.named_parameters()
            for name, param in params:
                named_size['FowardSTGCN.' + name] = param.size()
            total_params = total_params + sum(p.numel() for p in layer.parameters())    
        return named_size, total_params  
    
    @property
    def channels(self):
        return [self.s_out_channels]
    


class DctNet(nn.Module):   
    batch_first_order = True
    
    def __init__(self, n_len=125, n_pre=10, *args, **kwargs):     
        self.n_len, self.n_pre = n_len, n_pre
        super(DctNet, self).__init__()
        return
    
    def forward(self, x):
        if not hasattr(self, 'dct'):
            self.dct = Dct(n_len=self.n_len, n_pre=self.n_pre, device=x.device) 

        if x.ndim == 4:
            x = x[None, ...]
        ph, batch_size, length, channel, node_n = x.shape
        x = torch.swapaxes(x, 1, 2)  # shape: (ph, length, batch_size, length, channel, node_n)
        
        x_slices = []
        n_len = self.n_len if isinstance(self.n_len, int) else self.n_len[0]
        if length < n_len:
            idx_pad = [0]*(n_len-length) + list(range(length))
            x = x[idx_pad]
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
            x_dct = torch.reshape(x_dct, (ph, 1, batch_size, -1, node_n))
            x_dcts.append(x_dct)
        y = torch.concat(x_dcts, dim=1)    
        y = torch.squeeze(y)
        y = torch.swapaxes(y, 0, 1)
        return y


class DctSTGCN(STGCN, DctNet):
    def __init__(self, in_channel, channels, adjs, tks, wms, gcns, norms, dps,
                 spatial_act_list = ['ReLU'],
                 temporal_act_list = ['ReLU'], 
                 block_nums = 1,
                 n_len=25, n_pre=5,
                 *args, **kwargs) -> None:
        # super(DctSTGCN, self).__init__(channels, adjs, tks, wms, gcns, norms, dps,  \
        #                                spatial_act_list, temporal_act_list, block_nums,  \
        #                                in_channel=in_channel, n_len=n_len, n_pre=n_pre,  \
        #                                *args, **kwargs)
        super(DctSTGCN, self).__init__(in_channel=in_channel, channels=channels,                     
                                       adjs=adjs, tks=tks, wms=wms, gcns=gcns, norms=norms, dps=dps, 
                                       spatial_act_list=spatial_act_list, 
                                       temporal_act_list=temporal_act_list, 
                                       block_nums=block_nums, n_len=n_len, n_pre=n_pre,
                                       *args, **kwargs)
        return
    
    def forward(self, x):
        x1 = DctNet.forward(self, x)
        x2 = STGCN.forward(self, x1)
        return x2

        


if __name__ == "__main__":
    import argparse    
    import interface
    from utils.config import JsonConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default="20240428_1942")
    parser.add_argument('--hparams_path', default='results/prediction/H36Mso3N22Vel')
    parser.add_argument('--hparams_name', default='prediction_V3_VelonlyDistrib_V2_S2-2-4.json')
    parser.add_argument('--epoch', default='best')
    parser.add_argument('--actions', default='all')
    args = parser.parse_args()

    date         = args.date
    hparams_path = args.hparams_path
    hparams_name = args.hparams_name
    epoch        = args.epoch if args.epoch == 'best' else int(args.epoch)
    action       = args.actions

    hparams = os.path.join(os.getcwd(), hparams_path, "trained_"+date, hparams_name)
    hparams = JsonConfig(hparams)

    device = torch.device('cpu')

    np.random.seed(123)
    dataset_name = hparams.Dataset
    dataset = interface.dataset_interface(dataset_name, "test", hparams, actions=action)
    
    adj_matrix = torch.tensor(dataset.get_skeleton().adj_matrix_T)

    t_his = hparams.Data.t_his
    t_pred = hparams.Data.t_pred
    generator = dataset.sample_8()
    
    stgcn = STGCN(
                in_channel = 3,
                channels = [[32, 64, 32], [16, 32, 16], [8]],
                adjs = [adj_matrix, adj_matrix, adj_matrix],
                tks = [3, 3, 2],
                wms = ["sym_norm_adj", "sym_norm_adj", "sym_norm_adj"],
                gcns = ["fixed", "fixed", "fixed"],
                norms = ["None", "None", "None"],
                dps = [0.0, 0.0, 0.0],
                block_nums = 3,
                spatial_act_list = ['ReLU', 'ReLU', 'ReLU'],
                temporal_act_list = ['GLU', 'GLU', 'GLU'], 
            )

    named_size, total_params = stgcn.get_layer_params_size()
    d = max(map(len, named_size.keys()))
    for name, size in named_size.items():
        print(name.ljust(d), " : ", size)
    print("Total Params : {}".format(total_params))

    for x, label in generator:
        x = torch.tensor(x[:, :t_his, :3, :])
        y = stgcn(x)
        print()
    