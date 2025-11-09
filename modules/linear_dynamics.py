import torch
import torch.nn as nn
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from layers import GraphConvBlock
    
    
    
class LinearDynamics(nn.Module):
    norm_dict = {
        "LayerNorm": nn.LayerNorm   
    }
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, adj):
        raise NotImplementedError
    
    def get_layer_params_size(self):   
        raise NotImplementedError



class LinearDynamics_V1(LinearDynamics):
    def __init__(self, channels:tuple, 
                gcn = 'linear', 
                node_n = 18, 
                parts_list:tuple = [],
                activation = 'ReLU',
                norm = None,
                dropout = 0.0,
                skip_connect = None,
                device = 'cpu',
                **kwargs) -> None:
        super().__init__()
        
        self.gcn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            model = nn.ModuleList()
            for idx, channel in enumerate(channels):
                final_one = True if idx == len(channels)-1 else False
                _gcn = GraphConvBlock(channels_list = channel,
                                      node_n = len(part),
                                      gcn = gcn,
                                      norm = self.norm_dict[norm] if norm is not None and norm != "" \
                                                                    else None,
                                      dropout = dropout,
                                      skip_connect = False,
                                      final_block = final_one,
                                      activation = activation)
                model.append(_gcn)
            self.gcn_list.append(model)

        self.extract_lists, self.putback_lists = [], []
        with torch.no_grad():
            for part in parts_list:
                extr = torch.tensor(part).to(device)
                self.extract_lists.append(extr)
                pb = torch.zeros(len(part), node_n).to(device)
                for idx, node in enumerate(part):
                    pb[idx, node] = 1
                self.putback_lists.append(pb)

        self.node_n = node_n
        self.parts_list = parts_list
        self.skip_connect = skip_connect
        self.device = device
        self.r_node_n = len(parts_list)
        return
    
    
    def forward(self, x, adj):
        if self.node_n < self.r_node_n:
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1, self.r_node_n))        
        
        for idx, extr in enumerate(self.extract_lists):
            xpart = torch.index_select(x, -1, extr)
            _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))
                        
            _x = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[-1]))

            for _, layer in enumerate(self.gcn_list[idx]):
                _x = layer(_x, adj)
            # 将相应部位放回到应在位置，例如((3,3)->(3, 18))
            _x = torch.matmul(_x, self.putback_lists[idx]).to(self.device)
            
            y = _x if idx == 0 else torch.add(_x, y)      

        if self.node_n < self.r_node_n:
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1, self.node_n))     
        return y
    
    
    def get_layer_params_size(self):    
        named_size, total_params = {}, 0 
        for idx, gcn in enumerate(self.gcn_list):
            params = gcn.named_parameters()
            for name, param in params:
                named_size['lgcn.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in gcn.parameters())    
        return named_size, total_params



class LinearDynamics_V2(LinearDynamics):
    def __init__(self, channels:tuple, 
                gcn = 'separatable', 
                node_n = 18, 
                parts_list:tuple = [],
                activation = 'ReLU',
                norm = None,
                dropout = 0.0,
                skip_connect = None,
                res_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super().__init__()

        self.model = nn.ModuleList()
        for idx, channel in enumerate(channels):
            final_one = True if idx == len(channels)-1 else False
            _gcn = GraphConvBlock(channels_list = channel,
                                  node_n = len(parts_list),
                                  gcn = gcn,
                                  norm = self.norm_dict[norm] if norm is not None and norm != "" \
                                                              else None,
                                  dropout = dropout,
                                  skip_connect = skip_connect[idx] if skip_connect is not None else False,
                                  final_block = final_one,
                                  activation = activation)
            self.model.append(_gcn)

        self.node_n = node_n
        self.parts_list = parts_list
        self.skip_connect = skip_connect
        self.res_connect = res_connect
        self.device = device
        self.r_node_n = len(parts_list)
        return
    
    def forward(self, x, adj):
        if self.node_n < self.r_node_n:
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1, self.r_node_n))
        
        _x = x    
        for _, layer in enumerate(self.model):
            x = layer(x, adj)
        if self.res_connect:
            x = _x + x     

        if self.node_n < self.r_node_n:
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1, self.node_n))
        return x
    
    def get_layer_params_size(self):    
        named_size, total_params = {}, 0 
        for idx, layer in enumerate(self.model):
            params = layer.named_parameters()
            for name, param in params:
                named_size['lgcn.' + name] = param.size()
            total_params = total_params + sum(p.numel() for p in layer.parameters())    
        return named_size, total_params