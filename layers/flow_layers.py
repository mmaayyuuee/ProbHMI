import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy
from typing import Tuple

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from utils import thops



class myTanh(nn.Module):
    def forward(self, input):
        return torch.tanh(input) * 0.1


class ActNorm2d(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """    
    
    ActivationDict = {
        "LeakyReLU": (nn.LeakyReLU(), nn.LeakyReLU()),
        "ReLU": (nn.ReLU(), nn.ReLU()),
        "Tanh": (myTanh(), nn.Tanh()),
        "SiLU": (nn.SiLU(), nn.SiLU()),
        "Softplus": (nn.Softplus(), nn.Softplus()),       
        "Default": (lambda x: x, lambda x: torch.exp(x)),
        'Vacancy': (None, ),
    }    
    
    def __init__(self, num_features, scale=1., activations = ('Default', 'Default')):
        super().__init__()
        # register mean and scale        
        self.num_features = num_features
        self.scale = float(scale)
        
        bias_act, logs_act = activations
        self.bias_activation = self.ActivationDict[bias_act][0]
        self.logs_activation = self.ActivationDict[logs_act][1]   
             
        self.inited = False
        
        if self.bias_activation:
            self.register_parameter("bias", nn.Parameter(torch.zeros(*self.size)))
        self.register_parameter("logs", nn.Parameter(torch.ones(*self.size)))     
        return
    
    def initialize_parameters(self, input):        
        if self.training:
            logs = torch.log(self.scale/self.logs + 1e-6)
            self.logs.data.copy_(logs.reshape(shape=(1, -1)).data)
            self.inited = True
        return
    
    @property
    def size(self):
        return [1, self.num_features]
    
    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias_activation(self.bias)
        else:
            return input - self.bias_activation(self.bias)

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * self.logs_activation(logs)
        else:
            input = input / self.logs_activation(logs)
        if logdet is not None:
            dlogdet = thops.sum(logs) * int(input.size(-2))
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
            
        if not reverse:
            # center and scale
            if hasattr(self, "bias"):
                input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse)
            if hasattr(self, "bias"):
                input = self._center(input, reverse)
        return input, logdet


class ActNorm2d_channel(ActNorm2d):    
    def initialize_parameters(self, input):        
        if self.training:
            logs = torch.log(self.scale/self.logs + 1e-6)
            self.logs.data.copy_(logs.data)
            self.inited = True
        return

    @property
    def size(self):
        return [self.num_features, 1]
      
    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * self.logs_activation(logs)
        else:
            input = input / self.logs_activation(logs)
        if logdet is not None:
            dlogdet = thops.sum(logs) * int(input.size(-1))
            logdet = logdet + dlogdet
        return input, logdet
    

class ActNorm2d_plus(ActNorm2d):    
    def initialize_parameters(self, input):        
        if self.training:
            logs = torch.log(self.scale/self.logs + 1e-6)
            self.logs.data.copy_(logs.data)
            self.inited = True
        return

    @property
    def size(self):
        return self.num_features
      
    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * self.logs_activation(logs)
        else:
            input = input / self.logs_activation(logs)
        if logdet is not None:
            dlogdet = thops.sum(logs)
            logdet = logdet + dlogdet
        return input, logdet
        


def graph_split_matrix_rotation(node_n, init_invar_nidx):
    with torch.no_grad():
        mask1 = torch.ones(node_n, node_n)
        for i in init_invar_nidx:
            mask1[i, :] = torch.zeros(node_n)
        mask2 = torch.subtract(torch.ones(node_n, node_n), mask1)
        mask, cur_loc = [mask1, mask2], 0
        def real_funtion():
            nonlocal cur_loc
            z2_mask = mask[cur_loc]
            cur_loc = (cur_loc + 1) % len(mask)
            z1_mask = mask[cur_loc]
            
            # z1_mask, z1_m(invariant), z2_mask, z2_m
            # z1 = matual(z1_m, z) ; z2 = matual(z2_m, z)
            # 由于input_tensor是(C,N)，所以这里需要转置  
            return ( 
                    z1_mask.transpose(0, 1), \
                    torch.multiply(torch.eye(node_n, node_n), z1_mask).transpose(0, 1), \
                    z2_mask.transpose(0, 1), \
                    torch.multiply(torch.eye(node_n, node_n), z2_mask).transpose(0, 1) 
                )
    return real_funtion


class GraphSplitMatrix(object):
    def __init__(self, node_n, init_invar_nidx) -> None:
        super(GraphSplitMatrix, self).__init__()
        self.node_n = node_n
        self.init_invar_nidx = init_invar_nidx
        self.gsmr = graph_split_matrix_rotation(self.node_n, self.init_invar_nidx)
        return
    
    def get_split_matrix(self):
        return self.gsmr()


def split_feature(input_tensor, type='graph', z1_m=None, z2_m=None):
    if type == 'graph':
        z1, z2 = torch.matmul(input_tensor, z1_m), torch.matmul(input_tensor, z2_m)
        return z1, z2
    elif type == 'tensor':
        pass


class SplitSqueezeConfig(object):
    initial_invariant_node_index = {
        'bipartite0':{ 
            '2': [0, 2, 5, 8, 10, 12, 15, 14, 17],
            '1': [0, 4, 5],
            '0': [0] 
        },
        'bipartite1':{
            '2': [0, 10, 2, 4, 6, 8, 12, 22, 14, 16, 18, 20],
            '1': [0, 4, 5],
            '0': [0]
        },
        'bipartite2':{
            '2': [0, 9, 12, 17, 2, 5, 14, 4, 7],
            '1': [0, 4, 5],
            '0': [0] 
        },
        'bipartite3':{
            '2': [0, 9, 12, 2, 5, 14, 4, 7],
            '1': [0, 4, 5],
            '0': [0] 
        },
        'bipartite4':{
            '2': [0, 4, 10, 5, 11, 6, 12, 22, 16, 17, 20, 21],
            '1': [0, 3, 4],
            '0': [0]
        },
    }

    hierarchical_squeeze_pattern = {
        # (Human36M) 18 nodes for diverse motion prediction. 
        'squeeze0':{
            '1':[ [0,  7,  8], [ 1,  2,  3], [ 4,  5,  6], 
                  [9, 10, 11], [12, 13, 14], [15, 16, 17] ],
            '0':[ [0,  7,  8,  1,  2,  3,  4,  5,  6], 
                  [9, 10, 11, 12, 13, 14, 15, 16, 17] ]
        },
        # (Human36M) 22 nodes for deterministic motion prediction.
        'squeeze1':{
            '1':[ [0, 9, 10, 11], [1, 2, 3, 4], [5, 6, 7, 8],
                  [12, 13, 22, 23], [14, 15, 16, 17], [18, 19, 20, 21] ],
            '0':[ [0, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8],
                  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] ]
        },
        # (HumanEva-1) 18 nodes for diverse motion prediction.
        'squeeze2':{
            '1':[ [0, 16, 17], [ 8,  9, 10], [11, 12, 13], 
                  [1, 14, 15], [ 2,  3,  4], [ 5,  6,  7] ],
            '0':[ [0, 16, 17, 8,  9, 10, 11, 12, 13], 
                  [1, 14, 15, 2,  3,  4,  5,  6,  7] ]
        },
        # (HumanEva-1) 16 nodes for diverse motion prediction.
       'squeeze3':{
            '1':[ [0,  1, 14], [ 8,  9, 10], [11, 12, 13], 
                  [1, 14, 15], [ 2,  3,  4], [ 5,  6,  7] ],
            '0':[ [0,  1,  8,  9, 10, 11, 12, 13], 
                  [14, 15, 2,  3,  4,  5,  6,  7] ]
        },
        # (AMASS) 24 nodes for diverse motion prediction.
        'squeeze4':{
            '1':[ [0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11],
                  [14, 17, 19, 21], [13, 16, 18, 20], [12, 15, 22, 23] ],
            '0':[ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] ]
        },
    }
    
    def __init__(self, partition_type, squeeze_type) -> None:
        self.iini = self.initial_invariant_node_index[partition_type]
        self.hsq = self.hierarchical_squeeze_pattern[squeeze_type]
    
    
    def get(self, level):
        level = str(level)
        if level not in self.iini.keys() and level not in self.hsq.keys():
            return None, None
        elif level not in self.iini.keys():
            return None, self.hsq[level]
        elif level not in self.hsq.keys():
            return self.iini[level], None
        else:
            return self.iini[level], self.hsq[level]

    def next_level(self, level):
        level = int(level)
        return level - 1
    
    
    def compute_new_adjacency_matrix_after_squeeze(self, level, old_adj_matrix, masked=False):
        def mask(adj):
            nonlocal masked
            if masked is False:
                return adj
            next_level = self.next_level(level)
            _, mask_flag = self.get(next_level)
            if mask_flag is None:
                return adj
            
            location = [0] * adj.shape[0]
            for i, m_node in enumerate(mask_flag):
                for _, node in enumerate(m_node):
                    location[node] = i
            
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] > 0 and location[i] != location[j]:
                        adj[i, j] = 0
            return adj
        old_adj_matrix = mask(old_adj_matrix)
        
        _, squeeze_list = self.get(level)
        if squeeze_list is None:            
            return old_adj_matrix
        
        squeeze_list_np = np.array(squeeze_list)
        new_adj_matrix = torch.zeros(len(squeeze_list), len(squeeze_list))

        tmp_adj_matrix = torch.zeros(new_adj_matrix.shape[0], old_adj_matrix.shape[0])
        
        for i, m_node in enumerate(squeeze_list):
            for _, node in enumerate(m_node):
                tmp_adj_matrix[i] += old_adj_matrix[node]
        tmp_adj_matrix = torch.where(tmp_adj_matrix>=1, 1, 0)
        
        for i in range(tmp_adj_matrix.shape[0]):
            for j in range(tmp_adj_matrix.shape[1]):
                if tmp_adj_matrix[i, j] == 0:
                    continue
                idx = np.where(squeeze_list_np == j)
                new_adj_matrix[i, idx[0]] = 1
        new_adj_matrix = new_adj_matrix - torch.eye(new_adj_matrix.shape[0])
        
        return new_adj_matrix
 

class SqueezeBase(nn.Module):
    def __init__(self, hsp, adj_matrix, device):
        super().__init__()
        self.device = device
        self.squeeze_list = hsp
        if hsp == None:
            self.commutative_matrix_T = torch.eye(adj_matrix.shape[0]).to(self.device)
            self.commutative_matrix_T_inverse = self.commutative_matrix_T
            self.dim_times = 1
            return
        self.adj_matrix = adj_matrix
        self.dim_times = len(self.squeeze_list[0])
        with torch.no_grad():
            self._compute_commutative_matrix()
        return
    
    def forward(self, input, logdet, reverse=False):
        raise NotImplementedError
    
    def _compute_commutative_matrix(self):    
        temp = np.array(self.squeeze_list).flatten().tolist()
        commutative_matrix = torch.zeros(self.adj_matrix.shape[0], self.adj_matrix.shape[0])
        for i, n in enumerate(temp):
            commutative_matrix[i, n] = 1
        self.commutative_matrix = commutative_matrix
        # print(commutative_matrix)
        self.commutative_matrix_T = self.commutative_matrix.transpose(0, 1).to(self.device)
        self.commutative_matrix_T_inverse = torch.inverse(self.commutative_matrix_T)
        return
                
 
class Squeeze(SqueezeBase):
    def __init__(self, hsp, adj_matrix, device='cpu'):
        super(Squeeze, self).__init__(hsp, adj_matrix, device)
        
    def forward(self, input, logdet=None, reverse=False):
        if isinstance(input, (list, tuple)):
            output = []
            for x in input:
                if x is not None:
                    y, logdet = self.forward_onestep(x, logdet, reverse)
                    output.append(y)
                else:
                    output.append(x)
        else:
            output, logdet = self.forward_onestep(input, logdet, reverse)   
        return output, logdet 
    
    def forward_onestep(self, input, logdet=None, reverse=False):
        if reverse == False:
            x = input
            x = torch.matmul(x, self.commutative_matrix_T)
            if x.ndim == 3:
                x = torch.reshape(input=x, shape=(-1, int(x.shape[1]*self.dim_times), int(x.shape[2]/self.dim_times)))
            elif x.ndim == 4:
                x = torch.reshape(input=x, shape=(x.shape[0], x.shape[1], int(x.shape[2]*self.dim_times), int(x.shape[3]/self.dim_times)))
            else:
                print("SqueezeError : x.ndim is not 3 or 4 ......")
        else:
            x = input
            if x.ndim == 3:
                x = torch.reshape(input=x, shape=(-1, int(x.shape[1]/self.dim_times), int(x.shape[2]*self.dim_times)))
            elif x.ndim == 4:
                x = torch.reshape(input=x, shape=(x.shape[0], x.shape[1], int(x.shape[2]/self.dim_times), int(x.shape[3]*self.dim_times)))
            else:
                print("SqueezeError : x.ndim is not 3 or 4 ......")
            x = torch.matmul(x, self.commutative_matrix_T_inverse)
        return x, logdet
    

class UnSqueeze(SqueezeBase):
    def __init__(self, hsp, adj_matrix, device='cpu'):
        super(UnSqueeze, self).__init__(hsp, adj_matrix, device)

    def forward(self, input, logdet=None, reverse=False):
        if isinstance(input, (list, tuple)):
            output = []
            for x in input:
                if x is not None:
                    y, logdet = self.forward_onestep(x, logdet, reverse)
                    output.append(y)
                else:
                    output.append(x)
        else:
            output, logdet = self.forward_onestep(input, logdet, reverse)   
        return output, logdet 
    
    def forward_onestep(self, input, logdet=None, reverse=False):
        if reverse == False:
            x = input
            # x = torch.reshape(input=x, shape=(-1, int(x.shape[1]/self.dim_times), int(x.shape[2]*self.dim_times)))
            if x.ndim == 3:
                x = torch.reshape(input=x, shape=(-1, int(x.shape[1]/self.dim_times), int(x.shape[2]*self.dim_times)))
            elif x.ndim == 4:
                x = torch.reshape(input=x, shape=(x.shape[0], x.shape[1], int(x.shape[2]/self.dim_times), int(x.shape[3]*self.dim_times)))
            else:
                print("UnSqueezeError : x.ndim is not 3 or 4 ......")
            x = torch.matmul(x, self.commutative_matrix_T_inverse)
        else:
            x = input
            x = torch.matmul(x, self.commutative_matrix_T)
            # x = torch.reshape(input=x, shape=(-1, int(x.shape[1]*self.dim_times), int(x.shape[2]/self.dim_times)))
            if x.ndim == 3:
                x = torch.reshape(input=x, shape=(-1, int(x.shape[1]*self.dim_times), int(x.shape[2]/self.dim_times)))
            elif x.ndim == 4:
                x = torch.reshape(input=x, shape=(x.shape[0], x.shape[1], int(x.shape[2]*self.dim_times), int(x.shape[3]/self.dim_times)))
            else:
                print("UnSqueezeError : x.ndim is not 3 or 4 ......")
        return x, logdet