import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


def cal_adjmatrix_mode(adj, mode='unchanged', device='cpu'):
    eye = torch.eye(adj.shape[0], device=device)
    _adj = adj + eye
    
    if mode == 'unchanged':
        return adj
    elif mode == 'sym_norm_adj' or mode == 'sys_norm_lap':
        row_degree = torch.sum(_adj, dim=1) if mode == 'sym_norm_adj' else torch.sum(adj, dim=1)
        D = torch.pow(row_degree, -0.5)
        D = torch.diag(D, diagonal=0)
        if mode == 'sym_norm_adj':
            D = D = torch.matmul(torch.matmul(D, _adj), D)
        if mode == 'sys_norm_lap':
            D = torch.matmul(torch.matmul(D, adj), D)
            D = D + eye
        return D
    elif mode == 'row_norm_adj' or mode == 'row_norm_lap':
        row_degree = torch.sum(_adj, dim=1) if mode == 'sym_norm_adj' else torch.sum(adj, dim=1)
        D = torch.pow(row_degree, -1.0)
        D = torch.diag(D, diagonal=0)
        if mode == 'row_norm_adj':
            D = D = torch.matmul(D, _adj)
        if mode == 'row_norm_lap':
            D = torch.matmul(D, adj), D
            D = D + eye
        return D



class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x    
        return x
    
    

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """
    def __init__(self, in_channels, out_channels, node_n=-1, bias=False) -> None:   
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = Parameter(torch.FloatTensor(out_channels, in_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_channels, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        return

    def reset_parameters(self):        
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        return

    def forward(self, input, adj_matrix):
        support = torch.matmul(self.weight, input)
        output = torch.matmul(support, adj_matrix)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')' 


class AdaGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, node_n, bias=False) -> None:   
        super(AdaGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = Parameter(torch.FloatTensor(out_channels, in_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_channels, 1))
        else:
            self.register_parameter('bias', None)
        self.ada = Parameter(torch.FloatTensor(node_n, node_n))
        self.reset_parameters()
        return

    def reset_parameters(self):        
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
            self.ada.data.uniform_(-stdv, stdv)
        return

    def forward(self, input, adj_matrix):
        support = torch.matmul(self.weight, input)
        ada_adj_matrix = torch.multiply(self.ada, adj_matrix)
        output = torch.matmul(support, ada_adj_matrix)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'   


class RdnAdjGraphConv(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """
    def __init__(self, in_channels, out_channels, node_n, bias=False) -> None:   
        super(RdnAdjGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_n = node_n
        
        self.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_channels, 1))
        else:
            self.register_parameter('bias', None)
        self.adj_matrix = nn.Parameter(torch.FloatTensor(node_n, node_n))
        self.reset_parameters()
        return

    def reset_parameters(self):        
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            self.adj_matrix.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        return

    def forward(self, input, adj):
        support = torch.matmul(self.weight, input)
        output = torch.matmul(support, self.adj_matrix)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')' 


class Graph1x1Conv(nn.Module):
    def __init__(self, in_channels, out_channels, node_n=-1, bias=False) -> None:
        super(Graph1x1Conv, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.bias = bias
        self.layer = torch.nn.Conv1d(in_channels, out_channels, 1, 1, bias=bias)
        return

    def forward(self, input, adj_matrix):
        output = self.layer(input)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'     


class GraphLinear(nn.Module):
    def __init__(self, in_channels, out_channels, node_n, bias=False) -> None:
        super(GraphLinear, self).__init__()
        self.input_size, self.output_size = in_channels*node_n, out_channels*node_n
        self.layer = nn.Linear(self.input_size, self.output_size, bias)
        return 
    
    def forward(self, input, adj=None):
        _input = torch.reshape(input, (input.shape[1], -1, input.shape[-1]))
        _input = torch.reshape(_input, (_input.shape[0], -1))
        _output = self.layer(_input)
        output = torch.reshape(_output, (input.shape[0], input.shape[1], -1, input.shape[-1]))
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_size) + ' -> ' \
               + str(self.output_size) + ')'


class GraphSeparatableLinear(nn.Module):
    def __init__(self, in_channels, out_channels, node_n, bias=False) -> None:
        super(GraphSeparatableLinear, self).__init__()
        self.in_channels, self.out_channels, self.node_n = in_channels, out_channels, node_n
        
        self.weight = Parameter(torch.FloatTensor(out_channels, in_channels, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels, node_n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        return
    
    def forward(self, input, adj=None):
        if input.ndim == 4:
            output = torch.einsum("ijN,LBjN->LBiN", self.weight, input)
        elif input.ndim == 3:
            output = torch.einsum("ijN,BjN->BiN", self.weight, input)  
        if self.bias is not None:
            return output + self.bias
        return output

    def reset_parameters(self):
        # nn.init.uniform_(self.weight)
        # if self.bias is not None:
        #     nn.init.zeros_(self.bias)        
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)
        return

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')' 


class GraphComplexSeparatableLinear(nn.Module):
    def __init__(self, in_channels, out_channels, node_n, bias=False) -> None:
        super(GraphComplexSeparatableLinear, self).__init__()
        self.in_channels, self.out_channels, self.node_n = in_channels, out_channels, node_n
        
        self.weight = Parameter(torch.FloatTensor(out_channels, in_channels, node_n).to(torch.cfloat))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels, node_n).to(torch.cfloat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        return

    def forward(self, input, adj=None):
        if input.ndim == 4:
            output = torch.einsum("ijN,LBjN->LBiN", self.weight, input).to(torch.cfloat)
        elif input.ndim == 3:
            output = torch.einsum("ijN,BjN->BiN", self.weight, input).to(torch.cfloat)  
        if self.bias is not None:
            return output + self.bias
        return output

    def reset_parameters(self):        
        nn.init.xavier_uniform_(self.weight).to(torch.cfloat)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias).to(torch.cfloat)
        return

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')' 



class GraphConvBlockBase(nn.Module):
    NetworkModelDict = {
        "fixed": GraphConvolution,
        "learnable": RdnAdjGraphConv,
        "linear": GraphLinear,
        "separatable": GraphSeparatableLinear,
        "cseparatable": GraphComplexSeparatableLinear
    }
    ActivationDict = {
        "LeakyReLU": nn.LeakyReLU(),
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU(),
        "Linear": None,
    }
    
    def __init__(self, channels_list:list, node_n:int, 
                gcn = 'learnable', 
                norm = None,
                dropout = 0.0, 
                skip_connect = True,
                final_block = False,
                activation = 'LeakyReLU') -> None:
        super().__init__()
                
        self.channels_list = channels_list
        self.node_n = node_n
        self.gcn = gcn
        self.norm = norm
        self.skip_connect = skip_connect
        self.final_block = final_block
        self.activation = activation
        
        self.network_layer = self.NetworkModelDict[gcn]
        
        self.layers = nn.ModuleList()
        for idx, _ in enumerate(channels_list[:-2]):
            self.layers.append(
                self.network_layer(channels_list[idx], channels_list[idx+1], node_n, bias=True)
            )
            if norm is not None:
                self.layers.append( self.norm(normalized_shape=[channels_list[idx+1], node_n]) )
            if activation:
                act = self.ActivationDict[self.activation]
                if act:
                    self.layers.append(act)
            if dropout:
                self.layers.append( nn.Dropout(p=dropout) )
            
        self.layers.append(
            self.network_layer(channels_list[-2], channels_list[-1], node_n=node_n, bias=True)
        )
        if self.final_block is False:
            if norm is not None:
                self.layers.append( self.norm(normalized_shape=[channels_list[idx+1], node_n]) )
            if activation:
                act = self.ActivationDict[self.activation]
                if act:
                    self.layers.append(act)
            if dropout:
                self.layers.append( nn.Dropout(p=dropout) )
        return
        
    def forward(self, input, adj_matrix):
        raise NotImplementedError
    
    
class GraphConvBlock(GraphConvBlockBase):    
    def __init__(self, channels_list:list, node_n:int, 
                gcn = 'learnable', 
                norm = None,
                dropout = 0.0, 
                skip_connect = True,
                final_block = False,
                activation = 'LeakyReLU') -> None:
        super(GraphConvBlock, self).__init__(channels_list, node_n, gcn, norm, dropout, skip_connect, final_block, activation)
        return
    
    def forward(self, input, adj_matrix):
        x = input  
        for layer in self.layers:
            if (layer.__class__.__name__.find("Graph") >= 0):
                x = layer(x, adj_matrix)
            else:
                x = layer(x)
        if self.skip_connect:
            x = torch.add(x, input)
        return x



class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):
    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- casualconv2d --- sigmoid ---|                               
    #
    
    #param x: tensor, [bs, c_in, ts, n_vertex]
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.n_vertex = n_vertex
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'GLU' or self.act_func == 'GTU':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]
            if self.act_func == 'GLU':
                # Explanation of Gated Linear Units (GLU):
                # The concept of GLU was first introduced in the paper 
                # "Language Modeling with Gated Convolutional Networks". 
                # URL: https://arxiv.org/abs/1612.08083
                # In the GLU operation, the input tensor X is divided into two tensors, X_a and X_b, 
                # along a specific dimension.
                # In PyTorch, GLU is computed as the element-wise multiplication of X_a and sigmoid(X_b).
                # More information can be found here: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # The provided code snippet, (x_p + x_in) ⊙ sigmoid(x_q), is an example of GLU operation. 
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                # tanh(x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'ReLU':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'SiLU':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        return x    


class SpatialTemporalGraphConvBlock(nn.Module):
    NormDict = {
        "BatchNorm": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "None": None   
    }

    def __init__(self, in_channel, channels, adj_matrix, temporal_kernel_size,
                 weight_mode = 'sym_norm_adj', 
                 gcn = 'fixed', 
                 spatial_activation = 'ReLU',
                 temporal_activation = 'ReLU',
                 norm = None,
                 dropout = 0.0) -> None:
        self.in_channels, self.channels = in_channel, channels
        self.adj_matrix = adj_matrix
        self.temporal_kernel_size = temporal_kernel_size
        self.node_n = self.adj_matrix.shape[0]
        self.weight_matrix = cal_adjmatrix_mode(self.adj_matrix, weight_mode, device=self.adj_matrix.device)
        self.gcn = gcn
        self.spatial_activation, self.temporal_activation = spatial_activation, temporal_activation
        self.norm = norm
        self.dropout = dropout
        
        super(SpatialTemporalGraphConvBlock, self).__init__()
        
        channel_list = [in_channel] + channels
        layer_list = nn.ModuleList()
        for idx, channel in enumerate(channels):
            if idx % 2 == 0:    
                tcl = TemporalConvLayer(Kt = self.temporal_kernel_size,
                                        c_in = channel_list[idx],
                                        c_out = channel, 
                                        n_vertex = self.node_n,
                                        act_func = self.temporal_activation)
                layer_list.append(tcl)
            else:
                scl = GraphConvBlock(channels_list = [channel_list[idx], channel], 
                                     node_n = self.node_n,
                                     gcn = self.gcn,
                                     norm = None,
                                     dropout = 0.0,
                                     skip_connect = False,
                                     final_block = True,
                                     activation = self.spatial_activation)
                layer_list.append(scl)
        
        norm_layer = self.NormDict[self.norm]
        if norm_layer:
            norm_layer = norm_layer([self.node_n, channel_list[-1]])
            layer_list.append(norm_layer)
                    
        if self.dropout > 0.0:
            layer_list.append(nn.Dropout(self.dropout))
        
        self.model = layer_list
        return
        
        
    def forward(self, x):
        for layer in self.model:
            if (layer.__class__.__name__.find("Graph") >= 0):
                x = layer(x, self.weight_matrix)
            elif (layer.__class__.__name__.find("Temporal") >= 0):
                x = x.permute(0, 2, 1, 3)
                x = layer(x)
                x = x.permute(0, 2, 1, 3)
            else:
                x = layer(x)
        return x 



class ForwardSpatialTemporalGraphConvLayer(nn.Module):
    def __init__(self, t_in_channel, t_out_channel, s_in_channel, s_out_channel, bias=True) -> None:   
        super(ForwardSpatialTemporalGraphConvLayer, self).__init__()
        self.t_in_channel, self.t_out_channel = t_in_channel, t_out_channel
        self.s_in_channel, self.s_out_channel = s_in_channel, s_out_channel
        
        self.weight = Parameter(torch.FloatTensor(t_out_channel, t_in_channel))
        self.att = Parameter(torch.FloatTensor(s_in_channel, s_out_channel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, t_out_channel, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        return

    def reset_parameters(self):        
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(self.att.size(1))
            self.att.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        return

    def forward(self, input):
        ''' input : (B, T, C*N)'''
        support = torch.matmul(self.weight, input)
        output = torch.matmul(support, self.att)
        if self.bias is not None:
            output =  output + self.bias
        return output



class ForwardSpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True) -> None:   
        super(ForwardSpatialGraphConvLayer, self).__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        
        self.weight = Parameter(torch.FloatTensor(in_channel, out_channel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, 1, out_channel))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        return

    def reset_parameters(self):        
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        return

    def forward(self, input):
        ''' input : (B, T, C*N)'''
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            output =  output + self.bias
        return output