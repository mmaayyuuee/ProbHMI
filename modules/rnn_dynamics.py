import torch
import torch.nn as nn
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from layers import GraphConvBlock
from layers import LayerNormGRU


class RecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, 
                bias = True,
                dropout = 0.0,
                birectional = False,
                type = 'LSTM', 
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentLayer, self).__init__()
        self.layer_type = type
        
        if type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias, \
                               dropout = dropout,
                               bidirectional = birectional
                            ).to(device)
        elif type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, bias, \
                              dropout = dropout,
                              bidirectional = birectional
                            ).to(device)
        elif type == "LayerNormGRU":
            self.rnn = LayerNormGRU(input_size, hidden_size, bias, dropout, \
                                    using_layer_norm = True,
                                    norm_learnable = True,
                                    device = device
                                ).to(device)
        self.__init_parameters()
    
    '''
    # return: 1.output; 2.hn; 3.(hn, cn)/hn is the input of next iter
    '''
    def forward(self, x, h0c0=None):
        if self.layer_type == 'LSTM':
            output, (hn, cn) = self.rnn(x) if h0c0 is None else self.rnn(x, h0c0)
            return output, hn, (hn, cn)
        elif self.layer_type == 'GRU':
            output, hn = self.rnn(x) if h0c0 is None else self.rnn(x, h0c0)
            return output, hn, hn
        elif self.layer_type == 'LayerNormGRU':
            output, hn = self.rnn(x) if h0c0 is None else self.rnn(x, h0c0)
            return output, hn, hn
    
    
    def __init_parameters(self):
        def init_gru(layer, gain=1):
            for _, hh, _, _ in layer.all_weights:
                hs = layer.hidden_size
                for i in range(0, hh.size(0), hs):
                    nn.init.orthogonal_(hh[i:i + hs], gain=gain)
            return
        
        if self.layer_type == 'GRU':
            init_gru(self.rnn)
        elif self.layer_type == 'LSTM':
            init_gru(self.rnn)
            for _, _, ih_b, hh_b in self.rnn.all_weights:
                l = len(ih_b)
                ih_b[l // 4:l // 2].data.fill_(1.0)
                hh_b[l // 4:l // 2].data.fill_(1.0)
        elif self.layer_type == 'LayerNormGRU':
            self.rnn.init_parameters()



class RecurrentDynamics(nn.Module):
    norm_dict = {
        "LayerNorm": nn.LayerNorm   
    }
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, adj, h0c0=None):
        raise NotImplementedError
    
    def get_layer_params_size(self):   
        raise NotImplementedError



class RecurrentDynamics_V1(RecurrentDynamics):
    '''
    Args:
        gcn_channels: gcn模块输入的channel numbers,不包括输入和输出
        node_n: 输入图的节点数
        parts_list: list, 其中每个item也是一个list(包括若干skeleton node),表示输入的图需如何划分
        rnn_input_channels: rnn的input_size=rnn_input_channels * len(parts_list[i]), 即RNN输入的每个node的channel
        rnn_hidden_channels: rnn的hidden_size=rnn_hidden_channels * len(parts_list[i]), 即RNN输出的每个node的channel
        rnn_bias: boolen
        rnn_dropout: float
        rnn_birectional: boolen, rnn是否是双向的, 但现在不支持双向
        gcn: 使用gcn的类型, 类别见GraphConvBlock类
        rnn: 使用rnn的类型, LSTM/GRU, 但目前没有实现GRU
        gcn_skip_connect: GCN内部是否使用残差连接
        rnn_skip_connect: RNN输出与RGCN输出间是否添加残差连接
        device: GPU/CPU
    '''
    '''
    the input of graphconv is consisted of all rnn's output
    '''
    def __init__(self, gcn_channels, node_n, parts_list,
                rnn_input_channels, rnn_hidden_channels, rnn_num_layers, 
                rnn_bias = True,
                rnn_dropout = 0.0,
                rnn_birectional = False,
                gcn_norm = None, 
                rnn_norm = None,   
                gcn = 'learnable', 
                rnn = 'LSTM',
                gcn_skip_connect = None,
                rnn_skip_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentDynamics, self).__init__()
        
        self.rnn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            self.rnn_list.append(
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                            hidden_size = rnn_hidden_channels * len(part), 
                            num_layers = rnn_num_layers,
                            bias = rnn_bias, 
                            dropout = rnn_dropout, 
                            birectional = rnn_birectional,
                            type = rnn, 
                            device = device,
                            **kwargs)                
            )
        
        if rnn_norm is not None and rnn_norm != "":
            self.rnn_norm = nn.ModuleList()
            for _ in range(len(parts_list)):
                self.rnn_norm.append( self.norm_dict[rnn_norm](rnn_hidden_channels*len(part)) )
        
        
        if len(gcn_channels) <= 0 or isinstance(gcn_channels[0], int):
            # 兼容旧版本            
            self.gcn = GraphConvBlock(channels_list = [rnn_hidden_channels]+gcn_channels+[rnn_input_channels],
                                      node_n = node_n,
                                      gcn = gcn,
                                      norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                      else None,
                                      skip_connect = False,
                                      final_block = True)
        else:
            self.gcn = nn.ModuleList()
            for idx, channels in enumerate(gcn_channels):
                final_one = True if idx == len(gcn_channels) else False
                _gcn = GraphConvBlock(channels_list = channels,
                                      node_n = node_n,
                                      gcn = gcn,
                                      norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                    else None,
                                      skip_connect = gcn_skip_connect[idx],
                                      final_block = True)
                self.gcn.append(_gcn) 
        
        self.extract_lists, self.putback_lists = [], []
        with torch.no_grad():
            for part in parts_list:
                extr = torch.tensor([1 if x in part else 0 for x in list(np.arange(node_n))]).to(device)
                self.extract_lists.append(extr)
                
                pb = torch.zeros(len(part), node_n).to(device)
                for idx, node in enumerate(part):
                    pb[idx, node] = 1
                self.putback_lists.append(pb)
        
        self.node_n = node_n
        self.parts_list = parts_list
        self.gcn_skip_connect = gcn_skip_connect
        self.rnn_skip_connect = rnn_skip_connect
        self.device = device
        return
                
    
    # input_shape: (length, batch_size, channel, node_n)
    def forward(self, x, adj, h0c0=None):
        for idx, extr in enumerate(self.extract_lists):
            # 将对应部位提取出来，例如：提取spine部位(3,18)->(3, 3)
            xpart = torch.masked_select(x, extr.bool())
            xpart = torch.reshape(xpart, (x.shape[0], x.shape[1], x.shape[2], -1))
            _xpart = torch.reshape(xpart, (xpart.shape[0], xpart.shape[1], -1))
            
            _, hn, hncn = self.rnn_list[idx](_xpart) if h0c0 is None else self.rnn_list[idx](_xpart, h0c0[idx])
            _xpart = hn
            if hasattr(self, "rnn_norm"):
                _xpart = self.rnn_norm[idx](_xpart)
            
            # 将RNN的结果变回(channel, node)的形状，并将相应部位放回到应在位置，例如((3,3)->(3, 18))
            _xpart = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[3]))
            _xpart = torch.matmul(_xpart, self.putback_lists[idx]).to(self.device)
            
            _x = _xpart if idx == 0 else torch.add(_x, _xpart)
            _hncn = [hncn] if idx == 0 else _hncn + [hncn]
        
        if isinstance(self.gcn, nn.ModuleList):
            # for idx, layer in enumerate(self.gcn):
            #     _x = layer(_x, adj)
            #     if self.rnn_skip_connect and idx == 0:
            #         _x = torch.add(_x, x[-1])
            y = x[-1]
            for idx, layer in enumerate(self.gcn):
                _x = layer(_x, adj)
                if self.rnn_skip_connect:
                    y = torch.add(_x, y)
            if not self.rnn_skip_connect:
                y = _x
        else:
            y = self.gcn(_x, adj)
            if self.rnn_skip_connect:
                y = torch.add(y, x[-1])
                
        return y, _hncn


    def get_layer_params_size(self):    
        named_size, total_params = {}, 0
        for idx, rnn in enumerate(self.rnn_list):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.'+name+' ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())
        
        for name, params in self.gcn.named_parameters():
            named_size['rgcn.'+'gcn.'+name] = params.size()
        total_params = total_params + sum(p.numel() for p in self.gcn.parameters())
        return named_size, total_params



class RecurrentDynamics_V2(RecurrentDynamics):
    '''
    Different from V1, each RNN in V2 has one graphconv counterpart.
    For example, if there are 2 rnn and 2 graphconv(means the length of parts_list is 2), 
    the input of each graphconv is one rnn's output. 
    '''
    def __init__(self, gcn_channels, node_n, parts_list,
                rnn_input_channels, rnn_hidden_channels, rnn_num_layers, 
                rnn_bias = True,
                rnn_dropout = 0.0,
                rnn_birectional = False,
                gcn_norm = None, 
                rnn_norm = None,   
                gcn = 'learnable', 
                rnn = 'LSTM',
                gcn_skip_connect = None,
                rnn_skip_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentDynamics_V2, self).__init__()
        
        self.rnn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            self.rnn_list.append(
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                              hidden_size = rnn_hidden_channels * len(part), 
                              num_layers = rnn_num_layers,
                              bias = rnn_bias, 
                              dropout = rnn_dropout, 
                              birectional = rnn_birectional,
                              type = rnn, 
                              device = device,  **kwargs)
            )

        if rnn_norm is not None and rnn_norm != "":
            self.rnn_norm = nn.ModuleList()
            for _ in range(len(parts_list)):
                self.rnn_norm.append( self.norm_dict[rnn_norm](rnn_hidden_channels*len(part)) )
                
        gcn_activation = 'LeakyReLU' if "gcn_activation" not in kwargs else kwargs["gcn_activation"]
        
        self.gcn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            model = nn.ModuleList()
            for idx, channels in enumerate(gcn_channels):
                final_one = True if idx == len(gcn_channels) else False
                _gcn = GraphConvBlock(channels_list = channels,
                                    node_n = len(part),
                                    gcn = gcn,
                                    norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                    else None,
                                    skip_connect = gcn_skip_connect[idx],
                                    final_block = True,
                                    activation = gcn_activation)
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
        self.gcn_skip_connect = gcn_skip_connect
        self.rnn_skip_connect = rnn_skip_connect
        self.device = device
        return


    # input_shape: (length, batch_size, channel, node_n)
    def forward(self, x, adj, h0c0=None):
        for idx, extr in enumerate(self.extract_lists):
            # 将对应部位提取出来，例如：提取spine部位(3,18)->(3, 3)            
            xpart = torch.index_select(x, -1, extr)
            _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))
            
            _, hn, hncn = self.rnn_list[idx](_xpart) if h0c0 is None else self.rnn_list[idx](_xpart, h0c0[idx])
            _xpart = hn
            if hasattr(self, "rnn_norm"):
                _xpart = self.rnn_norm[idx](_xpart)
            
            # 将RNN的结果变回(channel, node)的形状
            _x = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[-1]))
            
            y = xpart[-1]
            for _, layer in enumerate(self.gcn_list[idx]):
                _x = layer(_x, adj)
                if self.rnn_skip_connect:
                    y = torch.add(_x, y)
            if not self.rnn_skip_connect:
                y = _x
                
            # 将相应部位放回到应在位置，例如((3,3)->(3, 18))
            y = torch.matmul(y, self.putback_lists[idx]).to(self.device)
            
            _y = y if idx == 0 else torch.add(_y, y)           
            _hncn = [hncn] if idx == 0 else _hncn + [hncn]
                
        return _y, _hncn
    

    def get_layer_params_size(self):    
        named_size, total_params = {}, 0
        for idx, rnn in enumerate(self.rnn_list):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())
        
        for idx, gcn in enumerate(self.gcn_list):
            params = gcn.named_parameters()
            for name, param in params:
                named_size['rgcn.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in gcn.parameters())    
        return named_size, total_params



class RecurrentDynamics_V3(RecurrentDynamics):
    '''
    V3 is a seq2seq model, thus has an encoder RNN and a decoder RNN.
    As same as V2, each part RNN has own graphconv counterpart.
    '''
    def __init__(self, gcn_channels, node_n, parts_list,
                rnn_input_channels, rnn_hidden_channels, rnn_num_layers, 
                rnn_bias = True,
                rnn_dropout = 0.0,
                rnn_birectional = False,
                gcn_norm = None, 
                rnn_norm = None,   
                gcn = 'learnable', 
                rnn = 'LSTM',
                gcn_skip_connect = None,
                rnn_skip_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentDynamics_V3, self).__init__()

        self.rnn_encoder, self.rnn_decoder = nn.ModuleList(), nn.ModuleList()      
        for _, part in enumerate(parts_list):
            self.rnn_encoder.append(
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                              hidden_size = rnn_hidden_channels * len(part), 
                              num_layers = rnn_num_layers,
                              bias = rnn_bias, 
                              dropout = rnn_dropout, 
                              birectional = rnn_birectional,
                              type = rnn, 
                              device = device,  **kwargs))
            self.rnn_decoder.append(                
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                              hidden_size = rnn_hidden_channels * len(part), 
                              num_layers = rnn_num_layers,
                              bias = rnn_bias, 
                              dropout = rnn_dropout, 
                              birectional = rnn_birectional,
                              type = rnn, 
                              device = device,  **kwargs))
        
        if rnn_norm is not None and rnn_norm != "":
            self.rnn_norm = nn.ModuleList()
            for _ in range(len(parts_list)):
                self.rnn_norm.append( self.norm_dict[rnn_norm](rnn_hidden_channels*len(part)) )
        
        self.gcn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            model = nn.ModuleList()
            for idx, channels in enumerate(gcn_channels):
                final_one = True if idx == len(gcn_channels) else False
                _gcn = GraphConvBlock(channels_list = channels,
                                     node_n = len(part),
                                     gcn = gcn,
                                     norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                     else None,
                                     skip_connect = gcn_skip_connect[idx],
                                     final_block = True)
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
        self.gcn_skip_connect = gcn_skip_connect
        self.rnn_skip_connect = rnn_skip_connect
        self.device = device
        return


    # input_shape: (length, batch_size, channel, node_n)
    def forward(self, x, adj, h0c0=None):
        if h0c0 is None:
            # h0c0 is None --> encode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))        
                _, _, hncn = self.rnn_encoder[idx](_xpart)
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            return None, _hncn
        
        else:
            # h0c0 is not None --> decode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))
                
                _, hn, hncn = self.rnn_decoder[idx](_xpart, h0c0[idx])
                _xpart = hn
                if hasattr(self, "rnn_norm"):
                    _xpart = self.rnn_norm[idx](_xpart)
                
                # 将RNN的结果变回(channel, node)的形状
                _x = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[-1]))

                y = xpart[-1]
                for _, layer in enumerate(self.gcn_list[idx]):
                    _x = layer(_x, adj)
                    if self.rnn_skip_connect:
                        y = torch.add(_x, y)
                if not self.rnn_skip_connect:
                    y = _x
                    
                # 将相应部位放回到应在位置，例如((3,3)->(3, 18))
                y = torch.matmul(y, self.putback_lists[idx]).to(self.device)
                
                _y = y if idx == 0 else torch.add(_y, y)           
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            return _y, _hncn
                    
                
    def get_layer_params_size(self):    
        named_size, total_params = {}, 0
        for idx, rnn in enumerate(self.rnn_encoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.encoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())

        for idx, rnn in enumerate(self.rnn_decoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.decoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())
        
        for idx, gcn in enumerate(self.gcn_list):
            params = gcn.named_parameters()
            for name, param in params:
                named_size['rgcn.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in gcn.parameters())    
        return named_size, total_params



class RecurrentDynamics_V4(RecurrentDynamics):    
    '''
    V4 is a variation of V3. The input of the decoder of V4 is the concatenation of the output of
    the encoder and the T time pose.
    '''
    def __init__(self, gcn_channels, node_n, parts_list,
                rnn_input_channels, 
                rnn_hidden_channels:tuple, 
                rnn_num_layers:tuple, 
                rnn_bias:tuple = (True, True),
                rnn_dropout:tuple = (0.0, 0,0),
                rnn_birectional:tuple = (False, False),
                gcn_norm = None, 
                rnn_norm = None,   
                gcn = 'learnable', 
                rnn = 'LSTM',
                gcn_skip_connect = None,
                rnn_skip_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentDynamics_V4, self).__init__()

        self.rnn_encoder, self.rnn_decoder = nn.ModuleList(), nn.ModuleList()      
        for _, part in enumerate(parts_list):
            self.rnn_encoder.append(
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                              hidden_size = rnn_hidden_channels[0] * len(part), 
                              num_layers  = rnn_num_layers[0],
                              bias        = rnn_bias[0], 
                              dropout     = rnn_dropout[0], 
                              birectional = rnn_birectional[0],
                              type        = rnn, 
                              device      = device,  
                              **kwargs))
            self.rnn_decoder.append(                
                RecurrentLayer(input_size = (rnn_input_channels+rnn_hidden_channels[1]) * len(part), 
                              hidden_size = rnn_hidden_channels[1] * len(part), 
                              num_layers  = rnn_num_layers[1],
                              bias        = rnn_bias[1], 
                              dropout     = rnn_dropout[1], 
                              birectional = rnn_birectional[1],
                              type        = rnn, 
                              device      = device,
                              **kwargs))
        
        if rnn_norm is not None and rnn_norm != "":
            self.rnn_norm = nn.ModuleList()
            for _ in range(len(parts_list)):
                self.rnn_norm.append( self.norm_dict[rnn_norm]( (rnn_hidden_channels[1]+rnn_hidden_channels[0]) * len(part) ) )
        
        ''' # OLD VERSION 
        # self.enc_gcn_list = nn.ModuleList()
        # if 'enc_gcn' in kwargs and kwargs['enc_gcn'] is not None:
        #     enc_gcn_channles = kwargs['enc_gcn']
        #     for idx, part in enumerate(parts_list):
        #         model = nn.ModuleList()
        #         for idx, channels in enumerate(enc_gcn_channles):
        #             _gcn = GraphConvBlock(channels_list = channels,
        #                                   node_n = len(part),
        #                                   gcn = gcn,
        #                                   norm = None,
        #                                   skip_connect = False,
        #                                   final_block = False,
        #                                   activation = activation)
        #             model.append(_gcn)
        #         self.enc_gcn_list.append(model)        
        '''
        
        gcn_activation = 'LeakyReLU' if "gcn_activation" not in kwargs else kwargs["gcn_activation"]
        
        self.enc_gcn_list = nn.ModuleList()
        if 'enc_gcn' in kwargs and kwargs['enc_gcn'] is not None and kwargs['enc_gcn'] != "":
            enc_gcn_channles = kwargs['enc_gcn']
            for idx, channels in enumerate(enc_gcn_channles):
                _gcn = GraphConvBlock(channels_list = channels,
                                      node_n = node_n,
                                      gcn = gcn,
                                      norm = None,
                                      skip_connect = False,
                                      final_block = False,
                                      activation = gcn_activation)
                self.enc_gcn_list.append(_gcn) 
        
        self.gcn_list = nn.ModuleList()
        for _, part in enumerate(parts_list):
            model = nn.ModuleList()
            for idx, channels in enumerate(gcn_channels):
                final_one = True if idx == len(gcn_channels) else False
                _gcn = GraphConvBlock(channels_list = channels,
                                      node_n = len(part),
                                      gcn = gcn,
                                      norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                      else None,
                                      skip_connect = gcn_skip_connect[idx],
                                      final_block = True,
                                      activation = gcn_activation)
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
        self.gcn_skip_connect = gcn_skip_connect
        self.rnn_skip_connect = rnn_skip_connect
        self.device = device
        return
    

    # input_shape: (length, batch_size, channel, node_n)
    def forward(self, x, adj, h0c0=None):
        if h0c0 is None:
            # h0c0 is None --> encode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))   # shape = (length, batch_size, -1)       
                _, _, hncn = self.rnn_encoder[idx](_xpart)
                '''# OLD VERSION             
                # if len(self.enc_gcn_list) > 0:
                #     for _, layer in enumerate(self.enc_gcn_list[idx]):
                #         hncn = layer(hncn, adj)
                '''
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            _hncn_ = _hncn    
            if len(self.enc_gcn_list) > 0:
                for idx, hncn in enumerate(_hncn):
                    hncn = torch.reshape(hncn, shape=(hncn.shape[0], hncn.shape[1], -1, len(self.extract_lists[idx])))
                    hncn = torch.matmul(hncn, self.putback_lists[idx]).to(self.device)
                    _hncn = hncn if idx == 0 else torch.add(_hncn, hncn)
                    
                for _, layer in enumerate(self.enc_gcn_list):     
                    _hncn = layer(_hncn, adj)
                # _hncn = torch.reshape(_hncn, shape=(x.shape[0], x.shape[1], -1, self.node_n))
                
                for idx, extr in enumerate(self.extract_lists): 
                    hncn_part = torch.index_select(_hncn, -1, extr)
                    hncn_part = torch.reshape(hncn_part, shape=(hncn_part.shape[0], hncn_part.shape[1], -1))
                    _hncn_ = [hncn_part] if idx == 0 else _hncn_ + [hncn_part]
            return None, _hncn_
        
        else:
            # h0c0 is not None --> decode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))
                
                _, hn, hncn = self.rnn_decoder[idx](_xpart, h0c0[idx])
                _xpart = hn
                if hasattr(self, "rnn_norm"):
                    _xpart = self.rnn_norm[idx](_xpart)
                
                # 将RNN的结果变回(channel, node)的形状
                _x = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[-1]))

                y = xpart[-1]
                for _, layer in enumerate(self.gcn_list[idx]):
                    _x = layer(_x, adj)
                    if self.rnn_skip_connect:
                        y = torch.add(_x, y)
                if not self.rnn_skip_connect:
                    y = _x
                    
                # 将相应部位放回到应在位置，例如((3,3)->(3, 18))
                y = torch.matmul(y, self.putback_lists[idx]).to(self.device)
                
                _y = y if idx == 0 else torch.add(_y, y)           
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            return _y, _hncn
                    
                
    def get_layer_params_size(self):    
        named_size, total_params = {}, 0
        for idx, rnn in enumerate(self.rnn_encoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.encoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())

        for idx, rnn in enumerate(self.rnn_decoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.decoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())
        
        for idx, gcn in enumerate(self.gcn_list):
            params = gcn.named_parameters()
            for name, param in params:
                named_size['rgcn.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in gcn.parameters())    
        return named_size, total_params
    


class RecurrentDynamics_V5(RecurrentDynamics):
    '''
    V5 is for data containing velocity and position together.
    '''
    def __init__(self, gcn_channels, node_n, parts_list,
                rnn_input_channels, 
                rnn_hidden_channels:tuple, 
                rnn_num_layers:tuple, 
                rnn_bias:tuple = (True, True),
                rnn_dropout:tuple = (0.0, 0,0),
                rnn_birectional:tuple = (False, False),
                gcn_norm = None, 
                rnn_norm = None,   
                gcn = 'learnable', 
                rnn = 'LSTM',
                gcn_skip_connect = None,
                rnn_skip_connect = False,
                device = 'cpu',
                **kwargs) -> None:
        super(RecurrentDynamics_V5, self).__init__()

        self.rnn_encoder, self.rnn_decoder = nn.ModuleList(), nn.ModuleList()      
        for _, part in enumerate(parts_list):
            self.rnn_encoder.append(
                RecurrentLayer(input_size = rnn_input_channels * len(part), 
                              hidden_size = rnn_hidden_channels[0] * len(part), 
                              num_layers  = rnn_num_layers[0],
                              bias        = rnn_bias[0], 
                              dropout     = rnn_dropout[0], 
                              birectional = rnn_birectional[0],
                              type        = rnn, 
                              device      = device,  
                              **kwargs))
            self.rnn_decoder.append(                
                RecurrentLayer(input_size = (rnn_input_channels+rnn_hidden_channels[1]) * len(part), 
                              hidden_size = rnn_hidden_channels[1] * len(part), 
                              num_layers  = rnn_num_layers[1],
                              bias        = rnn_bias[1], 
                              dropout     = rnn_dropout[1], 
                              birectional = rnn_birectional[1],
                              type        = rnn, 
                              device      = device,
                              **kwargs))
        
        if rnn_norm is not None and rnn_norm != "":
            self.rnn_norm = nn.ModuleList()
            for _ in range(len(parts_list)):
                self.rnn_norm.append( self.norm_dict[rnn_norm]( (rnn_hidden_channels[1]+rnn_hidden_channels[0]) * len(part) ) )
            
        gcn_activation = 'LeakyReLU' if "gcn_activation" not in kwargs else kwargs["gcn_activation"]
        
        self.enc_gcn_list = nn.ModuleList()
        if 'enc_gcn' in kwargs and kwargs['enc_gcn'] is not None and kwargs['enc_gcn'] != "":
            enc_gcn_channles = kwargs['enc_gcn']
            self.enc_gcn_list = nn.ModuleList()
            for idx, part in enumerate(parts_list):
                model = nn.ModuleList()
                for idx, channels in enumerate(enc_gcn_channles):
                    _gcn = GraphConvBlock(channels_list = channels,
                                          node_n = node_n,
                                          gcn = gcn,
                                          norm = None,
                                          skip_connect = False,
                                          final_block = False,
                                          activation = gcn_activation)
                    model.append(_gcn)
                self.enc_gcn_list.append(_gcn)
        
        '''    
        # self.gcn_list = nn.ModuleList()
        # for idx, part in enumerate(parts_list):
        #     model = nn.ModuleList()
        #     for idx, channels in enumerate(gcn_channels):
        #         final_one = True if idx == len(gcn_channels) else False
        #         _gcn = GraphConvBlock(channels_list = channels,
        #                               node_n = len(part),
        #                               gcn = gcn,
        #                               norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
        #                                                               else None,
        #                               skip_connect = gcn_skip_connect[idx],
        #                               final_block = True,
        #                               activation = gcn_activation)
        #         model.append(_gcn)
        #     self.gcn_list.append(model)
        '''
        
        self.vel_list = nn.ModuleList()
        for idx, part in enumerate(parts_list):
            model = nn.ModuleList()
            vel_channels = gcn_channels[0]
            for idx, channels in enumerate(vel_channels):
                _gcn = GraphConvBlock(channels_list = channels,
                                      node_n = len(part),
                                      gcn = gcn,
                                      norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                      else None,
                                      skip_connect = gcn_skip_connect[0][idx],
                                      final_block = True,
                                      activation = gcn_activation)
                model.append(_gcn)
            self.vel_list.append(model) 

        self.pos_list = nn.ModuleList()
        for idx, part in enumerate(parts_list):
            model = nn.ModuleList()
            pos_channels = gcn_channels[1]
            for idx, channels in enumerate(pos_channels):
                _gcn = GraphConvBlock(channels_list = channels,
                                      node_n = len(part),
                                      gcn = gcn,
                                      norm = self.norm_dict[gcn_norm] if gcn_norm is not None and gcn_norm != "" \
                                                                      else None,
                                      skip_connect = gcn_skip_connect[1][idx],
                                      final_block = True,
                                      activation = gcn_activation)
                model.append(_gcn)
            self.pos_list.append(model)        
        
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
        self.gcn_skip_connect = gcn_skip_connect
        # self.rnn_skip_connect = rnn_skip_connect
        self.rnn_skip_connect = False
        self.device = device
        return
    

    # input_shape: (length, batch_size, channel, node_n)
    def forward(self, x, adj, h0c0=None):
        if h0c0 is None:
            # h0c0 is None --> encode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))   # shape = (length, batch_size, -1)       
                _, _, hncn = self.rnn_encoder[idx](_xpart)
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            _hncn_ = _hncn    
            if len(self.enc_gcn_list) > 0:
                for idx, hncn in enumerate(_hncn):
                    hncn = torch.reshape(hncn, shape=(hncn.shape[0], hncn.shape[1], -1, len(self.extract_lists[idx])))
                    hncn = torch.matmul(hncn, self.putback_lists[idx]).to(self.device)
                    _hncn = hncn if idx == 0 else torch.add(_hncn, hncn)
                    
                for _, layer in enumerate(self.enc_gcn_list):     
                    _hncn = layer(_hncn, adj)
                # _hncn = torch.reshape(_hncn, shape=(x.shape[0], x.shape[1], -1, self.node_n))
                
                for idx, extr in enumerate(self.extract_lists): 
                    hncn_part = torch.index_select(_hncn, -1, extr)
                    hncn_part = torch.reshape(hncn_part, shape=(hncn_part.shape[0], hncn_part.shape[1], -1))
                    _hncn_ = [hncn_part] if idx == 0 else _hncn_ + [hncn_part]
            return None, _hncn_        
        else:
            # h0c0 is not None --> decode
            for idx, extr in enumerate(self.extract_lists):
                xpart = torch.index_select(x, -1, extr)
                _xpart = torch.reshape(xpart, shape=(xpart.shape[0], xpart.shape[1], -1))
                
                _, hn, hncn = self.rnn_decoder[idx](_xpart, h0c0[idx])
                _xpart = hn
                if hasattr(self, "rnn_norm"):
                    _xpart = self.rnn_norm[idx](_xpart)
                
                # 将RNN的结果变回(channel, node)的形状
                _x = torch.reshape(_xpart, (_xpart.shape[0], _xpart.shape[1], -1, xpart.shape[-1]))
                
                # 计算vel部分
                _x_vel = _x
                for _, layer in enumerate(self.vel_list[idx]):
                    _x_vel = layer(_x_vel, adj)
                y_vel = _x_vel

                # 计算pos部分
                _x_pos = _x
                for _, layer in enumerate(self.pos_list[idx]):
                    _x_pos = layer(_x_pos, adj)
                y_pos = _x_pos
                    
                # 将相应部位放回到应在位置，例如((3,3)->(3, 18))
                y_pos = torch.matmul(y_pos, self.putback_lists[idx]).to(self.device)
                y_vel = torch.matmul(y_vel, self.putback_lists[idx]).to(self.device)
                
                _y_pos = y_pos if idx == 0 else torch.add(_y_pos, y_pos)
                _y_vel = y_vel if idx == 0 else torch.add(_y_vel, y_vel)           
                _hncn = [hncn] if idx == 0 else _hncn + [hncn]
            return (_y_vel, _y_pos), _hncn
                    
                
    def get_layer_params_size(self):    
        named_size, total_params = {}, 0
        for idx, rnn in enumerate(self.rnn_encoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.encoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())

        for idx, rnn in enumerate(self.rnn_decoder):
            params = rnn.named_parameters()
            for name, param in params:
                named_size['rgcn.decoder.'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in rnn.parameters())
        
        for idx, vel in enumerate(self.vel_list):
            params = vel.named_parameters()
            for name, param in params:
                named_size['rgcn.velgcn'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in vel.parameters()) 

        for idx, pos in enumerate(self.pos_list):
            params = pos.named_parameters()
            for name, param in params:
                named_size['rgcn.posgcn'+name+'  ('+'part'+str(idx)+')'] = param.size()
            total_params = total_params + sum(p.numel() for p in pos.parameters())          
        return named_size, total_params


    ''' # No longer needed
    # def resample(self, mu, sigma):
    #     samp = torch.normal(mean=torch.zeros_like(mu), std=torch.ones_like(sigma), device=mu.device)
    #     rsamp = mu + sigma * samp
    #     return rsamp
    '''




if __name__ == "__main__":    
    device = 'cpu'
    test_data = torch.arange(0, 648).to(device) * 1.0
    test_data = torch.reshape(test_data, (6, 2, 3, 18))
    rgcn = RGCN_V1(gcn_channels = [6, 12, 6],
                   node_n = 18,
                   parts_list = [[0, 7, 8],[9, 10, 11],[12, 13, 14],[15, 16, 17],[4, 5, 6],[1, 2, 3]],
                   rnn_input_channels = 3,
                   rnn_hidden_channels = 6,
                   rnn_num_layers = 1)
    test_output, hncn = rgcn(test_data, torch.zeros(18, 18))
    test_output2, hncn = rgcn(test_output, torch.zeros(18, 18), hncn)
    named_size, total_params = rgcn.get_layer_params_size()
    d = max(map(len, named_size.keys()))
    for name, size in named_size.items():
        print(name.ljust(d), " : ", size)
    print("Flow Total Params : {}".format(total_params))
    print("")