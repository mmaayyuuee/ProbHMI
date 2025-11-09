import numpy as np
import torch
import torch.nn as nn
import math
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: pe编码维度, 一般与word embedding相同, 方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度, 即word embedding中的L
        """
        super(PositionEmbedding, self).__init__()
        # 计算pe编码
        if d_model % 2 != 0:
            td_model = d_model+1
        else:
            td_model = d_model
            
        pe = torch.zeros(max_len, td_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, td_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / td_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        if d_model % 2 != 0:
            pe = pe[:, :d_model]
            
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的


    def forward(self, x, st=0, ed=None):
        _x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        _x = torch.swapaxes(_x, 0, 1)
        
        if ed is not None:
            _x = _x + self.pe[:, st:ed, :_x.shape[-1]]
        else:
            _x = _x + self.pe[:, st:, :_x.shape[-1]]
        
        y = torch.swapaxes(_x, 0, 1)
        y = torch.reshape(y, x.shape) # size = [batch, L, d_model]
        return y



if __name__ == "__main__":
    ped = PositionEmbedding(d_model=3, max_len=3)
    zero = torch.zeros(size=(2, 3, 3))
    zped = ped(zero)
    print(zped)
    
    
