import torch
import torch.nn as nn

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

import math
from typing import Tuple

class LayerNormGRU(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        using_layer_norm: bool = True,
        norm_learnable: bool = True,
        device = 'cpu'
    ):
        super(LayerNormGRU, self).__init__()
        self.using_layer_norm = using_layer_norm
        self.device = device
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
            self.do_drop = True
        else:
            self.do_drop = False
        
        if using_layer_norm:
            self.ln_i2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=norm_learnable)
            self.ln_h2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=norm_learnable)
            
            self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def init_parameters(self, gain=1):
        nn.init.orthogonal_(self.i2h.weight.data, gain)
        nn.init.orthogonal_(self.h2h.weight.data, gain)
        nn.init.orthogonal_(self.h_hat_W.weight.data, gain)
        nn.init.orthogonal_(self.h_hat_U.weight.data, gain)        

    
    # (L, H) for unbatched input, and (L, N, H) for N batched input
    # (zt, rt) = sigmoid(LN(W1*x)) + sgimoid(LN(W2*h))
    # ht_hat = tanh( W3*x + rt对位乘法(W4*h) )
    # ht = (1-zt)对位乘法h + zt对位乘法ht_hat 
    def forward_one_step(self, x:torch.Tensor, h0:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h = x, h0
        
        # 计算(zt, rt)
        i2h, h2h = self.i2h(x), self.h2h(h)
        if self.using_layer_norm:
            i2h, h2h = self.ln_i2h(i2h), self.ln_h2h(h2h)
        # i2h_max, h2h_max = torch.max(i2h), torch.max(h2h)
        
        preact = i2h + h2h
        gates = preact[:, :].sigmoid()
        z_t = gates[..., :self.hidden_size]
        r_t = gates[..., -self.hidden_size:]

        # 计算ht_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        if self.using_layer_norm:
            h_hat_first_half = self.ln_cell_1(h_hat_first_half)
            h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))
        if self.do_drop:
            h_hat = self.dropout(h_hat)
        
        # 计算ht
        h_t = torch.mul(torch.sub(1, z_t), h) + torch.mul(z_t, h_hat)
        h_t = h_t.view(h_t.size(0), -1)
        return h_t
    
    
    def forward(self, x, h0=None):
        if x.ndim == 2:
            x = torch.unsqueeze(x, 0)
        h_t = torch.zeros((1, x.shape[1], self.hidden_size)).to(self.device) if h0 is None else h0
        h_t_list = []
        for idx in range(x.shape[0]):
            h_t = self.forward_one_step(x[idx], h_t[0])
            # if self.do_drop:
            #     h_t = self.dropout(h_t)
            h_t_list.append(torch.unsqueeze(h_t, 0))
        return torch.cat(h_t_list), torch.unsqueeze(h_t, 0)