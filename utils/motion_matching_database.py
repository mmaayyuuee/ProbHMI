import torch
import numpy as np
import argparse
from tqdm import tqdm

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
import interface


class MotionMatchingDatabase(object):  
    metric_dict = {
        "mae": lambda self, x, y, y_sigma: self._mae(x, y),
        "mse": lambda self, x, y, y_sigma: self._mse(x, y),
        "cosine": lambda self, x, y, y_sigma: self._cosine(x, y),
        "kl": lambda self, x, y, y_sigma: self._kl(x, y, y_sigma),
    }
    
    def __init__(self, model, generator, device):
        x_list, x_en_list, nll_list = [], [], []
        with torch.no_grad():
            for x, label in tqdm(generator):
                torch.cuda.empty_cache()
                x = x.to(device) if isinstance(x, torch.Tensor) \
                                else torch.tensor(x, device=device)  # (Batch, length, channel, nodes)
                _, x = torch.split(x, int(x.shape[-2]/2), dim=-2)
                x_en, nll = model.encode(x)  # (B, T, C, N) / (B, T)
                
                # x_list.append(x.reshape(-1, x.shape[-2], x.shape[-1]).to('cpu'))
                # x_en_list.append(x_en.reshape(-1, x_en.shape[-2], x_en.shape[-1]).to('cpu'))
                # nll_list.append(nll.reshape(-1,).to('cpu'))
                x_list.append(x.reshape(-1, x.shape[-2], x.shape[-1]))
                x_en_list.append(x_en.reshape(-1, x_en.shape[-2], x_en.shape[-1]))
                nll_list.append(nll.reshape(-1,))
        
        self.x = torch.concat(x_list, dim=0)    # (B*T, C, N)
        self.x_en = torch.concat(x_en_list, dim=0)  # (B*T, C, N)
        self.nll = torch.concat(nll_list, dim=0)    # (B*T,)
        
        self.device = device
        return
    
    
    def search(self, y, y_sigma=None, topk=50, metric="mae"):
        dist = self.metric_dict[metric](self, self.x_en, y, y_sigma)
        # dist = self.metric_dict[metric](self, self.x_en, y.to('cpu'), y_sigma.to('cpu'))
        _, indices = torch.topk(dist, k=topk, dim=-1, largest=False, sorted=True)
        x_m, x_en_m, nll_m = self.x[indices], self.x_en[indices], self.nll[indices]     # (B, S, C, N) / (B, S)
        return x_m, x_en_m, nll_m
        # return x_m.to(self.device), x_en_m.to(self.device), nll_m.to(self.device)   
    
    
    ### x: self.x_en / y: y_en (prediction) (B, 1, C, N)
    def _mae(self, x, y):
        diff = x.unsqueeze(0) - y
        return torch.sum(torch.abs(diff), dim=(-2, -1))
    
    def _mse(self, x, y):
        diff = x.unsqueeze(0) - y
        return torch.norm(diff, p=2, dim=(-2, -1))

    def _cosine(self, x, y):
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], y.shape[1], -1)
        return torch.nn.functional.cosine_similarity(x_flat.unsqueeze(0), y_flat, dim=-1)
    
    def _kl(self, x, y, y_sigma):
        distribution = torch.distributions.Normal(y, y_sigma)
        nll = -torch.mean(distribution.log_prob(x), dim=(-2, -1))        
        return nll

    def _pearson_correlation(self, x, y):
        pass
    