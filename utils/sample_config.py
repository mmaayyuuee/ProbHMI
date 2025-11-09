import torch
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())



class StdsConfig(object):
    @classmethod
    def empty(cls):
        return None 
    
    # return [value, value, value, ..., value]
    @classmethod
    def alpha(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            return [float(value)] * length
    
    # return [0, value, 0, value, ..., 0, value]
    @classmethod
    def beta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            stds = float(value) * np.ones(shape=(length, ))
            stds[1::2] = 0.0
            return stds.tolist()
    
    # return [0, ..., 0,  ..., value, ..., value]
    @classmethod
    def gamma(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            stds = float(value) * np.ones(shape=(length, ))
            stds[:int(length/2)] = 0.0
            return stds.tolist()

    # return [value, ..., value,  ..., 0, ..., 0]
    @classmethod
    def delta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            stds = float(value) * np.ones(shape=(length, ))
            stds[int(length/2):] = 0.0
            return stds.tolist()

    # return [value, 0, ..., 0]
    @classmethod
    def epsilon(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            stds = np.zeros(shape=(length, ))
            stds[0] = float(value)
            return stds.tolist()  

    # return [1*value/n, 2*value/n, 3*value/n, ..., n*value/n]
    @classmethod
    def zeta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            stds = ((np.arange(0, length) * 1.0) / length) * float(value) 
            return stds.tolist() 
        

class BiasConfig(object):    
    @classmethod
    def empty(cls):
        return None

    @classmethod
    def alpha(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                bias = value * torch.randn(size=(1,), device=value.device) * torch.ones(size=(length, ), device=value.device)
            else:
                bias = np.random.normal(loc=0, scale=float(value), size=(1,)) * np.ones(shape=(length, ))
            return bias

    @classmethod
    def beta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                bias = value * torch.randn(size=(1,), device=value.device) * torch.ones(size=(length, ), device=value.device)
            else:
                bias = np.random.normal(loc=0, scale=float(value), size=(1,)) * np.ones(shape=(length, ))
            bias[1::2] = 0.0
            return bias

    @classmethod
    def gamma(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                bias = value * torch.randn(size=(1,), device=value.device) * torch.ones(size=(length, ), device=value.device)
            else:
                bias = np.random.normal(loc=0, scale=float(value), size=(1,)) * np.ones(shape=(length, ))
            bias[:int(length/2)] = 0.0
            return bias.tolist()

    @classmethod
    def delta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                bias = value * torch.randn(size=(1,), device=value.device) * torch.ones(size=(length, ), device=value.device)
            else:
                bias = np.random.normal(loc=0, scale=float(value), size=(1,)) * np.ones(shape=(length, ))
            bias[int(length/2):] = 0.0
            return bias.tolist()

    @classmethod
    def epsilon(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                _bias = value * torch.randn(size=(1,), device=value.device)
                bias = torch.zeros(size=(length, ), device=_bias.device)
                bias[0] = _bias
            else:
                _bias = np.random.normal(loc=0, scale=float(value), size=(1,))
                bias = np.zeros(shape=(length, ))
                bias[0] = _bias
            return bias

    @classmethod
    def zeta(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                _bias = value * torch.randn(size=(1,), device=value.device)
                coef = ((torch.arange(0, length, device=_bias.device)+1) * 1.0) / length
                bias = coef * _bias
            else:
                _bias = np.random.normal(loc=0, scale=float(value), size=(1,))
                coef = ((np.arange(0, length)+1) * 1.0) / length
                bias = coef * _bias
            return bias
    
    @classmethod
    def eta(cls, length, value, *args):
        return args[0]

    @classmethod
    def theta(cls, length, value):
        if value is None:
            length = sum(length) if isinstance(length, list) else length
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                pass
            else:
                # compute value
                value = value.tolist() if isinstance(value, np.ndarray) else value
                if isinstance(value, list):
                    value = [0, value[0]] if len(value) == 1 else value
                else:
                    value = [0, value]

                # compute length
                length = length.tolist() if isinstance(length, np.ndarray) else length
                if isinstance(length, list):
                    length = [length[0], 0] if len(length) == 1 else length
                else:
                    length = [length, 0]
                
                # compute bias
                for i, valueues in enumerate(zip(value, length)):
                    _, le = valueues
                    if le <= 0:
                        continue
                    low, high = value[i], value[i+1] 
                    coef = ((np.arange(0, le)+1) * 1.0) / le
                    _bias = low + coef*(high-low)
                    bias = _bias if i==0 else np.append(bias, _bias)
            return bias

    @classmethod
    def aiot(cls, length, value):
        length = sum(length) if isinstance(length, list) else length
        if value is None:
            return [value] * length
        else:
            if isinstance(value, torch.Tensor):
                _bias = value * torch.randn(size=(1,), device=value.device)
                coef = ((torch.arange(0, length, device=_bias.device)+1) * 1.0) / length
                coef = torch.flip(coef, 0)
                bias = coef * _bias
            else:
                _bias = np.random.normal(loc=0, scale=float(value), size=(1,))
                coef = ((np.arange(0, length)+1) * 1.0) / length
                coef = np.flip(coef, 0)
                bias = coef * _bias
            return bias
    
    @classmethod
    def kappa(cls, length, value, *args):  
        return args[0]
    
    @classmethod
    def lambd(cls, length, value, *args):
        return value
    

class SampleConfig(object):
    alphabet = {
        '0': lambda cls, *args: cls.empty(),
        '1': lambda cls, length, value, *args: cls.alpha(length, value),
        '2': lambda cls, length, value, *args: cls.beta(length, value),
        '3': lambda cls, length, value, *args: cls.gamma(length, value),
        '4': lambda cls, length, value, *args: cls.delta(length, value),
        '5': lambda cls, length, value, *args: cls.epsilon(length, value),
        '6': lambda cls, length, value, *args: cls.zeta(length, value),
        '7': lambda cls, length, value, *args: cls.eta(length, value, SampleConfig.alphabet[args[0]](cls, length, value, False)),
        '8': lambda cls, length, value, *args: cls.theta(length, value),
        '9': lambda cls, length, value, *args: cls.aiot(length, value),
        'A': lambda cls, length, value, *args: cls.kappa(length, value, SampleConfig.alphabet[args[0]](StdsConfig, length, value, False)),
        'B': lambda cls, length, value, *args: cls.lambd(length, value)
    }
    
    def __init__(self) -> None:
        return

    @classmethod
    def get_config(cls, length, index, **kwargs):
        if index is None:
            return None, None
        else:
            index_s = str(index)
            stds_idx, bias_idx = index_s[0], index_s[1]
            destn = index_s[2:] if len(index_s)>=3 else None
            
            stds_value = None if "stds" not in kwargs else kwargs["stds"]
            stds = cls.alphabet[stds_idx](StdsConfig, length, stds_value, destn)
            
            bias_value = None if "bias" not in kwargs else kwargs["bias"]
            bias = cls.alphabet[bias_idx](BiasConfig, length, bias_value, destn)
            
            '''
            Return: 
                stds: 标准差序列, 用于决定隐空间重采样的方差
                bias: 采样序列, 从以bias_value为标准差的高斯分布采样
            '''
            return stds, bias   

   
          
if __name__ == "__main__":
    index = "678"
    stds, bias = SampleConfig.get_config(length=[4, 6, 0], index=index, stds=None, bias=[0, 0.5, 1.5])
    print()