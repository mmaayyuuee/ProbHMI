import scipy.fft
import scipy.fftpack
import torch
import scipy
import numpy as np


def get_dct_matrix(N, is_torch=True, device='cpu'):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m).type(torch.float32).to(device)
        idct_m = torch.from_numpy(idct_m).type(torch.float32).to(device)
    return dct_m, idct_m


class Dct(torch.nn.Module):
    def __init__(self, n_len, n_pre, device='cpu'):
        super().__init__()
        self.n_len, self.n_pre = n_len, n_pre
        self.device = device
        if isinstance(self.n_len, int):
            self.dct_m, self.idct_m = [], []
            dct_m, idct_m = get_dct_matrix(self.n_len, device=self.device)
            self.dct_m.append(dct_m)
            self.idct_m.append(idct_m)
        else:
            self.dct_m, self.idct_m = [], []
            for nl in self.n_len:
                dct_m, idct_m = get_dct_matrix(nl, device=self.device)
                self.dct_m.append(dct_m)
                self.idct_m.append(idct_m)
        return
    
    
    def forward(self, data, axis, inverse=False):               
        if len(self.dct_m) == 1:
            return self.dct1(data, axis, self.n_pre) if not inverse else self.idct1(data, axis, self.n_pre)
        elif len(self.dct_m) == 2:
            return self.dct2(data, axis) if not inverse else self.idct2(data, axis)
        return
        

    def dct1(self, data, axis, keep_dim, dctm_idx=0):
        axis = axis if axis > 0 else data.ndim + axis
        
        dims = torch.arange(0, data.ndim)
        dims[:axis+1] = torch.roll(dims[:axis+1], shifts=1)
        _data = torch.permute(data, dims=list(dims))
        
        shape = _data.shape
        _data = torch.matmul(self.dct_m[dctm_idx][:keep_dim, :], _data.reshape((_data.shape[0], -1)))
        _data = torch.reshape(_data, [_data.shape[0], *shape[1:]])
        
        dims = torch.arange(0, data.ndim)
        dims[:axis+1] = torch.roll(dims[:axis+1], shifts=-1)
        dct_data = torch.permute(_data, dims=list(dims))
        return dct_data
    
    
    def idct1(self, data, axis, keep_dim, dctm_idx=0):
        axis = axis if axis > 0 else data.ndim + axis

        dims = torch.arange(0, data.ndim)
        dims[:axis+1] = torch.roll(dims[:axis+1], shifts=1)
        _data = torch.permute(data, dims=list(dims))
        
        shape = _data.shape
        _data = torch.matmul(self.idct_m[dctm_idx][:, :keep_dim], _data.reshape((_data.shape[0], -1)))
        _data = torch.reshape(_data, [_data.shape[0], *shape[1:]])
        
        dims = torch.arange(0, data.ndim)
        dims[:axis+1] = torch.roll(dims[:axis+1], shifts=-1)
        idct_data = torch.permute(_data, dims=list(dims))        
        return idct_data
    
    
    def dct2(self, data, axis):
        dct_data_1 = self.dct1(data, axis[0], self.n_pre[0], dctm_idx=0)
        dct_data_2 = self.dct1(dct_data_1, axis[1], self.n_pre[1], dctm_idx=1)
        return dct_data_2
    
    
    def idct2(self, data, axis):
        idct_data_1 = self.idct1(data, axis[0], self.n_pre[0], dctm_idx=0)
        idct_data_2 = self.idct1(idct_data_1, axis[1], self.n_pre[1], dctm_idx=1)
        return idct_data_2




if __name__ == '__main__':
    '''    
    a, b = 5, 10
    dct_m1, idct_m1 = get_dct_matrix(5)
    dct_m2, idct_m2 = get_dct_matrix(10)
    data = torch.rand([5, 10])
    
    dct_data_1 = torch.matmul(dct_m1, data)
    _dct_data_1 = dct_data_1
    dct_data_1 = torch.transpose(dct_data_1, 0, 1)
    dct_data_2 = torch.matmul(dct_m2, dct_data_1)
    dct_data_2 = torch.transpose(dct_data_2, 0, 1)
    
    dct_data_i1 = torch.matmul(idct_m1, dct_data_2)
    dct_data_i1 = torch.transpose(dct_data_i1, 0, 1)
    dct_data_i2 = torch.matmul(idct_m2, dct_data_i1)
    dct_data_i2 = torch.transpose(dct_data_i2, 0, 1)
    
    res = torch.abs(data - dct_data_i2).max()
    
    data_np = data.numpy()
    dct_gt = scipy.fft.dctn(data_np, type=2, axes=(0, 1), norm='ortho')  
    print()
    
    data = torch.rand(5)
    dct_m, idct_m = get_dct_matrix(5)
    dct1 = torch.matmul(dct_m, data)
    
    data_np = data.numpy()
    dct_gt = scipy.fft.dct(data_np, norm='ortho')
    print()
    '''
    
    data = torch.rand([1, 2, 3, 4, 5], requires_grad=True)
    dct_func = Dct()
    dct_data = dct_func(data, axis=(-3, -2), keep_dim=(3, 4), inverse=False)
    # print(dct_data)
    dct_data.backward(torch.randn(dct_data.size()))
    print(data)
    
    data_np = data.numpy()
    dct_gt = scipy.fft.dctn(data_np, axes=(-3, -2), norm='ortho')
    print()
    
    