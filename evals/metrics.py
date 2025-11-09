import torch
import numpy as np
import argparse

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


def index_select_nodes(x, dn:list = None, device='cpu'):
    if dn is None or len(dn) <= 0:
        return x, torch.zeros_like(x, device=device)
    else:
        rn = []
        for i in list(np.arange(x.shape[-1])):
            if i not in dn:
                rn.append(i)
        yr = torch.index_select(x, dim=-1, index=torch.tensor(rn, device=device))
        yd = torch.index_select(x, dim=-1, index=torch.tensor(dn, device=device))
        return yr, yd
        


# ADE
def euler_position_mse(x, y, dn:list = None):
    diff = torch.subtract(x, y) * 1000
    res_diff, _ = index_select_nodes(diff, dn, diff.device)
    # res_diff_tmp = torch.reshape(res_diff, shape=(res_diff.shape[0], res_diff.shape[1], res_diff.shape[2], -1))
    res_dist = torch.linalg.norm(res_diff, dim=(-2, -1))
    # res_dist_tmp = torch.linalg.norm(res_diff_tmp, dim=-1)
    res_dist = torch.mean(res_dist, dim=-1)
    return res_dist, res_dist, 0

def single_average_displacement_error(x, y, dn=None):
    _, sde, _ = euler_position_mse(x, y, dn)
    sde = torch.mean(sde)
    return sde.to('cpu')

def average_displacement_error(x, y, dn=None):
    _, sde, _ = euler_position_mse(x, y, dn)
    ade = torch.min(sde, dim=0)[0]
    ade = torch.mean(ade)
    return ade.to('cpu')



# FDE
def euler_final_position_mse(x, y, dn:list = None):
    diff = torch.subtract(x, y) * 1000
    res_diff, _ = index_select_nodes(diff, dn, diff.device)
    f = res_diff[..., -1, :, :]
    res_dist = torch.linalg.norm(f, dim=(-2, -1))
    return res_dist, res_dist, 0

# x/y: (batch_size, length, channels, nodes)
def single_final_displacement_error(x, y, dn=None):
    _, sfde, _ = euler_final_position_mse(x, y, dn)
    sfde = torch.mean(sfde)
    return sfde.to('cpu')

# x/y: (nums, batch_size, length, channels, nodes)    
def final_displacement_error(x, y, dn=None):
    _, sfde, _ = euler_final_position_mse(x, y, dn)
    fde = torch.min(sfde, dim=0)[0]
    fde = torch.mean(fde)
    return fde.to('cpu')



def multimodal_displacement_error(gt_multi, y, dn=None):
    bs = len(gt_multi)
    mmade_list, mmfde_list = [], []
    for i in range(bs):
        torch.cuda.empty_cache()
        pred, gt = y[:, i], gt_multi[i]
        pred = torch.swapaxes(pred, -2, -1)
        gt = torch.tensor(gt, device=pred.device)
        # pred = torch.reshape(pred, shape=(pred.shape[0], 1, pred.shape[1], -1))
        # gt = torch.reshape(gt, shape=(1, gt.shape[0], gt.shape[1], -1))
        pred, gt = torch.reshape(pred, shape=(pred.shape[0], pred.shape[1], -1)), torch.reshape(gt, shape=(gt.shape[0], gt.shape[1], -1))
        pred, gt = pred[:, None, ...], gt[None, ...]
        diff_multi = (pred - gt) * 1000
        dist = torch.linalg.norm(diff_multi, dim=3)

        mmfde, _ = dist[:, :, -1].min(dim=0)
        mmfde = mmfde.mean()
        mmfde_list.append(mmfde)
        
        mmade, _ = dist.mean(dim=2).min(dim=0)
        mmade = mmade.mean()
        mmade_list.append(mmade)
    
    _mmade, _mmfde = torch.tensor(mmade_list).mean(), torch.tensor(mmfde_list).mean()
    return _mmade.to('cpu'), _mmfde.to('cpu')


# APD
def average_pairwise_distance(x, dn=None):
    _x = torch.permute(x, dims=(1, 0, 2, 3, 4))     # shape=(batch_size, samples, length, channels, nodes)
    _x, _ = index_select_nodes(_x, dn, _x.device)
    _x = torch.reshape(_x, shape=(_x.shape[0], _x.shape[1], -1))
    pd_list = []
    for idx in range(_x.shape[0]):
        pd = torch.pdist(_x[idx], p=2).mean()
        pd_list.append(pd)
    apd = torch.tensor(pd_list).mean()
    return apd.to('cpu')



# MPJPE
def MPJPE(type='last'):
    def mpjpe(x, y, dn=None):
        diff = torch.subtract(x, y) * 1000      
        res_diff, _ = index_select_nodes(diff, dn, diff.device)
        _mpjpe = torch.linalg.norm(res_diff, dim=-2)
        mpjpe = torch.mean(torch.mean(_mpjpe, dim=-3), dim=-1)
        if type == 'last':
            mpjpe = mpjpe[..., -1]
        elif type == 'mean':
            mpjpe = torch.mean(mpjpe)
        else:
            mpjpe = mpjpe
        return mpjpe.to('cpu')
    return mpjpe

mean_per_joint_position_error = MPJPE('last')



def MAE(type='last'):
    def mae_l2(x, y, dn=None):
        _x = torch.flatten(x, start_dim=-2)
        _y = torch.flatten(y, start_dim=-2)
        diff = torch.remainder(_x - _y + np.pi, 2 * np.pi) - np.pi
        res_diff, _ = index_select_nodes(diff, dn, diff.device)
        # mae = torch.mean(torch.sum(torch.abs(res_diff), dim=-1), dim=0)
        mae = torch.mean(res_diff.norm(dim=-1), dim=0)
        mae = torch.unsqueeze(mae, 0)
        if type == 'last':
            mae = mae[..., -1]
        elif type == 'mean':
            mae = torch.mean(mae)
        else:
            mae = mae
        return mae.to('cpu')
    return mae_l2

mean_angle_error_last = MAE('last')
mean_angle_error_mean = MAE('mean')

   
    
def single_modal_metrics(x, y, gt_multi=None, dn=None):
    sade = single_average_displacement_error(x, y, dn)
    sfde = single_final_displacement_error(x, y, dn)
    mpjpe = mean_per_joint_position_error(x, y, dn)
    return {
        "S-ADE": sade,
        "S-FDE": sfde,
        "MPJPE": mpjpe
    }

# x-ground truth ; y-prediction horizons
def multimodal_metrics(x, y, gt_multi=None, dn=None):
    ade = average_displacement_error(x, y, dn)
    fde = final_displacement_error(x, y, dn)
    apd = average_pairwise_distance(y, dn)
    
    if gt_multi is not None:
        mmade, mmfde = multimodal_displacement_error(gt_multi, y, dn)    
        return {"APD":apd, "ADE":ade, "FDE":fde, "MMADE":mmade, "MMFDE":mmfde}
    else:
        return {"APD":apd, "ADE":ade, "FDE":fde}


def only_apd_metric(y_fix, y_div, gt_multi, dn):
    fix_apd = average_pairwise_distance(y_fix, None)
    div_apd = average_pairwise_distance(y_div, None)
    return {"fix-APD":fix_apd, "div-APD":div_apd}



#''' # Version B
def metrics_wrapper(func):
    history_list = [{}, {}, {}, {}]   # loc1: all history | loc2: length | loc3: sum  | loc4: average
    def memos(x, y, dn, batchsize, gt_multi=None):
        result = func(x, y, gt_multi, dn)
        for key, value in result.items():
            _value = value * batchsize
            
            if key not in history_list[0]:
                history_list[0][key] = []
            history_list[0][key].append(_value)

            if key not in history_list[1]:
                history_list[1][key] = 0
            history_list[1][key] += batchsize
            
            if key not in history_list[2]:
                history_list[2][key] = 0.0
            history_list[2][key] += float(_value)
            
            if key not in history_list[3]:
                history_list[3][key] = 0.0
            history_list[3][key] = float(history_list[2][key] / history_list[1][key])
        return history_list
    return memos
#'''


''' # Version A
def metrics_wrapper(func):
    history_list = [{}, {}, {}]   # loc1: all history | loc2: sum | loc3: average
    def memos(x, y, dn):
        result = func(x, y, dn)
        for key, value in result.items():
            if key not in history_list[0]:
                history_list[0][key] = []
            history_list[0][key].append(value)
            
            if key not in history_list[1]:
                history_list[1][key] = 0.0
            history_list[1][key] += float(value)
            
            if key not in history_list[2]:
                history_list[2][key] = 0.0
            history_list[2][key] = float(history_list[1][key] / len(history_list[0][key]))
        return history_list
    return memos
'''