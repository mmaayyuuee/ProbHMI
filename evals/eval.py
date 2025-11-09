import torch
import numpy as np
import argparse
from inspect import isfunction
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from utils.config import JsonConfig
from utils.poisson_disk_sampling import PseudoGaussianDiskSampling, PseudoGaussianSampling, PseudoGaussianBiasSampling
from utils.poisson_disk_sampling import PseudoPossionDiskSampling, PseudoPossionDiskTruncSampling
from utils.sl import load
from utils.logger import Logger
import evals.metrics as metrics
from datasets import dataset_test_generator_factory
import interface



def get_multimodal_gt(dataset, data_gen, t_his):    
    all_data = []
    for data, label in tqdm(data_gen):
        data = np.swapaxes(data, -2, -1)
        if data.shape[-1] == 6:
            data, _ = np.split(data, 2, axis=-1)
        data = dataset.convert_to_position_mode(data, option=2)
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        data = np.reshape(data, newshape=(data.shape[0], data.shape[1], -1))
        all_data.append(data)
        
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his-1]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < 0.4) if hasattr(dataset, "AMASS") else np.nonzero(pd[i] < 0.5)
        traj_gt_arr.append(all_data[ind][:, t_his:])
    return traj_gt_arr


def meta_get_multimodal_gt(dataset, data_gen, t_his):    
    all_data = []
    for data, label in tqdm(data_gen):
        data = np.swapaxes(data, -2, -1)
        if data.shape[-1] == 6:
            data, _ = np.split(data, 2, axis=-1)
        # data = np.squeeze(data.numpy())
        data = dataset.convert_to_position_mode(data, option=2)
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        data = np.reshape(data, newshape=(data.shape[0], data.shape[1], -1))
        all_data.append(data)
        
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his-1]
    pd = squareform(pdist(all_start_pose))
    
    ind_list = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < 0.4) if hasattr(dataset, "AMASS") else np.nonzero(pd[i] < 0.5)
        ind_list.append(ind)
    
    def get_multimodal_gt(st, ed):
        traj_gt_arr = []
        for idx in range(st, ed, 1):
            traj_gt_arr.append(all_data[ind_list[idx]][:, t_his:])
        return traj_gt_arr
    return get_multimodal_gt



class EvalParamsConfig(object):
    def __init__(self) -> None:
        return
    
    '''
        Returns:
            loc1: nums: int
            loc2: sample: bool
            loc3: sample_cfgs: list for sample_config
            loc4: sample_stds
            loc5: sample_bias
    '''
    # sampling from the distribution with mu and sigma, where sigma is computed by the NN.
    @staticmethod
    def zero(*args):
        # return 1, False, [], None, None
        return 1, False, ["11"], None, 0.0
    
    @staticmethod
    def one(*args):
        return 1, True, ["11"], None, 0.0
    
    # sampling from the distribution with mu and sigma, where sigma is decided by fixed parameters (by cfgs).
    @staticmethod
    def two(*args):
        return 1, True, 1*["11"], None, None
    
    @staticmethod
    def three(*args):
        return 50, False, 50*["06"], 0.0, 1.0

    @staticmethod
    def four(*args):
        return 50, True, 50*["01"], 0.0, 1.0

    @staticmethod
    def five(*args):
        return 50, True, 50*["10"], 1.0, 0.0

    @staticmethod
    def six(*args):
        return 50, True, 50*["09"], 0.0, 1.0
    
    # sampling from the distribution where the sigma is either decided by cfgs or computed by the NN.
    @staticmethod
    def seven_wrapper(*args):
        nums, scale = args[0], args[1]
        def seven():
            return nums, None, nums*["0A1"], None, (((np.arange(nums)/nums - 0.5)) * scale).tolist()
        return seven
    
    @staticmethod
    def seven_one_wrapper(*args):
        nums, scale = args[0], args[1]
        def seven():
            return nums, None, nums*["0A6"], None, (((np.arange(nums)/nums - 0.5)) * scale).tolist()
        return seven
    
    @staticmethod
    def nine_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        ppds = PseudoPossionDiskSampling(nums, threshold)
        def nine(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, False, ratio)
        return nine

    @staticmethod
    def nine_one_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        ppds = PseudoPossionDiskSampling(nums, threshold)
        def nine(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, True, ratio)
        return nine

    @staticmethod
    def nine_two_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        ppds = PseudoPossionDiskSampling(nums, threshold)
        def nine(): 
            return nums, None, nums*["0B"], None, ppds.multi_mixture_sampling(100, None, ratio)
        return nine

    @staticmethod
    def nine_three_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        ppds = PseudoPossionDiskSampling(nums, threshold, option=1)
        def nine(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, False, ratio)
        return nine

    @staticmethod
    def ten_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        ppds = PseudoPossionDiskSampling(nums, threshold, 1)
        def ten(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, False, ratio)
        return ten

    @staticmethod
    def eleven_wrapper(*args):
        nums, threshold, ratio = args[0], args[1], args[2]
        pgds = PseudoGaussianDiskSampling(nums, threshold)
        def eleven(): 
            return nums, None, nums*["0B"], None, pgds.multi_sampling(100, False, ratio)
        return eleven

    @staticmethod
    def twelve_wrapper(*args):
        nums, threshold, ratio, p_nums = args[0], args[1], args[2], args[3]
        ppds = PseudoPossionDiskSampling(nums, threshold)
        def twelve(): 
            return p_nums, None, p_nums*["0B"], None, ppds.percentile_sampling(100, p_nums, False, ratio)
        return twelve

    @staticmethod
    def thirtheen_wrapper(*args):
        nums, threshold, ratio, p_nums = args[0], args[1], args[2], args[3]
        pgs = PseudoGaussianSampling(nums, threshold)
        def thirtheen(): 
            return nums, None, nums*["0B"], None, pgs.multi_sampling(100, False, ratio)
        return thirtheen

    @staticmethod
    def fourteen_wrapper(*args):
        nums, threshold, ratio, bias = args[0], args[1], args[2], args[3]
        ppds = PseudoPossionDiskSampling(nums, threshold, option=3, bias=bias)
        def fourteen(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, False, ratio)
        return fourteen

    @staticmethod
    def fifteen_wrapper(*args):
        nums, threshold, ratio, bias = args[0], args[1], args[2], args[3]
        ppds = PseudoGaussianBiasSampling(nums, threshold, bias=bias)
        def fifteen(): 
            return nums, None, nums*["0B"], None, ppds.multi_sampling(100, False, ratio)
        return fifteen

    @staticmethod
    def sixteen_wrapper(*args):
        nums, low_threshold, high_threshold, ratio = args[0], args[1], args[2], args[3]
        ppdts = PseudoPossionDiskTruncSampling(nums, low_threshold, high_threshold)
        def sixteen(): 
            return nums, None, nums*["0B"], None, ppdts.multi_sampling(100, False, ratio)
        return sixteen



eval_option_dict = {
    '0': (EvalParamsConfig.zero, metrics.single_modal_metrics),
    '1': (EvalParamsConfig.nine_wrapper(50, 2.0, 0.0), metrics.multimodal_metrics),
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()        
    parser.add_argument('--date', default="20250612_0401")
    parser.add_argument('--hparams_path', default='results/prediction/H36Mso3N18Vel_DL')
    parser.add_argument('--hparams_name', default='ct_dct_prediction_VelonlyDistrib_V1_3_S1_1.json')
    parser.add_argument('--epoch', default='best')
    parser.add_argument('--stride', default='t_his')
    parser.add_argument('--actions', default='all')
    parser.add_argument('--option', default='2')
    parser.add_argument('--logger', default='logger.xlsx')
    parser.add_argument('--horizon_coef', default=1.0)
    parser.add_argument('--multimodal_eval', default=True)
    parser.add_argument('--unsample_cond', default=False)
    parser.add_argument('--likelihood_flag', default=False)
    parser.add_argument('--sample_version', default='v1')
    parser.add_argument('--CNF_only', default=False)
    parser.add_argument('--ode_step_size', default=0.01)
    parser.add_argument('--ode_integ_method', default='dopri5')   ### euler, dopri5, rk4, midpoint or heun3

    args = parser.parse_args()
    
    date         = args.date
    hparams_path = args.hparams_path
    hparams_name = args.hparams_name
    epoch        = args.epoch if args.epoch == 'best' else int(args.epoch)
    action       = args.actions
    option       = str(args.option)
    logger       = args.logger
    horizon_coef  = float(args.horizon_coef)
    multimodal_eval = args.multimodal_eval
    stride = args.stride
    unsample_cond = args.unsample_cond
    sample_version = args.sample_version
    likelihood_flag = args.likelihood_flag
    CNF_only = args.CNF_only
    ode_step_size = float(args.ode_step_size)
    ode_integ_method = args.ode_integ_method

    hparams = os.path.join(os.getcwd(), hparams_path, "trained_"+date, hparams_name)
    hparams = JsonConfig(hparams)
    
    logger = os.path.join(os.getcwd(), logger)
    
    device = torch.device('cuda') if torch.cuda.is_available() and hparams.Device == 'cuda'  \
                                  else torch.device('cpu')
    torch.set_default_dtype(torch.float32)

    np.random.seed(123)
    dataset_name = hparams.Dataset
    dataset = interface.dataset_interface(dataset_name, "test", hparams, actions=action)
    
    t_his = hparams.Data.t_his
    t_pred = hparams.Data.t_pred

    steps = t_his if stride == 't_his' else int(stride)
    batch_size = min(32, hparams.Eval.batch_size)
    generator = dataset_test_generator_factory(dataset, batch_size, steps)
    
    model = interface.network_interface(dataset, device, hparams)
    model.to(device)
    model.eval()

    if "use_velocity" not in hparams.Data:
        discard_node = []
    else:
        vel_joint = hparams.Data.use_velocity
        # discard_node = [0] if vel_joint < 0 else [0, vel_joint]
        discard_node = [] if vel_joint < 0 else [0, vel_joint]
    
    load(
        epoch_or_path = epoch,
        model = model,
        optim = None,
        schedule = None,
        pkg_dir = os.path.join(hparams.Dir.trained_model_root, dataset_name, "trained_"+date)
    )

    eval_func = eval_option_dict[option]
    if isfunction(eval_func[0]):
        nums, sample, sample_cfgs, stds, bias = eval_func[0]()
    else:
        nums, sample, sample_cfgs, stds, bias = eval_func[0]
    metric_func = metrics.metrics_wrapper(eval_func[1])
    
    multimodal_eval = True if (multimodal_eval and eval_func[1] == metrics.multimodal_metrics) else False
    if multimodal_eval:
        gt_multi_func = meta_get_multimodal_gt(dataset, dataset.iter_generator(step=steps, batch_size=1), t_his)
    
    total_idx = 0
    with torch.no_grad():
        for x, label in tqdm(generator):
            torch.cuda.empty_cache()
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = torch.tensor(x, device=device)  # (Batch, length, channel, nodes)
                
            if sample_version == 'v1':
                sample_interface_func = interface.sample_interface
            elif sample_version == 'v2':
                sample_interface_func = interface.sample_interface_non_autoagression
                            
            # y/x_pred : (samples, batch_size, length, channel, node)/(batch_size, length, channel, node)
            results = sample_interface_func(model=model, x=x, t_his=t_his, t_pred=t_pred, nums=nums, 
                                            sample=sample, sample_config=sample_cfgs, sample_stds=stds, sample_bias=bias,
                                            zero_joints=torch.tensor(data=dataset.zero_joints, device=device),
                                            unsample_cond=unsample_cond, likelihood=likelihood_flag, CNF_only=CNF_only,
                                            step_size=ode_step_size, integ_method=ode_integ_method)
            if len(results) == 2:
                y, x_pred = results
            elif len(results) == 3:
                y, x_pred, ll = results
            
            horizons = int(horizon_coef * 1.0 * y.shape[-3])
            y, x_pred = y[:, :, :horizons, :, :], x_pred[:, :horizons, :, :]    
            y, x_pred = dataset.convert_to_position_mode(y, option=2), dataset.convert_to_position_mode(x_pred, option=2)
            
            if multimodal_eval:
                bs = x.shape[0]
                data = metric_func(x_pred, y, discard_node, bs, gt_multi_func(total_idx, total_idx+bs))
                # data = metric_func(x_pred, y, discard_node, bs, gt_multi[total_idx: total_idx+bs])
                total_idx += bs
            else:
                data = metric_func(x_pred, y, discard_node, x.shape[0])
        
        average_metric = data[3]
        average_metric['DATE'] = str(date)
        average_metric['DATASET'] = str(dataset_name)
        average_metric['ExpCfgs'] = str(option)
        average_metric['EPOCHS'] = str(epoch)
        average_metric['Haparams'] = str(hparams_name)
        average_metric['ACTIONS'] = str(action)
        
        if hasattr(dataset, "Human36M"):
            Logger.write(workbook=logger, sheet='prediction', overwrite=False, data=average_metric)
        elif hasattr(dataset, "HumanEva"):
            Logger.write(workbook=logger, sheet='prediction_Eva', overwrite=False, data=average_metric)
        elif hasattr(dataset, "AMASS"):
            Logger.write(workbook=logger, sheet='prediction_AMASS', overwrite=False, data=average_metric)
        else:
            Logger.write(workbook=logger, sheet='prediction', overwrite=False, data=average_metric)
        