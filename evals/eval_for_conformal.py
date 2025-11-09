import torch
import numpy as np
import argparse
from inspect import isfunction
from tqdm import tqdm

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from utils.config import JsonConfig
from utils.sl import load
from utils.logger import Logger
from utils.copulaCPTS import CopulaCPTS, StdCopulaCPTS
from eval import eval_option_dict, get_multimodal_gt
import evals.metrics as metrics
from datasets import dataset_test_generator_factory
import interface



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    # # ### Human36M 17nodes
    parser.add_argument('--date', default="20250612_0401")
    parser.add_argument('--hparams_path', default='results/prediction/H36Mso3N18Vel_DL')
    parser.add_argument('--hparams_name', default='ct_dct_prediction_VelonlyDistrib_V1_3_S1_1.json')
    parser.add_argument('--epoch', default='best')
    parser.add_argument('--stride', default='t_his')    
    parser.add_argument('--actions', default='all')
    parser.add_argument('--option', default='2')
    parser.add_argument('--logger', default='logger.xlsx')
    parser.add_argument('--horizon_coef', default=1.0)
    parser.add_argument('--multimodal_eval', default=False)
    parser.add_argument('--unsample_cond', default=False)
    parser.add_argument('--likelihood_flag', default=False)
    parser.add_argument('--sample_version', default='v1')
    parser.add_argument('--CNF_only', default=False)
    parser.add_argument('--using_conformal_prediction', default='stds')
    parser.add_argument('--epsilon', default=0.32)
    
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
    unsample_cond = args.unsample_cond
    sample_version = args.sample_version
    likelihood_flag = args.likelihood_flag
    CNF_only = args.CNF_only
    CP_flag = args.using_conformal_prediction
    epsilon = args.epsilon
    stride = args.stride

    hparams = os.path.join(os.getcwd(), hparams_path, "trained_"+date, hparams_name)
    hparams = JsonConfig(hparams)
    
    logger = os.path.join(os.getcwd(), logger)
    
    device = torch.device('cuda') if torch.cuda.is_available() and hparams.Device == 'cuda'  \
                                  else torch.device('cpu')
    torch.set_default_dtype(torch.float32)

    np.random.seed(123)
    dataset_name = hparams.Dataset
    dataset = interface.dataset_interface(dataset_name, "test", hparams, actions=action, mirror_augs=False)
        
    t_his = hparams.Data.t_his
    t_pred = hparams.Data.t_pred
    steps = t_his if stride == 't_his' else int(stride)
    batch_size = min(256, hparams.Eval.batch_size)
    generator = dataset_test_generator_factory(dataset, batch_size, steps)
    
    model = interface.network_interface(dataset, device, hparams)
    model.to(device)
    model.eval()

    if "use_velocity" not in hparams.Data:
        discard_node = []
    else:
        vel_joint = hparams.Data.use_velocity
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
        gt_multi = get_multimodal_gt(dataset, dataset.iter_generator(step=steps, batch_size=1), t_his)
    
    if not CP_flag:
        radius = None
    else: 
        if CP_flag == "mean":
            copula_cp = CopulaCPTS(model, dataset, t_his, t_pred, batchsize=256, device=device, metric="L2")
        elif CP_flag == "stds":
            copula_cp = StdCopulaCPTS(model, dataset, t_his, t_pred, batchsize=256, device=device, metric="L1")
            
        radius = copula_cp.get_threshold(epsilon=epsilon, cali_step=steps)
        radius = np.reshape(radius, newshape=(radius.shape[0], -1, len(dataset.kept_joints)))
        radius = torch.tensor(radius, device=device, dtype=torch.float32)
    
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
                                            copula_radius=radius, copula_type=CP_flag)
            if len(results) == 2:
                y, x_pred = results
            elif len(results) == 3:
                y, x_pred, ll = results
            
            horizons = int(horizon_coef * 1.0 * y.shape[-3])
            y, x_pred = y[:, :, :horizons, :, :], x_pred[:, :horizons, :, :]
            y, x_pred = dataset.convert_to_position_mode(y, option=2), dataset.convert_to_position_mode(x_pred, option=2)
            
            if multimodal_eval:
                bs = x.shape[0]
                data = metric_func(x_pred, y, discard_node, bs, gt_multi[total_idx:total_idx+bs])
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
        