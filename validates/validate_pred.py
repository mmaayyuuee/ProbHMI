import torch
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from loss_func.pred_func import PredictLoss
from utils.config import JsonConfig
from utils.sl import load
from utils.torch_empty_cache_switch import cuda_empty_cache_flag
import interface
from datasets import dataset_test_generator_factory


def validate_pred_using_iter_generator(model, dataset, pred_loss, t_his, t_pred, batch_size, device):
    model.eval()
    # generator = dataset.iter_generator(step=t_his, batch_size=batch_size)
    generator = dataset_test_generator_factory(dataset, batch_size, t_his, generator_type="iter_generator")
    with torch.no_grad():
        for x, label in generator:
            if cuda_empty_cache_flag.torch_cuda_empty_cache_flag:
                torch.cuda.empty_cache()
        
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = torch.tensor(x, device=device)  # (Batch, lenght, channel, nodes)
            y, x_pred, y_en, x_pred_en, nll, _ = interface.train_eval_interface(
                                                                model = model,
                                                                x = x,
                                                                t_his = t_his,
                                                                t_pred = t_pred,
                                                                step_by_step = None,
                                                                sample_config = "11",
                                                                sample_stds = None,
                                                                sample_bias = None,
                                                                sample = False,
                                                                zero_joints = torch.tensor(data=dataset.zero_joints, device=device),
                                                                label = label
                                                            )
            pred_loss.compute_loss(datas={'position_space': [y, x_pred], \
                                          'rotation_space': [y, x_pred], \
                                          'latent_space': [y_en, x_pred_en], \
                                          'latent_velocity_space' : [y_en[1], x_pred_en], \
                                          'likelihood_space': [nll]})
        avg_loss, avg_loss_list, _ = pred_loss.get_average_history()
        pred_loss.reset_memos()
    return avg_loss, avg_loss_list


def validate_pred_using_sampling_generator(model, generator, pred_loss, t_his, t_pred, device):
    model.eval()
    with torch.no_grad():
        for x, label in generator:
            if cuda_empty_cache_flag.torch_cuda_empty_cache_flag:
                torch.cuda.empty_cache()
            
            x = torch.tensor(x, device=device)                
            y, x_pred, y_en, x_pred_en, nll, _ = interface.train_eval_interface(
                                                                model = model,
                                                                x = x,
                                                                t_his = t_his,
                                                                t_pred = t_pred,
                                                                step_by_step = None,
                                                                sample_config = "11",
                                                                sample_stds = None,
                                                                sample_bias = None,
                                                                sample = False,
                                                                zero_joints = torch.tensor(data=dataset.zero_joints, device=device),
                                                                label = label
                                                            )
            pred_loss.compute_loss(datas={'position_space': [y, x_pred], \
                                          'rotation_space': [y, x_pred], \
                                          'latent_space': [y_en, x_pred_en], \
                                          'likelihood_space': [nll]})
        avg_loss, avg_loss_list, _ = pred_loss.get_average_history()
        pred_loss.reset_memos()
    return avg_loss, avg_loss_list



def validation_pred(model, generator, pred_loss, t_his, t_pred, device):
    if isinstance(generator, dict):
        avg_loss, avg_loss_list = validate_pred_using_iter_generator(model      = model, 
                                                                     dataset    = generator['dataset'], 
                                                                     pred_loss  = pred_loss, 
                                                                     t_his      = t_his, 
                                                                     t_pred     = t_pred, 
                                                                     batch_size = generator['batch_size'], 
                                                                     device     = device)
    else:
        avg_loss, avg_loss_list = validate_pred_using_sampling_generator(model, generator, pred_loss, t_his, t_pred, device)
    return avg_loss, avg_loss_list




if __name__ == "__main__":
    date = "20230415_1720"
    np.random.seed(123)
    
    hparams = os.path.join(os.getcwd(), 'results', 'prediction', "trained_" + date, 'prediction_V2.json')
    hparams = JsonConfig(hparams)
    
    device = torch.device('cuda') if torch.cuda.is_available() and hparams.Device == 'cuda'  \
                                  else torch.device('cpu')
    # device = 'cpu'

    dataset_name = hparams.Dataset
    dataset = interface.dataset_interface(dataset_name, "test", hparams, actions=None) 
    
    model = interface.network_interface(dataset, device, hparams)
    model.to(device)
    model.eval()
    
    named_size, total_params = model.get_layer_params_size()
    d = max(map(len, named_size.keys()))
    for name, size in named_size.items():
        print(name.ljust(d), " : ", size)
    print("Total Params : {}".format(total_params))
    
    pred_loss = PredictLoss(cmds = hparams.part_to_dict(hparams.Train.loss_and_weight), \
                            skeleton_parents = dataset.get_skeleton().parents(), \
                            hip_vel_location = dataset.use_vel)
    sampler = dataset.sampling_generator(1024, 256)
    t_his = hparams.Data.t_his
    t_pred = hparams.Data.t_pred
    
    load(
        epoch_or_path = 'best',
        model = model,
        optim = None,
        schedule = None,
        pkg_dir = os.path.join(hparams.Dir.trained_model_root, "trained_" + date)
    )
    avg_loss, avg_loss_list = validation_pred(model, sampler, pred_loss, t_his, t_pred, device)
        