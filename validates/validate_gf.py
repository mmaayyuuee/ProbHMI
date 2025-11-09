import torch
import numpy as np

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from utils.config import JsonConfig
from utils.sl import load
from utils.optim import build_optimizer_and_schedule
from utils.distributions import GaussianDiag
import interface


''' # OLD VERSION
def validation_gf(model, generator, device):
    model.eval()
    with torch.no_grad():
        total_val_loss, iter_nums = 0, 0
        for x, label in generator:
            x = torch.tensor(x).to(device)
            if x.ndim == 4:
                x = torch.reshape(x, shape=(-1, x.shape[-2], x.shape[-1]))
            _, nll = model(input=x, logdet=0.0, reverse=False)
            loss = model.loss_generative(nll)
            total_val_loss += loss
            iter_nums += 1
        val_loss = total_val_loss / iter_nums
    return val_loss
'''

def validation_gf(model, generator, flow_loss, z_alpha, device):
    model.eval()
    with torch.no_grad():
        for x, label in generator:
            torch.cuda.empty_cache()
            
            x = torch.tensor(x).to(device)
            if x.ndim == 4:
                x = torch.swapaxes(x, 0, 1)
            elif x.ndim == 3:
                x = x[None, ...]
                
            x = model.data_preprocessing(x)   
            z, nll = model(input=x, logdet=0.0, reverse=False, label=label, z_alpha=z_alpha)
            
            flow_loss.compute_loss(datas={'likelihood_space': [nll], \
                                        'velocity_latent_sapce': [z], \
                                        'velocity_position_sapce': [model, x, z]})  
        avg_loss, avg_loss_list, _ = flow_loss.get_average_history()
        flow_loss.reset_memos()
    return avg_loss, avg_loss_list


def sampling_gf(model, n=1, z_shape=(64, 3, 18), device='cpu'):
    gd = GaussianDiag() 
    model.eval()
    with torch.no_grad():
        samples_list = []
        for _ in range(n):
            z = gd.sample(z_shape).to(device)
            x = model(input=z, eps_std=1.0, reverse=True)
            samples_list.append(x)
    return samples_list
        
        
    
if __name__ == "__main__":         
    path = "H36Mso3N18VelOnly/trained_20250122_0813/conformal_dct_likelihood_V4_Velonly_S1_3.json"
    date = "20250122_0813"
    
    hparams = os.path.join(os.getcwd(), 'results', 'likelihood', path)
    hparams = JsonConfig(hparams)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    
    dataset = interface.dataset_interface(hparams.Dataset, "test", hparams, actions='all') 
        
    model = interface.flow_model_builder(dataset, device, hparams)
    model.to(device)
    
    ''' 模型可逆测试
    sampler = dataset.sampling_generator(1000, 8)
    for x, label in sampler:
        x1 = torch.tensor(x, device=device)
        z, nll = model(input=x1, logdet=0.0, reverse=False)
        x2 = model(input=z, logdet=0.0, reverse=True)
        x3 = x1 - x2
        maxx = torch.max(x3)
        mixx = torch.min(x3)
        print(x3)
        break
    '''
    
    load(
        epoch_or_path = 'best',
        # epoch_or_path = 1000,
        model = model,
        optim = None,
        schedule = None,
        pkg_dir = os.path.join(hparams.Dir.trained_model_root, hparams.Dataset, "trained_"+date)
    )
    
    from loss_func.flow_func import FlowLoss
    if "measures" in hparams.Val:
        flow_measure = FlowLoss(cmds = hparams.part_to_dict(hparams.Val.measures))
    else:
        flow_measure = FlowLoss(cmds = hparams.part_to_dict(hparams.Train.loss_and_weight))
    
    from datasets import dataset_test_generator_factory 
    val_loss, val_loss_list = validation_gf(model = model,
                                            generator = dataset_test_generator_factory(dataset, \
                                                                                    hparams.Val.batch_size, \
                                                                                    hparams.Val.num_samples, \
                                                                                    generator_type="sampling_generator"),
                                            flow_loss = flow_measure,
                                            device = device)
    print(val_loss)
    print()
        