import torch
# torch.autograd.set_detect_anomaly(True)
import numpy as np
import math
import datetime
import os
import argparse
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
import interface
from utils.config import JsonConfig
from utils.optim import build_optimizer_and_schedule
from utils.sl import save, load
from utils.torch_empty_cache_switch import cuda_empty_cache_flag
from loss_func.pred_func import PredictLoss
from validates import validation_pred
from datasets import dataset_train_generator_factory



def cosine_annealing_scheduler(cur_step=0, cycle_size=1000, std_length=100, random_length=False):
    if not random_length:
        return std_length
    
    if cycle_size > 0:
        cur_step = cur_step % cycle_size
        x_dp = ((cur_step*1.0) / cycle_size) * math.pi
        y_dp = (math.cos(x_dp) + 1) / 2
        max_length = max(int(np.rint((1.0 - y_dp) * std_length)), 1)
    else:
        max_length = std_length
        
    cur_length = np.random.randint(1, max_length) if max_length > 1 else max_length
    return cur_length


def nondecreasing_annealing_scheduler(cur_step=0, cycle_size=1000, std_length=100, random_length=False):
    if not random_length:
        return std_length

    if cycle_size > 0:        
        ratio = min(cur_step*1.0 / cycle_size, 1.0)
        max_length = min(max(int(np.rint(ratio * std_length)), 1), std_length)
    else:
        max_length = std_length
        
    cur_length = np.random.randint(1, max_length) if max_length > 1 else max_length
    return cur_length


annealing_scheduler = {
    'cosine': cosine_annealing_scheduler,
    'nondecreasing': nondecreasing_annealing_scheduler
}


phase_dict = {
    "0": "PhaseOne",
    "1": "PhaseTwo",
    "2": "PhaseThree",
    "3": "PhaseFour",
    "4": "PhaseFive"       
}


def train(model, hparams, phase, eval_hparams,
          train_dataset, val_dataset,  
          optim:list, schedule:list, 
          t_his:int, t_pred:int, 
          last_epoch:int, epochs:int, 
          writer=None, 
          checkpoints_gap=1, max_checkpoints=100, scalar_log_gap=2,
          num_workers=0):
    
    batch_size = hparams.Train.batch_size
    num_samples = hparams.Train.num_samples
    
    eval              = True      if "train_like_eval"   not in hparams.Train else hparams.Train.train_like_eval
    # for model.PredictNet    
    supervise_history = False     if "supervise_history" not in hparams.Train else hparams.Train.supervise_history
    # for model.Seq2SeqPredictNet/Seq2SeqPredictNet_V2
    nll_pred          = False     if "nll_pred"          not in hparams.Train else hparams.Train.nll_pred
    step_by_step      = None      if "step_by_step"      not in hparams.Train else hparams.Train.step_by_step
    sample_config     = "00"      if "sample_config"     not in hparams.Train else hparams.Train.sample_config
    # for model.Seq2SeqPredcitDistribNet
    step_by_step = step_by_step
    sample            = False     if "sample"            not in hparams.Train else hparams.Train.sample
    
    # controlling the length of the prediction horizons
    curriculum_type   = "cosine"  if "curriculum_type"   not in hparams.Train else hparams.Train.curriculum_type
    curriculum_iters  = -1        if "curriculum_iters"  not in hparams.Train else hparams.Train.curriculum_iters
    random_t_pred     = False     if "random_prediction_horizon" not in hparams.Train else hparams.Train.random_prediction_horizon
    
    # generator type sample/iter/loader/pseudo_sample ?
    generator_type    = None      if "generator_type"    not in hparams.Train else hparams.Train.generator_type
    
    # conformal prediction based loss parameters
    conformal_loss_params = {} if "conformal_loss_params" not in hparams.Train \
                               else hparams.part_to_dict(hparams.Train.conformal_loss_params)
    
    conformal_loss_params['model'] = model
    conformal_loss_params['dataset'] = val_dataset
    conformal_loss_params['t_his'] = t_his
    conformal_loss_params['t_pred'] = t_pred
    conformal_loss_params['device'] = device 

    conformal_loss_reborn = 0 if "conformal_loss_reborn" not in hparams.Train else hparams.Train.conformal_loss_reborn
    conformal_loss_quantile = [0.5] if "conformal_loss_quantile" not in hparams.Train else hparams.Train.conformal_loss_quantile
    conformal_loss_quantile = [torch.tensor(q, device=device) for q in conformal_loss_quantile]
        
    # initialize the loss (pred_loss)
    pred_loss = PredictLoss(cmds=hparams.part_to_dict(hparams.Train.loss_and_weight), **conformal_loss_params)
    # initialize the evaluation metrics (pred_measure)
    # if the hyparams of pred_measure is None, the pred_measure = pred_loss
    if "measures" in eval_hparams:
        pred_measure = PredictLoss(cmds=hparams.part_to_dict(eval_hparams.measures), **conformal_loss_params)
    else:
        pred_measure = pred_loss
                
    min_val_loss = float('inf')
    # try:
    sampler = dataset_train_generator_factory(train_dataset, batch_size, num_samples, generator_type=generator_type, num_workers=num_workers) 
    global_step = last_epoch * int(num_samples / batch_size)
    if hasattr(train_dataset, '__len__'):
        global_step = last_epoch * int(len(train_dataset) / batch_size)   
    
    for epoch in range(last_epoch, epochs):
        print("epoch", epoch+1)
        if cuda_empty_cache_flag.torch_cuda_empty_cache_flag:
            torch.cuda.empty_cache()
            
        model.train()
        
        if not isinstance(train_dataset, torch.utils.data.DataLoader):
            sampler = dataset_train_generator_factory(train_dataset, batch_size, num_samples, \
                                                    generator_type=generator_type, num_workers=num_workers)
        conformal_local_step = 0
        for x, label in tqdm(sampler):
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = torch.tensor(x, device=device)  # (Batch, length, channel, nodes)
            
            cur_t_pred = annealing_scheduler[curriculum_type](cur_step = global_step,
                                                            cycle_size = curriculum_iters,
                                                            std_length = t_pred, 
                                                            random_length = random_t_pred)
            x = x[:, :cur_t_pred+t_his, :, :]
            
            y, x_pred, y_en, x_pred_en, nll, x_en = interface.train_eval_interface(
                                                                model  = model,
                                                                x      = x,
                                                                t_his  = t_his,
                                                                t_pred = cur_t_pred,
                                                                eval   = eval,
                                                                # for model.PredictNet  
                                                                supervise_history = supervise_history,
                                                                # for model.Seq2SeqPredictNet/Seq2SeqPredictNet_V2
                                                                nll_pred      = nll_pred,
                                                                step_by_step  = step_by_step,
                                                                sample_config = sample_config,
                                                                sample_stds   = 1.0,
                                                                sample_bias   = None,
                                                                # for model.Seq2SeqPredcitDistribNet 
                                                                sample = sample,
                                                                # for model.Seq2SeqPredictDistribNetsetZeros
                                                                zero_joints = torch.tensor(data=train_dataset.zero_joints, device=device),                                                                
                                                                label = label
                                                            )
            _, mu, sigma = y_en
            # # is_tuple:model.Seq2SeqPredcitDistribNet / not_tuple: others
            # y_en = y_en[0] if isinstance(y_en, tuple) else y_en
            # # model.Seq2SeqPredcitVelDistribNet is also is_tuple  # OLD VERSION
            # at this time y_en = (vel_mu, vel_sigma, pos)
            # # model.Seq2SeqPredcitVelDistribNet is also is_tuple  # OLD VERSION
            # at this time y_en = (y_en, (vel_mu, vel_sigma, pos))    
            loss, loss_list = pred_loss.compute_loss(datas={'position_space'        : [y, x_pred], \
                                                            'rotation_space'        : [y, x_pred], \
                                                            'latent_space'          : [y_en, x_pred_en], \
                                                            'velocity_latent_space' : [y_en, x_pred_en], \
                                                            'latent_velocity_space' : [mu, x_pred_en], \
                                                            'likelihood_space'      : [nll],
                                                            'miscellaneous'         : [x_en],
                                                            'conformal_prediction_based_loss': [mu, sigma, conformal_loss_quantile, x_pred_en]})
            for opt in optim:
                opt.zero_grad()
            loss.backward()
            
            ''' # 查看梯度
            parameters = model.parameters()
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            total_norm = [p.grad.detach() for p in parameters]
            var_for_checkpoint = total_norm
            '''
            flow_grad = torch.nn.utils.clip_grad_norm_(parameters=model['0'].parameters(), max_norm=hparams.Train.max_grad_norm, norm_type=2)
            rnn_grad  = torch.nn.utils.clip_grad_norm_(parameters=model['1'].parameters(), max_norm=hparams.Train.max_grad_norm, norm_type=2)
            
            if global_step % scalar_log_gap == 0:
                writer.add_scalar("batch/sum_loss", loss, global_step)
                for key, value in loss_list.items():
                    writer.add_scalar("batch/"+str(key), value, global_step)
                writer.add_scalar("flow_grad", flow_grad, global_step)
                writer.add_scalar("rnn_grad", rnn_grad, global_step)
                writer.add_scalar("lr", optim[phase].state_dict()['param_groups'][0]['lr'], global_step)
            global_step += 1

            optim[phase].step()
            
            conformal_local_step += 1
            if conformal_loss_reborn > 0 and conformal_local_step % conformal_loss_reborn == 0:
                pred_loss.update(model=model)
            
        avg_loss, avg_loss_list, _ = pred_loss.get_average_history()
        pred_loss.reset_memos()
                
        val_loss, val_loss_list = validation_pred(model = model,
                                                generator = {'dataset': val_dataset, 'batch_size':hparams.Val.batch_size},
                                                pred_loss = pred_measure,
                                                t_his = t_his,
                                                t_pred = t_pred, 
                                                device = device) 
        
        writer.add_scalar("epoch_val/avg_loss", avg_loss, epoch+1)
        for key, value in val_loss_list.items():
            writer.add_scalar("epoch_val/"+str(key), value, epoch+1)
        
        print('Loss : %5f' % avg_loss, end=" / ")
        for _, value in avg_loss_list.items():
            print(' %5f ' % value, end=" ")
        print()
        
        print('Val_Loss : %5f' % val_loss, end=" / ")
        for _, value in val_loss_list.items():
            print(' %5f ' % value, end=" ")
        print()
        
        loss_for_schedule = metrics = list(val_loss_list.values())[0]
        # loss_for_schedule = metrics = list(val_loss_list.values())[1]
        if hparams.Optim.schedule.name == "plateau":
            if hparams.Optim.warmup > 1:
                schedule[phase].step(loss_for_schedule, epoch=epoch)
            else:
                schedule[phase].step(loss_for_schedule)
        else:
            schedule[phase].step()
            
        if (epoch+1) % checkpoints_gap == 0:
            if loss_for_schedule < min_val_loss:
                is_best_flag = True
                min_val_loss = loss_for_schedule
            else:
                is_best_flag = False
            save(
                epoch = epoch+1,
                model = model,
                optim = dict(zip(list(phase_dict.values())[:len(optim_list)], optim_list)),
                schedule = dict(zip(list(phase_dict.values())[:len(sched_list)], sched_list)),
                distribution = model.distribution_parameters,
                pkg_dir = trained_dir,
                is_best = is_best_flag,
                max_checkpoints = max_checkpoints
            )
    writer.close()  
         
    # except BaseException:
    #     writer.close()    



if __name__ == "__main__":           
    parser = argparse.ArgumentParser()      
    parser.add_argument('--hparams_path', default='hparams/PRED/Human36M/auto_predict_N18')
    parser.add_argument('--hparams_name', default='ct_dct_prediction_VelonlyDistrib_V1_3_S1_1.json')
    parser.add_argument('--cuda_cache_empty_flag', default=True)
    parser.add_argument('--num_workers', default=0)
    args = parser.parse_args()
    
    hparams_name = args.hparams_name
    hparams_path = os.path.join(os.getcwd(), args.hparams_path, hparams_name)
    assert os.path.exists(hparams_path), (
        "Failed to find hparams json `{}`".format(hparams_path))
    
    hparams = JsonConfig(hparams_path)
    
    dataset_name = hparams.Dataset
    
    device = torch.device('cuda') if torch.cuda.is_available() and hparams.Device == 'cuda' else torch.device('cpu')
    
    cuda_empty_cache_flag.update(args.cuda_cache_empty_flag)
    
    torch.set_default_dtype(torch.float32)
    
    if hparams.SavePoint.checkpoints is not None and hparams.SavePoint.checkpoints != "":
        checkpoint_date = hparams.SavePoint.checkpoints
        new_dir_flag = True if "new_dir" not in hparams.Dir else hparams.Dir.new_dir
        if new_dir_flag:
            date = str(datetime.datetime.now())
            date = date[:date.rfind(":")].replace("-", "")\
                                        .replace(":", "")\
                                        .replace(" ", "_")
        else:
            date = checkpoint_date 
    else:
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                    .replace(":", "")\
                                    .replace(" ", "_")
                                    
    log_root = os.path.join(hparams.Dir.log_root, dataset_name)
    log_dir = os.path.join(log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("log_dir:" + str(log_dir))
    trained_model_root = os.path.join(hparams.Dir.trained_model_root, dataset_name)
    trained_dir = os.path.join(trained_model_root, "trained_" + date)
    if not os.path.exists(trained_dir):
        os.makedirs(trained_dir)
    print("trained_dir:" + str(trained_dir))

    # copyfile(os.path.join(os.getcwd(), 
    #                       hparams.Model.Flow.hparams_path if hparams.Model.Flow.hparams_path is not None \
    #                       else os.path.join(hparams.Model.Flow.trained_model_root, hparams.Model.Flow.trained_model_subdir),
    #                       hparams.Model.Flow.trained_model_hparam), \
    #          os.path.join(trained_dir, hparams.Model.Flow.trained_model_hparam))
    # print("copy likelihood.json to:" + str(trained_dir))
    copyfile(hparams_path, os.path.join(trained_dir, hparams_name))
    print("copy prediction.json to:" + str(trained_dir))
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train_dataset = interface.dataset_interface(dataset_name, "train", hparams)
    val_dataset = interface.dataset_interface(dataset_name, "test", hparams, mirror_augs=False)
    
    torch.manual_seed(0)
    model = interface.network_interface(train_dataset, device, hparams)
    # # ERROR !!! !!! WARNING: 写死的参数 !!!
    # writer.add_graph(model['RNN'], (torch.rand(1, 1, 3, 18).to(device), torch.rand(18, 18).to(device)), use_strict_trace=False)
    
    # 打印flow模型和rnn模型参数
    named_size, total_params = model.get_layer_params_size()
    d = max(map(len, named_size.keys()))
    for name, size in named_size.items():
        print(name.ljust(d), " : ", size)
    print("Total Params : {}".format(total_params))

    last_epoch = hparams.SavePoint.last_epoch
    
    hparams_train = hparams.Train
    phase_nums = 2 if "PhaseNums" not in hparams_train else hparams_train.PhaseNums

    hparams_phase_list = []
    epochs = [0]
    for i in range(phase_nums):
        hparams_phase = hparams_train[phase_dict[str(i)]]
        hparams_phase_list.append(hparams_phase)
        epochs.append(hparams_phase.Train.epochs + epochs[i])
    epochs = epochs[1:]
    
    optim_list, sched_list = [], []
    for i in range(phase_nums):
        if "train_part" not in hparams_phase_list[i].Train:
            params = model.parameters()
        else:
            if hparams_phase_list[i].Train.train_part != "total":
                trained_model = model[hparams_phase_list[i].Train.train_part]
                if isinstance(trained_model, list):
                    params = list(trained_model[0].parameters())
                    for t_model in trained_model[1:]:
                        params = params + list(t_model.parameters())
                else:
                    params = model[hparams_phase_list[i].Train.train_part].parameters()
            else:
                params = model.parameters()
        
        if i == 0:
            _last_epoch = -1
        else:
            _last_epoch = -1 if last_epoch == 0 or last_epoch == "best" or (last_epoch - epochs[i-1]) <= 0  \
                            else (last_epoch - epochs[i-1])
        
        optim, schedule = build_optimizer_and_schedule(hparams_phase_list[i], params, _last_epoch)
        optim_list.append(optim)
        sched_list.append(schedule)
        
    # if hparams.SavePoint.checkpoints is not None and hparams.SavePoint.checkpoints != "":
    if hparams.SavePoint.checkpoints:
        load(
            epoch_or_path = last_epoch,
            model = model,
            optim = dict(zip(list(phase_dict.values())[:len(optim_list)], optim_list)),
            schedule = dict(zip(list(phase_dict.values())[:len(sched_list)], sched_list)),
            pkg_dir = os.path.join(hparams.Dir.trained_model_root, dataset_name, "trained_" + checkpoint_date)
        )
    
    last_epoch = 0 if last_epoch == "best" else last_epoch
    for i in range(phase_nums):
        if last_epoch > epochs[i]:
            continue
        train(
            model = model,
            hparams = hparams_phase_list[i],
            phase = i,
            eval_hparams = hparams.Eval,
            train_dataset = train_dataset,
            val_dataset = val_dataset, 
            optim = optim_list,
            schedule = sched_list,
            t_his = hparams.Data.t_his,
            t_pred = hparams.Data.t_pred,
            last_epoch = last_epoch if (i==0 or last_epoch>epochs[i-1]) else epochs[i-1],
            epochs = epochs[i],  
            writer = writer,
            checkpoints_gap = hparams.SavePoint.checkpoints_gap,
            max_checkpoints = hparams.SavePoint.max_checkpoints,
            scalar_log_gap = hparams.SavePoint.scalar_log_gap,
            num_workers = int(args.num_workers)
        )