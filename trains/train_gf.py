import torch
import numpy as np
import datetime
import os
import argparse
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.cluster import KMeans

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

import interface
from utils.config import JsonConfig
from utils.optim import build_optimizer_and_schedule
from utils.sl import save, load
from validates import validation_gf
from loss_func.flow_func import FlowLoss
from datasets import dataset_train_generator_factory, dataset_test_generator_factory


def train(model, train_dataset, val_dataset,  
          optim, schedule,
          t_his:int, t_pred:int, num_samples, batch_size, 
          last_epoch:int, epochs:int, 
          writer=None,
          augs=0, checkpoints_gap=1, max_checkpoints=100, scalar_log_gap=2):
    
    global_step = last_epoch * num_samples
    dlen = t_his + t_pred

    # conformal prediction based loss parameters
    contrastive_loss_params = {} if "contrastive_loss_params" not in hparams.Train \
                                else hparams.part_to_dict(hparams.Train.contrastive_loss_params)
    clustering = 0 if "clustering" not in hparams.Train else int(hparams.Train.clustering)
    pre_clustering = False if "pre_clustering" not in hparams.Train else int(hparams.Train.pre_clustering)
    hard_clustering = False if "hard_clustering" not in hparams.Train else int(hparams.Train.hard_clustering)
    em_clustering = False if "em_clustering" not in hparams.Train else int(hparams.Train.em_clustering)
    z_alpha = 1.0 if "z_alpha" not in hparams.Train else float(hparams.Train.z_alpha)
    flow_loss = FlowLoss(cmds=hparams.part_to_dict(hparams.Train.loss_and_weight), **contrastive_loss_params)

    if "measures" in hparams.Val:
        flow_measure = FlowLoss(cmds = hparams.part_to_dict(hparams.Val.measures))
    else:
        flow_measure = flow_loss

    if pre_clustering:
        # sampler = dataset_test_generator_factory(train_dataset, batch_size, int(0.2*dlen))
        sampler = dataset_train_generator_factory(train_dataset, batch_size, num_samples)
        z_list = []
        for idx, (x, label) in enumerate(sampler):
            x = x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
            x = torch.swapaxes(x, 0, 1) if x.ndim == 4 else x[None, ...]        
            x = model.data_preprocessing(x)
            x = x.to('cpu').numpy()
            channel, node_n = x.shape[-2], x.shape[-1]
            x = x.reshape(-1, channel*node_n)
            z_list.append(x)
        z_list = np.concatenate(z_list, axis=0)     # (Batch, length, channel, nodes)

        kmeans = KMeans(n_clusters=train_dataset.get_action_nums(), 
                        init='k-means++',  # 更好的初始化方法
                        n_init=10,         # 多次初始化选择最佳结果
                        max_iter=300,
                        random_state=42)
        kmeans.fit(z_list)
        centers = kmeans.cluster_centers_
        centers = torch.from_numpy(centers).to(device)
        
        z_list = torch.from_numpy(z_list).to(device)
        dist = torch.cdist(z_list, centers, p=2)
        dist = torch.argmin(dist, dim=1)
        weights = torch.zeros(size=(centers.shape[0],), device=centers.device)
        for i in range(centers.shape[0]):
            weights[i] = torch.sum(torch.where(dist==i, 1, 0))
        weights = weights / z_list.shape[0]
                
        centers = centers.reshape(train_dataset.get_action_nums(), channel, node_n)
        model.distribution.set_distribution_parameters([centers[i][None] for i in range(centers.shape[0])], weights=weights)
        
    min_val_loss = float('inf')
    # try:
    for epoch in range(last_epoch, epochs):
        torch.cuda.empty_cache()
        print("epoch", epoch+1)
        
        model.train()
        sampler = dataset_train_generator_factory(train_dataset, batch_size, num_samples)
        for x, label in sampler:
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = torch.tensor(x, device=device)  # (Batch, length, channel, nodes)
                
            if x.ndim == 4:
                x = torch.swapaxes(x, 0, 1)
            elif x.ndim == 3:
                x = x[None, ...]
                               
            # z, nll = model(input=x, logdet=0.0, reverse=False, label=label)
            z, nll = interface.flow_interface(model=model, x=x, logdet=0.0, reverse=False, label=label, z_alpha=z_alpha)
            loss, loss_list = flow_loss.compute_loss(datas={'likelihood_space': [nll], \
                                                            'velocity_latent_sapce': [z], \
                                                            'contrastive_based_loss': [z, train_dataset.get_index_labels_without_empty(label)]})                
            '''
            # a penalty about linear relationship between pose sequences
            itp_loss = 0
            if itp_coef > 0:
                z = torch.reshape(z, shape=old_shape)
                
                _indice = indice - indice[0]
                coef2 = torch.reshape(_indice, shape=(select_nums, 1, 1, 1)).broadcast_to(z.shape) * (1.0/_indice[-1])
                coef1 = torch.flip(coef2, dims=[0])
                
                _z = coef1*z[0] + coef2*z[-1]
                itp_loss = torch.sum(torch.pow(torch.subtract(z, _z), 2), dim=(0, 2, 3))
                itp_loss = torch.mean(itp_loss)
            itp_total_loss += itp_loss
                
            s_nll, s_mse = 0, 0
            if snll_coef > 0 or smse_coef > 0:
                curr_epoch, max_epoch = epoch-last_epoch, epochs-last_epoch
                s_nll, s_mse = gaussian_distrib_sample_loss(model, x, z, max_epoch, curr_epoch, 1)
            
            loss = nll_loss + itp_coef*itp_loss + snll_coef*s_nll + smse_coef*s_mse
            '''
            optim.zero_grad()
            loss.backward()
            
            total_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=hparams.Train.max_grad_norm, norm_type=2)
            ''' # 查看梯度
            parameters = model.parameters()
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            total_norm = [p.grad.detach() for p in parameters]
            '''
            optim.step()

            if global_step % scalar_log_gap == 0:
                for key, value in loss_list.items():
                    writer.add_scalar("batch/"+str(key), value, global_step)
                writer.add_scalar("total_norm", total_norm, global_step)
                writer.add_scalar("lr", optim.state_dict()['param_groups'][0]['lr'], global_step)
            global_step += 1
            
        avg_loss, avg_loss_list, _ = flow_loss.get_average_history()
        flow_loss.reset_memos()
    
        val_loss, val_loss_list = validation_gf(model = model,
                                                generator = dataset_test_generator_factory(val_dataset, \
                                                                                           hparams.Val.batch_size, \
                                                                                           hparams.Val.num_samples, \
                                                                                           generator_type="sampling_generator"),
                                                flow_loss = flow_measure,
                                                z_alpha = z_alpha,
                                                device = device)
        
        if clustering > 0 and (epoch+1) % clustering == 0:
            with torch.no_grad():
                # sampler = dataset_test_generator_factory(train_dataset, batch_size, int(0.2*dlen))
                sampler = dataset_train_generator_factory(train_dataset, batch_size, num_samples)
                z_list = {}
                for x, label in sampler:
                    if isinstance(x, torch.Tensor):
                        x = x.to(device)
                    else:
                        x = torch.tensor(x, device=device)  # (Batch, length, channel, nodes)
                        
                    if x.ndim == 4:
                        x = torch.swapaxes(x, 0, 1)
                    elif x.ndim == 3:
                        x = x[None, ...]
                        
                    z, _ = interface.flow_interface(model=model, x=x, logdet=0.0, reverse=False, label=label)
                    z = z[0] if z.ndim == 4 else z
                    for idx in range(len(label)):
                        if label[idx] not in z_list:
                            z_list[label[idx]] = []
                        z_list[label[idx]].append(z[idx:idx+1])

                for key, value in z_list.items():
                    z_list[key] = torch.concat(value, dim=0)
                
                if hard_clustering:
                    model.distribution.hard_reclustering(z_list)
                elif em_clustering:
                    model.distribution.em_reclustering(z_list)
                else:
                    model.distribution.reclustering(z_list)
            
        if hparams.Optim.schedule.name == "plateau":
            if hparams.Optim.warmup > 1:
                schedule.step(metrics=val_loss, epoch=epoch)
            else:
                schedule.step(val_loss)
        else:
            schedule.step()

        writer.add_scalar("epoch/avg_loss", avg_loss, epoch+1)
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
        
        '''
        if (epoch+1) % sampling_gap == 0 or epoch == 0:
            sampling_subdir = os.path.join(sampling_dir, str(epoch+1))
            if not os.path.exists(sampling_subdir):
                os.makedirs(sampling_subdir)
            xs = sampling_gf(model, 100, (1, 3, 18), device)
            for id, x in enumerate(xs):
                x = np.swapaxes(x.to('cpu').numpy(), 1, 2)
                
                if val_dataset.use_vel>=0:
                    vel = val_dataset.use_vel
                    x[:, vel:(vel+1), :] = 0.0
            
                plt_one(train_dataset.get_skeleton(True), x, 
                        save_dir=sampling_subdir, 
                        save_name=str(id)+'.jpg', 
                        add_labels=True)
            print("No.{} sampling is over.".format(epoch+1))
        '''    
        
        if (epoch+1) % checkpoints_gap == 0:
            if val_loss < min_val_loss:
                is_best_flag = True
                min_val_loss = val_loss
            else:
                is_best_flag = False
                
            save(
                epoch = epoch+1,
                model = model,
                optim = optim,
                schedule = schedule,
                distribution = model.distribution_parameters,
                pkg_dir = trained_dir,
                is_best = is_best_flag,
                max_checkpoints = max_checkpoints,
            )
        
    print(
        f'Loss: {avg_loss:.5f}/ Validation Loss: {val_loss:.5f} '
    )  
    writer.close()     
    # except BaseException:
    #     writer.close()  
         
         

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_path', default='hparams/PRED/Human36M/auto_likelihood_N18')
    parser.add_argument('--hparams_name', default='conformal_dct_likelihood_V4_Velonly_S1_3.json')
    parser.add_argument('--clustering', default=0)
    parser.add_argument('--logger', default='logger.xlsx')
    args = parser.parse_args()
        
    hparams_path = os.path.join(os.getcwd(), args.hparams_path, args.hparams_name)    
    assert os.path.exists(hparams_path), (
        "Failed to find hparams josn `{}`".format(hparams_path))
    hparams = JsonConfig(hparams_path)
    
    dataset_name = hparams.Dataset
    
    device = torch.device('cuda') if torch.cuda.is_available() and hparams.Device == 'cuda' else torch.device('cpu')
    
    if hparams.Train.checkpoints is not None and hparams.Train.checkpoints != "":
        date = hparams.Train.checkpoints
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
    
    sampling_dir = os.path.join(hparams.Dir.sampling_root, "sampling_" + date)
    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)
    print("sampling_dir:" + str(sampling_dir))
    
    copyfile(hparams_path, os.path.join(trained_dir, args.hparams_name))
    print("copy likelihood.json to:" + str(trained_dir))
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train_dataset = interface.dataset_interface(dataset_name, "train", hparams)
    val_dataset = interface.dataset_interface(dataset_name, "test", hparams, mirror_augs=False) 
    
    model = interface.flow_model_builder(train_dataset, device, hparams)

    named_size, total_params = model.get_layer_params_size()
    d = max(map(len, named_size.keys()))
    for name, size in named_size.items():
        print(name.ljust(d), " : ", size)
    print("Total Params : {}".format(total_params))
    
    if hparams.Train.last_epoch == 0:
        last_epoch = -1
    optimizer, scheduler = build_optimizer_and_schedule(hparams, filter(lambda p: p.requires_grad, model.parameters()), last_epoch=last_epoch)
    
    # if hparams.Train.checkpoints is not None and hparams.Train.checkpoints != "":
    if hparams.Train.checkpoints:
        load(
            epoch_or_path = hparams.Train.last_epoch,
            model = model,
            optim = optimizer,
            schedule = scheduler,
            pkg_dir = os.path.join(hparams.Dir.trained_model_root, "trained_" + date)
        )
    
    train(
        model = model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        optim = optimizer,
        schedule = scheduler,
        t_his = hparams.Data.t_his,
        t_pred = hparams.Data.t_pred,
        num_samples = hparams.Train.num_samples,
        batch_size = hparams.Train.batch_size,
        last_epoch = hparams.Train.last_epoch,
        epochs = hparams.Train.epochs,  
        writer = writer,
        augs = hparams.Train.augmentation if "augmentation" in hparams.Train else 0,
        checkpoints_gap = hparams.Train.checkpoints_gap,
        max_checkpoints = hparams.Train.max_checkpoints,
        scalar_log_gap = hparams.Train.scalar_log_gap
    )