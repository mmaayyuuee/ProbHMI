import torch
import numpy as np
import os
import math
import time
from torch.func import functional_call

import datasets
import models, modules
from models import load_prednet, load_flowmatching_prednet
from modules import recurrent_dyna_dict, linear_dyna_dict, flowmatching_dyna_dict
from modules import flow_model_dict
from layers import ContinuousTimesteps, ContinuousTimestepEmbedder
from utils.flow_matching import flow_path_dict
from utils.config import JsonConfig
from utils.sl import load
from utils.pe import PositionEmbedding
from utils.sample_config import SampleConfig



def dataset_interface(dataset_name, mode, hparams, actions=None, **kwargs):
    assert dataset_name in datasets.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset_name, datasets.Datasets.keys()))
    
    dataset = datasets.Datasets[dataset_name]
    
    if isinstance(dataset, list):
        dataset = dataset[0](dataset[1])
       
    if hasattr(dataset, "PW3D"):
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        return dataset(t_his, t_pred, mode, False)
    
    elif hasattr(dataset, "CMU"):
        data_path = hparams.Dir.data_path
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        action = hparams.Data.actions.replace(" ", "").split(",") if actions is None else actions.replace(" ", "").split(",")
        frame_rate = hparams.Data.frame_rate if "frame_rate" in hparams.Data else 1
        downsample = hparams.Data.downsample if "downsample" in hparams.Data else False
        return dataset(data_path, mode, t_his, t_pred, action, frame_rate, downsample)
    
    elif hasattr(dataset, "AMASS_BMLrub"):
        data_path = hparams.Dir.data_path
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        frame_rate = hparams.Data.frame_rate if "frame_rate" in hparams.Data else 1
        return dataset(data_path, mode, t_his, t_pred, frame_rate=frame_rate)
    
    elif hasattr(dataset, "AMASS"):
        data_path = hparams.Dir.data_path
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        stride = hparams.Data.stride if "stride" in hparams.Data else 1
        norm_type = hparams.Data.norm_type if "norm_type" in hparams.Data else None
        cali_subject = hparams.Data.cali_subject if "cali_subject" in hparams.Data else 0.0
        return dataset(data_path, mode, t_his, t_pred, stride, cali_subject=cali_subject, norm_type=norm_type)
            
    elif hasattr(dataset, "Human36M"):    
        data_path = hparams.Dir.data_path
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        
        action = hparams.Data.actions.replace(" ", "").split(",") if actions is None else actions.replace(" ", "").split(",")
        
        local_coord  = hparams.Data.local_coord  if "local_coord"  in hparams.Data else True
        hip_zeros    = hparams.Data.hip_zeros    if "hip_zeros"    in hparams.Data else True
        frame_rate   = hparams.Data.frame_rate   if "frame_rate"   in hparams.Data else 1
        cali_subject = hparams.Data.cali_subject if "cali_subject" in hparams.Data else None
                        
        return dataset(data_path, mode, t_his, t_pred, action, frame_rate, 
                       local_coord = local_coord, 
                       hip_zeros = hip_zeros,
                       cali_subject = cali_subject)
    
    elif hasattr(dataset, "HumanEva"):
        data_path = hparams.Dir.data_path
        t_his, t_pred = hparams.Data.t_his, hparams.Data.t_pred
        
        action = hparams.Data.actions.replace(" ", "").split(",") if actions is None else actions.replace(" ", "").split(",")
        
        local_coord  = hparams.Data.local_coord  if "local_coord"  in hparams.Data else True
        hip_zeros    = hparams.Data.hip_zeros    if "hip_zeros"    in hparams.Data else True
        frame_rate   = hparams.Data.frame_rate   if "frame_rate"   in hparams.Data else 1
        cali_subject = hparams.Data.cali_subject if "cali_subject" in hparams.Data else None
        cali_params = JsonConfig.to_dict(hparams.Data.cali_params) if "cali_params" in hparams.Data else {}
                
        return dataset(data_path, mode, t_his, t_pred, action, frame_rate, 
                       local_coord = local_coord, 
                       hip_zeros = hip_zeros,
                       cali_subject = cali_subject,
                       cali_params = cali_params)




def flow_model_builder(dataset, device, hparams, **kwargs):
    if hparams.Type == "graph" or hparams.Type == "graphflow" or \
       hparams.Type == "channel" or hparams.Type == "channelflow":
                   
        version = 1 if "Version" not in hparams else hparams.Version
        
        flow_model = flow_model_dict[version]
        
        n_len, n_pre = None, None
        if 'n_len' in hparams.Flow and 'n_pre' in hparams.Flow:      
            n_len, n_pre = hparams.Flow.n_len, hparams.Flow.n_pre

        flow_coupling_params = {}        
        if "flow_coupling_params" in hparams.Flow:
            flow_coupling_params =  JsonConfig.to_dict(hparams.Flow.flow_coupling_params)
            
        mixture_expert_params = {}
        if "mixture_expert_params" in hparams.Flow:
            mixture_expert_params =  JsonConfig.to_dict(hparams.Flow.mixture_expert_params)
        
        cond_model_type, cond_model_params = "None", {}
        if "CondModel" in hparams:
            cond_model_type = hparams.CondModel.type
            cond_model_params = JsonConfig.to_dict(hparams.CondModel.hparams)
            cond_model_params['adjs'] = torch.tensor(dataset.get_skeleton().adj_matrix_T, device=device)
                                
        graphflow = flow_model(
                        in_channels = dataset.data_channels if "in_channel" not in hparams.Flow else hparams.Flow.in_channel,
                        K = hparams.Flow.K,
                        depth = hparams.Flow.depth,
                        imc_list = hparams.Flow.imc,
                        actnorm_scale = hparams.Flow.actnorm_scale,
                        flow_type = hparams.Type,
                        flow_step = hparams.Flow.flow_step,
                        flow_coupling = hparams.Flow.flow_coupling,
                        flow_coupling_params = flow_coupling_params,
                        mixture_expert_params = mixture_expert_params,
                        network_model = hparams.Flow.network_model,
                        distribution = hparams.Flow.distribution,
                        device = device,
                        
                        node_n = dataset.get_skeleton().num_joints(),
                        
                        split = hparams.Flow.partition,
                        squeeze = hparams.Flow.squeeze,
                        adj_matrix = torch.tensor(dataset.get_skeleton().adj_matrix_T),
                        restricted = False if 'restricted' not in hparams.Flow else hparams.Flow.restricted,

                        t_his = hparams.Data.t_his,
                        t_pred = hparams.Data.t_pred, 
                                                
                        n_len = n_len,
                        n_pre = n_pre,
                        
                        condition_model_type = cond_model_type,
                        condition_model_params = cond_model_params,
                        
                        mixture_actions = dataset.get_action_types(),
                        mixture_config = "default" if "mixture_config" not in hparams.Flow else hparams.Flow.mixture_config,
                        mixture_initialize_coef = 5.0 if "mixture_initialize_coef" not in hparams.Flow \
                                                    else hparams.Flow.mixture_initialize_coef
                    )
        return graphflow.to(device)
    else:
        raise NotImplementedError
 
 

def recurrent_dynamics_builder(dataset, dataset_name, device, hparams):
    version = hparams.Version
    rnn_dyna_model = recurrent_dyna_dict[version]

    dataset_semantics = datasets.DataSetSemantics[dataset_name]
    if isinstance(dataset_semantics, list):
        dataset_semantics = dataset_semantics[0](dataset_semantics[1])
    semantics_dict = dataset_semantics().get(hparams.GCN.parts_level) 
    semantics = []
    for _, value in semantics_dict.items():
        semantics.append(value)
        
    rnn_dyna = rnn_dyna_model(
                    gcn = hparams.GCN.type,
                    gcn_channels = hparams.GCN.imc,
                    node_n = dataset.get_skeleton().num_joints(),
                    parts_list = semantics,
                    gcn_skip_connect = hparams.GCN.skip_connect,
                    gcn_norm = None if "norm" not in hparams.GCN else hparams.GCN.norm,
                    gcn_activation = "LeakyReLU" if "activation" not in hparams.GCN else hparams.GCN.activation,
                    enc_gcn = None if "enc_gcn" not in hparams.GCN else hparams.GCN.enc_gcn,
                    
                    rnn = hparams.RNN.type,
                    rnn_input_channels = dataset.data_channels if "in_channel" not in hparams.RNN else hparams.RNN.in_channel,
                    rnn_hidden_channels = hparams.RNN.hidden_channels,
                    rnn_num_layers = hparams.RNN.num_layers,
                    rnn_bias = hparams.RNN.bias,
                    rnn_dropout = hparams.RNN.dropout,
                    rnn_birectional = hparams.RNN.birectional,
                    rnn_skip_connect = hparams.RNN.skip_connect,
                    rnn_norm = None if "norm" not in hparams.RNN else hparams.RNN.norm,
                    
                    device = device
                )
    rnn_dyna.to(device)
    return rnn_dyna



def flowmatching_dynamics_builder(dataset, dataset_name, device, hparams):
    version = hparams.Version
    flowmatching_dyna_model = flowmatching_dyna_dict[version]
    flowmatching_dyna = flowmatching_dyna_model(
                            flow_model_params = JsonConfig.to_dict(hparams.FlowModel),
                            flow_path_params = JsonConfig.to_dict(hparams.FlowPath),
                            mean_std = dataset.dataset_mean_std if hasattr(dataset, 'dataset_mean_std') else None,
                            adj_matrix = dataset.get_skeleton().adj_matrix_T,
                        )
    flowmatching_dyna.to(device)
    return flowmatching_dyna



def dynamics_model_builder(dataset, dataset_name, device, hparams):
    hparams_model = hparams
    dataset_flow, dataset_full = dataset[0], dataset[1]
    
    if "RGCN" in hparams_model:
        if "FlowFollwer" not in hparams_model.RGCN or hparams_model.RGCN.FlowFollwer:
            rnn_dynamics_model = recurrent_dynamics_builder(dataset_flow, dataset_name, device, hparams_model.RGCN)
        else:
            rnn_dynamics_model = recurrent_dynamics_builder(dataset_full, dataset_name, device, hparams_model.RGCN)
        return rnn_dynamics_model
    
    elif "FlowMatchingDynamics" in hparams_model:
        flowmatching_dynamics_model = flowmatching_dynamics_builder(dataset_full, dataset_name, device, hparams_model.FlowMatchingDynamics)
        return flowmatching_dynamics_model
        
    else:
        raise NotImplementedError




def normalizingflow_based_network_interface(dataset, device, hparams, **kwargs):
    hparams_model = hparams.Model
    
    # build a flow and load the trained parameters.
    hparams_flow = hparams_model.Flow        
    hparams_flow_path = os.path.join(hparams_flow.trained_model_root, hparams_flow.trained_model_subdir)
    config_flow = os.path.join(os.getcwd(), hparams_flow_path, hparams_flow.trained_model_hparam)
    config_flow = JsonConfig(config_flow)
    
    dataset_flow = dataset_interface(config_flow.Dataset, "test", config_flow)
    
    flow = flow_model_builder(dataset_flow, device, config_flow)
        
    if hparams_flow.epoch_or_path is None or hparams_flow.epoch_or_path != "":
        load(
            epoch_or_path = hparams_flow.epoch_or_path,
            model = flow,
            optim = None,
            schedule = None,
            pkg_dir = os.path.join(hparams_flow.trained_model_root, hparams_flow.trained_model_subdir)
        )
            
    # build a dynamic model.
    dyna_model = dynamics_model_builder(dataset=(dataset_flow, dataset), dataset_name=hparams.Dataset, device=device, hparams=hparams_model)
    
    # build a full network based on the flow and the dynamic model. 
    hparams_pred = hparams_model.GeneralSettings
    hparams_pred_type = hparams_pred.type
    '''
    hparams_pred_pe = hparams_pred.positional_embedding
    # hparams_pred_norm = hparams_pred.normalize_embedding           
    if hparams_pred_pe is True:
        pe = PositionEmbedding(d_model = dataset.get_skeleton().num_joints() * dataset.data_channels, \
                                max_len = hparams.Data.t_his + hparams.Data.t_pred)
    else:
        pe = None
    '''
    drop_coef = None if 'drop_coef' not in hparams_pred else hparams_pred.drop_coef
    var_temp  = None if 'var_temp'  not in hparams_pred else hparams_pred.var_temp
    
    if hparams.Type == "prednet":
        prednet_model = load_prednet(hparams_pred_type)
        prednet = prednet_model(flow, dyna_model, None, positional_embedding=None, drop_coef=drop_coef, var_temp=var_temp)
        return prednet.to(device) 
         
    elif hparams.Type == "flowmatching_prednet":
        sampling_method = hparams_pred.sampling_method if "sampling_method" in hparams_pred else "Uniform"
        sampling_coef = hparams_pred.sampling_coef if "sampling_coef" in hparams_pred else 1.0
        random_drop = hparams_pred.random_drop if "random_drop" in hparams_pred else 0
        anisotropic_coef = hparams_pred.anisotropic_coef if "anisotropic_coef" in hparams_pred else 0.0
        temporal_correlation = hparams_pred.temporal_correlation if "temporal_correlation" in hparams_pred else 1.0
        
        fm_prednet_model = load_flowmatching_prednet(hparams_pred_type)
        if "encoder_retrain" not in hparams_pred:
            fm_prednet = fm_prednet_model(flow, dyna_model,
                                          adj_matrix=torch.tensor(dataset.get_skeleton().adj_matrix_T, device=device),
                                          anisotropic_coef=anisotropic_coef,                 
                                          drop_coef=drop_coef,
                                          sampling_method=sampling_method,
                                          sampling_coef=sampling_coef,
                                          random_drop=random_drop,
                                          temporal_correlation=temporal_correlation)
        else:
            flow_encoder = flow_model_builder(dataset_flow, device, config_flow)
            flow_decoder = flow
            if not hparams_pred.encoder_retrain:
                load(
                    epoch_or_path = hparams_flow.epoch_or_path,
                    model = flow_encoder,
                    optim = None,
                    schedule = None,
                    pkg_dir = os.path.join(hparams_flow.trained_model_root, hparams_flow.trained_model_subdir)
                )
                if hasattr(flow_encoder, "set_actnorm_init"):
                    flow_encoder.set_actnorm_init(inited=True)
            flow_encoder.load_distribution_parameters(flow_decoder.distribution_parameters)
            fm_prednet = fm_prednet_model(flow, dyna_model,
                                          adj_matrix=torch.tensor(dataset.get_skeleton().adj_matrix_T, device=device), 
                                          anisotropic_coef=anisotropic_coef,                
                                          drop_coef=drop_coef,
                                          sampling_method=sampling_method,
                                          sampling_coef=sampling_coef,
                                          random_drop=random_drop,
                                          temporal_correlation=temporal_correlation)
                
        return fm_prednet.to(device)   
          
    else:
        raise NotImplementedError  



def network_interface(dataset, device, hparams, **kwargs):
    hparams_model = hparams.Model
    if "Flow" in hparams_model:
        return normalizingflow_based_network_interface(dataset, device, hparams, **kwargs)
    elif "FlowMatching" in hparams_model:
        raise NotImplementedError
    else:
        raise NotImplementedError
    



def train_eval_interface(model, x, t_his:int, t_pred:int, **kwargs):
    label = None if "label" not in kwargs else kwargs["label"]
    
    if x.ndim == 3:
        x = torch.unsqueeze(x, dim=0)   # from (B, C, N) to (B, 1, C, N)
    batchsize, _, _, _ = x.shape   

    if isinstance(model, models.Seq2SeqPredictVelonlyDistribNet):
        sample      = False if "sample"      not in kwargs else kwargs["sample"]
        zero_joints = None  if "zero_joints" not in kwargs else kwargs["zero_joints"]

        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2)
        
        x_vel_en, nll = model.encode(x, label) if hasattr(model, 'pose_cond') else model.encode(x_vel, label)
        _, f_pred = model.flow.get_seq_length()
        sub_group_num_his = math.ceil(t_his / f_pred)
        nll = nll.reshape((x_vel_en.shape[0], x_vel_en.shape[1]))[sub_group_num_his:].reshape((1, -1)) \
                if hasattr(model.flow, 'cond_model') else nll
        
        y, y_vel_en, (vel_en_mus, vel_en_sigmas), _ = model((x_pos, x_vel, x_vel_en), t_pred, t_his, 
                                                            eval = False, 
                                                            step_by_step = False, 
                                                            sample = sample, 
                                                            bias = None, 
                                                            zero_joints = zero_joints,
                                                            label = label)
        y_en = (y_vel_en, vel_en_mus, vel_en_sigmas)
        x_pred = x[t_his:]
        x_pred_en = x_vel_en[sub_group_num_his:]
        x_en = x_vel_en
    
    elif isinstance(model, models.FlowMatchingPredict_V1):
        step_size = 0.1 if "step_size" not in kwargs else kwargs["step_size"]
        
        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2)
        
        x_en, nll = model.encode(x_pos, label) if hasattr(model, 'output_pose') else model.encode(x_vel, label)
        _, f_pred = model.flow.get_seq_length()
        group_t_his = math.ceil(t_his / f_pred)
        nll = nll.reshape((x_en.shape[0], x_en.shape[1]))[group_t_his:].reshape((1, -1)) \
                if hasattr(model.flow, 'cond_model') else nll

        dx_t, dx_t_pred = model((x_pos, x_vel, x_en), t_pred, t_his, eval=False, step_size=step_size, label=None)
        
        y, x_pred = None, None
        y_en, x_pred_en = (None, dx_t_pred, None), dx_t   
    
    elif isinstance(model, models.FlowMatchingPredict_V3):
        step_size = 0.1 if "step_size" not in kwargs else kwargs["step_size"]
        flow_model_params = None if "outer_model_params" not in kwargs else kwargs["outer_model_params"]
        matching_model_params = None if "inner_model_params" not in kwargs else kwargs["inner_model_params"]
        
        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2)
        
        if flow_model_params is not None and matching_model_params is not None:
            dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss = model((x_pos, x_vel), t_pred, t_his, eval=False, step_size=step_size, label=None, \
                                                                          flow_model_params=flow_model_params, matching_model_params=matching_model_params)
        else:
            dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss = model((x_pos, x_vel), t_pred, t_his, eval=False, step_size=step_size, label=None)
        
        y, x_pred = x_0, x_1
        x_en = aux_loss   # 占用x_en的位置，从而不改变API，但在train_pred中没有实现aux_loss  ## 之前占用nll位置，使用MLE训练aux_loss 
        y_en, x_pred_en = (None, dx_t_pred, None), dx_t  
        nll = x_0_nll

    elif isinstance(model, models.FlowMatchingPredict_V2) or \
         isinstance(model, models.FlowMatchingPredict_V5):
        step_size = 0.1 if "step_size" not in kwargs else kwargs["step_size"]
        flow_model_params = None if "outer_model_params" not in kwargs else kwargs["outer_model_params"]
        matching_model_params = None if "inner_model_params" not in kwargs else kwargs["inner_model_params"]
        
        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2)
        
        if flow_model_params is not None and matching_model_params is not None:
            dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss = model((x_pos, x_vel), t_pred, t_his, eval=False, step_size=step_size, label=None, \
                                                                          flow_model_params=flow_model_params, matching_model_params=matching_model_params)  
        else:          
            dx_t, dx_t_pred, x_0, x_1, x_0_nll, x_1_nll, aux_loss = model((x_pos, x_vel), t_pred, t_his, eval=False, step_size=step_size, label=None)
        
        y, x_pred = x_0, x_1
        x_en = aux_loss   # 占用x_en的位置，从而不改变API，但在train_pred中没有实现aux_loss  ## 之前占用nll位置，使用MLE训练aux_loss 
        y_en, x_pred_en = (None, dx_t_pred, None), dx_t  
        nll = torch.concat((x_0_nll, x_1_nll), dim=0)
           
    '''
    Return:
        y: 预测出的运动序列
        x_pred: y对应的Ground Truth
        y_en: 预测出的隐变量序列
        x_pred_en: y_en对应的Ground Truth
        nll: 输入序列x在flow模型下的负对数似然
        x_en: 输入序列x对应的隐变量序列
    '''
    return (y, x_pred, y_en, x_pred_en, nll, x_en)



def sample_interface(model, x, t_his:int, t_pred:int, **kwargs):
    '''
    Args:
        # x: (length(t_his+t_pred), batch_size, channel, node_n)
         x: (batch_size, length(t_his+t_pred), channel, node_n)
    Returns:
        samples: (nums, batch_size, t_pred, channel, node_n), device=x.device
        gt:      (      batch_size, t_pred, channel, node_n), device=x.device
    '''
    if x.ndim == 3:
        x = torch.unsqueeze(x, dim=1)   # from (B, C, N) to (B, 1, C, N)
    batchsize, _, channel, nodes = x.shape    
    
    label = None if "label" not in kwargs else kwargs["label"]
    likelihood = False if "likelihood" not in kwargs else kwargs["likelihood"]
    CNF_only = False if "CNF_only" not in kwargs else kwargs["CNF_only"]
    ll_seq = None
    
    if isinstance(model, models.Seq2SeqPredictVelonlyDistribNet):              
        # parameters for sampling from the distribution with mu and sigma.
        nums   = 1     if "nums"   not in kwargs else kwargs["nums"]
        sample = False if "sample" not in kwargs else kwargs["sample"]
        # parameters for sampling from the list of bias.
        # 9-0-3 (uniform sampling from [-1, +1] and multiply the copula_radius) for conformal prediction 
        sample_cfgs = ["11"]  if "sample_config" not in kwargs else kwargs["sample_config"]
        sample_bias = [0] if "sample_bias"   not in kwargs else kwargs["sample_bias"]
        # parameters for set specific joints as zero
        zero_joints = None if "zero_joints" not in kwargs else kwargs["zero_joints"]
        # mask for body-part controllable predictions
        mask = None if 'mask' not in kwargs else kwargs['mask']
        unsample_cond = kwargs['unsample_cond'] if 'unsample_cond' in kwargs else False   
        # copula conformal prediction parameters     
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
        copula_type   = kwargs['copula_type']   if 'copula_type'   in kwargs else None

        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(channel/2), dim=-2)
        # x_vel_en, nll = model.encode(x_vel)
        x_vel_en, _ = model.encode(x, label) if hasattr(model, 'pose_cond') else model.encode(x_vel, label)
        
        def get_sample_bias(nums, sample_cfgs, sample_bias):  
            nums = max(nums, len(sample_cfgs))
            if nums > len(sample_cfgs):
                sample_cfgs = sample_cfgs + [0] * (nums-len(sample_cfgs))
                
            bias_list = []
            for i in range(nums):
                if isinstance(sample_bias, list):
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias[i])
                else:
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias)
                if bias is not None:
                    bias_list.append(bias)             
            with torch.no_grad():        
                bias_list = torch.tensor(np.array(bias_list), device=x.device, dtype=torch.float32)
                bias_list = bias_list[:, None, :].expand(-1, batchsize, -1).reshape((-1, bias_list.shape[-1]))
            return bias_list
        
        input_bias = get_sample_bias(nums, sample_cfgs, sample_bias)
                     
        # samples = torch.zeros(size=(nums, t_pred, batchsize, int(channel/2), nodes), device=x.device)
        with torch.no_grad():
            repeat_shape = (nums, -1, -1, -1, -1)
            _x_pos, _x_vel, _x_vel_en = x_pos[None, ...], x_vel[None, ...], x_vel_en[None, ...]
            _x_pos, _x_vel, _x_vel_en = _x_pos.expand(repeat_shape), _x_vel.expand(repeat_shape), _x_vel_en.expand(repeat_shape)

            y, _, (_, sigmas), ll_seq = model((_x_pos, _x_vel, _x_vel_en), t_pred, t_his, 
                                            eval = True, 
                                            step_by_step = False, 
                                            sample = sample, 
                                            bias = input_bias, 
                                            zero_joints = zero_joints,
                                            mask = mask,
                                            unsample_cond = unsample_cond,
                                            CNF_only = CNF_only,
                                            copula_radius = copula_radius,
                                            copula_type = copula_type)
            
            '''
            print('warm up ... \n')
            runtime_cost_list = []
            for _ in range(1000):                
                start = time.time()
                y, _, (_, sigmas), ll_seq = model((_x_pos, _x_vel, _x_vel_en), t_pred, t_his, 
                                                eval = True, 
                                                step_by_step = False, 
                                                sample = sample, 
                                                bias = input_bias, 
                                                zero_joints = zero_joints,
                                                mask = mask,
                                                unsample_cond = unsample_cond,
                                                CNF_only = CNF_only)
                torch.cuda.synchronize()
                end = time.time()
                print('Time:{}ms'.format((end-start)*1000))
                runtime_cost_list.append((end-start)*1000)
            runtime_cost = np.array(runtime_cost_list)
            runtime_mean, runtime_std = np.mean(runtime_cost), np.std(runtime_cost)
            print(runtime_mean)
            print(runtime_std)  
            '''
                
            samples, _ = torch.split(y, int(channel/2), dim=-2)
            if 'var_list' in kwargs:
                kwargs['var_list'].append(sigmas)

            samples = torch.swapaxes(samples, 1, 2)
            x_future_gt = torch.swapaxes(x_pos[t_his:], 0, 1)
            if ll_seq is not None:
                ll_seq = torch.swapaxes(ll_seq, 1, 2)
    
    
    elif isinstance(model, models.FlowMatchingPredict_V1):
        step_size = 0.01 if "step_size" not in kwargs else kwargs["step_size"]
        nums   = 1     if "nums"   not in kwargs else kwargs["nums"]
        sample_cfgs = ["11"]  if "sample_config" not in kwargs else kwargs["sample_config"]
        sample_bias = [0] if "sample_bias"   not in kwargs else kwargs["sample_bias"]
    
        def get_sample_bias(nums, sample_cfgs, sample_bias):  
            nums = max(nums, len(sample_cfgs))
            if nums > len(sample_cfgs):
                sample_cfgs = sample_cfgs + [0] * (nums-len(sample_cfgs))
                
            bias_list = []
            for i in range(nums):
                if isinstance(sample_bias, list):
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias[i])
                else:
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias)
                if bias is not None:
                    bias_list.append(bias)             
            with torch.no_grad():        
                bias_list = torch.tensor(np.array(bias_list), device=x.device, dtype=torch.float32)
                bias_list = bias_list[:, None, :].expand(-1, batchsize, -1).reshape((-1, bias_list.shape[-1]))
            return bias_list
        input_bias = get_sample_bias(nums, sample_cfgs, sample_bias)
        
        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        with torch.no_grad():
            x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2)
            x_en, nll = model.encode(x_pos, label) if hasattr(model, 'pose_cond') else model.encode(x_vel, label)
            
            repeat_shape = (nums, -1, -1, -1, -1)
            _x_pos, _x_vel, _x_en = x_pos[None, ...], x_vel[None, ...], x_en[None, ...]
            _x_pos, _x_vel, _x_en = _x_pos.expand(repeat_shape), _x_vel.expand(repeat_shape), _x_en.expand(repeat_shape)
                
            y = model((_x_pos, _x_vel, _x_en), t_pred, t_his, eval=True, step_size=step_size, \
                       bias=input_bias, label=None)
            ll_seq = None
        
            x_future_gt = torch.swapaxes(x_pos[t_his:], 0, 1)
            samples = y    


    elif isinstance(model, models.FlowMatchingPredict_V2) or \
         isinstance(model, models.FlowMatchingPredict_V3) or \
         isinstance(model, models.FlowMatchingPredict_V5):
        step_size = 0.01 if "step_size" not in kwargs else kwargs["step_size"]
        ode_method = 'euler' if "integ_method" not in kwargs else kwargs["integ_method"]
        nums = 1 if "nums" not in kwargs else kwargs["nums"]
        sample_cfgs = ["11"] if "sample_config" not in kwargs else kwargs["sample_config"]
        sample_bias = [0] if "sample_bias" not in kwargs else kwargs["sample_bias"]
        fm_nll = False if "fm_nll" not in kwargs else kwargs["fm_nll"]
            
        def get_sample_bias(nums, sample_cfgs, sample_bias):  
            nums = max(nums, len(sample_cfgs))
            if nums > len(sample_cfgs):
                sample_cfgs = sample_cfgs + [0] * (nums-len(sample_cfgs))
                
            bias_list = []
            for i in range(nums):
                if isinstance(sample_bias, list):
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias[i])
                else:
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias)
                if bias is not None:
                    bias_list.append(bias)             
            with torch.no_grad():        
                bias_list = torch.tensor(np.array(bias_list), device=x.device, dtype=torch.float32)
                bias_list = bias_list[:, None, :].expand(-1, batchsize, -1).reshape((-1, bias_list.shape[-1]))
            return bias_list
        input_bias = get_sample_bias(nums, sample_cfgs, sample_bias)
        
        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        with torch.no_grad():
            x_pos, x_vel = torch.split(x, int(x.shape[-2]/2), dim=-2) 
                       
            repeat_shape = (nums, -1, -1, -1, -1)
            _x_pos, _x_vel = x_pos[None, ...], x_vel[None, ...]
            _x_pos, _x_vel = _x_pos.expand(repeat_shape), _x_vel.expand(repeat_shape)

            '''
            print('warm up ... \n')
            runtime_cost_list = []
            for _ in range(100):                
                start = time.time()
                y, ll_seq = model((_x_pos, _x_vel), t_pred, t_his, eval=True, step_size=step_size, ode_method=ode_method, \
                                bias=input_bias, label=None, fm_nll=fm_nll)
                torch.cuda.synchronize()
                end = time.time()
                print('Time:{}ms'.format((end-start)*1000))
                runtime_cost_list.append((end-start)*1000)
            runtime_cost = np.array(runtime_cost_list)
            runtime_mean, runtime_std = np.mean(runtime_cost), np.std(runtime_cost)
            print(runtime_mean)
            print(runtime_std)
            '''
                
            y, ll_seq = model((_x_pos, _x_vel), t_pred, t_his, eval=True, step_size=step_size, ode_method=ode_method, \
                              bias=input_bias, label=None, fm_nll=fm_nll)
            y = model.unnormalize_data(y)
        
            x_future_gt = torch.swapaxes(x_pos[t_his:], 0, 1)
            samples = y    
    

    if likelihood:                 
        return samples, x_future_gt, ll_seq
    else:
        return samples, x_future_gt



def sample_interface_non_autoagression(model, x, t_his:int, t_pred:int, **kwargs):
    '''
    Args:
        # x: (length(t_his+t_pred), batch_size, channel, node_n)
         x: (batch_size, length(t_his+t_pred), channel, node_n)
    Returns:
        samples: (nums, batch_size, t_pred, channel, node_n), device=x.device
        gt:      (      batch_size, t_pred, channel, node_n), device=x.device
    '''
    if x.ndim == 3:
        x = torch.unsqueeze(x, dim=1)   # from (B, C, N) to (B, 1, C, N)
    batchsize, _, channel, nodes = x.shape   
       
    label = None if "label" not in kwargs else kwargs["label"]
    likelihood = False if "likelihood" not in kwargs else kwargs["likelihood"]
    ll_seq = None

    if isinstance(model, models.Seq2SeqPredictVelonlyDistribNet):              
        # parameters for sampling from the distribution with mu and sigma.
        nums   = 1     if "nums"   not in kwargs else kwargs["nums"]
        sample = False if "sample" not in kwargs else kwargs["sample"]
        # parameters for sampling from the list of bias.
        # 9-0-3 (uniform sampling from [-1, +1] and multiply the copula_radius) for conformal prediction 
        sample_cfgs = []  if "sample_config" not in kwargs else kwargs["sample_config"]
        sample_bias = [0] if "sample_bias"   not in kwargs else kwargs["sample_bias"]
        # parameters for set specific joints as zero
        zero_joints = None if "zero_joints" not in kwargs else kwargs["zero_joints"]
        # mask for body-part controllable predictions
        mask = [] if 'mask' not in kwargs else kwargs['mask']
        unsample_cond = kwargs['unsample_cond'] if 'unsample_cond' in kwargs else False
        # copula radius for conformal prediction
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None

        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(channel/2), dim=-2)
        x_vel_en, _ = model.encode(x, label) if hasattr(model, 'pose_cond') else model.encode(x_vel, label)
                
        def get_sample_bias(nums, sample_cfgs, sample_bias):  
            nums = max(nums, len(sample_cfgs))
            if nums > len(sample_cfgs):
                sample_cfgs = sample_cfgs + [0] * (nums-len(sample_cfgs))
                
            bias_list = []
            for i in range(nums):
                if isinstance(sample_bias, list):
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias[i])
                else:
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias)
                if bias is not None:
                    bias_list.append(bias)             
            with torch.no_grad():        
                bias_list = torch.tensor(np.array(bias_list), device=x.device, dtype=torch.float32)
                bias_list = bias_list[:, None, :].expand(-1, batchsize, -1).reshape((-1, bias_list.shape[-1]))
            return bias_list
        
        deterministic_bias = get_sample_bias(1, ['11'], 0.0)
        input_bias = get_sample_bias(nums, sample_cfgs, sample_bias)
                       
        with torch.no_grad():
            repeat_shape = (1, -1, -1, -1, -1)
            _x_pos, _x_vel, _x_vel_en = x_pos[None, ...], x_vel[None, ...], x_vel_en[None, ...]
            _x_pos, _x_vel, _x_vel_en = _x_pos.expand(repeat_shape), _x_vel.expand(repeat_shape), _x_vel_en.expand(repeat_shape)

            y, _, (_, sigmas), ll_seq = model.non_autoregressive_forward((_x_pos, _x_vel, _x_vel_en), t_pred, t_his, 
                                                                        eval = True, 
                                                                        step_by_step = False, 
                                                                        sample = sample, 
                                                                        bias = input_bias,
                                                                        nums = nums, 
                                                                        zero_joints = zero_joints,
                                                                        mask = mask,
                                                                        unsample_cond = unsample_cond,
                                                                        deterministic_bias = deterministic_bias,
                                                                        copula_radius = copula_radius)
            samples, _ = torch.split(y, int(channel/2), dim=-2)                
            samples = torch.swapaxes(samples, 1, 2)
            x_future_gt = torch.swapaxes(x_pos[t_his:], 0, 1) 
            if ll_seq is not None:
                ll_seq = torch.swapaxes(ll_seq, 1, 2)
            
    if likelihood:                 
        return samples, x_future_gt, ll_seq
    else:
        return samples, x_future_gt



def motion_matching_sample_interface(model, x, t_his:int, t_pred:int, **kwargs):
    '''
    Args:
        # x: (length(t_his+t_pred), batch_size, channel, node_n)
         x: (batch_size, length(t_his+t_pred), channel, node_n)
    Returns:
        samples: (nums, batch_size, t_pred, channel, node_n), device=x.device
        gt:      (      batch_size, t_pred, channel, node_n), device=x.device
    '''
    if x.ndim == 3:
        x = torch.unsqueeze(x, dim=1)   # from (B, C, N) to (B, 1, C, N)
    batchsize, _, channel, nodes = x.shape    
    
    label = None if "label" not in kwargs else kwargs["label"]
    likelihood = False if "likelihood" not in kwargs else kwargs["likelihood"]
    ll_seq = None

    if isinstance(model, models.Seq2SeqPredictVelonlyDistribNet):              
        nums = 1 if "nums" not in kwargs else kwargs["nums"]
        # motion matching database
        database = None if "database" not in kwargs else kwargs["database"]
        matching_metric = None if "matching_metric" not in kwargs else kwargs["matching_metric"]
        single_matching = True if "single_matching" not in kwargs else kwargs["single_matching"]
        # sample configs
        sample_cfgs = ["11"]  if "sample_config" not in kwargs else kwargs["sample_config"]
        sample_bias = [0] if "sample_bias"   not in kwargs else kwargs["sample_bias"]        
        # mask for body-part controllable predictions
        mask = [] if 'mask' not in kwargs else kwargs['mask']
        # copula conformal prediction parameters     
        copula_radius = kwargs['copula_radius'] if 'copula_radius' in kwargs else None
        copula_type   = kwargs['copula_type']   if 'copula_type'   in kwargs else None

        def get_sample_bias(nums, sample_cfgs, sample_bias):  
            nums = max(nums, len(sample_cfgs))
            if nums > len(sample_cfgs):
                sample_cfgs = sample_cfgs + [0] * (nums-len(sample_cfgs))
                
            bias_list = []
            for i in range(nums):
                if isinstance(sample_bias, list):
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias[i])
                else:
                    _, bias = SampleConfig.get_config(length=t_pred, index=sample_cfgs[i], stds=None, bias=sample_bias)
                if bias is not None:
                    bias_list.append(bias)             
            with torch.no_grad():        
                bias_list = torch.tensor(np.array(bias_list), device=x.device, dtype=torch.float32)
                bias_list = bias_list[:, None, :].expand(-1, batchsize, -1).reshape((-1, bias_list.shape[-1]))
            return bias_list
        
        input_bias = get_sample_bias(nums, sample_cfgs, sample_bias) if not single_matching else None

        x = torch.swapaxes(x, 0, 1) # from (B, L, C, N) to (L, B, C, N)
        x_pos, x_vel = torch.split(x, int(channel/2), dim=-2)
        x_vel_en, _ = model.encode(x, label) if hasattr(model, 'pose_cond') else model.encode(x_vel, label)
                             
        with torch.no_grad():
            if single_matching:
                _x_pos, _x_vel, _x_vel_en = x_pos, x_vel, x_vel_en      
                     
                y, _, (_, sigmas), ll_seq = model.motion_matching_single_forward(
                                                (_x_pos, _x_vel, _x_vel_en), t_pred, t_his, 
                                                eval = True, 
                                                step_by_step = False, 
                                                sample_nums = nums,
                                                bias = input_bias,
                                                database = database,
                                                matching_metric = matching_metric,
                                                mask = mask,
                                                copula_radius = copula_radius,
                                                copula_type = copula_type)
            
            else:
                repeat_shape = (nums, -1, -1, -1, -1)
                _x_pos, _x_vel, _x_vel_en = x_pos[None, ...], x_vel[None, ...], x_vel_en[None, ...]
                _x_pos, _x_vel, _x_vel_en = _x_pos.expand(repeat_shape), _x_vel.expand(repeat_shape), _x_vel_en.expand(repeat_shape) 

                y, _, (_, sigmas), ll_seq = model.motion_matching_forward(
                                                (_x_pos, _x_vel, _x_vel_en), t_pred, t_his, 
                                                eval = True, 
                                                step_by_step = False, 
                                                sample_nums = nums,
                                                bias = input_bias,
                                                database = database,
                                                matching_metric = matching_metric,
                                                mask = mask,
                                                copula_radius = copula_radius,
                                                copula_type = copula_type)
                            
            samples, _ = torch.split(y, int(channel/2), dim=-2)
            if 'var_list' in kwargs:
                kwargs['var_list'].append(sigmas)

            samples = torch.swapaxes(samples, 1, 2)
            x_future_gt = torch.swapaxes(x_pos[t_his:], 0, 1)
            if ll_seq is not None:
                ll_seq = torch.swapaxes(ll_seq, 1, 2)
                
    if likelihood:                 
        return samples, x_future_gt, ll_seq
    else:
        return samples, x_future_gt



def flow_interface(model, x, logdet, reverse, label=None, **kwargs):
    if hasattr(model, 'cond_model'):
        if isinstance(x, list) or isinstance(x, tuple):
            _input = x
        else:        
            # x: (length, group_size, batch_size, channel, node_n) or (length/group_size, batch_size, channel, node_n)      
            if x.ndim == 4:
                x = x[None, ...]
            cond_his, cond_pred = model.get_condition_length()           
            input, cond_input = x[:, cond_his:], x[:, :cond_his]
                
            if 2 * model.get_channels() == x.shape[-2]:
                _, input = torch.split(input, int(input.shape[-2]/2), dim=-2)
                cond_input, _ = torch.split(cond_input, int(cond_input.shape[-2]/2), dim=-2) if min(cond_input.shape) > 0 \
                                                                                            else (cond_input, _)                                                                                                           
            input = model.data_preprocessing(input)
            _input = (input, cond_input)
    else:
        input = model.data_preprocessing(x) if not reverse else x
        _input = input    
    
    z_alpha = 1.0 if "z_alpha" not in kwargs else kwargs["z_alpha"]
    if "model_params" in kwargs:
        z, nll = functional_call(model, kwargs["model_params"], args=(_input, logdet), kwargs={'reverse':reverse, 'label':label, 'z_alpha':z_alpha})
    else:
        z, nll = model(input=_input, logdet=logdet, reverse=reverse, label=label, z_alpha=z_alpha)
    return z, nll