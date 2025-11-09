# Human36M
from .Human36M.dataset_h36m import StochasticDatasetH36MexpMap
from .Human36M.dataset_h36m import DeterministicDatasetH36MexpMap
from .Human36M.dataset_h36m import StochasticTorchDatasetH36MexpMap
from .Human36M.dataset_h36m import DeterministicTorchDatasetH36MexpMap
from .Human36M.dataset_h36m import StandardStochasticDatasetH36MexpMap
from .Human36M.skeleton_semantics_h36m import Human36m18expmapSkeletonSemantics
from .Human36M.skeleton_semantics_h36m import Human36m24expmapSkeletonSemantics

# HumanEva-1
from .HumanEva.dataset_humaneva import StochasticDatasetHumanEvaExpMap, StochasticDatasetHumanEvaPos
from .HumanEva.dataset_humaneva import StochasticTorchDatasetHumanEvaExpMap, StochasticTorchDatasetHumanEvaPos
from .HumanEva.skeleton_semantics_humanEva import HumanEva18expmapSkeletonSemantics
from .HumanEva.skeleton_semantics_humanEva import HumanEva15expmapSkeletonSemantics

# AMASS
from .AMASS.data_amass import DataSetAMASSexpMapDataLoader, DataSetAMASSexpMap
from .AMASS.data_amass import DataSetAMASSPosDataLoader
from .AMASS.AMASS_semantics import AmassSemantics

from utils.wrapper import this_is_wrapper
import torch


Datasets = {
    ### Human36M
    # stochastic motion prediction    
    "H36Mso3N18"       :  StochasticDatasetH36MexpMap,
    "H36Mso3N18Vel"    :  [this_is_wrapper({"velocityFlag": "velocity"}), StochasticDatasetH36MexpMap],
    "H36Mso3N18VelOnly":  [this_is_wrapper({"velocityFlag": "velocityOnly"}), StochasticDatasetH36MexpMap],
    "H36Mso3N18_DL"    :  StochasticTorchDatasetH36MexpMap,
    "H36Mso3N18Vel_DL" :  [this_is_wrapper({"velocityFlag": "velocity"}), StochasticTorchDatasetH36MexpMap],
    
    # deterministic motion prediction        
    "H36Mso3N22"       :  DeterministicDatasetH36MexpMap,
    "H36Mso3N22Vel"    :  [this_is_wrapper({"velocityFlag": "velocity"}), DeterministicDatasetH36MexpMap],
    "H36Mso3N22VelOnly":  [this_is_wrapper({"velocityFlag": "velocityOnly"}), DeterministicDatasetH36MexpMap],
    "H36Mso3N22_DL"    :  DeterministicTorchDatasetH36MexpMap,
    "H36Mso3N22Vel_DL" :  [this_is_wrapper({"velocityFlag": "velocity"}), DeterministicTorchDatasetH36MexpMap],
        
    ### HumanEva-1    
    "HumanEVAposN18"       :  [this_is_wrapper({"18N": True}), StochasticDatasetHumanEvaPos],
    "HumanEVAposN18Vel"    :  [this_is_wrapper({"18N": True, "velocityFlag": "velocity"}), StochasticDatasetHumanEvaPos],
    "HumanEVAposN18VelOnly":  [this_is_wrapper({"18N": True, "velocityFlag": "velocityOnly"}), StochasticDatasetHumanEvaPos],
    "HumanEVAposN18Vel_DL"    :  [this_is_wrapper({"18N": True, "velocityFlag": "velocity"}), StochasticTorchDatasetHumanEvaPos],
    
    ### AMASS
    "AMASS": DataSetAMASSexpMap,    
    "AMASSVel": [this_is_wrapper({"velocityFlag": "velocity"}), DataSetAMASSexpMap],
    "AMASSVelOnly": [this_is_wrapper({"velocityFlag": "velocityOnly"}), DataSetAMASSexpMap],
    "AMASSVel_DL": [this_is_wrapper({"velocityFlag": "velocity"}), DataSetAMASSexpMapDataLoader],
}



DataSetSemantics = {
    ### Human36M
    # stochastic motion prediction    
    "H36Mso3N18"         :  Human36m18expmapSkeletonSemantics,
    "H36Mso3N18Vel"      :  Human36m18expmapSkeletonSemantics,
    "H36Mso3N18VelOnly"  :  Human36m18expmapSkeletonSemantics,
    "H36Mso3N18Vel_DL"   :  Human36m18expmapSkeletonSemantics,
    
    # deterministic motion prediction    
    "H36Mso3N22"         :  Human36m24expmapSkeletonSemantics,
    "H36Mso3N22Vel"      :  Human36m24expmapSkeletonSemantics,
    "H36Mso3N22VelOnly"  :  Human36m24expmapSkeletonSemantics,
    "H36Mso3N22Vel_DL"   :  Human36m24expmapSkeletonSemantics,
    
    ### HumanEva-1
    "HumanEVAso3N18"       :  HumanEva18expmapSkeletonSemantics,
    "HumanEVAso3N18Vel"    :  HumanEva18expmapSkeletonSemantics,
    "HumanEVAso3N18VelOnly":  HumanEva18expmapSkeletonSemantics,
    
    "HumanEVAso3N15"       :  HumanEva15expmapSkeletonSemantics,
    "HumanEVAso3N15Vel"    :  HumanEva15expmapSkeletonSemantics,
    "HumanEVAso3N15VelOnly":  HumanEva15expmapSkeletonSemantics,

    ### AMASS
    "AMASS": AmassSemantics,
    "AMASSVel": AmassSemantics,
    "AMASSVelOnly": AmassSemantics,
    "AMASSVel_DL": AmassSemantics,
}



def dataset_train_generator_factory(dataset, *args, **kwargs):
    '''
        args[0]: batch_size
        args[1]: sample numbers
        kwargs[0]: generator type
        kwargs[1]: number_workers
    '''
    generator_type = None if 'generator_type' not in kwargs else kwargs['generator_type']
    if generator_type:
        generator = getattr(dataset, generator_type)
        sampler = generator(args[1], args[0])
    else:
        if hasattr(dataset, 'DataLoader'):
            import torch
            num_workers = 0 if 'num_workers' not in kwargs else kwargs['num_workers']
            dataset.generate_segments(step=1)
            sampler = torch.utils.data.DataLoader(dataset, args[0], shuffle=True, drop_last=False, pin_memory=False, num_workers=num_workers)
        else:
            generator = getattr(dataset, 'sampling_generator')
            if "augs" in kwargs:
                sampler = generator(args[1], args[0], augmentation=kwargs["augs"]) if kwargs["augs"]>=0 \
                        else generator(args[1], args[0])
            else:
                sampler = generator(args[1], args[0])
    return sampler


def dataset_test_generator_factory(dataset, *args, **kwargs):
    '''
        args[0]: batch_size
        args[1]: step
        kwargs[0]: generator type
        kwargs[1]: number_workers
    '''
    generator_type = None if 'generator_type' not in kwargs else kwargs['generator_type']
    if generator_type:
        generator = getattr(dataset, generator_type)
        if generator_type == 'sample_256' and callable(dataset.sample_256):
            sampler = generator()
        
        elif generator_type == 'sample_8' and callable(dataset.sample_8):
            sampler = generator()
            
        elif generator_type == 'sampling_generator' and callable(dataset.sampling_generator):   
            sampler = generator(args[1], args[0])
            
        elif generator_type == 'iter_generator' and callable(dataset.iter_generator):
            sampler = generator(args[1], args[0])  

        elif hasattr(dataset, 'DataLoader'):
            import torch
            num_workers = 0 if 'num_workers' not in kwargs else kwargs['num_workers']
            dataset.generate_segments(step=args[1])
            sampler = torch.utils.data.DataLoader(dataset, args[0], shuffle=False, drop_last=False, pin_memory=False, num_workers=num_workers)    
              
    else:
        # if hasattr(dataset, 'sample_256'):
        #     sampler = getattr(dataset, 'sample_256')() 
        if hasattr(dataset, 'sample_8'):
            sampler = getattr(dataset, 'sample_8')() 
        elif hasattr(dataset, 'DataLoader'):
            import torch
            num_workers = 0 if 'num_workers' not in kwargs else kwargs['num_workers']
            dataset.generate_segments(step=args[1])
            sampler = torch.utils.data.DataLoader(dataset, args[0], shuffle=False, drop_last=False, pin_memory=False, num_workers=num_workers)                                
        else:
            generator = getattr(dataset, 'iter_generator')
            sampler = generator(args[1], args[0])  
    return sampler


def dataset_vis_generator_factory(dataset, *args, **kwargs):
    '''
        args[0]: batch_size
        args[1]: step
        kwargs[0]: generator type
        kwargs[1]: number_workers
    '''
    generator_type = None if 'generator_type' not in kwargs else kwargs['generator_type']
    if generator_type:
        generator = getattr(dataset, generator_type)  
        if generator_type == 'sampling_generator':
            sampler = generator(args[1], args[0])
            
        elif generator_type == 'iter_generator':
            sampler = generator(args[1], args[0])
            
    else:
        if hasattr(dataset, 'iter_generator') and callable(dataset.iter_generator):
            generator = getattr(dataset, 'iter_generator')
            sampler = generator(args[1], args[0]) 
        else:
            import torch
            num_workers = 0 if 'num_workers' not in kwargs else kwargs['num_workers']
            dataset.generate_segments(step=args[1])
            sampler = torch.utils.data.DataLoader(dataset, args[0], shuffle=False, drop_last=False, pin_memory=False, num_workers=num_workers)
    return sampler