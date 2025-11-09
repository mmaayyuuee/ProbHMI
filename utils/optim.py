import torch
from utils.config import JsonConfig
from utils.warmup import GradualWarmupScheduler

'''
# https://github.com/tensorflow/tensor2tensor/issues/280#issuecomment-339110329
def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000, minimum=None):
     # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    if minimum is not None and global_step > warmup_steps:
        if lr < minimum:
            lr = minimum
    return lr


def step_learning_rate_decay(init_lr, global_step, anneal_rate=0.98, anneal_interval=30000):
    return init_lr * anneal_rate ** (global_step // anneal_interval)
'''


optimizer_dict = {
    "SGD": lambda params, kwargs: torch.optim.SGD(params, **kwargs),
    "adam": lambda params, kwargs: torch.optim.Adam(params, **kwargs),
    "adamW": lambda params, kwargs: torch.optim.AdamW(params, **kwargs),
    "RMSprop": lambda params, kwargs: torch.optim.RMSprop(params, **kwargs)
}



def build_optimizer_and_schedule(hparams, model_params, last_epoch=None):
    optim_name = hparams.Optim.name
    optim_args = hparams.part_to_dict(hparams.Optim.args)
    last_epoch = hparams.Train.last_epoch if last_epoch is None else last_epoch

    warmup = hparams.Optim.warmup
    schedule_name = hparams.Optim.schedule.name
    schedule_args = hparams.Optim.schedule.args
    
    if schedule_name == "step":
        optimizer = optimizer_dict[optim_name]([{'params':model_params, 'initial_lr':optim_args['lr']}], optim_args)
    else:
        optimizer = optimizer_dict[optim_name](model_params, optim_args)
                
    if schedule_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, 
                                                    step_size = schedule_args.step_size, 
                                                    gamma = schedule_args.gamma,
                                                    last_epoch = last_epoch,
                                                    verbose = True)
    elif schedule_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                               mode = schedule_args.mode,
                                                               factor = schedule_args.factor,
                                                               patience = schedule_args.patience,
                                                               verbose = True,
                                                               threshold = schedule_args.threshold,
                                                               threshold_mode = 'abs' if 'thres_mode' not in schedule_args \
                                                                                      else schedule_args.thres_mode,
                                                               cooldown = schedule_args.cooldown,
                                                               min_lr = schedule_args.min_lr)   
    elif schedule_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, 
                                                         milestones = schedule_args.milestones, 
                                                         gamma = schedule_args.gamma,
                                                         last_epoch = last_epoch,
                                                         verbose = True)
    elif schedule_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer,
                                                           gamma = schedule_args.gamma,
                                                           last_epoch = last_epoch,
                                                           verbose = True)
    elif schedule_name == "lambda":
        pass
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', schedule_name)
    
    if warmup > 1:
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup, after_scheduler=scheduler)
        return optimizer, warmup_scheduler
    
    return optimizer, scheduler



def build_optimizer_and_schedule_abbrev(hparams, model_params, last_epoch=None):
    optim_name = hparams.name
    optim_args = hparams.part_to_dict(hparams.args)
    schedule_name = hparams.schedule.name
    schedule_args = hparams.schedule.args
    
    if schedule_name == "step":
        optimizer = optimizer_dict[optim_name]([{'params':model_params, 'initial_lr':optim_args['lr']}], optim_args)
    else:
        optimizer = optimizer_dict[optim_name](model_params, optim_args)
                
    if schedule_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, 
                                                    step_size = schedule_args.step_size, 
                                                    gamma = schedule_args.gamma,
                                                    last_epoch = last_epoch,
                                                    verbose = True)
    elif schedule_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                               mode = schedule_args.mode,
                                                               factor = schedule_args.factor,
                                                               patience = schedule_args.patience,
                                                               verbose = True,
                                                               threshold = schedule_args.threshold,
                                                               threshold_mode = 'abs' if 'thres_mode' not in schedule_args \
                                                                                      else schedule_args.thres_mode,
                                                               cooldown = schedule_args.cooldown,
                                                               min_lr = schedule_args.min_lr)   
    elif schedule_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, 
                                                         milestones = schedule_args.milestones, 
                                                         gamma = schedule_args.gamma,
                                                         last_epoch = last_epoch,
                                                         verbose = True)
    elif schedule_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer,
                                                           gamma = schedule_args.gamma,
                                                           last_epoch = last_epoch,
                                                           verbose = True)
    elif schedule_name == "lambda":
        pass
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', schedule_name)
    
    return optimizer, scheduler