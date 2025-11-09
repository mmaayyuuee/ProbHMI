import os
import re
import copy
import torch
import numpy as np
from shutil import copyfile



def _file_at_step(epoch):
    return "save_{}.pkg".format(epoch)



def _file_best():
    return "trained.pkg"



# def save(epoch, model, optim, schedule, pkg_dir="", is_best=False, max_checkpoints=None):
def save(epoch, model, optim, schedule, distribution=None, pkg_dir="", is_best=False, max_checkpoints=None):
    if optim is None:
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": {},
            "schedule": {},
            "distribution": {} if distribution is None else distribution
        }
    else:    
        if not isinstance(optim, dict) and not isinstance(schedule, dict):
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "schedule": schedule.state_dict(),
                "distribution": {} if distribution is None else distribution
            }
        elif isinstance(optim, dict) and isinstance(schedule, dict):
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": {},
                "schedule": {},
                "distribution": {} if distribution is None else distribution
            }
            for key, opt in optim.items():
                state["optim"][key] = opt.state_dict()
            for key, sch in schedule.items():
                state["schedule"][key] = sch.state_dict()
        else:
            raise ValueError("optim and schedule must be or not be dict simultaneously")
    
    save_path = os.path.join(pkg_dir, _file_at_step(epoch))
    best_path = os.path.join(pkg_dir, _file_best())
    
    torch.save(state, save_path)
    
    if is_best:
        copyfile(save_path, best_path)
    
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_\d*\.pkg", file_name):
                digits = file_name.replace("save_", "").replace(".pkg", "")
                number = int(digits)
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0]))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)
    return



def load(epoch_or_path, model, optim=None, schedule=None, pkg_dir=""):
    step = epoch_or_path
    save_path = None
	
    print("LOADING FROM pkg_dir: " + pkg_dir)
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best())
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint with given step or filename or `best`")
        return

    # begin to load
    state = torch.load(save_path, weights_only=False)
    # state = torch.load(save_path)
    epoch = state["epoch"]
    model.load_state_dict(state["model"], strict=True)
        
    if optim is not None:        
        if isinstance(optim, dict):
            for key, opt in optim.items():
                # opt.load_state_dict(state["optim"][key].state_dict())
                if key in state["optim"]:
                    opt.load_state_dict(state["optim"][key])
        else:
            optim.load_state_dict(state["optim"])
        
    if schedule is not None:
        if isinstance(schedule, dict):
            for key, sche in schedule.items():
                # sche.load_state_dict(state["schedule"][key].state_dict())
                if key in state["schedule"]:
                    sche.load_state_dict(state["schedule"][key])
        else:
            schedule.load_state_dict(state["schedule"])
    
    # if "distribution" in state:        
    #     model.load_distribution_parameters(state["distribution"])
    # model.set_actnorm_init(inited=True)
    if "distribution" in state:
        if isinstance(state["distribution"], dict):
            if len(state["distribution"]) > 0:
                model.load_distribution_parameters(state["distribution"])
        else:
            model.load_distribution_parameters(state["distribution"])
    
    if hasattr(model, "set_actnorm_init"):
        model.set_actnorm_init(inited=True)

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return epoch



if __name__ == "__main__":
    file_name = _file_at_step(12)
    digits = file_name.replace("save_", "").replace(".pkg", "")
    digits = int(digits)
    print(digits)