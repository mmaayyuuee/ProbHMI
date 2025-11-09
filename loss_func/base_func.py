import torch
import numpy as np
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


class BaseLoss(object):
    loss_func_dict = {
    }
    
    def __init__(self, cmds:dict) -> None:        
        self.loss_func = {}
        for key, value in cmds.items():
            if key not in self.loss_func_dict:
                continue
            self.loss_func[key] = []
            for func, weight in value.items():
                self.loss_func[key].append([func, self.loss_func_dict[key][func], weight])
        self.reset_memos()
        return
    
        
    def compute_loss(self, datas:dict):
        sum_loss = 0.0
        loss_list = {}
        for loss_type, data in datas.items():
            if loss_type in self.loss_func:
                funcs_weights = self.loss_func[loss_type]
                for func_weight in funcs_weights:
                    if func_weight[2] > 0:
                        loss = func_weight[2] * func_weight[1](*data)
                        loss_list[func_weight[0]] = loss
                        # sum_loss += loss.item()
                        sum_loss += loss
        self.__memos(sum_loss, loss_list)
        return sum_loss, loss_list

    
    def __memos(self, sum_loss, loss_list:dict):
        self.history_lens += 1
        self.history_loss += sum_loss.item()
        for name, loss in loss_list.items():
            self.history_loss_list[name] = loss.item() if name not in self.history_loss_list \
                                        else self.history_loss_list[name] + loss.item()
        return
    
    
    def reset_memos(self):
        self.history_loss = 0.0
        self.history_loss_list = {}
        self.history_lens = 0
        return
    

    def get_history(self):
        return [self.history_loss, self.history_loss_list, self.history_lens]
    
    
    def get_average_history(self):
        average_loss = self.history_loss / self.history_lens
        average_loss_list = {}
        for name, loss in self.history_loss_list.items():
            average_loss_list[name] = loss / self.history_lens
        return [average_loss, average_loss_list, self.history_lens]