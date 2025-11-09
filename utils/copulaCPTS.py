import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import time

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
import interface


class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()
        return

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(torch.sigmoid((self.alphas - pseudo_data) * 1000), dim=1)
            )
        )
        return torch.abs(coverage - 1 + self.epsilon)


def search_alpha(alpha_input, epsilon, epochs=500):
    pseudo_data = torch.tensor(alpha_input)
    if alpha_input.ndim == 2:
        dim = alpha_input.shape[-1]
    elif alpha_input.ndim == 3:
        dim = (alpha_input.shape[-2], alpha_input.shape[-1])
    cp = CP(dim, epsilon)
    optimizer = torch.optim.Adam(cp.parameters(), weight_decay=1e-4)

    with trange(epochs, desc="training", unit="epochs") as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = cp(pseudo_data)

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.detach().numpy())
    return cp.alphas.detach().numpy()



class CopulaCPTS(object):
    def __init__(self, model, dataset, t_his, t_pred, batchsize=256, metric=None, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.t_his, self.t_pred = t_his, t_pred
        self.batchsize = batchsize
        self.metric = metric
        self.device = device
        return
    
    
    def get_threshold(self, epsilon=0.1, cali_step=0):        
        cali_data = self.dataset.get_cali_data(step=cali_step)
        if cali_data:
            cali_data, copula_data = cali_data
        else:
            return 0.0
        
        start = time.time()
        nonconformity = self.calibrate(cali_data)
        
        scores_on_copula = self.compute_scores(copula_data)
        alphas = []
        for i in range(scores_on_copula.shape[0]):
            a = (scores_on_copula[i] > nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)

        threshold = search_alpha(alphas, epsilon, epochs=1000)

        mapping_shape = nonconformity.shape[0]
        mapping = np.sort(nonconformity.transpose(1, 2, 0), axis=-1)
        
        idx = (np.floor(threshold * mapping_shape) + 1).astype(int)
        idx = np.where(idx>=mapping_shape, mapping_shape-1, idx)
        
        idx = idx[..., np.newaxis].repeat(mapping_shape, axis=-1)
        support_idx = np.arange(mapping_shape)[np.newaxis, np.newaxis, ...].repeat(idx.shape[0], axis=0).repeat(idx.shape[1], axis=1)
        bool_idx = np.where(idx == support_idx, 1.0, 0.0)
        
        quantile = mapping * bool_idx
        quantile = np.sum(quantile, axis=-1)
        pos_threshold = self.post_process_threshold(quantile)
        pos_threshold = np.repeat(pos_threshold, self.data_channles, axis=-1) if pos_threshold.shape[-1] == 1 else pos_threshold
        
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))
        return pos_threshold

    
    def post_process_threshold(self, threshold):
        if self.metric == "mae":
            threshold = threshold
        elif self.metric == "mse":  # absolutely equal to "mae" when using the sqrt function.
            threshold = np.sqrt(threshold)
        elif self.metric == "L2":
            threshold = threshold / np.sqrt(self.data_channles)
        return threshold
        

    def calibrate(self, data):
        return self.compute_scores(data)
    
    
    def compute_scores(self, data):
        pred, gt = [], []
        st, ed = 0, min(self.batchsize, data.shape[0])
        with torch.no_grad():
            while True:
                batch_data = data[st:ed]
                batch_data = batch_data.to(self.device) if isinstance(batch_data, torch.Tensor) \
                                                        else torch.tensor(batch_data, device=self.device)
                batch_data = torch.swapaxes(batch_data, -2, -1)

                _, _, y_en, x_pred_en, _, _ = interface.train_eval_interface(model = self.model, 
                                                                             x = batch_data, 
                                                                             t_his = self.t_his, 
                                                                             t_pred = self.t_pred,
                                                                             zero_joints = torch.tensor(data=self.dataset.zero_joints, device=self.device) 
                                                                            )
                _, en_mus, _ = y_en
                en_mus = torch.squeeze(en_mus)
                
                pred.append(en_mus)
                gt.append(x_pred_en)
                
                st, ed = st+self.batchsize, min(ed+self.batchsize, data.shape[0])
                if st > data.shape[0]:
                    break
                
            pred = torch.swapaxes(torch.concat(pred, dim=1), 0, 1)
            gt = torch.swapaxes(torch.concat(gt, dim=1), 0, 1)
        
        if not hasattr(self, "data_channles") or not hasattr(self, "data_nodes"):
            self.data_channles = pred.shape[-2]
            self.data_nodes = pred.shape[-1]
                
        if self.metric == "mae":
            pred, gt = pred.reshape(pred.shape[0], pred.shape[1], -1), gt.reshape(gt.shape[0], gt.shape[1], -1)
            scores = self.mae(pred, gt)
            
        elif self.metric == "mse":
            pred, gt = pred.reshape(pred.shape[0], pred.shape[1], -1), gt.reshape(gt.shape[0], gt.shape[1], -1)
            scores = self.mse(pred, gt)
            
        elif self.metric == "L2":
            scores = self.L2(pred, gt)            
        return scores
    
    
    def mse(self, pred, gt):
        scores = torch.pow((pred - gt), 2).detach().cpu().numpy()
        return scores 

    def mae(self, pred, gt):
        scores = torch.abs((pred - gt)).detach().cpu().numpy()
        return scores
    
    def L2(self, pred, gt):
        scores = torch.norm((pred - gt), p=2, dim=-2).detach().cpu().numpy()
        return scores



class StdCopulaCPTS(CopulaCPTS):           
    def compute_scores(self, data):
        pred, stds, gt = [], [], []
        st, ed = 0, min(self.batchsize, data.shape[0])
        with torch.no_grad():
            while True:
                batch_data = data[st:ed]
                batch_data = batch_data.to(self.device) if isinstance(batch_data, torch.Tensor) \
                                                        else torch.tensor(batch_data, device=self.device)
                batch_data = torch.swapaxes(batch_data, -2, -1)

                _, _, y_en, x_pred_en, _, _ = interface.train_eval_interface(model = self.model, 
                                                                             x = batch_data, 
                                                                             t_his = self.t_his, 
                                                                             t_pred = self.t_pred,
                                                                             zero_joints = torch.tensor(data=self.dataset.zero_joints, device=self.device) 
                                                                            )
                _, en_mus, en_sigmas = y_en
                en_mus, en_sigmas = torch.squeeze(en_mus), torch.squeeze(en_sigmas)
                
                pred.append(en_mus)
                stds.append(en_sigmas)
                gt.append(x_pred_en)
                
                st, ed = st+self.batchsize, min(ed+self.batchsize, data.shape[0])
                if st > data.shape[0] or st == data.shape[0]:
                    break
                                
            pred = torch.swapaxes(torch.concat(pred, dim=1), 0, 1)
            pred = torch.reshape(pred, shape=(pred.shape[0], pred.shape[1], -1))
            
            gt = torch.swapaxes(torch.concat(gt, dim=1), 0, 1)
            gt = torch.reshape(gt, shape=(gt.shape[0], gt.shape[1], -1))
            
            stds = torch.swapaxes(torch.concat(stds, dim=1), 0, 1)
            stds = torch.reshape(stds, shape=(stds.shape[0], stds.shape[1], -1))

            if not hasattr(self, "data_channles"):
                self.data_channles = pred.shape[-1]
                        
        if self.metric == "mae":
            scores = self.mae(pred, gt, stds)   
        elif self.metric == "mse":
            scores = self.mse(pred, gt, stds)      
        elif self.metric == "L1":
            scores = self.L1(pred, gt, stds)
        return scores    
    

    def post_process_threshold(self, threshold):
        if self.metric == "mae":
            threshold = threshold
        elif self.metric == "mse":  # absolutely equal to "mae" when using the sqrt function.
            threshold = threshold
        elif self.metric == "L1":
            threshold = threshold / self.data_channles
        return threshold


    def mse(self, pred, gt, stds):
        scores = (torch.pow((pred - gt), 2) / stds).detach().cpu().numpy()
        return scores 

    def mae(self, pred, gt, stds):
        scores = (torch.abs((pred - gt)) / stds).detach().cpu().numpy()
        return scores
    
    def L1(self, pred, gt, stds):
        scores = torch.sum((torch.abs((pred - gt)) / stds), dim=-1)
        scores = scores[..., None]
        return scores.detach().cpu().numpy() 