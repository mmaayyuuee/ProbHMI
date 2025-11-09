import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import scipy
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from utils import thops
from utils.pca import TorchPCA


class GaussianDiag(object):
    Log2PI = float(np.log(2 * np.pi))

    def likelihood(self, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (((x) ** 2) + GaussianDiag.Log2PI)

    def logp(self, x, *args, **kwargs):
        likelihood = self.likelihood(x)
        return torch.sum(likelihood, dim=(-2, -1))

    def sample(self, z_shape, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros(z_shape),
                           std=torch.ones(z_shape) * eps_std)
        return eps

    @property
    def distrib_parameters(self):
       return {}
    
    def load_distribution_parameters(self, parameters_dict):
        return    
    
       
            
class StudentT(object):
    def __init__(self, df, d):
        self.df=df
        self.d=d
        self.norm_const = scipy.special.loggamma(0.5*(df+d))-scipy.special.loggamma(0.5*df)-0.5*d*np.log(np.pi*df)
        return

    def logp(self,x,**kwargs):
        '''
        Multivariate t-student density:
        output:
            the sum density of the given element
        '''
        x_norms = thops.sum(((x) ** 2), dim=[1])
        likelihood = self.norm_const-0.5*(self.df+self.d)*torch.log(1+(1/self.df)*x_norms)
        return thops.sum(likelihood, dim=[1])

    def sample(self,z_shape, eps_std=None, device=None):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        n : int
            number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        x_shape = torch.Size((z_shape[0], 1, z_shape[2]))
        x = np.random.chisquare(self.df, x_shape)/self.df
        x = np.tile(x, (1,z_shape[1],1))
        x = torch.Tensor(x.astype(np.float32))
        z = torch.normal(mean=torch.zeros(z_shape),std=torch.ones(z_shape) * eps_std)
        return (z/torch.sqrt(x)).to(device)



'''
class GaussianMixture(torch.distributions.Distribution):
    def __init__(self, means, inv_cov_stds=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds

        self.weights = torch.ones((len(means)), device=device)
        self.device = device
        return

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
                          for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians


    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples
        
        
    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs


    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted


    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)


    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)
'''



class GaussianMixture(object):
    Log2PI = float(np.log(2 * np.pi))
    
    f''' # deprecated version
    def __init__(self, component, config='default', device='cpu'):
        self.component, self.config = component, config
        self.device = device
        self.n_components = len(self.component)
        self.means = []
        return
    '''
    
    def __init__(self, component, config='default', initial_coef=5.0, channel=3, node_n=18, device='cpu'):
        self.component, self.config = component, config
        self.initial_coef = initial_coef
        self.channel, self.node_n = channel, node_n
        self.device = device
        self.n_components = len(self.component)
        self.initialize_parameters()
        return
    
    def initialize_parameters(self):
        if self.config == 'default':
            means = torch.normal(mean=torch.zeros((self.n_components,), device=self.device), std=1.0)
            self.means = [mean.expand(1, self.channel, self.node_n) for mean in means]
            self.weights = torch.ones((self.n_components), device=self.device) / self.n_components
            self.gaussians = [distributions.Normal(mean, 1.0) for mean in self.means]
            
        elif self.config == 'uniform' or self.config == 'single_uniform':
            means = torch.rand(self.n_components, device=self.device) * self.initial_coef - self.initial_coef / 2
            self.means = [mean.expand(1, self.channel, self.node_n) for mean in means]
            self.weights = torch.ones((self.n_components), device=self.device) / self.n_components
            self.gaussians = [distributions.Normal(mean, 1.0) for mean in self.means]
            
        else:                
            raise NotImplementedError
        return
    
    
    def logp(self, x, label=None):
        if len(self.means) <= 0:
            self.initialize_parameters()
        
        if x.ndim == 4:
            length, batch_size, channel, node_n = x.shape
        elif x.ndim == 3:
            batch_size, channel, node_n = x.shape
        
        if label is not None:
            means = [self._get_component(lab) for lab in label]
            if None in means:
                means = []
        else:
            means = []
           
        if len(means) > 0:
            means = torch.concat(means, dim=0)
            if means.ndim == 1:
                means = means.reshape(-1, 1, 1).expand(-1, channel, node_n)
            
            mixture_log_probs = self._likelihood(x, means)
            mixture_log_probs = torch.sum(mixture_log_probs, dim=(-2, -1))
            ''' # For validation
            distrib = [self.gaussians[self.component.index(lab)] for lab in label]
            log_probs = []
            for idx in range(len(distrib)):
                log_probs.append(torch.sum(distrib[idx].log_prob(x[:, idx]), dim=(-2, -1)))
            print()
            '''
        else:
            means = [mean.data for mean in self.means]
            means = torch.concat(means, dim=0)
            if means.ndim == 1:
                all_means = means.reshape(-1, 1, 1, 1).expand(-1, batch_size, channel, node_n)
            else:
                all_means = means[:, None, ...]
                all_means = all_means.expand(size=(self.n_components, batch_size, channel, node_n))
                
            all_log_probs = self._likelihood(x[:, None, ...], all_means) if x.ndim == 4 else \
                            self._likelihood(x[None, None, ...], all_means)
            weights_log_probs = torch.log(F.softmax(self.weights, dim=0))[:, None, None, None].expand_as(all_means)
            mixture_log_probs = torch.logsumexp(all_log_probs + weights_log_probs, dim=1)
            mixture_log_probs = torch.sum(mixture_log_probs, dim=(-2, -1))
            ''' # For validation
            log_probs = torch.cat([g.log_prob(x) for g in self.gaussians], dim=0)
            log_probs = torch.logsumexp(log_probs + weights_log_probs, dim=0)
            log_probs = torch.sum(log_probs, dim=(-2, -1))
            print()
            '''
        return mixture_log_probs


    def _likelihood(self, x, mu):
        return -0.5 * (((x - mu) ** 2) + GaussianMixture.Log2PI)
    
    def _get_component(self, label):
        if label in self.component:
            return self.means[self.component.index(label)]
        else:
            return None
    
    @property
    def distrib_parameters(self):
       return {
           'mean': self.means,
           'weights': self.weights
        }
    
    def load_distribution_parameters(self, parameters_dict):
        means, weights = parameters_dict['mean'], parameters_dict['weights']
        if self.config == 'default' or self.config == 'uniform':
            self.means, self.weights = means, weights
            self.gaussians = [distributions.Normal(mean, 1.0) for mean in self.means]
        return
    
    def set_distribution_parameters(self, means, weights=None, **kwargs):
        self.means = means
        if weights is not None:
            self.weights = weights
        return
    
    
    def classification(self, x, metric='mse', pca_model:TorchPCA=None, use_stats=False, **kwargs): 
        batch_size, feat_channels = x.shape

        if use_stats:
            all_means = kwargs['mean']
        else:
            means = [mean.data for mean in self.means]
            all_means = torch.concat(means, dim=0)
            all_means = torch.reshape(all_means, shape=(all_means.shape[0], -1))
    
        if pca_model is not None:
            x = pca_model.transform(x)
            if not use_stats:
                all_means = pca_model.transform(all_means)
                
        all_means = all_means[:, None, ...]
        all_means = all_means.expand(size=(-1, batch_size, -1))
        
        if metric == 'mse' or metric == 'identity_gaussian':
            distrib = distributions.Normal(loc=all_means, scale=1.0)
            likelihood = distrib.log_prob(x)
            score = torch.mean(likelihood, dim=-1)
        
        elif metric == 'multi_gaussian':
            std = kwargs['std']
            std = torch.clamp(std, min=1e-6)
            std = std[:, None, :].expand(-1, batch_size, -1)
            distrib = distributions.Normal(loc=all_means, scale=std)
            likelihood = distrib.log_prob(x)
            score = torch.mean(likelihood, dim=-1)
            
        elif metric == 'cosine':
            score_ = F.cosine_similarity(x, all_means, dim=-1)
            score = 1 - score_
            
        elif metric == 'mahalanobis':
            cov = kwargs['cov']
            U, S, Vh = torch.linalg.svd(cov, full_matrices=False)                
            S_regularized = torch.where(S>1e-4, S, 1e-4)
            diag_inv = torch.diag_embed(1.0 / S_regularized)
            cov_pinv = Vh.transpose(-2, -1) @ diag_inv @ U.transpose(-2, -1)          
                    
            diff = x - all_means
            left = torch.einsum('bnd,bdd->bnd', diff, cov_pinv)    # (batch_size, n, d)
            distance_sq = torch.einsum('bnd,bnd->bn', left, diff)  # (batch_size, n)
            score_ = torch.sqrt(torch.clamp(distance_sq, min=0))
            
            score = score_ / score_.sum(dim=0)
            score = 1.0 - score
            
        label_idx = (torch.argmax(score, dim=0)).detach().cpu().numpy()
        label = np.array(self.component)[label_idx]
        uncertainty = score.swapaxes(0, 1)[np.arange(label_idx.shape[0]), label_idx]
        return label, label_idx, uncertainty

    
    def reclustering(self, data):
        mean = [torch.mean(data[action], dim=0)[None] for action in self.component]
        self.set_distribution_parameters(mean)
        return
    
 

class LearnableGaussianMixture(nn.Module, GaussianMixture):
    def __init__(self, component, config='default', initial_coef=5.0, channel=3, node_n=18, device='cpu'):
        nn.Module.__init__(self)
        GaussianMixture.__init__(self, component, config, initial_coef, channel, node_n, device)
        return    
    
    def initialize_parameters(self):
        if self.config == 'default':
            means = torch.normal(mean=torch.zeros((self.n_components,), device=self.device), std=1.0)  
        elif self.config == 'uniform' or self.config == 'single_uniform':
            means = torch.rand(self.n_components, device=self.device) * self.initial_coef - self.initial_coef / 2  
        else:                
            raise NotImplementedError

        if self.config == 'single_uniform':
            self.means = nn.ParameterList([nn.Parameter(mean.clone().reshape(-1,)).to(self.device) for mean in means])
        else:
            self.means = nn.ParameterList([nn.Parameter(mean.expand(1, self.channel, self.node_n).clone()).to(self.device) 
                                           for mean in means])
        self.weights = torch.ones((self.n_components), device=self.device)
        self.gaussians = [distributions.Normal(mean, 1.0) for mean in self.means]
        return



class ClusteringGaussianMixture(GaussianMixture):  
    def logp(self, x, label=None):
        if len(self.means) <= 0:
            self.initialize_parameters()
        
        if x.ndim == 4:
            length, batch_size, channel, node_n = x.shape
        elif x.ndim == 3:
            length, batch_size, channel, node_n = (1, *x.shape)
        x = x.reshape(length*batch_size, channel, node_n)
        
        if label is not None:
            all_means = torch.concat([mean.data for mean in self.means], dim=0)
            if all_means.ndim == 1:
                all_means = all_means.reshape(-1, 1, 1).expand(-1, channel, node_n)
            means = self._get_component(x, all_means)
            mixture_log_probs = self._likelihood(x, means)
            mixture_log_probs = torch.sum(mixture_log_probs, dim=(-2, -1))
           
        else:
            means = [mean.data for mean in self.means]
            means = torch.concat(means, dim=0)
            if means.ndim == 1:
                all_means = means.reshape(-1, 1, 1, 1).expand(-1, length*batch_size, channel, node_n)
            else:
                all_means = means[:, None, ...]
                all_means = all_means.expand(size=(self.n_components, length*batch_size, channel, node_n))
                
            all_log_probs = self._likelihood(x[None, ...], all_means)
            weights_log_probs = torch.log(F.softmax(self.weights, dim=0))[:, None, None, None].expand_as(all_means)
            mixture_log_probs = torch.logsumexp(all_log_probs + weights_log_probs, dim=0)
            mixture_log_probs = torch.sum(mixture_log_probs, dim=(-2, -1))
        
        mixture_log_probs = mixture_log_probs.reshape(length, batch_size)
        return mixture_log_probs    
    
      
    def _get_component(self, x, means):
        x_flat = x.view(x.shape[0], -1)
        means_flat = means.view(means.shape[0], -1)
        
        dist = torch.cdist(x_flat, means_flat, p=2)
        min_dist_indices = torch.argmin(dist, dim=1)
        means_min_dist = means[min_dist_indices]
        return means_min_dist
    
    
    def get_cluster_num(self, x):
        if x.ndim == 4:
            length, batch_size, channel, node_n = x.shape
        elif x.ndim == 3:
            length, batch_size, channel, node_n = (1, *x.shape)
        x = x.reshape(length*batch_size, channel, node_n)

        all_means = torch.concat([mean.data for mean in self.means], dim=0)
        if all_means.ndim == 1:
            all_means = all_means.reshape(-1, 1, 1).expand(-1, channel, node_n)
            
        x_flat = x.view(x.shape[0], -1)
        means_flat = all_means.view(all_means.shape[0], -1)
        
        dist = torch.cdist(x_flat, means_flat, p=2)
        min_dist_indices = torch.argmin(dist, dim=1)
        return min_dist_indices
        
    
    def sampling(self, x=None, label=None, stds=None, method="Gaussian"):
        if len(self.means) <= 0:
            self.initialize_parameters()        
 
        if x.ndim == 4:
            length, batch_size, channel, node_n = x.shape
        elif x.ndim == 3:
            length, batch_size, channel, node_n = (1, *x.shape)
        x = x.reshape(length*batch_size, channel, node_n)
                
        all_means = torch.concat([mean.data for mean in self.means], dim=0)
        if all_means.ndim == 1:
            all_means = all_means.reshape(-1, 1, 1).expand(-1, channel, node_n)     
        
        if label is not None:                
            batch_means = self._get_component(x, all_means)
        else:
            component_idx = torch.multinomial(self.weights, length*batch_size, replacement=True)
            batch_means = all_means[component_idx]

        if method == "Gaussian":
            samples = batch_means + stds * torch.randn_like(x) if stds is not None else \
                    batch_means + 1.0 * torch.randn_like(x)
        elif method == "Uniform":
            samples = batch_means + stds * torch.rand_like(x) if stds is not None else \
                    batch_means + 1.0 * torch.rand_like(x)
        else:
            raise NotImplementedError         
        
        samples = samples.reshape(length, batch_size, channel, node_n)    
        return samples
        

    def reclustering(self, data):
        data_list = []
        for key, value in data.items():
            data_list.append(value)
        new_data = torch.concat(data_list, dim=0)
        
        new_data_flat = new_data.view(new_data.shape[0], -1)

        all_means = torch.concat([mean.data for mean in self.means], dim=0)
        if all_means.ndim == 1:
            all_means = all_means.reshape(-1, 1, 1).expand(-1, new_data.shape[-2], new_data.shape[-1])
        all_means_flat = all_means.view(all_means.shape[0], -1)
        
        dist = torch.cdist(new_data_flat, all_means_flat, p=2)
        dist = torch.softmax(dist, dim=-1)
                
        mean = []
        for idx in range(self.n_components):
            tmp = (dist[:, idx][:, None, None].expand(-1, new_data.shape[-2], new_data.shape[-1]) * new_data) 
            tmp = torch.sum(tmp, dim=0) / torch.sum(dist[:, idx])
            mean.append(tmp[None])
        self.set_distribution_parameters(mean)
        return


    def hard_reclustering(self, data):
        data_list = []
        for key, value in data.items():
            data_list.append(value)
        new_data = torch.concat(data_list, dim=0)
        
        new_data_flat = new_data.view(new_data.shape[0], -1)

        all_means = torch.concat([mean.data for mean in self.means], dim=0)
        if all_means.ndim == 1:
            all_means = all_means.reshape(-1, 1, 1).expand(-1, new_data.shape[-2], new_data.shape[-1])
        all_means_flat = all_means.view(all_means.shape[0], -1)
        
        dist = torch.cdist(new_data_flat, all_means_flat, p=2)
        dist = torch.softmax(dist, dim=-1)
        max_vals = torch.max(dist, dim=-1, keepdim=True)[0]
        dist = torch.where(dist == max_vals, torch.tensor(1.0), torch.tensor(0.0))
        
        mean = []
        for idx in range(self.n_components):
            sum = torch.sum(dist[:, idx])
            if sum <= 0:
                mean.append(self.means[idx])
            else:
                tmp = (dist[:, idx][:, None, None].expand(-1, new_data.shape[-2], new_data.shape[-1]) * new_data) 
                tmp = torch.sum(tmp, dim=0) / torch.sum(dist[:, idx])
                mean.append(tmp[None])
        self.set_distribution_parameters(mean)
        return


    def em_reclustering(self, data):
        data_list = []
        for key, value in data.items():
            data_list.append(value)
        new_data = torch.concat(data_list, dim=0)
        
        new_data_flat = new_data.view(new_data.shape[0], -1)

        all_means = torch.concat([mean.data for mean in self.means], dim=0)
        if all_means.ndim == 1:
            all_means = all_means.reshape(-1, 1, 1).expand(-1, new_data.shape[-2], new_data.shape[-1])
        all_means_flat = all_means.view(all_means.shape[0], -1)
        
        dist = torch.cdist(new_data_flat, all_means_flat, p=2)
        dist = dist * self.weights
        score = dist / torch.sum(dist, dim=1)[:, None].expand(-1, dist.shape[-1])
        
        weight = torch.sum(score, dim=0) / score.shape[0]
                 
        mean = []
        for idx in range(self.n_components):
            tmp = (dist[:, idx][:, None, None].expand(-1, new_data.shape[-2], new_data.shape[-1]) * new_data) 
            tmp = torch.sum(tmp, dim=0) / torch.sum(dist[:, idx])
            mean.append(tmp[None])
        self.set_distribution_parameters(mean, weights=weight)
        return



if __name__ == "__main__":
    gmm = GaussianMixture(['A', 'B', 'C', 'D'], 'default')
    data = torch.normal(torch.zeros(size=(1, 4, 3, 18)), 1.0)
    label = ['A', 'B', 'C', 'D']
    gmm.logp(data, None)
        
        