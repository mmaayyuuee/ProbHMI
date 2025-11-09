import torch
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import math
import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from utils.thops import forward_kinematics, expmap2rotmat_torch, expmap2rotmat_torch_V2
from utils.copulaCPTS import CopulaCPTS, StdCopulaCPTS
from loss_func.base_func import BaseLoss
from loss_func.geodesic_loss import GeodesicLoss


''' loss function in eular coordinate representation space. '''
# X,Y size: (length, batch_size, channels, node_n)
def euler_position_mse_loss(X, Y):
    diff = torch.subtract(X, Y)
    # dist = torch.sum(torch.pow(diff, 2), dim=[0, 2, 3])
    dist = torch.sum(torch.pow(diff, 2), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def euler_position_mae_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.sum(torch.abs(diff), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss    


def bone_length_wrapper(parents, hip_vel_location):
    def euler_bone_length_mse_loss(X, Y):
        if parents is None:
            return 0.0
        else:
            if hip_vel_location >= 0:
                X[..., hip_vel_location] = 0
                Y[..., hip_vel_location] = 0
            
            bones_end_index = torch.tensor(parents, device=X.device)
            bones_end_index[0] = 0
            
            def compute_bone_length(data):
                bone_st = data[..., 1:]
                bone_ed = torch.index_select(data, -1, bones_end_index)[..., 1:]
                diff = torch.subtract(bone_ed, bone_st)
                dist = torch.sum(torch.pow(diff, 2), dim=-2)
                bone_length = torch.sqrt(dist)
                return bone_length
            
            x_bone_length = compute_bone_length(X)
            y_bone_length = compute_bone_length(Y)
            
            bone_diff = torch.subtract(x_bone_length, y_bone_length)
            # bone_length_mse = torch.sum(torch.pow(bone_diff, 2), dim=[0, 2])
            bone_length_mse = torch.sum(torch.pow(bone_diff, 2), dim=[2])
            bone_length_mse = torch.mean(bone_length_mse)
            return bone_length_mse   
    return euler_bone_length_mse_loss


def linear_velocity_wrapper(hip_vel_location):
    def euler_velocity_mse_loss(X, Y):
        if X.shape[0] <= 1 or Y.shape[0] <=1:
            return 0.0
        
        if hip_vel_location >= 0:
            X[..., hip_vel_location] = 0
            Y[..., hip_vel_location] = 0
        
        def roll(data):
            return torch.roll(data, shifts=1, dims=0)
        _X, _Y = roll(X), roll(Y)
        
        x_vel = torch.subtract(_X, X)[1:, ...]
        y_vel = torch.subtract(_Y, Y)[1:, ...]

        diff = torch.subtract(x_vel, y_vel)
        # dist = torch.sum(torch.pow(diff, 2), dim=[0, 2, 3])
        dist = torch.sum(torch.pow(diff, 2), dim=[2, 3])
        loss = torch.mean(dist)
        return loss
    return euler_velocity_mse_loss
    
    
       
''' loss function about likelihood. '''
def maximum_likelihood_estimation(nll):
    return torch.mean(nll)


class FlowPriorloss(object):
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def preprocess(X, Y):
        x = X[0] if isinstance(X, tuple) else X
        y = Y
        return x, y
    
    # X,Y size: (length, batch_size, channels, node_n)
    @classmethod
    def mse(cls, X, Y):
        # X = FlowPriorloss.preprocess(X)
        X, Y = cls.preprocess(X, Y)
        diff = torch.subtract(X, Y)
        dist = torch.sum(torch.pow(diff, 2), dim=[2, 3])
        loss = torch.mean(dist)
        return loss
    
    @classmethod
    def mae(cls, X, Y):
        # X = FlowPriorloss.preprocess(X)
        X, Y = cls.preprocess(X, Y)
        diff = torch.subtract(X, Y)
        dist = torch.sum(torch.abs(diff), dim=[2, 3])
        loss = torch.mean(dist)
        return loss
    
    @classmethod
    def weight_mse(cls, X, Y):
        # X = FlowPriorloss.preprocess(X)
        X, Y = cls.preprocess(X, Y)
        diff = torch.subtract(X, Y)
        
        with torch.no_grad():
            abs = torch.abs(diff)
            abs = torch.reshape(abs, shape=(abs.shape[0], abs.shape[1], -1))
            MIN = torch.unsqueeze(torch.min(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
            MAX = torch.unsqueeze(torch.max(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
            weight = (abs - MIN) / (MAX - MIN) + 0.1   
            weight = torch.pow(weight, 2)
            weight = torch.reshape(weight, shape=diff.shape)
            
        dist = torch.sum(weight*torch.pow(diff, 2), dim=[2, 3])
        loss = torch.mean(dist)
        return loss
        
    @classmethod
    def linear_constraint_mse(cls, X, Y):
        # X = FlowPriorloss.preprocess(X)
        X, Y = cls.preprocess(X, Y)
        # X: results of prediction ;  Y: ground truth
        # X/Y.shape = (length, batchsize, channel, node)
        length, _, _, _ = Y.shape
        if length < 3:
            return 0.0
        
        select_nums = max(int(length/4), 3)
        
        indice = np.sort(np.random.choice(length, select_nums, replace=False))
        indice = torch.tensor(indice, device=Y.device)
        x = torch.index_select(Y, 0, indice)
        
        _indice = indice - indice[0]
        coef2 = torch.reshape(_indice, shape=(select_nums, 1, 1, 1)).broadcast_to(x.shape) * (1.0/_indice[-1])
        coef1 = torch.flip(coef2, dims=[0])
        
        _x = coef1*x[0] + coef2*x[-1]
        itp_loss = torch.sum(torch.pow(torch.subtract(x, _x), 2), dim=(2, 3))
        itp_loss = torch.mean(itp_loss)
        return itp_loss

    # from DeepAR https://github.com/husnejahan/DeepAR-pytorch
    @classmethod
    def negative_log_likelihood(cls, X, Y):
        mu, sigma = X[1], X[2]
        distribution = torch.distributions.Normal(mu, sigma)
        nll = -torch.mean(distribution.log_prob(Y))        
        return nll
    
    @classmethod
    def inverse_negative_log_likelihood(cls, X, Y):
        mu, _ = X[1], X[2]
        distribution = torch.distributions.Normal(Y, 1.0)
        nll = -torch.mean(distribution.log_prob(mu))        
        return nll


@DeprecationWarning
class VelocityFlowPriorLoss(FlowPriorloss):
    @staticmethod
    def preprocess(X, Y):
        '''
        Args: 
            X: (y_en, (vel_mu, vel_sigma, pos))
            Y: (..., pos+vel)
        '''
        y_en, (vel_mu, vel_sigma, pos) = X
        x_pred_en_pos, x_pred_en_vel = torch.split(Y, int(Y.shape[-2]/2), dim=-2)
        
        if vel_sigma:
            x = torch.concat((y_en, pos), dim=-2)
            y = torch.concat((x_pred_en_pos, x_pred_en_vel, x_pred_en_pos), dim=-2)
        else:
            vel_sample = vel_mu
            x = torch.concat((pos, vel_sample), dim=-2)
            x = torch.concat((y_en, x), dim=0)
            y = Y
        return x, y
    
    @classmethod
    def negative_log_likelihood(cls, X, Y):
        _, (mu, sigma, _) = X
        y = torch.split(Y, int(Y.shape[-2]/2), dim=-2)[1]   # get velocity part in Y
        distribution = torch.distributions.Normal(mu, sigma)
        nll = -torch.mean(distribution.log_prob(y))        
        return nll



''' loss function in angle representation space. '''
''' MSE loss functions '''
def expmap_mse_loss(X, Y):
    diff = torch.subtract(X, Y)
    # dist = torch.sum(torch.pow(diff, 2), dim=[0, 2, 3])
    dist = torch.sum(torch.pow(diff, 2), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def expmap_weight_mse_loss(X, Y):
    diff = torch.subtract(X, Y)    
    with torch.no_grad():
        abs = torch.abs(diff)
        abs = torch.reshape(abs, shape=(abs.shape[0], abs.shape[1], -1))
        MIN = torch.unsqueeze(torch.min(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
        MAX = torch.unsqueeze(torch.max(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
        weight = (abs - MIN) / (MAX - MIN) + 0.1   
        weight = torch.pow(weight, 2)
        weight = torch.reshape(weight, shape=diff.shape)
        
    # dist = torch.sum(weight*torch.pow(diff, 2), dim=[0, 2, 3])
    dist = torch.sum(weight*torch.pow(diff, 2), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def expmap_emphasize_hip_mse_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.pow(diff, 2)
    dist[..., 0] *= 10
    # dist = torch.sum(dist, dim=[0, 2, 3])
    dist = torch.sum(dist, dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def expmap_angle_mse_loss(X, Y):
    x_angle = torch.linalg.norm(X, dim=-2)
    y_angle = torch.linalg.norm(Y, dim=-2)
    diff = x_angle - y_angle
    # dist = torch.sum(torch.pow(diff, 2), dim=[0, 2])
    dist = torch.sum(torch.pow(diff, 2), dim=[2])
    loss = torch.mean(dist)
    return loss



''' MAE loss functions '''
def expmap_mae_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.sum(torch.abs(diff), dim=[-2, -1])
    # dist2 = torch.mean(torch.abs(diff), dim=[2, 3])  ## 20240309 test
    loss = torch.mean(dist)
    return loss


def expmap_weight_mae_loss(X, Y):
    diff = torch.subtract(X, Y)    
    with torch.no_grad():
        abs = torch.abs(diff)
        abs = torch.reshape(abs, shape=(abs.shape[0], abs.shape[1], -1))
        MIN = torch.unsqueeze(torch.min(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
        MAX = torch.unsqueeze(torch.max(abs, dim=-1).values, dim=-1).broadcast_to(abs.shape)
        weight = (abs - MIN) / (MAX - MIN) + 0.1   
        weight = torch.pow(weight, 2)
        weight = torch.reshape(weight, shape=diff.shape)
        
    # dist = torch.sum(weight*torch.abs(diff), dim=[0, 2, 3])
    dist = torch.sum(weight*torch.abs(diff), dim=[-2, -1])
    loss = torch.mean(dist)
    return loss


def expmap_emphasize_hip_mae_loss(X, Y):
    diff = torch.subtract(X, Y)
    dist = torch.abs(diff)
    dist[..., 0] *= 10
    # dist = torch.sum(dist, dim=[0, 2, 3])
    dist = torch.sum(dist, dim=[-2, -1])
    loss = torch.mean(dist)
    return loss



''' reweight loss functions '''
class ReweightLoss(object):
    coef_18nodes = [10, 5, 2, 0, 5, 2, 0, 10, 10, 5, 2, 0, 5, 2, 0, 5, 2, 0]
    # coef_24nodes = [10, 10, 5, 2, 1, 10, 5, 2, 1, 10, 10, 5, 2, 1, 10, 5, 2, 1, 10, 5, 2, 1, 0, 0] ## for human3.6m
    coef_24nodes = [10, 5, 5, 5, 2, 2, 2, 1, 1, 5, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # coef_24nodes = [25, 5, 5, 5, 2, 2, 2, 1, 1, 5, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    def __init__(self) -> None:
        pass

    @classmethod
    def reweight(cls, loss):
        if loss.shape[-1] == 18:
            coef = torch.tensor(cls.coef_18nodes, device=loss.device)
        elif loss.shape[-1] == 24:
            coef = torch.tensor(cls.coef_24nodes, device=loss.device)
        # coef = torch.unsqueeze(coef, dim=[0, 1, 2]).broadcast_to(loss.shape)
        coef = torch.broadcast_to(coef, loss.shape)
        reweight_loss = coef * loss
        return reweight_loss
    
    @classmethod
    def reweight_mse_loss(cls, X, Y):
        diff = torch.subtract(X, Y)
        reweight_dist = cls.reweight(torch.pow(diff, 2))
        # loss = torch.sum(reweight_dist, dim=[0, 2, 3])
        loss = torch.sum(reweight_dist, dim=[-2, -1])
        loss = torch.mean(loss)
        return loss

    @classmethod
    def reweight_mae_loss(cls, X, Y):
        diff = torch.subtract(X, Y)
        reweight_dist = cls.reweight(torch.abs(diff))
        # loss = torch.sum(reweight_dist, dim=[0, 2, 3])
        loss = torch.sum(reweight_dist, dim=[-2, -1])
        loss = torch.mean(loss)
        return loss


expmap_reweight_mse_loss = ReweightLoss.reweight_mse_loss
expmap_reweight_mae_loss = ReweightLoss.reweight_mae_loss



''' exmap - geodesic loss '''
class ExpmapGeodesicLoss(object):
    geodesic_loss_none = GeodesicLoss(reduction="none")
    geodesic_loss_mean = GeodesicLoss(reduction="mean")
    geodesic_loss_sum  = GeodesicLoss(reduction="sum")

    @staticmethod
    def preprocess(X, Y):
        X = torch.swapaxes(X, -2, -1)
        X = torch.reshape(X, shape=(-1, X.shape[-1]))
        if X.shape[-1] > 3:
            X_pos, _ = torch.split(X, int(X.shape[-1]/2), dim=-1)
        else:
            X_pos = X
        
        Y = torch.swapaxes(Y, -2, -1)
        Y = torch.reshape(Y, shape=(-1, Y.shape[-1]))
        if Y.shape[-1] > 3:
            Y_pos, _ = torch.split(Y, int(Y.shape[-1]/2), dim=-1)
        else:
            Y_pos = Y
        
        X_pos, Y_pos = expmap2rotmat_torch(X_pos), expmap2rotmat_torch(Y_pos)
        return X_pos, Y_pos
    
    @classmethod
    def compute_none(cls, X, Y):
        X, Y = cls.preprocess(X, Y)
        dists = cls.geodesic_loss_none(X, Y)
        return dists

    @classmethod
    def compute_mean(cls, X, Y):
        X, Y = cls.preprocess(X, Y)
        dists = cls.geodesic_loss_mean(X, Y)
        return dists            

    @classmethod
    def compute_sum(cls, X, Y):
        X, Y = cls.preprocess(X, Y)
        dists = cls.geodesic_loss_sum(X, Y)
        return dists


class RotMatGeodesicLoss(ExpmapGeodesicLoss):    
    @staticmethod
    def preprocess(X, Y):
        X_pos, _ = torch.split(X, int(X.shape[-3]/2), dim=-3)
        
        Y = torch.swapaxes(Y, -2, -1)
        Y_pos, _ = torch.split(Y, int(Y.shape[-1]/2), dim=-1)
        Y_pos = expmap2rotmat_torch_V2(Y_pos)
        return X_pos, Y_pos



''' miscellaneous loss '''
class SampleLoss(object):
    nll = 0.0
    mse = 0.0
    mae = 0.0
    x_flag = None
    
    @classmethod
    def gaussian_distrib_sample_loss(cls, model, x, z, sample_nums=1):
        cls.x_flag = torch.clone(x)
        
        nll, mse, mae = 0.0, 0.0, 0.0
        for _ in range(0, sample_nums):
            _z = torch.reshape(z, shape=(-1, z.shape[-2], z.shape[-1]))
            _z = torch.normal(mean=_z, std=1.0)
            _x = model(input=_z, reverse=True)
            _, _nll = model(input=_x, logdet=0.0, reverse=False)
            nll += maximum_likelihood_estimation(_nll)
            _x = torch.reshape(_x, x.shape)
            _mse = torch.mean(torch.sum(torch.pow(torch.subtract(x, _x), 2), dim=[1, 2]))
            _mae = torch.mean(torch.sum(torch.abs(torch.subtract(x, _x)), dim=[1, 2]))
            mse += _mse
            mae += _mae
        cls.nll = nll
        cls.mse = mse
        cls.mae = mae
        return nll, mse, mae
    
    @classmethod
    def nll_loss(cls, model, x, z, sample_nums=1):
        if cls.x_flag is not None and torch.equal(x, cls.x_flag):
            return cls.nll
        else:
            nll, _, _ = cls.gaussian_distrib_sample_loss(model, x, z, sample_nums)
            return nll
    
    @classmethod
    def mse_loss(cls, model, x, z, sample_nums=1):
        if cls.x_flag is not None and torch.equal(x, cls.x_flag):
            return cls.mse
        else:
            _, mse, _ = cls.gaussian_distrib_sample_loss(model, x, z, sample_nums)
            return mse
        
    @classmethod
    def mae_loss(cls, model, x, z, sample_nums=1):
        if cls.x_flag is not None and torch.equal(x, cls.x_flag):
            return cls.mae
        else:
            _, _, mae = cls.gaussian_distrib_sample_loss(model, x, z, sample_nums)
            return mae



''' Conformal Prediction based Loss '''
class CopulaConformalLoss(object):
    def __init__(self, model, dataset, t_his, t_pred, type="mean", metric="L2", epsilon=0.1, \
                 decay_type='linear', decay_cycle_size=1000, device="cpu", **kwargs):
        self.model, self.dataset = model, dataset
        self.t_his, self.t_pred = t_his, t_pred
        self.type, self.metric, self.epsilon = type, metric, epsilon
        self.decay_type, self.decay_cycle_size = decay_type, decay_cycle_size
        self.decay_step_count = 0
        self.device = device
        self.normal = Normal(loc=0.0, scale=1.0)
        self.get_radius()        
        return
    
    def get_radius(self):
        if self.type == "mean":
            copula_cp = CopulaCPTS(self.model, self.dataset, self.t_his, self.t_pred, batchsize=256, device=self.device, metric=self.metric)
        elif self.type == "stds":
            copula_cp = StdCopulaCPTS(self.model, self.dataset, self.t_his, self.t_pred, batchsize=256, device=self.device, metric=self.metric)
            
        radius = copula_cp.get_threshold(epsilon=self.epsilon, cali_step=self.t_his)
        radius = np.reshape(radius, newshape=(radius.shape[0], 1, -1, len(self.dataset.kept_joints)))
        radius = torch.tensor(radius, device=self.device, dtype=torch.float32)
        self.radius = radius
        self.decay_step_count = 0
        return    
    
    def update(self, model, **kwargs):
        self.model = model
        self.get_radius()
        return
    
    def _update_decay_counts(self):
        self.decay_step_count += 1
        if self.decay_step_count > self.decay_cycle_size:
            self.decay_step_count = 0
        return
    
    def _get_decay_coef(self):
        if self.decay_type == "linear":
            return 1.0 - (float(self.decay_step_count) / self.decay_cycle_size) 
        elif self.decay_type == "exponential":
            return math.exp(-1.0*self.decay_step_count)
        else:
            return 1.0    

        
    def MeanQuantileRegression(self, mu, sigma, qs:list, Y, bilateral=True):
        modified_quantile_1 = mu + self.radius
        error_1 = Y - modified_quantile_1
        q = 1 - self.epsilon
        loss_1 = torch.mean(torch.max(q * error_1, (q-1) * error_1))
        if bilateral:
            modified_quantile_2 = mu - self.radius
            error_2 = Y - modified_quantile_2
            loss_2 = torch.mean(torch.max(q * error_2, (q-1) * error_2))
            loss_1 = 0.5 * (loss_1 + loss_2)
        
        decay_coef = self._get_decay_coef()
        loss_1 = decay_coef * loss_1
            
        self._update_decay_counts()
        return loss_1

        
    def VarianceQuantileRegression(self, mu, sigma, qs:list, Y):            
        loss_list = []
        for q in qs:
            d = self.normal.icdf(q)
            expand_radius = self.radius.expand(-1, Y.shape[1], -1, -1)
            quantile = sigma * d + mu
            error = torch.exp(torch.abs(expand_radius - 1.0)) * (Y - quantile)
            loss = torch.mean(torch.max(q * error, (q-1) * error))
            loss_list.append(loss)
    
        loss_list = torch.tensor(loss_list)
        L = torch.mean(loss_list)

        decay_coef = self._get_decay_coef()
        L = decay_coef * L
        
        self._update_decay_counts()
        return L


    def UnilateralVarianceQuantileRegression(self, mu, sigma, qs:list, Y):
        loss_list = []
        for q in qs:
            d_u = self.normal.icdf(q)
            d_l = self.normal.icdf(1.0-q)
            
            expand_radius = self.radius.expand(-1, Y.shape[1], -1, -1)
            
            imd_modif_quantile = torch.where(mu>0, sigma*d_u+mu, expand_radius)
            imd_modif_quantile = torch.where(mu<0, sigma*d_l+mu, imd_modif_quantile)
            modified_quantile = imd_modif_quantile
            error = torch.exp(torch.abs(expand_radius - 1.0)) * (Y - modified_quantile)
            
            loss = torch.mean(torch.max(q * error, (q-1) * error))
            loss_list.append(loss)
    
        loss_list = torch.tensor(loss_list)
        L = torch.mean(loss_list)

        decay_coef = self._get_decay_coef()
        L = decay_coef * L
        
        self._update_decay_counts()
        return L



def velocity_direction_loss(X, Y):
    X = X.reshape(-1, X.shape[-2] * X.shape[-1])
    Y = Y.reshape(-1, Y.shape[-2] * Y.shape[-1])
    score = F.cosine_similarity(X, Y, dim=-1)
    score = 1.0 - torch.mean(score)
    return score


   
class PredictLoss(BaseLoss):
    loss_func_dict = {
        "position_space":{
            'euler_position_mse': euler_position_mse_loss,
            'euler_position_mae': euler_position_mae_loss,
            'euler_bone_length_mse': None,
            'euler_velocity_mse_loss': None
        },
        "rotation_space":{
            'expmap_angle_mse': expmap_angle_mse_loss,
            
            'expmap_mse': expmap_mse_loss,
            'expmap_weight_mse': expmap_weight_mse_loss,
            'expmap_emphasize_hip_mse': expmap_emphasize_hip_mse_loss,
            
            'expmap_mae': expmap_mae_loss,
            'expmap_weight_mae': expmap_weight_mae_loss,
            'expmap_emphasize_hip_mae': expmap_emphasize_hip_mae_loss,
            
            'expmap_reweight_mse': expmap_reweight_mse_loss,
            'expmap_reweight_mae': expmap_reweight_mae_loss,
            
            'geodesic_none': ExpmapGeodesicLoss.compute_none,
            'geodesic_mean': ExpmapGeodesicLoss.compute_mean,
            'geodesic_sum' : ExpmapGeodesicLoss.compute_sum,
            
            'geodesic_none_rm': RotMatGeodesicLoss.compute_none,
            'geodesic_mean_rm': RotMatGeodesicLoss.compute_mean,
            'geodesic_sum_rm' : RotMatGeodesicLoss.compute_sum,
        },
        "latent_space":{
            "flow_prior_mse": FlowPriorloss.mse,
            "flow_prior_mae": FlowPriorloss.mae,
            "flow_prior_weight_mse": FlowPriorloss.weight_mse,
            "linear_constraint": FlowPriorloss.linear_constraint_mse,
            "negative_log_likelihood": FlowPriorloss.negative_log_likelihood,
            "inverse_negative_log_likelihood": FlowPriorloss.inverse_negative_log_likelihood,   
        },
        "latent_velocity_space":{
            "mse": euler_position_mse_loss,
            "mae": euler_position_mae_loss,
            'reweight_mse': expmap_reweight_mse_loss,
            'reweight_mae': expmap_reweight_mae_loss,
            'velocity_direction_loss': velocity_direction_loss,
        },
        "likelihood_space":{
            "MLE": maximum_likelihood_estimation         
        },
        "miscellaneous":{
            "gaussian_distrib_sample_nll": SampleLoss.nll_loss,
            "gaussian_distrib_sample_mse": SampleLoss.mse_loss,
            "gaussian_distrib_sample_mae": SampleLoss.mae_loss,
            "minimize": maximum_likelihood_estimation
        },
        "conformal_prediction_based_loss":{
            "mean_quantile_regression": None,
            "variance_quantile_regression": None,
            "unilateral_variance_quantile_regression": None
        },
        "displace_space":{
            "displace_mse": euler_position_mse_loss,
            "displace_mae": euler_position_mae_loss,
            "displace_geodesic_mean": ExpmapGeodesicLoss.compute_mean
        },
    }
    
    '''
    def __init__(self, cmds:dict, skeleton_parents=None, hip_vel_location=-1) -> None:
        self.skeleton_parents = skeleton_parents
        self.hip_vel_location = hip_vel_location
        if self.loss_func_dict["position_space"]["euler_bone_length_mse"] is None:
            self.loss_func_dict["position_space"]["euler_bone_length_mse"] = bone_length_wrapper(skeleton_parents, hip_vel_location)
        if self.loss_func_dict["position_space"]["euler_velocity_mse_loss"] is None:
            self.loss_func_dict["position_space"]["euler_velocity_mse_loss"] = linear_velocity_wrapper(hip_vel_location)
        super(PredictLoss, self).__init__(cmds)
        return
    '''    
    def __init__(self, cmds:dict, **kwargs) -> None:
        if "conformal_prediction_based_loss" in cmds:
            self.copula_conformal_loss = CopulaConformalLoss(**kwargs)
            self.loss_func_dict["conformal_prediction_based_loss"]["mean_quantile_regression"] = self.copula_conformal_loss.MeanQuantileRegression
            self.loss_func_dict["conformal_prediction_based_loss"]["variance_quantile_regression"] = self.copula_conformal_loss.VarianceQuantileRegression
            self.loss_func_dict["conformal_prediction_based_loss"]["unilateral_variance_quantile_regression"] = self.copula_conformal_loss.UnilateralVarianceQuantileRegression
        super(PredictLoss, self).__init__(cmds)
        return
    
    def update(self, **kwargs):
        if hasattr(self, "copula_conformal_loss"):
            self.copula_conformal_loss.update(**kwargs)
        return
 
