import numpy as np
import scipy.stats as stats
import math

class PseudoPossionDiskSampling(object):
    def __init__(self, sample_nums=50, threshold=0.5, option=0, **kwargs) -> None:
        # self.threshold = threshold
        self.sample_nums = sample_nums
        self.interval = (1.0 * threshold) / self.sample_nums
        
        if option == 0:
            self.ranges = ((np.arange(self.sample_nums) * 1.0 / self.sample_nums) - 0.5) * threshold
        elif option == 1:
            self.ranges = (np.arange(self.sample_nums) * 1.0 / self.sample_nums) * threshold
        elif option == 2:
            self.ranges = (np.arange(self.sample_nums) * 1.0 / self.sample_nums) * threshold * (-1)
        elif option == 3:
            bias = kwargs['bias']
            self.ranges = ((np.arange(self.sample_nums) * 1.0 / self.sample_nums) - bias) * threshold
        return
    
    def _sampling(self, reshuffled=True):
        samples = np.random.uniform(low=0.0, high=1.0, size=(self.sample_nums,))
        samples = np.multiply(samples, self.interval)
        samples = self.ranges + samples
        if reshuffled:
            np.random.shuffle(samples)
        return samples
        
    ''' # V1
    def multi_sampling(self, n, reshuffled=True):
        # multi_samples = []
        # for _ in range(n):
        #     multi_samples.append(self.sampling(reshuffled))
        # return multi_samples
        multi_samples = np.zeros(shape=(self.sample_nums, n))
        for i in range(n):
            s = self.sampling(reshuffled)
            multi_samples[:, i] = s
        multi_samples = multi_samples.tolist()
        return multi_samples
    '''

    def multi_sampling(self, n, reshuffled=True, ratio=0.0):
        multi_samples = np.zeros(shape=(self.sample_nums, n))
        for i in range(n):
            s = self._sampling(reshuffled)
            multi_samples[:, i] = s
        
        mask = np.random.uniform(low=0.0, high=1.0, size=multi_samples.shape)
        mask = np.where(mask>=ratio, 1, 0)
        
        multi_samples = np.multiply(multi_samples, mask)
        multi_samples = multi_samples.tolist()        
        # zeros = [0.0] * n
        # multi_samples.append(zeros)
        return multi_samples
    
    
    @DeprecationWarning
    def multi_mixture_sampling(self, n, reshuffled=True, ratio=0.0):
        ''' Deprecated Function '''
        multi_samples_1 = np.zeros(shape=(self.sample_nums, n))
        multi_samples_2 = np.zeros(shape=(self.sample_nums, n))
        for i in range(n):
            s1, s2 = self._sampling(True), self._sampling(False)
            multi_samples_1[:, i] = s1
            multi_samples_2[:, i] = s2
        
        multi_samples_2 = np.sort(multi_samples_2, axis=0)
        
        multi_samples = multi_samples_1
        multi_samples[22:28, :] = multi_samples_2[22:28, :]
                
        mask = np.random.uniform(low=0.0, high=1.0, size=multi_samples.shape)
        mask = np.where(mask>=ratio, 1, 0)
        
        multi_samples = np.multiply(multi_samples, mask)
        multi_samples = multi_samples.tolist()
        return multi_samples


    def percentile_sampling(self, n, pn, reshuffled=True, ratio=0.0):
        samples = self.multi_sampling(n, reshuffled, ratio)
        skip_num = math.floor(self.sample_nums*1.0 / (pn-1))        
        percentile_samples = [ samples[skip_num*i] for i in range(pn) ]
        return percentile_samples


class PseudoPossionDiskTruncSampling(PseudoPossionDiskSampling):
    def __init__(self, sample_nums=50, low_threshold=-0.5, high_threshold=0.5, **kwargs) -> None:
        self.low_threshold, self.high_threshold = low_threshold, high_threshold
        self.sample_nums = sample_nums
        self.interval = 1.0 * (high_threshold - low_threshold) / self.sample_nums        
        self.ranges = (np.arange(self.sample_nums) * (high_threshold - low_threshold) / self.sample_nums) - (-low_threshold)       
        return


class PseudoGaussianDiskSampling(object):
    def __init__(self, sample_nums=50, threshold=0.5, **kwargs) -> None:    
        self.sample_nums = sample_nums
        self.threshold = threshold * 0.5
        return

    def _sampling(self, reshuffled=True):
        trunc_norm = stats.truncnorm(-self.threshold, self.threshold, loc=0.0, scale=1.0)
        samples = trunc_norm.rvs(self.sample_nums)
        samples = np.sort(samples)
        if reshuffled:
            np.random.shuffle(samples)
        return samples

    def multi_sampling(self, n, reshuffled=True, ratio=0.0):
        multi_samples = np.zeros(shape=(self.sample_nums, n))
        for i in range(n):
            s = self._sampling(reshuffled)
            multi_samples[:, i] = s
        
        mask = np.random.uniform(low=0.0, high=1.0, size=multi_samples.shape)
        mask = np.where(mask>=ratio, 1, 0)
        
        multi_samples = np.multiply(multi_samples, mask)
        multi_samples = multi_samples.tolist()
        return multi_samples


class PseudoGaussianSampling(PseudoGaussianDiskSampling):
    def __init__(self, sample_nums=50, threshold=1.0, **kwargs) -> None:    
        self.sample_nums = sample_nums
        self.threshold = threshold
        return
    
    def _sampling(self, reshuffled=True):
        samples = np.random.normal(loc=0, scale=self.threshold, size=(self.sample_nums,))
        samples = np.sort(samples)
        if reshuffled:
            np.random.shuffle(samples)
        return samples


class PseudoGaussianBiasSampling(PseudoGaussianDiskSampling):
    def __init__(self, sample_nums=50, threshold=0.5, bias=-3.0, **kwargs) -> None:    
        self.sample_nums = sample_nums
        self.threshold = threshold
        self.bias = bias
        return

    def _sampling(self, reshuffled=True):
        trunc_norm = stats.truncnorm(-self.bias, 3.0, loc=0.0, scale=self.threshold)
        samples = trunc_norm.rvs(self.sample_nums)
        samples = np.sort(samples)
        if reshuffled:
            np.random.shuffle(samples)
        return samples

    

if __name__ == "__main__":
    # ppds = PseudoPossionDiskSampling(threshold=1.0)
    # sample = ppds.multi_sampling(100, False, 0.0)
    # p_sample = ppds.percentile_sampling(100, 5, False, 0.0)
    # print()
    
    # pgds = PseudoGaussianDiskSampling(threshold=1.0)
    # sample = pgds.multi_sampling(100, False, 0.0)
    # print()
    
    ppdts = PseudoPossionDiskTruncSampling(sample_nums=50, low_threshold=-0.3, high_threshold=0.5)
    sample =ppdts.multi_sampling(10, False, 0.0)
    print()
        
        
        