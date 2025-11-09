
class CudaEmptyCacheFlagStatus(object):
    def __init__(self) -> None:
        self.cuda_empty_cache_flag = True
        self.have_alterd = False
    
    def update(self, status):
        if self.have_alterd is False:
            self.have_alterd = True
            self.cuda_empty_cache_flag = status
        else:
            print("ERROR: cuda_empty_cache_flag have been altered.")
    
    @property
    def torch_cuda_empty_cache_flag(self):
        return self.cuda_empty_cache_flag


cuda_empty_cache_flag = CudaEmptyCacheFlagStatus()

    