import torch

class TorchPCA:
    def __init__(self, n_components=2, method='svd'):
        """
        PyTorch PCA实现
        参数:
        n_components: 要保留的主成分数量
        method: 计算方法 ('svd' 或 'eigen')
        """
        self.n_components = n_components
        self.method = method
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        return
        
        
    def fit(self, X):
        """拟合PCA模型"""
        # 计算均值
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        if self.method == 'svd':
            # SVD方法
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components, :]
            self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        else:
            # 特征分解方法
            covariance_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
            
            # 按特征值降序排列
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]
        return self


    def transform(self, X):
        """转换数据"""
        X_centered = X - self.mean_
        return torch.mm(X_centered, self.components_.T)

    
    def fit_transform(self, X):
        """拟合并转换数据"""
        self.fit(X)
        return self.transform(X)

    
    def inverse_transform(self, X_transformed):
        """将降维后的数据转换回原始空间"""
        return torch.mm(X_transformed, self.components_) + self.mean_