import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.scaler = None
        self.fit_data = None
        self.cov = None
        self.eigenvectors = None
        self.eigenvalues = None
        
    def fit(self, data):
        self.scaler = StandardScaler()
        self.fit_data = self.scaler.fit_transform(data)
        self.cov = np.cov(self.fit_data, rowvar=False, bias=True)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov)
        
    def transfrom(self, data):
        self.transform_data = self.scaler.transform(data)
        self.transform_data = self.transform_data.dot(self.eigenvectors)
        return self.transform_data[:, :self.n_components]
            
    def fit_transform(self, data):
        self.fit(data)
        return self.transfrom(data)