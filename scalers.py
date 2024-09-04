import numpy as np

class Scaler:
    ''' Base para todos os escaladores '''
    
    def fit(self, X: np.ndarray):
        ''' Ajusta o escalador '''
        
        raise NotImplementedError()

    def transform(self, X: np.ndarray) -> np.ndarray:
        ''' Transforma os dados '''
        
        raise NotImplementedError()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ''' Ajusta e transforma os dados '''
        
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray):
        ''' Inverte a transformação dos dados '''
        
        raise NotImplementedError()

class StandardScaler(Scaler):
    ''' Escalador normal padrão '''
    
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray):
        return X * self.std + self.mean
    
class MinMaxScaler(Scaler):
    ''' Escalador Min-Max '''
    
    def __init__(self):
        self.min = 0
        self.max = 1
        
    def fit(self, X: np.ndarray):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        
    def transform(self, X: np.ndarray):
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform(self, X: np.ndarray):
        return X * (self.max - self.min) + self.min