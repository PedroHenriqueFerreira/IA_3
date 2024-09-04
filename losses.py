import numpy as np

class Loss:
    ''' Base para todas as funções de perda da rede neural '''
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        ''' Chamada da função de perda '''
        
        raise NotImplementedError()
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        ''' Propagação do gradiente pela função de perda '''
        
        raise NotImplementedError()

class MSE(Loss):
    ''' Função de perda de erro quadrático médio '''
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return 0.5 * ((y_pred - y_true) ** 2)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_pred - y_true
    
class BCE(Loss):
    ''' Função de perda de entropia cruzada binária '''
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)