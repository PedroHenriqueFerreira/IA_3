import numpy as np

class Optimizer:
    ''' Base para todos os otimizadores '''
    
    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        ''' Atualiza os parâmetros da rede neural '''
        
        raise NotImplementedError()

class SGD(Optimizer):
    ''' Otimizador de descida de gradiente estocástica '''
    
    def __init__(self, lr=0.01, **kwargs):
        self.lr = lr # Taxa de aprendizado

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]):
        for param, grad in zip(params, grads):
            param -= self.lr * grad
            
class Momentum(Optimizer):
    ''' Otimizador de momento '''
    
    def __init__(self, lr=0.01, momentum=0.9, **kwargs):
        self.lr = lr # Taxa de aprendizado
        self.momentum = momentum # Momento
        self.m = None # Momento

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * grad
            
            param -= self.m[i] * self.lr

class Adam(Optimizer):
    ''' Otimizador Adam '''
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, **kwargs):
        self.lr = lr # Taxa de aprendizado
        self.beta1 = beta1 # Fator de decaimento do primeiro momento
        self.beta2 = beta2 # Fator de decaimento do segundo momento
        
        self.m = None # Primeiro momento
        self.v = None # Segundo momento
        
        self.t = 0 # Iteração

    def __call__(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
            
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)