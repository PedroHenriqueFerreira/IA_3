import numpy as np

class Layer:
    ''' Base para todas as camadas da rede neural '''
    
    def parameters(self) -> list[np.ndarray]:
        ''' Retorna os parâmetros da camada '''
        
        return []
    
    def gradients(self) -> list[np.ndarray]:
        ''' Retorna os gradientes da camada '''
        
        return []
    
    def __call__(self, input: np.ndarray):
        ''' Chamada da camada '''
        
        return self.__call__(input)
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        ''' Propagação do gradiente pela camada '''
        
        raise NotImplementedError()

class Linear(Layer):
    ''' Camada linear da rede neural '''
    
    def __init__(self, in_features: int, out_features: int):
        limit = (1 / in_features) ** 0.5
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.random.uniform(-limit, limit, (1, out_features))
        
        self.weight_grad = np.zeros((in_features, out_features))
        self.bias_grad = np.zeros((1, out_features))
        
    def parameters(self):
        return [self.weight, self.bias]
    
    def gradients(self):
        return [self.weight_grad, self.bias_grad]
        
    def __call__(self, input: np.ndarray):
        self.input = input
        
        return self.input @ self.weight + self.bias
    
    def backward(self, gradient: np.ndarray):
        self.weight_grad = self.input.T @ gradient
        self.bias_grad = gradient.sum(axis=0, keepdims=True)
        
        return gradient @ self.weight.T
        
class Sigmoid(Layer):
    ''' Camada sigmoide da rede neural '''
    
    def __call__(self, input: np.ndarray):
        self.output = 1 / (1 + np.exp(-input))
        
        return self.output
    
    def backward(self, gradient: np.ndarray):
        return self.output * (1 - self.output) * gradient
       
class ReLU(Layer):
    ''' Camada ReLU da rede neural '''
    
    def __call__(self, input: np.ndarray):
        self.input = input
        
        return np.maximum(0, input)
    
    def backward(self, gradient: np.ndarray):
        return (self.input > 0) * gradient
        
class SoftMax(Layer):
    ''' Camada SoftMax da rede neural '''
    
    def __call__(self, input: np.ndarray):
        exp = np.exp(input - input.max(axis=1, keepdims=True))
        self.output = exp / exp.sum(axis=1, keepdims=True)
        
        return self.output
    
    def backward(self, gradient: np.ndarray):
        return (gradient - (gradient * self.output).sum(axis=1, keepdims=True)) * self.output
    