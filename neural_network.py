import numpy as np

from layers import Layer
from losses import Loss
from optimizers import Optimizer

class NeuralNetwork:
    ''' Rede neural '''
    
    def __init__(self, loss: Loss, optimizer: Optimizer, layers: list[Layer]):
        self.loss = loss # Função de perda
        self.optimizer = optimizer # Otimizador
        self.layers = layers # Camadas

    def __call__(self, input: np.ndarray):
        ''' Propagação da entrada pela rede neural '''

        for layer in self.layers:
            input = layer(input)
            
        return input

    def backward(self, gradient: np.ndarray):
        ''' Propagação do gradiente pela rede neural '''
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            
        return gradient

    def parameters(self):
        ''' Retorna os parâmetros da rede neural '''
        
        return [param for layer in self.layers for param in layer.parameters()]
    
    def gradients(self):
        ''' Retorna os gradientes da rede neural '''
        
        return [grad for layer in self.layers for grad in layer.gradients()]

    def train(self, X: np.ndarray, y: np.ndarray):
        ''' Treina a rede neural sobre um lote de dados '''
        
        y_pred = self(X)
        
        self.backward(self.loss.backward(y, y_pred))
        
        self.optimizer(self.parameters(), self.gradients())
        
        return self.loss(y, y_pred).mean()
    
    def fit(
        self, 
        X: np.ndarray, # Dados de entrada
        y: np.ndarray, # Dados de saída
        epochs: int = 1000, # Número de épocas
        batch_size: int = 50, # Tamanho do lote
        n_iter_no_change = 10, # Número de iterações sem melhora
        tol: float = 0.0001, # Tolerância
        verbose: bool = False, # Exibir mensagens
        verbose_interval: int = 100, # Intervalo de mensagens
        **kwargs
    ):
        ''' Treina a rede neural sobre os dados '''
      
        no_change = 0 # Número de iterações sem melhora
        min_loss = float('inf') # Menor perda
        
        batchs = len(X) // batch_size
        
        for i in range(epochs):
            loss = 0
            
            for j in range(0, len(X), batch_size):
                X_batch = X[j:j + batch_size]
                y_batch = y[j:j + batch_size]
                
                loss += self.train(X_batch, y_batch)
            
            loss /= batchs
            
            if verbose and i == 0 or (i + 1) % verbose_interval == 0: 
                print(f'Epoch {i + 1}/{epochs} - Loss: {loss}')
                
            if loss < min_loss:
                min_loss = loss
                no_change = 0
            elif loss - min_loss > tol:
                no_change += 1
                
                if verbose:
                    print(f'Epoch {i + 1}/{epochs} - Loss: {loss} - No change: {no_change}')
                
                if no_change >= n_iter_no_change:
                    if verbose:
                        print(f'Epoch {i + 1}/{epochs} - Loss: {loss} - Early stopping')
                    
                    break