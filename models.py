import numpy as np

from losses import MSE, BCE
from optimizers import Adam, SGD

from neural_network import NeuralNetwork
from layers import Linear, ReLU, Sigmoid, SoftMax

from utils import timer

class Model:
    ''' Classe base para os modelos '''

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        ''' Treina o modelo '''
        
        raise NotImplementedError()
    
    def __call__(self, X: np.ndarray):
        ''' Realiza a predição '''
        
        raise NotImplementedError()
    
    def score(self, X: np.ndarray, y: np.ndarray):
        ''' Calcula a acurácia do modelo '''
        
        raise NotImplementedError()

class Classifier(Model):
    ''' Classe base para os modelos de classificação '''
    
    def score(self, X: np.ndarray, y: np.ndarray):
        ''' Calcula a acurácia do modelo '''
        
        return np.mean(self(X) == y)
    
class Regression(Model):
    ''' Classe base para os modelos de regressão '''
    
    def score(self, X: np.ndarray, y: np.ndarray):
        ''' Calcula as métricas '''
        
        return np.mean((self(X) - y) ** 2)
    
class NeuralNetworkRegression(Regression):
    ''' Rede neural para regressão '''
    
    def __init__(self):
        self.neural_network: NeuralNetwork | None = None
    
    @timer    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        in_features = X.shape[1]
        out_features = y.shape[1]
        
        self.neural_network = NeuralNetwork(
            MSE(), 
            Adam(**kwargs),
            [
                Linear(in_features, 24),
                ReLU(),
                Linear(24, 12),
                ReLU(),
                Linear(12, 6),
                ReLU(),
                Linear(6, out_features)
            ]
        )
        
        self.neural_network.fit(X, y, **kwargs)
    
    def __call__(self, X: np.ndarray):
        return self.neural_network(X)
    
class NeuralNetworkClassifier(Classifier):
    ''' Rede neural para classificação '''
    
    def __init__(self):
        self.neural_network: NeuralNetwork | None = None
        
    @timer
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        in_features = X.shape[1]
        out_features = len(np.unique(y))
        
        self.neural_network = NeuralNetwork(
            BCE(), 
            Adam(**kwargs),
            [
                Linear(in_features, 12),
                ReLU(),
                Linear(12, 6),
                ReLU(),
                Linear(6, out_features),
                SoftMax()
            ]
        )
        
        y_onehot = np.eye(out_features)[y.flatten()]
        
        self.neural_network.fit(X, y_onehot, **kwargs)
    
    def __call__(self, X: np.ndarray):
        y_pred = self.neural_network(X)
        
        return np.argmax(y_pred, axis=1).reshape(-1, 1)
    
class LinearRegression(Regression):
    ''' Regressão linear '''
    
    def __init__(self):
        self.neural_network: NeuralNetwork | None = None
    
    @timer
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        in_features = X.shape[1]
        out_features = y.shape[1]
        
        self.neural_network = NeuralNetwork(
            MSE(), 
            SGD(**kwargs),
            [
                Linear(in_features, out_features)
            ]
        )
        
        self.neural_network.fit(X, y, **kwargs)
    
    def __call__(self, X: np.ndarray):
        return self.neural_network(X)
    
class LogisticRegression(Classifier):
    ''' Regressão logística '''
    
    def __init__(self):
        self.neural_networks: dict[int, NeuralNetwork] = {}
    
    @timer
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        in_features = X.shape[1]
        
        for classification in np.unique(y):
            self.neural_networks[classification] = NeuralNetwork(
                BCE(), 
                SGD(**kwargs),
                [
                    Linear(in_features, 1),
                    Sigmoid()
                ]
            )
            
            y_binary = (y == classification).astype(int)
            
            self.neural_networks[classification].fit(X, y_binary, **kwargs)
    
    def __call__(self, X: np.ndarray):
        y_pred = np.zeros((X.shape[0], len(self.neural_networks)))
        
        for classification, neural_network in self.neural_networks.items():
            y_pred[:, classification] = neural_network(X).flatten()
            
        return np.argmax(y_pred, axis=1).reshape(-1, 1)