import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    ''' Separa os dados em treino e teste '''
    
    split = int(len(X) * test_size)
    
    X_train, X_test = X[split:], X[:split]
    y_train, y_test = y[split:], y[:split]
    
    return X_train, X_test, y_train, y_test