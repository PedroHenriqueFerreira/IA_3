from typing import Any
import numpy as np

from scalers import StandardScaler

from time import time

def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_size: float = 0.8, 
    shuffle: bool = False
):
    ''' Separa os dados em treino e teste '''
    
    split = int(len(X) * train_size)
    
    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test

def preprocess_database(database: Any): 
    ''' Pré-processa a base de dados '''
    
    X, y = database.data, database.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return scaler, X_train, X_test, y_train, y_test

def timer(func: callable):
    ''' Calcula o tempo de execução de uma função '''
    
    def sub(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        
        return time() - start
    
    return sub