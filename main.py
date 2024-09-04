from sklearn.datasets import fetch_california_housing, load_iris

from models import (
    Model, 
    Regression,
    Classifier,
    LinearRegression, 
    LogisticRegression, 
    NeuralNetworkRegression, 
    NeuralNetworkClassifier
)

from utils import train_test_split

california = fetch_california_housing()
iris = load_iris()

models: dict[str, Model] = {
    'LINEAR REGRESSION': LinearRegression(),
    'NEURAL REGRESSION': NeuralNetworkRegression(),
    'LOGISTIC REGRESSION': LogisticRegression(),
    'NEURAL CLASSIFIER': NeuralNetworkClassifier(),
}

for model in models:
    print(f'TRAINING {model}...')
    
    if isinstance(models[model], Regression):
        X = california.data
        y = california.target.reshape(-1, 1)
    else:
        X = iris.data
        y = iris.target.reshape(-1, 1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    models[model].fit(X_train, y_train, verbose=100)
    
    print(f'TRAINING {model} DONE!')
    
    print(f'SCORE {model}: {models[model].score(X_test, y_test)}')
    
    print()
