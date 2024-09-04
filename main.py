from sklearn.datasets import fetch_california_housing, load_iris

california = fetch_california_housing()
iris = load_iris()

from models import (
    LinearRegression, 
    LogisticRegression, 
    NeuralNetworkRegression, 
    NeuralNetworkClassifier
)

from utils import preprocess_database

linear_regression = LinearRegression()
neural_regression = NeuralNetworkRegression()

logistic_regression = LogisticRegression()
neural_classifier = NeuralNetworkClassifier()

# Preprocessing

c_scaler, c_X_train, c_X_test, c_y_train, c_y_test = preprocess_database(california)
i_scaler, i_X_train, i_X_test, i_y_train, i_y_test = preprocess_database(iris)

# Training Regressions

# linear_regression.fit(c_X_train, c_y_train, lr=0.0001, epochs=200, verbose=True)
# neural_regression.fit(c_X_train, c_y_train, lr=0.0001, epochs=200, verbose=True)

# Training Classifiers

logistic_regression.fit(i_X_train, i_y_train, lr=0.01, epochs=10000, verbose=True)
neural_classifier.fit(i_X_train, i_y_train, lr=0.01, epochs=10000, verbose=True)

# Scores

print()

# print(f'LINEAR REGRESSION SCORE: {linear_regression.score(c_X_test, c_y_test)}')
# print(f'NEURAL REGRESSION SCORE: {neural_regression.score(c_X_test, c_y_test)}')

print()

print(f'LOGISTIC REGRESSION SCORE: {logistic_regression.score(i_X_test, i_y_test)}')
print(f'NEURAL CLASSIFIER SCORE: {neural_classifier.score(i_X_test, i_y_test)}')