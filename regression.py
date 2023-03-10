import numpy as np
from sklearn import linear_model

class BaseRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr # Learning Rate
        self.n_iters = n_iters # no. of iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        '''
            Train alogrithm
                get sample size and feature size
                init weights and bias
                for n_iters:
                    predicted y
                    calculate derivatives
                    update weights and bias
        '''
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, weights, bias):
        raise NotImplementedError()

    def _approximation(self, X, weights, bias):
        raise NotImplementedError()


class LinearRegression(BaseRegression):
    def _approximation(self, X, weights, bias):
        return np.dot(X, weights) + bias

    def _predict(self, X, weights, bias):
        return np.dot(X, weights) + bias


class LogisticRegression(BaseRegression):
    def _approximation(self, X, weights, bias):
        linear_model = np.dot(X, weights) + bias
        return self._sigmoid(linear_model)

    def _predict(self, X, weights, bias):
        linear_model = np.dot(X, weights) + bias
        y_predicted = self._sigmoid(linear_model)
        predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return predicted_cls

    def _sigmoid(self, x):
         return 1 / (1 + np.exp(-x))