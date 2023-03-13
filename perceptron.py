import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _unit_step_func(self, x):
        np.where(x>=0, 1, 0)