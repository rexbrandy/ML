import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weigths = None
        self.bias = None

    def fit(self, X, y): 
        # Init parameters
        n_samples, n_features = X.shape
        self.weigths = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weigths) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # calculate derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
            linear_model = np.dot(X, self.weigths) + self.bias
            y_predicted = self._sigmoid(linear_model)
            predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
            return predicted_cls

    def _sigmoid(self, x):
         return 1 / (1 + np.exp(-x))