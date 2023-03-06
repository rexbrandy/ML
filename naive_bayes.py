import numpy as np
from sympy import denom

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, variance, priors
        # for each classes we need means,variance and priors for each features
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for cls_label in self._classes:
            X_cls = X[cls_label==y]
            self._mean[cls_label, :] = X_cls.mean(axis=0)
            self._var[cls_label, :] = X_cls.var(axis=0)
            self._priors[cls_label] = X_cls.shape[0] / float(n_samples) #X_cls.shape[0] = samples with this label / total number of samples


    def predict(self, X):
        y_predict = [self._predict(sample) for sample in X]
        return y_predict

    def _predict(self, x):
        posteriors = []

        for idx, cls_label in enumerate(self._classes):
            prior = self._priors[idx]
            cls_conditional = np.sum(np.log(self._probability_density(idx, x)))
            posterior = prior + cls_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _probability_density(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
