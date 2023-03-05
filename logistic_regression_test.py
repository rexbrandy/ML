import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
from tests.linear_regression_tests import X_train

bc = datasets.load_breast_cancer() # import data set

# X is the data we will use to train/predict 
#   it is an np.ndarray of size(n, f) 
#   n = number of samples, f = number of features
# y is the target (in this case if a tumor benign or not, 0 or 1)
#    this is size (n, ) one label per sample
X, y = bc.data, bc.target 

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)








'''
print("Keys of bc: \n{}".format(bc.keys()))
print(bc['DESCR'] + "\n...")
print("Target names: {}".format(bc['feature_names']))
print("Shape of data: {}".format(bc['data'].shape))
'''