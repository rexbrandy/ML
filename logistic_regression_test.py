import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

#from logistic_regression import LogisticRegression
from regression import LogisticRegression
from tests.linear_regression_tests import X_train

bc = datasets.load_breast_cancer() # import data set

# X is the data we will use to train/predict 
#   it is an np.ndarray of size(n, f) 
#   n = number of samples, f = number of features
#   (569, 30)
# y is the target (in this case if a tumor benign or not, 0 or 1)
#    this is size (n, ) one label per sample
#    (569,)
X, y = bc.data, bc.target 

# train_test_split() will split the data into train and test
#   X_train, y_train is 80% of the data which will be used to train the algorithm
#   X_test, y_test is 20% of the data which we try to predict the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# initalise our class
regressor = LogisticRegression(lr=0.0001, n_iters=1000)

# here we are training our model
regressor.fit(X_train, y_train)

# using what we have learnt now predict the remain samples
y_pred = regressor.predict(X_test)

accuracy = np.sum(y_test == y_pred) / len(y_test)

print("Classification accuracy: {}".format(accuracy))




'''
print("Keys of bc: \n{}".format(bc.keys()))
print(bc['DESCR'] + "\n...")
print("Target names: {}".format(bc['feature_names']))
print("Shape of data: {}".format(bc['data'].shape))
'''