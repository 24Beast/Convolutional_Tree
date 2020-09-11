# Importing Libraries
import numpy as np
import tensorflow
from Conv_Tree import *

# Loading Data
(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data(path="mnist.npz")

# X_train = np.reshape(X_train,(*X_train.shape,1))
# X_test = np.reshape(X_test,(*X_test.shape,1))

# Initializing Classifier
clf = Tree(0.05,X_train[0].shape)

# Training
clf.fit(X_train,y_train)

# Testing
y_pred = clf.predict(X_test)
