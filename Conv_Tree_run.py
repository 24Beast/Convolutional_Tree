# Importing Libraries
import tensorflow
import numpy as np
from Convolutional_Tree import Tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm

# Loading Data
(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data(path="mnist.npz")

X_train,_,y_train,_ = train_test_split(X_train,y_train,test_size=0.9)
X_test,_,y_test,_ = train_test_split(X_test,y_test,test_size=0.9)

X_train = np.reshape(X_train,(*X_train.shape,1)).astype(np.float32)
X_test = np.reshape(X_test,(*X_test.shape,1)).astype(np.float32)

# Initializing Classifier
clf = Tree(0.05,X_train[0].shape,max_depth =4,ksize=5)

# Training
clf.fit(X_train,y_train,num_iter=10)

# Testing
y_pred = clf.predict(X_test)

print(cm(y_test,y_pred))

# Saving Model Image
print("Saving Image")
clf.save_img(clf.root)