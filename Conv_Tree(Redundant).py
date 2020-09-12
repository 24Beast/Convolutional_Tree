# Importing Libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.models import Model

# Helper Functions

def entropy(labels):      
    _,_,counts = tf.unique_with_counts(labels)
    probs = counts / K.sum(counts)
    class_ent = -1*probs*K.log(probs)
    ent = K.sum(class_ent)
    z = tf.divide(labels,labels) * tf.cast(ent,dtype = tf.float32)
    print(z.shape)
    return ent

def custom_loss(y_true,y_pred):
    print(y_true[y_pred==False])
    return entropy(tf.cast(y_true[y_pred==True],dtype =tf.int8)) + entropy(tf.cast(y_true[y_pred==False],dtype =tf.int8))


# Class Definitions

class Node():
    
    def __init__(self,input_shape,ksize=3,name="Root"):
        self.name = name
        self.left = None
        self.right = None
        self.model = None
        self.history = None
        self.loss = 0
        self.input_shape = input_shape
        self.initialize_model(input_shape,ksize)
        print("Node Created: "+str(name))
    
    def initialize_model(self,input_shape,ksize):
        img = Input(shape = input_shape)
        conv_img = Conv2D(1,kernel_size=ksize,activation ="relu")(img)
        sum_img = Lambda(lambda x: K.sum(x,axis=1))(conv_img)
        sum_vect = Lambda(lambda x: K.sum(x,axis=1))(sum_img)
        out = Lambda(lambda x: 2*K.sigmoid(x)>1)(sum_vect)
        self.model = Model(img,out)
        self.model.compile(optimizer="rmsprop",loss=custom_loss)
        print(self.model.summary())
        
    def train(self,X,y,num_epochs=25):
        self.history = self.model.fit(X,y,epochs=num_epochs)
        self.loss =  self.history.history["loss"]
        return self.loss[-1]
        
    def predict(self,X):
        return self.model.predict(X)
    

class Tree():
    
    def __init__(self,threshold,in_shape,num_epochs = 100,max_depth=5):
        self.thresh = threshold
        self.ksize = 3
        self.input_shape = in_shape
        self.root = Node(in_shape,self.ksize)
        self.nodemap = None
        self.max_d = max_depth
    
    def train(self,X,y,new_root,count=0):
        loss = new_root.train(X,y)
        pred_y = new_root.predict(X)
        X0 = X[pred_y==0]
        X1 = X[pred_y==1]
        y0 = y[pred_y==0]
        y1 = y[pred_y==1]
        loss0 = entropy(y0)
        loss1 = entropy(y1)
        if((loss0<=self.thresh) or (count>=self.max_d)):
            (unique0,count0) = np.unique(y0,return_counts=True)
            new_root.left = unique0[np.argmax(count0)]
        else:
            new_root.left = Node(self.input_shape,self.ksize,name=new_root.name+"0")
            self.train(X0,y0,new_root.left,count+1)
        if((loss1<=self.thresh) or (count>=self.max_d)):
            (unique1,count1) = np.unique(y1,return_counts=True)
            new_root.right = unique1[np.argmax(count1)]
        else:
            new_root.right = Node(self.input_shape,self.ksize,name=new_root.name+"1")
            self.train(X1,y1,new_root.right,count+1)
    
    def fit(self,X,y):
        self.train(X,y,self.root,0)
        
    def predict(self,X):
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            temp = self.root
            img = np.reshape(X[i],(1,*self.input_shape))
            while(type(temp)!=int):
                pred = temp.predict(img)
                if(pred ==0):
                    temp = temp.left
                else:
                    temp = temp.right
            predictions[i] = temp
        return predictions