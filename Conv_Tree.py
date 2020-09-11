# Importing Libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Helper Functions

def entropy(labels):      
    _,counts = np.unique(labels,return_counts=True)
    probs = counts / np.sum(counts)
    class_ent = -1*probs*K.log(probs)
    ent = K.sum(class_ent)
    return ent

def custom_loss(y_true,y_pred):
    return entropy(y_true[y_pred==1]) + entropy(y_true[y_pred==0])

def sigmoid(x):
    return 1/(1+ np.exp(-1*x))

# Class Definitions

class Node():
    
    def __init__(self,input_shape,ksize=3,name="Root",alpha=0.1):
        self.loss = 0
        self.lr = alpha
        self.name = name
        self.left = None
        self.right = None
        self.history = []
        self.step_f = 1.0
        self.step_d = 1.0
        self.step_c = 1.0
        self.ksize = ksize
        self.filter = np.random.rand(ksize,ksize)
        self.c = -1 * np.random.rand(1)[0]
        self.d = -1 * np.random.rand(1)[0]
        self.input_shape = input_shape
        print("Node Created: "+str(name))
    
    def predict(self,X):
        out = np.zeros((X.shape[0],1+X.shape[1]-self.ksize,1+X.shape[2]-self.ksize))
        for num in range(X.shape[0]):
            for i in range(0,1+X.shape[1]-self.ksize):
                for j in range(0,1+X.shape[2]-self.ksize):
                    out[num,i,j] = np.sum(X[num,i:i+self.ksize,j:j+self.ksize] * self.filter)
        out += self.c
        out = np.sum(np.sum(out * (out>0),axis=1),axis=1) - self.d
        output = (sigmoid(out)>0.5).astype(np.int8)
        return output
    
    def train(self,X,y,num_epochs=5):
        for n in range(num_epochs):
            print("\rIteration no."+str(n),end="\r")
            y_pred = self.predict(X)
            loss = custom_loss(y, y_pred)
            self.history.append(loss)
            for i in range(self.ksize):
                for j in range(self.ksize):
                    self.filter[i,j] += self.step_f
                    y_pred_new = self.predict(X)
                    loss_new = custom_loss(y, y_pred_new)
                    slope = (loss_new-loss)/self.step_f
                    self.filter[i,j] -= (self.lr*slope + self.step_f)
            self.c +=self.step_c
            y_pred_new = self.predict(X)
            loss_new = custom_loss(y, y_pred_new)
            slope = (loss_new-loss)/self.step_c
            self.c -= (self.lr*slope + self.step_c)
            self.d +=self.step_d
            y_pred_new = self.predict(X)
            loss_new = custom_loss(y, y_pred_new)
            slope = (loss_new-loss)/self.step_d
            self.d -= (self.lr*slope + self.step_d)
        y_pred = self.predict(X)
        loss = custom_loss(y,y_pred)
        self.history.append(loss)
        
                    
class Tree():
    
    def __init__(self,threshold,in_shape,max_depth=5):
        self.thresh = threshold
        self.ksize = 3
        self.input_shape = in_shape
        self.root = Node(input_shape = in_shape,ksize = self.ksize)
        self.nodemap = None
        self.max_d = max_depth
    
    def train(self,X,y,new_root,count=0,num_epochs = 5):
        new_root.train(X,y,num_epochs)
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