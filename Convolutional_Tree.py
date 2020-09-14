# Importing Libraries
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten

# Helper functions
def make_labels(y):
    y_new = np.zeros(len(y))
    unique = np.unique(y)
    unique = unique[int(len(unique)/2):]
    y_new[[x in unique for x in y]] = 1
    return y_new.astype(np.float32)

def entropy(labels):      
    _,counts = np.unique(labels,return_counts=True)
    probs = counts / np.sum(counts)
    class_ent = -1*probs*np.log(probs)
    ent = np.sum(class_ent)
    return ent

# Class Definitions

class Node():
    
    def __init__(self,input_shape,ksize=3,name="Root",printer=0):
        self.name = name
        self.left = None
        self.right = None
        self.model = None
        self.history = None
        self.loss = 0
        self.ksize = ksize
        self.input_shape = input_shape
        self.initialize_model(input_shape,ksize,printer)
        print("Node Created: "+str(name))
    
    def initialize_model(self,input_shape,ksize,printer):
        img = Input(shape = input_shape)
        conv_1 = Conv2D(1,kernel_size=ksize,activation ="relu")(img)
        flat = Flatten()(conv_1)
        out = Dense(1,activation="sigmoid")(flat)
        self.model = Model(img,out)
        self.model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
        if(printer):
            print(self.model.summary())
        
    def train(self,X,y,num_epochs=25):
        y_new = make_labels(y)
        self.history = self.model.fit(X,y_new,epochs=num_epochs)
        
    def predict(self,X):
        return self.model.predict(X)
    
    def save_img(self):
        w = self.model.get_weights()
        w = np.array(w)
        np.save(self.name+".npy",w)

class Tree():
    
    def __init__(self,threshold,in_shape,max_depth=5,ksize=3):
        self.thresh = threshold
        self.ksize = ksize
        self.input_shape = in_shape
        self.root = Node(in_shape,self.ksize,printer=1)
        self.max_d = max_depth
    
    def train(self,X,y,new_root,num_epochs = 30,count=0):
        new_root.train(X,y,num_epochs)
        pred_y = new_root.predict(X)
        pred_y = np.reshape(pred_y,(len(pred_y)))
        X0 = X[[x<=0.5 for x in pred_y]]
        y0 = y[[x<=0.5 for x in pred_y]]
        X1 = X[[x>0.5 for x in pred_y]]
        y1 = y[[x>0.5 for x in pred_y]]
        loss0 = entropy(y0)
        loss1 = entropy(y1)
        if((len(y0)==0) or (len(y1)==0)):
            (unique,counts) = np.unique(y,return_counts=True)
            new_root.left = unique[np.argmax(counts)]
            new_root.right = unique[np.argmax(counts)]
        else:
            if((loss0<=self.thresh) or (count>=self.max_d)):
                (unique0,count0) = np.unique(y0,return_counts=True)
                new_root.left = unique0[np.argmax(count0)]
            else:
                new_root.left = Node(self.input_shape,self.ksize,name=new_root.name+"0")
                self.train(X0,y0,new_root.left,num_epochs,count+1)
            if((loss1<=self.thresh) or (count>=self.max_d)):
                (unique1,count1) = np.unique(y1,return_counts=True)
                new_root.right = unique1[np.argmax(count1)]
            else:
                new_root.right = Node(self.input_shape,self.ksize,name=new_root.name+"1")
                self.train(X1,y1,new_root.right,num_epochs,count+1)
    
    def fit(self,X,y,num_iter=50):
        self.train(X,y,self.root, num_epochs= num_iter,count=0)
        
    def predict(self,X):
        predictions = np.zeros(len(X))
        print("Prediction Status:")
        for i in range(len(X)):
            print("\r"+str(i)+"/"+str(len(X)),end="")
            temp = self.root
            img = np.reshape(X[i],(1,*self.input_shape))
            while(type(temp)!=np.uint8):
                pred = temp.predict(img)
                if(pred <=0.5):
                    temp = temp.left
                else:
                    temp = temp.right
            predictions[i] = temp
        print("Predictions Complete.")
        return predictions
    
    def save_img(self,root):
        temp = root
        if(type(root)==np.uint8):
            return
        else:
            print(root.name)
            temp.save_img()
            self.save_img(temp.left)
            self.save_img(temp.right)