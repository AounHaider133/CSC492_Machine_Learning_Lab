import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def sigmoid(z):
 return 1 / (1 + np.exp(-z))

#W*X+B
def model(X,Y,learning_rate,iterations):
 m = X_train.shape[1]
 n = X.shape[0]
 cost_list = [] 
 W = np.zeros((n,1))
 B = 0
 
 for i in range(iterations):
  z = np.dot(W.T,X)+B
  A = sigmoid(z)
  
  #cost function
  cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) 
  
  #Gradient descent
  dw = (1/m)*np.dot(A-Y,X.T)
  db = (1/m)*np.sum(A-Y)
  
  W = W - learning_rate*dw.T
  B = B - learning_rate*db
  
  cost_list.append(cost)
  if (i%(iterations/10)==0):
   print("Cost after ",i,"iteration is:",cost )
 return B,W,cost_list

def accuracy(X,Y,W,B):
 z = np.dot(W.T,X)+B
 A = sigmoid(z)
 
 A = A>=0.5
 A = np.array(A,dtype='int64')
 acc = (1-np.sum(np.absolute(A-Y))/Y.shape[1])*100
 
 print('Accuracy of our model is:',acc,'%')


X_train = pd.read_csv("train_X.csv")
Y_train = pd.read_csv("train_Y.csv")

X_test = pd.read_csv("test_X.csv")
Y_test = pd.read_csv("test_Y.csv")

X_train.head()

X_train = X_train.drop("Id",axis=1)
Y_train = Y_train.drop("Id",axis=1)

X_test = X_test.drop("Id",axis=1)
Y_test = Y_test.drop("Id",axis=1)

X_train = X_train.values
Y_train = Y_train.values

X_test = X_test.values
Y_test = Y_test.values

X_train = X_train.T
Y_train = Y_train.reshape(1,X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1,X_test.shape[1])



W,B,cost_list = model(X_train,Y_train,0.0015,100000)

plt.plot(np.arange(100000),cost_list)
plt.show()

accuracy(X_test,Y_test,W,B)