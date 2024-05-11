#Regularization implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

teams = pd.read_csv("teams.csv")


train,test = train_test_split(teams,test_size = 0.2, random_state = 1)

predictors = ["athletes","events"]
target = "medals"

alpha = 2

#Implementing ridge regression

ridge = Ridge(alpha=alpha)
X = train[predictors]
y = train[target]
ridge.fit(X[predictors],y)

X_test = test[predictors]
sklearn_prediction = ridge.predict(X_test[predictors])
#print(ridge.coef_)
#print(ridge.intercept_)
#print(theta_best)

#enclosing all the stuff inside a function
def ridge_fit(train,predictors,target,alpha):
 X = train[predictors]
 Y = train[target]
 
 #feature scaling
 x_mean = X.mean()
 x_std = X.std()
 
 X = (X - x_mean)/x_std
 
 X["intercept"] = 1
 X = X[["intercept"]+predictors]
 
 penalty = alpha*np.identity(X.shape[1])
 penalty[0][0] = 0
 
 theta_best = np.linalg.inv(X.T@X+penalty)@X.T@Y
 theta_best.index = ["intercept","athletes","events"]
 
 return theta_best,x_mean,x_std

def predict(test,Y,predictos,x_mean,x_std,theta_best):
 X_test = test[predictors]
 X_test = (X_test - x_mean)/x_std
 X_test["intercept"] = 1
 X_test = X_test[["intercept"]+predictors]

 return X_test@theta_best
 

theta,mean,std =ridge_fit(train,predictors,target,alpha)

prediction = predict(test,y,predictors,mean,std,theta)
print(prediction)