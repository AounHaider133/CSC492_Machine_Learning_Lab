from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

X,Y = make_regression(n_samples = 100, n_features =2, n_informative = 2,n_targets = 1,noise = 50)
df = pd.DataFrame({'feature1':X[:,0],'feature2':X[:,1],'target':Y})
df.head()


X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)

print("MAE",mean_absolute_error(y_test,y_predict))
print("MSE",mean_squared_error(y_test,y_predict))
print("R2 score",r2_score(y_test,y_predict))

x = np.linspace(-5,5,10)
x = np.linspace(-5,5,10)
xGrid, yGrid = np.meshgrid(y,x)

z_final = lr.predict(final)