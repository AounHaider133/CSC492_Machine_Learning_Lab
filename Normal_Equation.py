import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)

# y = x0+ ax1 + bx2

X = 3 * np.random.rand(100,1)
Y = 4 + 2 * X + np.random.rand(100,1)

X_P = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_P.T.dot(X_P)).dot(X_P.T).dot(Y)

#print("X_P=")
#print(X_P)
#print("Normal equation result = ")
#print(theta_best)

plt.plot(X,Y,"bo")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


X1 = np.array([[0],[16]])
X_test = np.c_[np.ones((2,1)),X1]

Y1 = X_test.dot(theta_best)

plt.plot(X1,Y1,'r--',label= "Regression line")
plt.plot(X,Y,"bo")
plt.legend()
plt.show()