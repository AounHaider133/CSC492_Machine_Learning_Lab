import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.rand(50,1)

Y1 = 3+ 2*np.random.rand(50,1)

plt.plot(X1,Y1,"bo")
print(X1)
print(Y1)

plt.show()

X1 = [2104,1416,1534,852]
X2 = [5,3,3,2]
X3 = [1,2,2,1]
X4 = [45,40,30,36]

Y = [460,232,315,178]

h = 80 + 0.1 * X1 + 0.01*X2 + 3*X3 - 2*X4