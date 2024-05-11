import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(X, Y, learning_rate, num_iterations):
    a = 0
    b = 0
    m = len(X)
    
    for i in range(num_iterations):
        predicted_y = a * X + b
        cost_function = (1 / m) * sum([(y - y_pred) ** 2 for y, y_pred in zip(Y, predicted_y)])
        
        da = (-2 / m) * sum(predicted_y - Y)
        db = (-2 / m) * sum(X * (predicted_y - Y))
        
        a = a - learning_rate * da
        b = b - learning_rate * db
    
    return a, b

# Example data
#x1 = np.array([82, 56, 88, 70, 80, 49, 65, 35, 66, 67])
#y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])


x1 = np.random.randint(0, 100, size=100)  # Generate 100 random integers between 0 and 100
y = x1 * 2 + np.random.randint(-10, 10, size=100)  # Create the target variable with some noise


a,b = gradient_descent(x1, y, 0.001, 1000)

y = a*x1 + b

plt.scatter(x1, y, color='green', marker='o', s=20)

plt.plot(x1, y, color='blue', label='Regression Line')
plt.xlabel('Math marks')
plt.ylabel('CS marks')
plt.title('Gradient Descent')
plt.legend()
plt.show()
