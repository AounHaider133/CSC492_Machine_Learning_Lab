import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate regression dataset
X, Y = make_regression(n_samples=100, n_features=2, n_informative=2, noise=10, random_state=25)

# Perform multiple linear regression
regression_model = LinearRegression()
regression_model.fit(X, Y)

# Predict using the multiple linear regression model
predicted_ml = regression_model.predict(X)

# Perform prediction using the normal equation
X_new = np.concatenate([np.ones((len(X), 1)), X], axis=1)
theta_best = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(Y)
predicted_ne = X_new.dot(theta_best)

# Plotting the results
plt.subplots(figsize=(8, 5))

# Plot the original data
plt.scatter(X[:, 1], Y, marker='o', label='Original Data')

# Plot the predicted data from multiple linear regression
plt.scatter(X[:, 1], predicted_ml, marker='+',s=80, color='r', label='Predicted (Multiple Linear Regression)')

# Plot the predicted data from normal equation
plt.scatter(X[:, 1], predicted_ne, marker='s', color='g', label='Predicted (Normal Equation)')

plt.xlabel("Features at index 1")
plt.ylabel("Target")
plt.legend()
plt.show()
