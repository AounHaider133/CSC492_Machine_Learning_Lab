import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {'area': [2600, 3000, 3200, 3600, 4000],
        'bedrooms': [3, 4, 0, 3, 5],
        'age': [20, 15, 18, 30, 8],
        'price': [550000, 565000, 610000, 595000, 760000]}

table = pd.DataFrame(data)
print(table)

X = table[['area', 'bedrooms', 'age']]  # Independent variables
Y = table['price']  # Dependent variable

lr = LinearRegression()
lr.fit(X, Y)

# Coefficients and intercept
coefficients = lr.coef_
intercept = lr.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'c', 'm']
markers = ['o', 's', 'v', '*', 'x']

for i in range(len(X)):
    ax.scatter(X.iloc[i, 0], X.iloc[i, 1], X.iloc[i, 2], c=colors[i], marker=markers[i], label=f'Point {i+1}')

ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Age')
ax.set_title('Scatter Plot of Data')
ax.legend()

plt.show()
