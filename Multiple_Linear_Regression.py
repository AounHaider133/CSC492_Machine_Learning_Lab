import pandas as pd
from sklearn.linear_model import LinearRegression

# Create the dataset
data = {'House Size': [1000, 1500, 1200, 1800, 1350],
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Green'],
        'Height': [3.5, 4.2, 3.9, 4.5, 4.0],
        'Price': [250000, 350000, 300000, 400000, 320000]}
df = pd.DataFrame(data)

# Convert categorical variable 'Color' to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Color'])

# Separate independent variables (features) and dependent variable (target)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Test the model with new data
new_data = {'House Size': [1100, 1300],
            'Color_Red': [1, 0],
            'Color_Blue': [0, 1],
            'Color_Green': [0, 0],
            'Height': [3.7, 4.1]}
df_new = pd.DataFrame(new_data, columns=X.columns)  # Match the column names of X

# Predict the price for new data
predictions = model.predict(df_new)

print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Predictions:", predictions)
print(df)