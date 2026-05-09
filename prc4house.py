# Practical No. 04

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read dataset
df = pd.read_csv("BostonHousing.csv")

# Remove null values
df = df.dropna()

# Display first 5 rows
print(df.head())

# Display column names
print(df.columns)

# Independent variables
x = df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
        'age', 'dis', 'rad', 'tax',
        'ptratio', 'b', 'lstat']]

print(x.head())

# Dependent variable
y = df['medv']

print(y.head())

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

# Predict values
y_pred = model.predict(x_test)

print(y_pred)

# Training accuracy
print(model.score(x_train, y_train))

# Testing accuracy
print(model.score(x_test, y_test))

# RMSE value
print(np.sqrt(mean_squared_error(y_test, y_pred)))
