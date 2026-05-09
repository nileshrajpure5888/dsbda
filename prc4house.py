# Practical No. 04

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("BostonHousing.csv")
df = df.dropna()
df.head()
df.columns
x = df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
x.head()
y = df['medv']
y.head()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
model.score(x_train,y_train)
model.score(x_test,y_test)
np.sqrt(mean_squared_error(y_test,y_pred))
