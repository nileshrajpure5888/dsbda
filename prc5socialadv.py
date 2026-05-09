# Practical No. 05

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
df = pd.read_csv("Social_Network_Ads.csv")
df['Gender'] = df['Gender'].replace({"Male":0,"Female":1})
df.head()
df.columns
x = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
y = df[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=29)
model = LogisticRegression()
model.fit (x_train,y_train)
y_pred = model.predict(x_test)
y_pred
model.score(x_train,y_train)
model.score(x,y)
cm = confusion_matrix(y_test,y_pred)
cm
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print (tn, fp, fn, tp)
a = accuracy_score(y_test,y_pred)
a
e = 1 - a
e
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
