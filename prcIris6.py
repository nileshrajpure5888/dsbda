# Practical No. 06
# Naive Bayes Classification on Iris Dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.naive_bayes import GaussianNB

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("Iris.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

# -------------------------------
# SPLIT FEATURES & TARGET
# -------------------------------

# Independent variables
x = df.drop(['Species'], axis=1)

# Dependent variable
y = df['Species']

# -------------------------------
# FEATURE SCALING
# -------------------------------

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

print("\nScaled Data:\n")
print(x_scaled)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled,
    y,
    test_size=0.2,
    random_state=43
)

# -------------------------------
# CREATE & TRAIN MODEL
# -------------------------------

gnb = GaussianNB()

gnb.fit(x_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------

y_pred = gnb.predict(x_test)

print("\nPredicted Values:\n")
print(y_pred)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# -------------------------------
# EVALUATION METRICS
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(
    y_test,
    y_pred,
    average='micro'
)

recall = recall_score(
    y_test,
    y_pred,
    average='micro'
)

print("\nAccuracy Score:")
print(accuracy)

print("\nError Rate:")
print(error_rate)

print("\nPrecision Score:")
print(precision)

print("\nRecall Score:")
print(recall)
