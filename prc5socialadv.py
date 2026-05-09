# Practical No. 05
# Logistic Regression on Social Network Ads Dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("Social_Network_Ads.csv")

# Display first 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Display column names
print("\nColumn Names:\n")
print(df.columns)

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# Convert Gender column into numeric values
# Male = 0, Female = 1

df['Gender'] = df['Gender'].map({
    "Male": 0,
    "Female": 1
})

# Independent Variables (Features)
x = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]

# Dependent Variable (Target)
y = df['Purchased']

# -------------------------------
# SPLIT DATASET
# -------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=29
)

# -------------------------------
# CREATE & TRAIN MODEL
# -------------------------------

model = LogisticRegression()

model.fit(x_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------

y_pred = model.predict(x_test)

print("\nPredicted Values:\n")
print(y_pred)

# -------------------------------
# MODEL ACCURACY
# -------------------------------

train_accuracy = model.score(x_train, y_train)

test_accuracy = model.score(x_test, y_test)

print("\nTraining Accuracy:")
print(train_accuracy)

print("\nTesting Accuracy:")
print(test_accuracy)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# Extract values

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

# -------------------------------
# EVALUATION METRICS
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("\nAccuracy Score:")
print(accuracy)

print("\nError Rate:")
print(error_rate)

print("\nPrecision Score:")
print(precision)

print("\nRecall Score:")
print(recall)
