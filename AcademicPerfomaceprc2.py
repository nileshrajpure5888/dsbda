# Practical No. 01
# Data Preprocessing and Outlier Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CHECK NUMPY VERSION
# -------------------------------

print("\nNumPy Version:\n")
print(np.version.version)

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("StudentsPerformance.csv")

# -------------------------------
# DISPLAY FIRST 5 ROWS
# -------------------------------

print("\nFirst 5 Rows:\n")
print(df.head())

# -------------------------------
# CHECK NULL VALUES
# -------------------------------

print("\nNull Values:\n")
print(df.isnull().sum())

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------

# Calculate mean of math score
math_score_mean = df["math score"].mean()

# Replace null values with mean
df["math score"] = df["math score"].fillna(
    math_score_mean
)

# Remove remaining null values
df = df.dropna()

# -------------------------------
# DISPLAY UPDATED DATA
# -------------------------------

print("\nUpdated Dataset:\n")
print(df.head())

# -------------------------------
# BOXPLOT BEFORE REMOVING OUTLIERS
# -------------------------------

print("\nDisplaying Boxplot Before Removing Outliers...")

df.boxplot()

plt.show()

# -------------------------------
# REMOVE OUTLIERS
# -------------------------------

# Keep records where math score > 30
newdf = df[df["math score"] > 30]

# -------------------------------
# DISPLAY FILTERED DATA
# -------------------------------

print("\nFiltered Dataset:\n")
print(newdf.head())

# -------------------------------
# BOXPLOT AFTER REMOVING OUTLIERS
# -------------------------------

print("\nDisplaying Boxplot After Removing Outliers...")

newdf.boxplot()

plt.show()
