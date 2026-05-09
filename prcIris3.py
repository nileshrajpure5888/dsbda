# Practical No. 03

import pandas as pd

# Read dataset
df = pd.read_csv("Iris.csv")

# Display first 5 rows
print(df.head())

# Statistical summary of dataset
print(df.describe())

# Statistical summary of SepalLengthCm column
print(df["SepalLengthCm"].describe())

# Group data according to Species
print(df.groupby("Species").describe())

# Sum of grouped statistical values
print(df.groupby("Species").describe().sum())
