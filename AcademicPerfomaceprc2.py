

import pandas as pd
import numpy as np

# Check NumPy version
print(np.version.version)

# Read CSV file
df = pd.read_csv("StudentsPerformance.csv")

# Display first 5 rows
print(df.head())

# Check null values
print(df.isnull().sum())

# Drop null values (optional)
df = df.dropna()

# Find mean of math score
math_score_mean = df["math score"].mean()

# Fill missing values with mean
df["math score"] = df["math score"].fillna(math_score_mean)

# Display updated data
print(df.head())

# Import matplotlib properly
import matplotlib.pyplot as plt

# Boxplot
df.boxplot()
plt.show()

# Remove outliers / filter records
newdf = df[df["math score"] > 30]

# Display filtered data
print(newdf.head())

# Boxplot after filtering
newdf.boxplot()
plt.show()
