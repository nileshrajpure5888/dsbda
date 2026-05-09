import pandas as pd
import numpy as np

df = pd.read_csv('StudentsPerformance.csv')

print(df.head())

print(df.isnull().sum())

print(df.describe())

print(df.notnull().sum())

print(df.size)

print(df.ndim)

print(df.shape)

print(df.info())

df['writing score'] = df['writing score'].astype(int)

df = df.dropna()

print(df.isnull().sum())

df['writing score'] = df['writing score'].astype(float)

# Replace gender values
df["gender"] = df["gender"].map({
    "female": 0,
    "male": 1
})

print(df.head())

print(df.sample(5))
