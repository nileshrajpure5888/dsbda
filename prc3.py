# Practical No. 03


import pandas as pd
df = pd.read_csv("Iris.csv")
df.head()
df.describe()
df["SepalLengthCm"].describe()
df.groupby("Species").describe()
df.groupby("Species").describe().sum()
