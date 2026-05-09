

import pandas as pd
import numpy as np
df = pd.read_csv('StudentsPerformance.csv')
df.head()
df.isnull().sum()
df.describe()
df.notnull().sum()
df.size
df.ndim
df.shape
df.info()
df['writing score'].astype(int)
df.dropna()
df.isnull().sum()
df['writing score'] = df['writing score'].astype(float)
df.head()
df["gender"] = df['gender'].replace({"female":0,"male":1})
df.sample(5)
