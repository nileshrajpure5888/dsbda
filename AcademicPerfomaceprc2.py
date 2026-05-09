# Practical No. 02

import pandas as pd
import numpy as np
np.version.version
df = pd.read_csv("StudentsPerformance.csv")
df.head()
df.isnull().sum()
df.dropna()
math_score_mean = df["math score"].mean()
df["math score"] = df["math score"].fillna(math_score_mean)
df.head()
import matplotlib as plt
df.boxplot()
newdf = df[df["math score"] > 30]
newdf.head()
newdf.boxplot()
pip uninstall numpy
Y
pip install numpy==2.0
