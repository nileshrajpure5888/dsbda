# Practical No. 10

import seaborn as sb
import matplotlib.pyplot as plt

# Load Iris dataset
ds = sb.load_dataset('iris')

# Display first 5 rows
print(ds.head())

# --------------------------------
# Histograms
# --------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

fig.suptitle("Histogram of Iris Features")

sb.histplot(ds['sepal_length'], ax=axes[0, 0])

sb.histplot(ds['sepal_width'], ax=axes[0, 1])

sb.histplot(ds['petal_length'], ax=axes[1, 0])

sb.histplot(ds['petal_width'], ax=axes[1, 1])

plt.show()

# --------------------------------
# Boxplots
# --------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

fig.suptitle("Boxplot of Iris Features")

sb.boxplot(
    x='species',
    y='petal_length',
    data=ds,
    ax=axes[0, 0]
)

sb.boxplot(
    x='species',
    y='petal_width',
    data=ds,
    ax=axes[0, 1]
)

sb.boxplot(
    x='species',
    y='sepal_length',
    data=ds,
    ax=axes[1, 0]
)

sb.boxplot(
    x='species',
    y='sepal_width',
    data=ds,
    ax=axes[1, 1]
)

plt.show()


