

# Practical No. 09

import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display first rows
print(df.head())

# Create boxplot
sns.boxplot(
    x='sex',
    y='age',
    data=df,
    hue='survived'
)

# Title
plt.title("Age Distribution by Gender and Survival")

# Show plot
plt.show()
