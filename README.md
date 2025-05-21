# Jupyter-notebook

# iris_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# Exploration
print("First 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

# Basic Analysis
print("\nClass distribution:")
print(df['species'].value_counts())

# Visualization
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Observations
print("\nObservations:")
print("- Setosa is clearly separable from the other species.")
print("- Petal length and width are more informative than sepal features.")
print("- Versicolor and Virginica overlap more compared to Setosa.")
