import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Series Object Demonstration
# ---------------------------

# Creating a simple Pandas Series
s1 = pd.Series([11, 2, 32, 41, 59])
print("Series s1:")
print(s1)

# Series with custom index
s2 = pd.Series([21, 22, 33, 89, 77], index=["a", "b", "c", "d", "e"])
print("\nSeries s2 with custom index:")
print(s2)

# Series from dictionary
s3 = pd.Series({'a': 13, 'b': 20, 'c': 90, 'd': 21})
print("\nSeries s3 from dictionary:")
print(s3)

# Series with additional indexes (missing values will be NaN)
s4 = pd.Series({'a': 13, 'b': 20, 'c': 90, 'd': 21}, index=['a', 'b', 'c', 'd', 'e', 'f'])
print("\nSeries s4 with additional indexes:")
print(s4)

# Extracting individual elements from Series
s5 = pd.Series([13, 20, 90, 21])
print("\nFirst element of s5:", s5[0])

# Slicing a Series
print("Slicing s5 from index 2 onward:")
print(s5[2:])

# Basic math operations on Series
print("\ns5 + 5:\n", s5 + 5)
print("s5 - 2:\n", s5 - 2)
print("s5 * 2:\n", s5 * 2)
print("s5 / 2:\n", s5 / 2)
print("s5 + s1:\n", s5 + s1)

# ---------------------------
# DataFrame Demonstration
# ---------------------------

# Creating a simple DataFrame
df1 = pd.DataFrame({"name": ["batool", "arooj", "unsa"], "marks": [9, 10, 9]})
print("\nSimple DataFrame:")
print(df1)

# ---------------------------
# Working with Iris Dataset
# ---------------------------

# Reading dataset
iris = pd.read_csv("Iris.csv")
print("\nFirst 5 rows of Iris dataset:")
print(iris.head())

# DataFrame slicing using iloc
print("\nUsing iloc to extract first 2 rows and first 3 columns:")
print(iris.iloc[0:2, 0:3])

# Last 5 rows
print("\nLast 5 rows:")
print(iris.tail())

# Shape of the DataFrame
print("\nShape of Iris dataset:", iris.shape)

# Statistical summary
print("\nDescriptive statistics:")
print(iris.describe())

# Drop a column (example: SepalWidthCm)
iris_dropped = iris.drop("SepalWidthCm", axis=1)
print("\nDataFrame after dropping 'SepalWidthCm' column:")
print(iris_dropped.head())

# ---------------------------
# Aggregate Functions
# ---------------------------

print("\nMean of numerical columns:")
print(iris.mean(numeric_only=True))

print("\nMedian of numerical columns:")
print(iris.median(numeric_only=True))

print("\nMinimum values of each column:")
print(iris.min(numeric_only=True))

print("\nMaximum values of each column:")
print(iris.max(numeric_only=True))

# ---------------------------
# Column-wise Data Extraction
# ---------------------------

iris_length = iris[['SepalLengthCm', 'PetalLengthCm']]
print("\nExtracted columns 'SepalLengthCm' and 'PetalLengthCm':")
print(iris_length)

# ---------------------------
# Row-wise Filtering
# ---------------------------

long_sepal = iris['SepalLengthCm'] > 6
print("\nBoolean mask for SepalLengthCm > 6:")
print(long_sepal)

# ---------------------------
# Data Visualization
# ---------------------------

print("\nPlotting histogram of Iris dataset...")
iris.plot(kind="hist", figsize=(10, 6), bins=20, alpha=0.7, edgecolor='black', title="Iris Dataset Histogram")
plt.xlabel("Values")
plt.tight_layout()
plt.show()







