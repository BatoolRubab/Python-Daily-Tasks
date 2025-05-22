import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ------------------------
# LINE PLOTS
# ------------------------

# Basic Line Plot
x = np.arange(1, 11)
y = 2 * x
plt.plot(x, y, color='red', linestyle='--', linewidth=2)
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# Multiple Lines in Same Plot
y2 = 2 * y
plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='2x')
plt.plot(x, y2, color='blue', linestyle=':', linewidth=2, label='4x')
plt.title("Multiple Lines in One Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# Subplots: 1 row, 2 columns
plt.subplot(1, 2, 1)
plt.plot(x, y, color='red', linestyle='--', linewidth=2)
plt.title("Subplot 1")

plt.subplot(1, 2, 2)
plt.plot(x, y2, color='blue', linestyle=':', linewidth=2)
plt.title("Subplot 2")
plt.tight_layout()
plt.show()

# Subplots: 2 rows, 1 column
plt.subplot(2, 1, 1)
plt.plot(x, y, color='red', linestyle='--', linewidth=2)
plt.title("Subplot 1")

plt.subplot(2, 1, 2)
plt.plot(x, y2, color='blue', linestyle=':', linewidth=2)
plt.title("Subplot 2")
plt.tight_layout()
plt.show()

# ------------------------
# BAR PLOTS
# ------------------------

students = {"Ali": 54, "Sajid": 91, "Abbas": 50}
names = list(students.keys())
values = list(students.values())

# Vertical Bar Plot
plt.bar(names, values, color='skyblue')
plt.title("Student Marks - Bar Plot")
plt.xlabel("Names")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

# Horizontal Bar Plot
plt.barh(names, values, color="orange")
plt.title("Student Marks - Horizontal Bar Plot")
plt.xlabel("Marks")
plt.ylabel("Names")
plt.grid(True)
plt.show()

# ------------------------
# SCATTER PLOTS
# ------------------------

list1 = [1, 2, 3, 4, 5, 7, 9, 11, 12]
list2 = [11, 2, 1, 7, 21, 20, 13, 1, 15]
list3 = [5, 4, 9, 1, 10, 2, 11, 17, 19]

# Multiple Scatter Plots on One Graph
plt.scatter(list1, list2, marker='*', color='black', s=200, label='List2')
plt.scatter(list1, list3, marker='o', color='blue', s=200, label='List3')
plt.title("Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()

# Subplot of Scatter Plots
plt.subplot(1, 2, 1)
plt.scatter(list1, list2, marker='o', s=50, c="g")
plt.title("List1 vs List2")

plt.subplot(2, 2, 2)
plt.scatter(list2, list3, marker='o', s=50, c="y")
plt.title("List2 vs List3")
plt.tight_layout()
plt.show()

# ------------------------
# HISTOGRAMS
# ------------------------

data = [1, 2, 3, 4, 4, 3, 3, 2, 6, 7, 8, 9]
plt.hist(data, color='b', bins=5)
plt.title("Histogram")
plt.show()

# Histogram from Dataset
# iris = pd.read_csv('iris.csv')
# plt.hist(iris['Species'], bins=20, color='g')
# plt.title("Histogram - Iris Species")
# plt.show()

# ------------------------
# BOX PLOTS
# ------------------------

data = [list1, list2, list3]
plt.boxplot(data)
plt.title("Box Plot")
plt.show()

# ------------------------
# VIOLIN PLOTS
# ------------------------

plt.violinplot(data, showmeans=True)
plt.title("Violin Plot")
plt.show()

# ------------------------
# PIE CHARTS
# ------------------------

fruits = ["Apple", "Orange", "Banana", "Strawberry"]
quantity = [10, 15, 24, 30]

# Basic Pie Chart
plt.pie(quantity, labels=fruits, autopct='%1.1f%%', colors=['green', 'orange', 'yellow', 'red'])
plt.title("Fruit Distribution - Pie Chart")
plt.show()

# Doughnut Chart
plt.pie(quantity, labels=fruits, autopct='%1.1f%%', colors=['green', 'orange', 'yellow', 'red'], radius=2)
plt.pie([1], colors=["white"], radius=1)  # Inner circle
plt.title("Fruit Distribution - Doughnut Chart")
plt.show()










