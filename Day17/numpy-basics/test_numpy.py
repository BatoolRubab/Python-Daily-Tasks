import numpy as np
from numpy.ma.extras import vstack, hstack

# 1D NumPy array
n1 = np.array([1, 2, 3, 4, 5])
print("n1:", n1)

# 2D NumPy array
n2 = np.array([[1, 2, 3, 4, 5], [5, 70, 3, 9, 1]])
print("n2:\n", n2)

# Arrays of zeros
n3 = np.zeros((1, 2))
print("n3 (1x2 zeros):\n", n3)

n4 = np.zeros((5, 5))
print("n4 (5x5 zeros):\n", n4)

# Array filled with a constant value
n5 = np.full((3, 3), 8)
print("n5 (3x3 filled with 8):\n", n5)

# Array with a range of values
n6 = np.arange(10, 20)
print("n6 (range 10-19):", n6)

# Array with step size
n7 = np.arange(10, 51, 5)
print("n7 (range 10-50 with step 5):", n7)

# Random integers in a specific range
n8 = np.random.randint(20, 30, 4)
print("n8 (4 random integers between 20-29):", n8)

# Shape of array
print("Shape of n2:", n2.shape)

# Reshaping the array
n2.shape = (5, 2)
print("n2 reshaped to 5x2:\n", n2)

# Vertical stacking of arrays
n9 = np.array([1, 2, 3, 4])
n10 = np.array([2, 6, 3, 8])
print("Vertical Stack:\n", np.vstack((n9, n10)))

# Horizontal stacking of arrays
print("Horizontal Stack:", hstack((n9, n10)))

# Column-wise stacking
print("Column Stack:\n", np.column_stack((n9, n10)))

# Intersection and difference
print("Intersection of n9 and n10:", np.intersect1d(n9, n10))
print("Elements in n9 but not in n10:", np.setdiff1d(n9, n10))

# Array mathematics
print("Total Sum:", np.sum([n9, n10]))
print("Column-wise Sum:", np.sum([n9, n10], axis=0))
print("Row-wise Sum:", np.sum([n9, n10], axis=1))

# Scalar operations
n9 = n9 + 1
print("n9 after +1:", n9)

n10 = n10 - 1
print("n10 after -1:", n10)

n1 = n1 * 2
print("n1 after *2:", n1)

n1 = n1 / 2
print("n1 after /2:", n1)

# Statistical functions: mean, standard deviation, median
n11 = np.array([1, 2, 33, 44, 55, 23, 42])
print("Mean of n11:", np.mean(n11))
print("Standard Deviation of n11:", np.std(n11))
print("Median of n11:", np.median(n11))

# Matrix operations
n12 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n13 = np.array([[4, 6, 7], [10, 1, 0], [3, 2, 11]])

print("Matrix n12:\n", n12)
print("First row of n12:", n12[0])
print("Third row of n12:", n12[2])
print("Third column of n12:", n12[:, 2])
print("Transpose of n12:\n", np.transpose(n12))

# Matrix multiplication
n14 = n12.dot(n13)
print("Matrix multiplication (n12 · n13):\n", n14)

# Save and load array
np.save("my_array", n13)
print("Loaded array from file:\n", np.load("my_array.npy"))

