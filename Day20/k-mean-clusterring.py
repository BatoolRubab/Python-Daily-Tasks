# Importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings

# Ignore warnings to keep the output clean
warnings.filterwarnings('ignore')

# Fixing random seed so the results are same every time
np.random.seed(0)

# Create fake data (5000 points) with 4 cluster centers
X, y = make_blobs(
    n_samples=5000,                          # Total number of data points
    centers=[[4, 4], [-2, -1], [2, -3], [1, 1]],  # Coordinates of cluster centers
    cluster_std=0.9                          # Spread of points around each center
)

# Plotting the generated data points (before clustering)
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, edgecolors='k', s=80)
plt.title("Original Data (Before Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Creating KMeans object with:
# - 4 clusters
# - 'k-means++' for smarter initialization
# - 12 different starting points to choose the best one
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Training (fitting) KMeans on the data
k_means.fit(X)

# Getting labels (which cluster each point belongs to)
k_means_labels = k_means.labels_
print("Cluster Labels for Each Point:\n", k_means_labels)

# Getting the center of each cluster
k_means_cluster_centers = k_means.cluster_centers_
print("Coordinates of Cluster Centers:\n", k_means_cluster_centers)

# Set the size of the plot
fig = plt.figure(figsize=(6, 4))

# Generate different colors for each cluster using colormap
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Add a subplot to the figure
ax = fig.add_subplot(1, 1, 1)

# Loop through each cluster
for k, col in zip(range(4), colors):
    # Check which data points belong to the current cluster
    my_members = (k_means_labels == k)

    # Get the center of the current cluster
    cluster_center = k_means_cluster_centers[k]

    # Plot the points in the current cluster with color
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.', ms=10)

    # Plot the cluster center
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col, markeredgecolor='k', markersize=6)

# Set the title of the plot
ax.set_title('KMeans Clustering Result')

# Hide the x and y axis ticks for clean look
ax.set_xticks(())
ax.set_yticks(())

# Show the final plot with clusters and centers
plt.show()
