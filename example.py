# %matplotlib inline
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ExKMC.Tree import Tree
from utils import calc_cost, plot_kmeans, plot_tree_boundary

# n = 100
# d = 10
# k = 3
# # X, _ = make_blobs(n, d, k, cluster_std=3.0)
# X, y = make_blobs(n_samples=n, centers=k, n_features=d, cluster_std=3.0,random_state=0)


# # Initialize tree with up to 6 leaves, predicting 3 clusters
# tree = Tree(k=k, max_leaves=2 * k)

# # Construct the tree, and return cluster labels
# prediction = tree.fit_predict(X)

# # Tree plot saved to filename
# tree.plot("example")


n = 1000
d = 2
k = 3
X, y = make_blobs(n_samples=n, centers=k, n_features=3, cluster_std=3.0, random_state=0)


kmeans = KMeans(k, random_state=42)
kmeans.fit(X)

plot_kmeans(kmeans, X)
