from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs


n = 100
d = 10
k = 3
# X, _ = make_blobs(n, d, k, cluster_std=3.0)
X, y = make_blobs(n_samples=n, centers=k, n_features=d, cluster_std=3.0,random_state=0)


# Initialize tree with up to 6 leaves, predicting 3 clusters
tree = Tree(k=k, max_leaves=2 * k)

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X)

# Tree plot saved to filename
tree.plot("example")