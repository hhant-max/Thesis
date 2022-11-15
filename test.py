from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs
import gdown
import pandas as pd
import copy

# input data
def getDataDrive(url, output, isImport=False):
    """
    return pandas dataframe
    """
    if isImport:
        gdown.download(url=url, output=output, quiet=False)
    res = pd.read_csv(output)
    return res


neg = getDataDrive(
    url="https://drive.google.com/uc?id=1ocidTn7jUvCrLG_XJ6H9MiNUDexCkjFG",
    output="negtive.csv",
)
pos = getDataDrive(
    url="https://drive.google.com/uc?id=1IyMPjACBkz96giGJ-Z4IMk-qzM-1CJ9G",
    output="positive.csv",
)

X_neg = copy.deepcopy(neg)
X_pos = copy.deepcopy(pos)
#########################################################
# test KMeans

# process name column
# X = X_neg.loc[:, X_neg.columns!='name']

k = 3

# Initialize tree with up to 6 leaves, predicting 3 clusters
tree = Tree(k=k, max_leaves=2 * k)

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X_neg)

# Tree plot saved to filename
tree.plot("filename")
