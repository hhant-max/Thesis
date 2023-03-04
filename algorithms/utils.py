import copy
import functools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state, check_X_y, _safe_indexing
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder

import gdown

import pandas as pd
import warnings

import os

import subprocess
import time

warnings.simplefilter("ignore", UserWarning)


def calc_cost(tree, k, x_data):
    clusters = tree.predict(x_data)
    cost = 0
    for c in range(k):
        cluster_data = x_data[clusters == c]
        if cluster_data.shape[0] > 0:
            center = cluster_data.mean(axis=0)
            for x in cluster_data:
                cost += np.linalg.norm(x - center) ** 2
    return cost


def plot_kmeans(kmeans, x_data):
    cmap = plt.cm.get_cmap("PuBuGn")

    k = kmeans.n_clusters
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    values = np.c_[xx.ravel(), yy.ravel()]

    ########### K-MEANS Clustering ###########
    plt.figure(figsize=(4, 4))
    Z = kmeans.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmap,
        aspect="auto",
        origin="lower",
        alpha=0.4,
    )

    y_kmeans = kmeans.predict(x_data)
    plt.scatter(
        [x[0] for x in x_data],
        [x[1] for x in x_data],
        c=y_kmeans,
        s=20,
        edgecolors="black",
        cmap=cmap,
    )
    for c in range(k):
        center = x_data[y_kmeans == c].mean(axis=0)
        plt.scatter(
            [center[0]],
            [center[1]],
            c="white",
            marker="$%s$" % c,
            s=350,
            linewidths=0.5,
            zorder=10,
            edgecolors="black",
        )

    plt.xticks([])
    plt.yticks([])
    plt.title("Near Optimal Baseline", fontsize=14)
    plt.show()


def plot_tree_boundary(cluster_tree, k, x_data, kmeans, plot_mistakes=False):
    cmap = plt.cm.get_cmap("PuBuGn")

    ########### IMM leaves ###########
    plt.figure(figsize=(4, 4))

    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    values = np.c_[xx.ravel(), yy.ravel()]

    y_cluster_tree = cluster_tree.predict(x_data)

    Z = cluster_tree.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmap,
        aspect="auto",
        origin="lower",
        alpha=0.4,
    )

    plt.scatter(
        [x[0] for x in x_data],
        [x[1] for x in x_data],
        c=y_cluster_tree,
        edgecolors="black",
        s=20,
        cmap=cmap,
    )
    for c in range(k):
        center = x_data[y_cluster_tree == c].mean(axis=0)
        plt.scatter(
            [center[0]],
            [center[1]],
            c="white",
            marker="$%s$" % c,
            s=350,
            linewidths=0.5,
            zorder=10,
            edgecolors="black",
        )

    if plot_mistakes:
        y = kmeans.predict(x_data)
        mistakes = x_data[y_cluster_tree != y]
        plt.scatter(
            [x[0] for x in mistakes],
            [x[1] for x in mistakes],
            marker="x",
            c="red",
            s=60,
            edgecolors="black",
            cmap=cmap,
        )

    plt.xticks([])
    plt.yticks([])
    plt.title(
        "Approximation Ratio: %.2f"
        % (cluster_tree.score(x_data) / -kmeans.score(x_data)),
        fontsize=14,
    )
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=np.array(["Cluster %d" % i for i in range(len(classes))]),
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Cluster label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def plot_df(df, dataset, step=1, ylim=None):
    k = int(df.iloc[0].leaves)
    cols = ["CART", "KDTree", "ExKMC", "ExKMC (base: IMM)"]

    flatui = ["#3498db", "#e74c3c", "#2ecc71", "#34495e"]
    palette = sns.color_palette(flatui)

    plt.figure(figsize=(4, 3))
    ax = sns.lineplot(
        data=df[::step][cols], linewidth=4, palette=palette, markers=True, dashes=False
    )
    plt.yticks(fontsize=14)

    plt.xticks(
        np.arange(0, 1.01, 1 / 3) * (df.shape[0] - 1),
        [
            "$k$\n$(=%s)$" % k,
            r"$2 \cdot k$",
            r"$3 \cdot k$",
            "$4 \cdot k$\n$(=%s)$" % (4 * k),
        ],
        fontsize=14,
    )

    if ylim is not None:
        axes = plt.gca()
        axes.set_ylim(ylim)

    plt.title(dataset, fontsize=22)
    plt.xlabel("# Leaves", fontsize=18)
    ax.xaxis.set_label_coords(0.5, -0.15)
    plt.ylabel("Cost Ratio", fontsize=18)
    plt.legend(fontsize=12)
    plt.show()


def import_all_data():
    """
    get data from online and downoad and label them and merge into one big dataset contains positive and negtive
    """
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
        output="/home/sfy/Documents/VScodeProject/Thesis/data/negtive.csv",
    )
    pos = getDataDrive(
        url="https://drive.google.com/uc?id=1IyMPjACBkz96giGJ-Z4IMk-qzM-1CJ9G",
        output="/home/sfy/Documents/VScodeProject/Thesis/data/positive.csv",
    )

    X_neg = copy.deepcopy(neg)
    X_pos = copy.deepcopy(pos)

    pos_target = [0 for _ in range(X_pos.shape[0])]
    neg_target = [1 for _ in range(X_neg.shape[0])]

    X_pos["y"] = pos_target
    X_neg["y"] = neg_target

    X__ = pd.concat([X_pos, X_neg])
    return X__


def add_family(filePath="/home/sfy/Documents/VScodeProject/Thesis/data/negtive.csv"):
    """
    add families column to negtive dataset(AMD)
    """

    """
    # read the whole file
    DF = pd.read_csv(filepath)
    name = DF["name"]

    # create a new dataframe
    data = pd.DataFrame()

    # FAMILY = 'Airpush'
    familyList = ["Airpush", "BankBot", "Boqx"]


    def addFam(data, family):
        # extract lines contating this family
        targetLines = DF[name.str.contains(f"({family})")]

        # add family name into it
        addFamil = [family for _ in range(len(targetLines))]
        targetLines.insert(1, "family", addFamil)

        res = pd.concat([data, targetLines], ignore_index=True)
        return res


    for family in familyList:
        data = addFam(data, family)
    """
    # read the whole file

    DF = pd.read_csv(filePath)
    name = DF["name"]

    # create a new dataframe
    data = pd.DataFrame()

    familyList = [
        "Airpush",  # 6652
        "AndroRAT",  # 46 less
        "Andup",  # 43
        "Aples",  # 20
        "BankBot",  # 647
        "Bankun",  # 70
        "Boqx",  # 215
        "Boxer",  # 44 less
        "Cova",
        "Dowgin",
        "DroidKungFu",
        "Erop",
        "FakeAngry",  # 9210 less
        "FakeAV",  # 9220 less
        "FakeDoc",  # 9225
        "FakeInst",  # 9246
        "FakePlayer",  # 11414 less
        "FakeTimer",  # 11435 less
        "FakeUpdates",  # 11447 less
        "Finspy",  # 11452 less
        "Fjcon",  # 11461 less
        "Fobus",  # 11477 less
        "Fusob",  # 11481
        "GingerMaster",  # 12743 less
        "GoldDream",  # 12871 less
        "Gorpo",  # 12924 less
        "Gumen",  # 12947 less
        "Jisut",  # 13092 less
        "Kemoge",  # 13650 less
        "Koler",  # 13664 less
        "Ksapp",  # 13733 less
        "Kuguo",  # 13768
        "Kyview",  # 14966
        "Leech",  # 15141 less
        "Lnk",  # 15250 less
        "Lotoor",  # 15255
        "Mecor",  # 15584
        "Minimob",  # 17404
        "Mmarketpay",  # 17607 less
        "MobileTX",  # 17620 less
        "Mseg",  # 235
        "Mtk",  # 17872 less
        "Nandrobox",  # 17939 less(76)
        "Obad",  # 18015 less
        "Ogel",  # 18024 less
        "Opfake",  # 18030 less
        "Penetho",  # 18040 less
        "Ramnit",  # 18058 less
        "Roop",  # 18058 less
        "RuMMS",  # 402
        "SimpleLocker",  # 18516
        "SlemBunk",  # (174)
        "SmsKey",  # (165)
        "SmsZombie",  # 9 less
        "Spambot",  # 15 less
        "SpyBubble",  # 10 less
        "Stealer",  # 25 less
        "Steek",  # 12 less
        "Svpeng",  # 13 less
        "Tesbo",  # 5 less
        "Triada",  # 197
        "Univert",  # 10 less
        "UpdtKiller",  # 24
        "Utchi",  # 12
        "Vidro",  # 23
        "VikingHorde",  # 7
        "Vmvol",  # 13
        "Winge",  # 19
        "Youmi",  # 1300
        "Zitmo",  # 24 less
        "Ztorg",  # 17 less
    ]
    # 6654 6700 6743 6763 7410 7480 7695 7739 7756 8618 9164 9210

    def addFam(data, family):
        # extract lines contating this family
        targetLines = DF[name.str.contains(f"({family})")]

        # add family name into it
        addFamil = [family for _ in range(len(targetLines))]
        targetLines.insert(1, "family", addFamil)

        res = pd.concat([data, targetLines], ignore_index=True)
        return res

    for family in familyList:
        data = addFam(data, family)

    data.to_csv("family.csv", index=False)

    print(data.head(5))


def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : array-like of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(
            labels, weights=D_chunk[i], minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start : start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def silhouette_score(
    X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds
):
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    else:
        scores = silhouette_samples(X, labels, metric=metric, **kwds)
    return np.mean(scores, axis=1)


def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample.

    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 ``<= n_labels <= n_samples - 1``.

    This function returns the Silhouette Coefficient for each sample.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    """
    X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = np.finfo(X.dtype).eps * 100
            if np.any(np.abs(np.diagonal(X)) > atol):
                raise ValueError(error_msg)
        elif np.any(np.diagonal(X) != 0):  # integral dtype
            raise ValueError(error_msg)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    # intra:35655.59,inter:1.559
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    # unknow operatino of denom # after denom the dists become 1.49 before is 30234
    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return [intra_clust_dists, inter_clust_dists, np.nan_to_num(sil_samples)]

# calculate the scores with jaccard

def f_a_score(true_labels, kmenas_labels, tree_labels):
    from sklearn.metrics import f1_score, accuracy_score

    print(f"fsocre of kmenas is {f1_score(y_true=true_labels,y_pred=kmenas_labels,average='macro')}")
    print(f"fsocre of tree score is {f1_score(y_true=true_labels,y_pred=tree_labels,average='macro')}")

    print(
        f"accuracy of kmenas is {accuracy_score(y_true=true_labels,y_pred=kmenas_labels)}"
    )
    print(
        f"accuracy of tree score is {accuracy_score(y_true=true_labels,y_pred=tree_labels)}"
    )
