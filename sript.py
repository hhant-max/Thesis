# %%
# %matplotlib inline
# from ExKMC.Tree import Tree # import from cloned local library followed by installing manually
from utils import plot_confusion_matrix, silhouette_score, f_a_score, to_csv, get_distance
from evaluation import getMojofm, get_similarity, plot_count
from sklearn.cluster import KMeans, DBSCAN
from multi_imbalance.resampling.mdo import MDO
from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP
from dataPre import fs_FRUFS
import seaborn as sns
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from IPython.display import Image
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
from ExKMC_M.ExKMC.Tree import Tree
import sys
sys.path.append('../')


# %% [markdown]
# ## Data preprocessing

# %%
# input data
DF = pd.read_csv(
    '/home/sfy/Documents/VScodeProject/Thesis/data/family.csv', index_col=False)
print(f'Data shape is {DF.shape}')

grouped = DF.groupby(['family'])
# get numbers of each group
groupCount = grouped['family'].count()
table_count = groupCount.sort_values(ascending=False)[:20]
print(table_count)
# select by numbers of the top
# TOP = 11
TOP = 10

selected = groupCount.sort_values(ascending=False)[:TOP]
# plot_count(selected)

# get names
selectedNames = list(selected.index)
print(f"selected family are {selectedNames}")

# select trained
train = DF.loc[DF["family"].isin(selectedNames)]
train.reset_index(inplace=True, drop=True)
print(f"train dataset shape {train.shape}")


# encode into encoder
le = LabelEncoder()
label = le.fit_transform(train["family"])
# print(train.head(5))

# exclue name -> X_
train = train.loc[:, train.columns != "name"]
train = train.loc[:, train.columns != "family"]

# remove duplicates -> X
# parameter with first and last ? !!!! can be discussed
# X = train.drop_duplicates(keep='last')
X = train

# with X.index canbe reserved to original labels
Y = np.take(label, X.index)

print(f"Without duplications wi shape {X.shape}")
print(f"With duplications wi shape {train.shape}")

X.reset_index(inplace=True, drop=True)

# X.head(5)
# X.to_csv('testDup.csv',index=False)

# %% [markdown]
# ## Train -kmeans

# %%
# from sklearn.feature_selection import VarianceThreshold,GenericUnivariateSelect,chi2
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0)

K = len(selectedNames)
X_train = X
X_train.shape

# success with variance
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X_train)

# imbalanced data

# smote_enn = SMOTEENN(random_state=0)

# X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
# print(sorted(Counter(y_resampled).items()))
############
clf = NearMiss(version=1)
# clf = SMOTEENN(random_state=42)

# clf = GlobalCS()
print('before sampling')
print(sorted(Counter(Y).items()))

resampled_X, resampled_y = clf.fit_resample(scaled_df, Y)
# resampled_X, resampled_y = nm1.fit_resample(X, Y)

print('after sampling')
print(sorted(Counter(resampled_y).items()))
# checek resampeld data count


# Converting the numpy array into a pandas DataFrame
X_train_ = pd.DataFrame(resampled_X)

# values = np.arange(0.10,0.13,0.005)
# for i in values:
#     print(f'starting {i} ')


# feature selection
# k = float(0.1155555555555555555555)
# 0.035-67.83  0.08- 67.43
X_train_prued = fs_FRUFS(X_train_, k=0.3, display=True, iter=0)
# X_train_prued = fs_FRUFS(X_train_,k=0.12,display=False)


# ---------------------training --------------------------------------
# models
'''
kmeans = KMeans(K,random_state=0)
kmeans.fit(X_train_prued)


# f_a_score(Y, kmeans_labels, tree_labels=tree_labels)
kmeans.inertia_
get_similarity(y_true = resampled_y,y_predict = kmeans.labels_,le = le,wholeList= selectedNames,display=True)



# class_names = np.array(list(le.inverse_transform([_ for _ in range(TOP)])))
class_names = np.array(list(le.inverse_transform([_ for _ in range(TOP)])))
plot_confusion_matrix(resampled_y,kmeans.labels_,class_names,normalize=True)
# plt.savefig(f"confusion_mat_fs{k}.png")
# plt.show()

### MoJofm for result

getMojofm(index_X=X_train_prued.index,predict=kmeans.labels_,y_true=resampled_y)

'''

# output dataset to table
# to_csv(X_train_prued, resampled_y, train)


# %%
# to_csv(X_train_prued, resampled_y)
