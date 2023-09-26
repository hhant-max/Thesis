
from ExKMC_M.ExKMC.Tree import Tree

import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from utils import plot_confusion_matrix,silhouette_score,f_a_score
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from IPython.display import Image

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN
from collections import Counter

import seaborn as sns

from evaluation import getMojofm,get_similarity,plot_count
from dataPre import fs_FRUFS
from multi_imbalance.resampling.soup import SOUP

# %%
# input data
DF = pd.read_csv('/home/sfy/Documents/VScodeProject/Thesis/data/family.csv',index_col=False)
print(f'Data shape is {DF.shape}')

grouped = DF.groupby(['family'])
# get numbers of each group 
groupCount = grouped['family'].count()
table_count = groupCount.sort_values(ascending=False)[:20]
print(table_count)
# select by numbers of the top 
TOP = 11

selected = groupCount.sort_values(ascending=False)[1:TOP]
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

# %%
def setMajMin():
    maj_int_min = {}
    lst_maj,lst_min = [],[]

    for i in range(10):
        if i < 3:
            lst_maj.append(i)
        else:
            lst_min.append(i)
    maj_int_min['maj'] = lst_maj
    maj_int_min['min'] = lst_min
    return maj_int_min

# from sklearn.feature_selection import VarianceThreshold,GenericUnivariateSelect,chi2
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0)


K = len(selectedNames)
X_train = X
X_train.shape

# success with variance
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X_train)

# imbalanced data
soup = SOUP(maj_int_min = setMajMin())
resampled_X,resampled_y  = soup.fit_resample(scaled_df, Y)
print(f'resampeld X shape is {resampled_X.shape}')

# Converting the numpy array into a pandas DataFrame
# X_train_ = pd.DataFrame(resampled_X)

# feature selection
X_train_prued = fs_FRUFS(resampled_X,k=0.35)

# 
###training --------------------------------------
#models
kmeans = KMeans(K,random_state=0)
kmeans.fit(X_train_prued)


# f_a_score(Y, kmeans_labels, tree_labels=tree_labels)
kmeans.inertia_

get_similarity(y_true = resampled_y,y_predict = kmeans.labels_,le = le,wholeList= selectedNames )


class_names = np.array(list(le.inverse_transform([_ for _ in range(TOP-1)])))
plot_confusion_matrix(resampled_y,kmeans.labels_,class_names,normalize=True)
plt.savefig("confusion_mat")
plt.show()

### MoJofm for result

getMojofm(index_X=X_train_prued.index,predict=kmeans.labels_,y_true=resampled_y)