# feature relevance
# install ~/miniconda3/envs/thesis/bin/pip install FRUFS
import matplotlib.pyplot as plt

from FRUFS import FRUFS

from lightgbm import LGBMClassifier, LGBMRegressor
# ['FakeInst', 'Mecor', 'Youmi', 'Fusob', 'Kuguo', 'Dowgin', 'BankBot', 'Jisut', 'DroidKungFu', 'RuMMS']


def fs_FRUFS(x_train, k, display, iter):
    # 0.38 best
    fea_model = FRUFS(model_c=LGBMClassifier(random_state=25), k=k, n_jobs=-1)

    # fea_model = FRUFS(model_c=LGBMRegressor(random_state=28),k=0.5,n_jobs=-1) # 84%

    X_train_prued = fea_model.fit_transform(x_train)
    if display:
        # plt.figure(figsize=(8, 110), dpi=100)
        fea_model.feature_importance()
        # plt.savefig(f'feature_selection{iter}.png')

    return X_train_prued
