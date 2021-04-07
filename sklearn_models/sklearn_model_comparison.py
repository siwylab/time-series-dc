import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import df_utils

feature_dict = df_utils.read_feats()
feature_list = list(feature_dict)
feature_list_cleaned = [feature_dict[i] for i in list(feature_dict.keys())]

# Load dataset
df = pd.read_pickle(os.path.join(ROOT_DIR, 'FINAL_DF_light'))
df = df_utils.filter_df(df, ymax=5, max_ar=1.1, radius_std=3)
df = df[(df.cell == 'hl60') | (df.cell == 'hl60d')]
df = df[np.logical_not((df.cell == 'hl60') & (df.date == '11-3-20') & (df.run == '0'))]
df = df[np.logical_not((df.cell == 'hl60') & (df.date == '11-5-20') & (df.run == '3'))]
df.dropna(inplace=True)

# Extract features
x = df[feature_list].to_numpy()
y = df.apply(lambda a: int(a['cell'] == 'hl60'), axis=1).to_numpy()

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_std = scalar.fit_transform(x)
x_train, x_val, y_train, y_val = train_test_split(x_std, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

# Iterate over models and plot roc
model_dict = {}
ax = plt.subplot(222)
for rel_dir in ['knn', 'logistic_regression', 'random_forest', 'svm']:
    current_dir = os.getcwd()
    model_dir = os.path.join(current_dir, rel_dir, rel_dir+'.pkl')
    # Load model
    with open(model_dir, 'rb') as pkl_file:
        model = pickle.load(pkl_file)

    # Obtain fpr, tpr
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    auc = sklearn.metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    plt.plot(fpr, tpr, label=rel_dir + ' (AUC: ' + str(round(auc, 2)) + ')')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title('ROC Curve', fontsize=24)
plt.legend(loc='best', prop={'size': 12})
plt.tight_layout()
plt.savefig('sklearn_roc.eps', format='eps')
plt.show()