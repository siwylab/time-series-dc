import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/Full_filt_101_cx_el.pkl')

feature_list = ['peak_to_peak', 'mean_aspect', 'lfitr0p0', 'lfitr0p1', 'lfitr1p0', 'lfitr1p1', 'nar1_asp', 'nar2_asp',
                'cav1_asp', 'cav2_asp', 'mean_area', 'mean_perimeter']
x = df[feature_list].to_numpy()
y = df[['y']].to_numpy()

# Cross validation
kf = sklearn.model_selection.KFold(n_splits=5, random_state=123, shuffle=True)

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_std = scalar.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.25, random_state=123)

cv_score = []
for train_index, test_index in kf.split(x_std):
    x_train, x_test = x_std[train_index], x_std[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svm_clf = sklearn.svm.SVC()
    svm_clf.fit(x_train, y_train)
    cv_score.append(svm_clf.score(x_test, y_test))
print(np.mean(cv_score)*100, ' +-', np.std(cv_score)*100)

