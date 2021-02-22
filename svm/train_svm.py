import sklearn
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/Full_filt_101_cx_el.pkl')

feature_list = ['peak_to_peak', 'mean_aspect', 'lfitr0p0', 'lfitr0p1', 'lfitr1p0', 'lfitr1p1', 'nar1_asp', 'nar2_asp',
                'cav1_asp', 'cav2_asp', 'mean_area', 'mean_perimeter']

# Extract features
x = df[feature_list].to_numpy()
y = df[['y']].to_numpy()

# Cross validation
kf = sklearn.model_selection.KFold(n_splits=5, random_state=123, shuffle=True)

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_std = scalar.fit_transform(x)
x_train, x_val, y_train, y_val = train_test_split(x_std, y, test_size=0.25, random_state=123)

score = {}
# Grid search all hyperparameters
c_list = np.linspace(0.1, 1.5, 10)
degree = np.arange(1, 10)
k_list = ['linear', 'poly', 'rbf', 'sigmoid']
for c in c_list:
    for k in k_list:
        if k == 'poly':
            for d in degree:
                svm_clf = svm.SVC(C=c, kernel=k, degree=d)
                svm_clf.fit(x_train, y_train)
                score[(str(c), k, str(d))] = svm_clf.score(x_val, y_val)
        svm_clf = svm.SVC(kernel=k)
        svm_clf.fit(x_train, y_train)
        score[(str(c), k, 'N/A')] = svm_clf.score(x_val, y_val)

# Select best weights
c, k, d = max(score)
print('Optimized hyper params:')
print('C: ', c, '\n', 'Kernel: ', k, '\n', 'D: ', d, '\n')

# Report accuracy using 5-fold CV
cv_score = []
for train_index, test_index in kf.split(x_std):
    x_train, x_test = x_std[train_index], x_std[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svm_clf = sklearn.svm.SVC(C=c, kernel=k, degree=d)
    svm_clf.fit(x_train, y_train)
    cv_score.append(svm_clf.score(x_test, y_test))
print(np.mean(cv_score)*100, ' +-', np.std(cv_score)*100)

sklearn.metrics.plot_roc_curve(svm_clf, x_test, y_test)
plt.title('SVM' + 'C: '+ c + ' Kernel: ', k, '\n', 'D: ', d, '\n')
plt.savefig('svm_roc.png', dpi=300)