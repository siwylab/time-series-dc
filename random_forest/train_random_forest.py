import sklearn
from sklearn.ensemble import RandomForestClassifier
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

# Split test and train data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=123)

score = {}

depth = np.arange(1, 30, 5)
for d in depth:
    clf = RandomForestClassifier(max_depth=d, criterion='gini', random_state=123)
    clf.fit(x_train, y_train)
    score[str(d)] = clf.score(x_val, y_val)

# Choose depth with best validation score
d = max(score)

# Report accuracy using 5-fold CV
kf = sklearn.model_selection.KFold(n_splits=5, random_state=123, shuffle=True)
cv_score = []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = RandomForestClassifier(max_depth=d, criterion='gini', random_state=123)
    clf.fit(x_train, y_train)
    cv_score.append(clf.score(x_test, y_test))
print(np.mean(cv_score)*100, ' +-', np.std(cv_score)*100)

sklearn.metrics.plot_roc_curve(clf, x_test, y_test)
plt.title('RF' + 'D: ' + str(d))
plt.savefig('random_forest_roc.png', dpi=300)
