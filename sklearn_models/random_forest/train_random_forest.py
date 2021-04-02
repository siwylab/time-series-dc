import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.abspath("../../")
# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/df_with_features.pkl')
sklearn_dir = os.path.join(ROOT_DIR, 'sklearn_models')

with open(os.path.join(sklearn_dir, 'feature_list.pkl'), 'rb') as file:
    feature_dict = pickle.load(file)
feature_list = list(feature_dict)
# Extract features
x = df[feature_list].to_numpy()
y = df[['y']].to_numpy()

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)

# Fit scalar on training, apply transformation to val/test
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_val = scalar.transform(x_val)

x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

score = {}

depth = np.arange(1, 30, 5)
for d in depth:
    clf = RandomForestClassifier(max_depth=d, criterion='gini', random_state=123)
    clf.fit(x_train, y_train.ravel())
    score[str(d)] = clf.score(x_val, y_val)

# Choose depth with best validation score
d = max(score, key=lambda key: score[key])

# Report accuracy using test set
print('Optimized hyper params:')
print('Depth: ', d)
clf = RandomForestClassifier(max_depth=int(d), criterion='gini', random_state=123)
clf.fit(x_train, y_train.ravel())
print(clf.score(x_test, y_test))

sklearn.metrics.plot_roc_curve(clf, x_test, y_test)
plt.title('RF' + ' D: ' + str(d))
plt.savefig('random_forest_roc.png', dpi=300)

pickle.dump(clf, open('random_forest.pkl', 'wb'))
