import pickle
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/df_with_features.pkl')

feature_list = ['peak_to_peak', 'mean_aspect', 'lfitr0p0', 'lfitr0p1', 'lfitr1p0', 'lfitr1p1', 'nar1_asp', 'nar2_asp',
                'cav1_asp', 'cav2_asp', 'mean_area', 'mean_perimeter']

# Extract features
x = df[feature_list].to_numpy()
y = df[['y']].to_numpy()

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_std = scalar.fit_transform(x)
x_train, x_val, y_train, y_val = train_test_split(x_std, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

# Grid search all hyperparameters
leaf_size = list(range(1, 200, 5))
neighbors = np.arange(1, 25)
score = {}
for leaf in leaf_size:
    for n in neighbors:
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, leaf_size=leaf)
        clf.fit(x_train, y_train)
        score[(str(n), str(leaf))] = clf.score(x_val, y_val)


# Select best weights
n, leaf = max(score)
print('Optimized hyper params:')
print('N: ', n, '\n', 'Leaf size: ', leaf)
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, leaf_size=leaf)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

sklearn.metrics.plot_roc_curve(clf, x_test, y_test)
plt.title('k Nearest Neighbors' + 'K: ' + str(n) + ' Leaf Size: ' + str(leaf))
plt.savefig('knn_roc.png', dpi=300)

pickle.dump(clf, open('knn.pkl', 'wb'))
