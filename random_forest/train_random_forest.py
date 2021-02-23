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
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

score = {}

depth = np.arange(1, 30, 5)
for d in depth:
    clf = RandomForestClassifier(max_depth=d, criterion='gini', random_state=123)
    clf.fit(x_train, y_train)
    score[str(d)] = clf.score(x_val, y_val)

# Choose depth with best validation score
d = max(score)

# Report accuracy using test set
print('Optimized hyper params:')
print('Depth: ', d)
clf = RandomForestClassifier(max_depth=d, criterion='gini', random_state=123)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

sklearn.metrics.plot_roc_curve(clf, x_test, y_test)
plt.title('RF' + 'D: ' + str(d))
plt.savefig('random_forest_roc.png', dpi=300)
