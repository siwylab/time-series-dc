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
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y, test_size=0.25, random_state=123)

depth = np.arange(1, 30, 5)
for d in depth:
    clf = RandomForestClassifier(max_depth=5, criterion='gini', random_state=0)
    clf.fit(x_train_t, y_train_t)

# TODO Finish hyperparam looping and add performance plotting