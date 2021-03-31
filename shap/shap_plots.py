from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import numpy as np
import os
import sklearn
import pickle
import pandas as pd

# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/df_with_features.pkl')

feature_list = ['peak_to_peak', 'mean_aspect', 'lfitr0p0', 'lfitr0p1', 'lfitr1p0', 'lfitr1p1', 'nar1_asp', 'nar2_asp',
                'cav1_asp', 'cav2_asp', 'mean_area', 'mean_perimeter']

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

# Load trained random forest model
os.chdir('..')
current_dir = os.getcwd()
rel_dir = 'random_forest'
model_dir = os.path.join(current_dir, rel_dir, rel_dir+'.pkl')
with open(model_dir, 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# Create object to calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(x_test)
feature_list_cleaned = ['peak to peak',
                        'mean aspect',
                        'linear fit r0 slope',
                        'linear fit r0 y-int',
                        'linear fit r1 slope',
                        'linear fit r1 y-int',
                        'narrow 1 mean aspect',
                        'narrow 2 mean aspect',
                        'cavity 1 mean aspect',
                        'cavity 2 mean aspect',
                        'mean area',
                        'mean perimeter']

# Make plot
os.chdir('shap/')
# row = 45
# shap.force_plot(np.around(explainer.expected_value[1], decimals=2), np.around(shap_values[1][row, :], decimals=2), np.around(x_test[row, :], decimals=2),
#                 feature_names=feature_list, matplotlib=True, show=False)
# plt.tight_layout()
# plt.savefig('force_plot_row' + str(row) + '.png', format='png')
# plt.figure()

# shap.summary_plot(shap_values[1], x_test, feature_names=feature_list_cleaned, show=False)
# plt.yticks(fontsize=14)
# plt.tight_layout()

# plt.figure()
#

shap.summary_plot(shap_values[1], x_test, feature_names=feature_list_cleaned, plot_type='bar', show=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('feature_importance.eps', format='eps')

