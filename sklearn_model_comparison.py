import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os


script_dir = os.path.dirname(__file__)

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
# plt.show()
plt.savefig('sklearn_roc.eps', format='eps')
