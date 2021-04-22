import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from matplotlib import rcParams
import sklearn
from sklearn.model_selection import train_test_split

rcParams['font.family'] = 'arial'
ROOT_DIR = os.path.abspath("../")
current_dir = os.getcwd()
# Load gru fpr/tpr
gru_fpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_fpr.csv'))
gru_tpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_tpr.csv'))
gru_auc = float(np.loadtxt(os.path.join(current_dir, 'gru', 'gru_auc.csv')))
gru_pred = np.load(os.path.join(current_dir, 'gru', 'gru_pred.npy'))
seq_y_true = np.load(os.path.join(current_dir, 'seq_y_true.npy'))
gru_accuracy = sklearn.metrics.accuracy_score(seq_y_true, gru_pred[0].round())

# Load CNN-gru fpr/tpr
cnn_gru_fpr = np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_fpr.csv'))
cnn_gru_tpr = np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_tpr.csv'))
cnn_gru_auc = float(np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_auc.csv')))
cnn_gru_pred_true = np.load(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_pred.npy'))
cnn_gru_accuracy = sklearn.metrics.accuracy_score(cnn_gru_pred_true[1], cnn_gru_pred_true[0].round())

# Load SVM perf
svm_pred = np.load(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_pred.npy'))
svm_y_true = np.load(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_ground_truth.npy'))
svm_accuracy = sklearn.metrics.accuracy_score(svm_y_true, svm_pred.round())
svm_fpr = np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_fpr.csv'))
svm_tpr = np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_tpr.csv'))
svm_auc = float(np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_auc.csv')))

# cnn_gru_folds = [0.9401993155,
#                   0.9399999976,
#                   0.926666677,
#                   0.9433333278,
#                   0.9399999976]

# Obtain fpr, tpr
plt.plot(svm_fpr, svm_tpr, label='SVM' + ' (AUC: ' + str(round(svm_auc, 2)) + ')')
plt.plot(gru_fpr, gru_tpr, label='GRU' + ' (AUC: ' + str(round(gru_auc, 2)) + ')')
plt.plot(cnn_gru_fpr, cnn_gru_tpr, label='CNN-GRU' + ' (AUC: ' + str(round(cnn_gru_auc, 2)) + ')')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=16)
plt.title('Deep Learning ROC Curve', fontsize=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')
plt.savefig('deep_learning_roc.eps', format='eps')
plt.show()
plt.close()

plt.bar(range(3), [svm_accuracy, gru_accuracy, cnn_gru_accuracy], tick_label=['SVM', 'GRU', 'CNN_GRU'])
plt.ylabel('Accuracy (Percent)', fontsize=9)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylim((0.5, 1.0))
plt.title('Deep Learning Model Accuracy', fontsize=10)
plt.tight_layout()
plt.savefig('deep_learning_bar.eps', format='eps')
plt.savefig('deep_learning_bar.png', format='png')
plt.show()
