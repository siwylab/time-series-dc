import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

rcParams['font.family'] = 'arial'

current_dir = os.getcwd()
# Load LSTM fpr/tpr
lstm_fpr = np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_fpr.csv'))
lstm_tpr = np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_tpr.csv'))
lstm_auc = float(np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_auc.csv')))
lstm_pred = np.load(os.path.join(current_dir, 'lstm', 'lstm_pred.npy'))
seq_y_true = np.load(os.path.join(current_dir, 'seq_y_true.npy'))
print(sklearn.metrics.accuracy_score(seq_y_true, lstm_pred[0].round()))
lstm_accuracy = sklearn.metrics.accuracy_score(seq_y_true, lstm_pred[0].round())

# Load 1D_CNN fpr/tpr
# cnn_fpr = np.loadtxt(os.path.join(current_dir, '1D_CNN', '1dCNN_fpr.csv'))
# cnn_tpr = np.loadtxt(os.path.join(current_dir, '1D_CNN', '1dCNN_tpr.csv'))
# cnn_auc = float(np.loadtxt(os.path.join(current_dir, '1D_CNN', '1dCNN_auc.csv')))

# Load GRU fpr/tpr
gru_fpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_fpr.csv'))
gru_tpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_tpr.csv'))
gru_auc = float(np.loadtxt(os.path.join(current_dir, 'gru', 'gru_auc.csv')))
gru_pred = np.load(os.path.join(current_dir, 'gru', 'gru_pred.npy'))
gru_accuracy = sklearn.metrics.accuracy_score(gru_pred[1], gru_pred[0].round())

# Obtain fpr, tpr
plt.plot(lstm_fpr, lstm_tpr, label='LSTM' + ' (AUC: ' + str(round(lstm_auc, 2)) + ')')
# plt.plot(cnn_fpr, cnn_tpr, label='1D-CNN' + ' (AUC: ' + str(round(cnn_auc, 2)) + ')')
plt.plot(gru_fpr, gru_tpr, label='GRU' + ' (AUC: ' + str(round(gru_auc, 2)) + ')')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('sequential_roc.eps', format='eps')
plt.close()

plt.bar(range(2), [lstm_accuracy, gru_accuracy], tick_label=['LSTM', 'GRU'])
plt.show()
