import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

rcParams['font.family'] = 'arial'

current_dir = os.getcwd()
# Load LSTM fpr/tpr
lstm_fpr = np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_fpr.csv'))
lstm_tpr = np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_tpr.csv'))
lstm_auc = float(np.loadtxt(os.path.join(current_dir, 'lstm', 'lstm_auc.csv')))


# Load CNN-LSTM fpr/tpr
cnn_lstm_fpr = np.loadtxt(os.path.join(current_dir, 'CNN_LSTM', 'lstm_fpr.csv'))
cnn_lstm_tpr = np.loadtxt(os.path.join(current_dir, 'CNN_LSTM', 'lstm_tpr.csv'))
cnn_lstm_auc = float(np.loadtxt(os.path.join(current_dir, 'CNN_LSTM', 'lstm_auc.csv')))


# Obtain fpr, tpr
plt.plot(lstm_fpr, lstm_tpr, label='LSTM' + ' (AUC: ' + str(round(lstm_auc, 2)) + ')')
plt.plot(cnn_lstm_fpr, cnn_lstm_tpr, label='CNN-LSTM' + ' (AUC: ' + str(round(cnn_lstm_auc, 2)) + ')')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('deep_learning_roc.eps', format='eps')
plt.show()
