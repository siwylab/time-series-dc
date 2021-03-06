from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import pandas as pd
import sklearn
import sys
# Import df_utils
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
import df_utils

# Load dataset
df = pd.read_pickle(os.path.join(ROOT_DIR, 'FINAL_DF_light'))

x, y = df_utils.extract_sequential_features(df)

# Split test and train data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

print(x_train.shape)
print(y_test.shape)
print(y_train.shape)
print(y_val.shape)
# sys.exit()

# Use function for making specific models, allows model architectures to be recreated with random parameters for
# testing purposes


def create_model():
    _model = tf.keras.models.Sequential()
    _model.add(layers.Conv1D(filters=16, kernel_size=10, activation='relu', input_shape=x_train.shape[1:]))
    _model.add(layers.Conv1D(filters=16, kernel_size=10, activation='relu'))
    _model.add(layers.Conv1D(filters=16, kernel_size=10, activation='relu'))
    _model.add(layers.Flatten())
    _model.add(layers.Dense(16, activation='relu'))
    _model.add(layers.Dense(16, activation='relu'))
    _model.add(layers.Dense(1, activation='sigmoid'))
    _model.compile(optimizer='rmsprop',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                   metrics=['accuracy']
                   )
    return _model


model = create_model()
print(model.summary())
base_path = os.getcwd()
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=1,
                                                 verbose=0)

history = model.fit(x_train, y_train, epochs=350,
                    validation_data=(x_val, y_val),
                    callbacks=[cp_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('1dcnn_training.eps', format='eps')
plt.figure()
os.chdir(os.path.join(base_path, 'training_1'))

# Create base model and load best validation weights
cp = np.argmax(history.history['val_accuracy'])
val_model = create_model()
model.load_weights('cp-' + str.zfill(str(cp), 4) + '.ckpt')
test_acc = model.evaluate(x_test, y_test)[1]
print(test_acc)

os.chdir(base_path)
pred = model.predict(x_test).ravel()
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, pred)
auc = np.expand_dims(sklearn.metrics.auc(fpr, tpr), -1)

# Save tpr and fpr for lstm model comparison script
np.savetxt('1dcnn_fpr.csv', fpr)
np.savetxt('1dcnn_tpr.csv', tpr)
np.savetxt('1dcnn_auc.csv', auc)

# Save npy of y_pred/y_true
np.save('1dcnn_pred.npy', np.vstack((pred, y_test)))

plt.plot(fpr, tpr, label='1D CNN' + ' (AUC: ' + str(round(float(auc), 2)) + ')')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('1dcnn_roc.png', dpi=300)