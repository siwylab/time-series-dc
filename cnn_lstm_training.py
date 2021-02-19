#!/usr/bin/env python
# coding: utf-8
#Git is so fun!
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

# Load data from format script
x = np.load('x_data_ellipse2.npy')
y = np.load('y_data_ellipse2.npy')

lstm_x_train, lstm_x_val, lstm_y_train, lstm_y_val = train_test_split(x, y, test_size=0.3, random_state=123)
lstm_x_val, lstm_x_test, lstm_y_val, lstm_y_test = train_test_split(lstm_x_val, lstm_y_val, test_size=0.5, random_state=321)

in_shape = lstm_x_train.shape[1:]

tf.keras.backend.clear_session()

def create_model(input_shape):
    model = models.Sequential()
    model.add(
        TimeDistributed(
            layers.Conv2D(8, (3, 3), activation='relu'), 
            input_shape=input_shape
        )
    ) # Add in position scalar
    model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(TimeDistributed(layers.Conv2D(16, (3,3), activation='relu')))
    model.add(TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(layers.Conv2D(16, (3,3), activation='relu')))
    model.add(TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2))))

    # extract features and dropout 
    model.add(TimeDistributed(layers.Flatten()))
    model.add(layers.Dropout(0.4))

    # input to LSTM
    model.add(layers.LSTM(input_shape[0], return_sequences=True, dropout=0.5))
    model.add(layers.LSTM(input_shape[0], return_sequences=False, dropout=0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                 metrics=['accuracy']
                 )
    return model
model = create_model(in_shape)
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=1,
                                                 verbose=0)

# Find GPU
device_lib.list_local_devices()

with tf.device('/device:GPU:0'):
    lstm_history = model.fit(lstm_x_train, lstm_y_train, epochs=250,
                        validation_data=(lstm_x_val, lstm_y_val),
                             verbose=1,
              callbacks=[cp_callback]
                            )

# Point to training directory to access weights
os.chdir('training_1')

dif_dict = {}
acc_list = []
for cp in np.arange(1,250):
    val_model = create_model(in_shape)
    model.load_weights('cp-' + str.zfill(str(cp),4) + '.ckpt')
    test_acc = model.evaluate(lstm_x_test, lstm_y_test)[1]
    acc_list.append(test_acc)
    dif_dict[cp] = test_acc-lstm_history.history['val_accuracy'][cp]

# Plot training curve
plt.plot(lstm_history.history['accuracy'], label='accuracy')
plt.plot(lstm_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
# plt.figure(figsize=(12,16))
plt.savefig('lstm_training.png', dpi=300)
# plt.show()