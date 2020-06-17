import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, \
    Conv1D, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

from datetime import datetime

import data

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=3, input_shape=(180, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4))
model.add(Activation('softmax'))

heartbeats = data.load_train_heartbeats_mean()
y = data.load_labels()
y = data.one_hot(y)
X_train, X_val, y_train, y_val = train_test_split(heartbeats, y)

resampler = RandomUnderSampler(sampling_strategy='not majority')
X_train, y_train = resampler.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = X_train.reshape(-1, 180, 1)
X_val = X_val.reshape(-1,180,1)

f1 = tfa.metrics.F1Score(num_classes=4, average='micro')
# sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', f1])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/1d_cnn_avg-{0}'.format(datetime.now()), histogram_freq=1)

nb_epoch = 50
model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_val, y_val), batch_size=32,
          callbacks=[tensorboard_callback])

print(f1_score(np.argmax(y_val, axis=1), np.argmax(model.predict(X_val), axis=1), average='micro'))