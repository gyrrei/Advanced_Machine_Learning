import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

from datetime import datetime

import data

heartbeats = data.load_train_heartbeats()
y = data.load_labels()
X_train, X_val, y_train, y_val = train_test_split(heartbeats, y)

max_signal_length = 50
X_train = sequence.pad_sequences(X_train, maxlen=max_signal_length, dtype='float64')
X_val = sequence.pad_sequences(X_val, maxlen=max_signal_length, dtype='float64')


y_train = data.one_hot(y_train)
y_val = data.one_hot(y_val)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(max_signal_length,180)))
model.add(RepeatVector(max_signal_length))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(180)))
model.compile(optimizer='adam', loss='mse')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/lstm-{0}'.format(datetime.now()), histogram_freq=1)

model.fit(X_train, X_train, epochs=50, validation_data=(X_val, X_val), callbacks=[tensorboard_callback])
print("Reconstruction Error={0}".format(mean_squared_error(X_val, model.predict(X_val))))



# print(f1_score(np.argmax(y_val, axis=1), np.argmax(model.predict(X_val), axis=1), average='micro'))