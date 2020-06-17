import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

import data
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, dropout=0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    return model


model = build_model()
heartbeats = data.load_train_heartbeats()
y = data.load_labels()
X_train, X_val, y_train, y_val = train_test_split(heartbeats, y)

max_signal_length = 50
X_train = sequence.pad_sequences(X_train, maxlen=max_signal_length, dtype='float64')
X_val = sequence.pad_sequences(X_val, maxlen=max_signal_length, dtype='float64')

resampler = SMOTE()
X_train, y_train = resampler.fit_resample(X_train, y_train)

y_train = data.one_hot(y_train)
y_val = data.one_hot(y_val)




# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/encoder-decoder', histogram_freq=1)

model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

print(f1_score(np.argmax(y_val, axis=1), np.argmax(model.predict(X_val), axis=1), average='micro'))