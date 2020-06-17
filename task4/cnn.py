import numpy as np
import tensorflow as tf
import data
from sklearn.metrics import balanced_accuracy_score as bmac
from sklearn.preprocessing import OneHotEncoder

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 8,
                               activation='relu',
                               input_shape=(512, 2)),
        tf.keras.layers.Conv1D(32, 4, activation='relu'),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Conv1D(64, 2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam'
                  )

    return model

model = build_model()
X_train = np.concatenate([data.load_train_eeg1([0,1], True).reshape(-1, 1), data.load_train_eeg2([0,1], True).reshape(-1,1)], axis=1)
y_train = data.load_train_labels([0,1])
y_train -= 1
y_train = data.one_hot(y_train)


X_val = np.concatenate([data.load_train_eeg1([2], True).reshape(-1, 1), data.load_train_eeg2([2], True).reshape(-1,1)], axis=1)
y_val = data.load_train_labels([2])
y_val -= 1
y_val = data.one_hot(y_val)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

preds = np.argmax(model.predict(X_val), axis=1)
preds += 1

y_val = np.argmax(y_val, axis=1)
y_val += 1

print(bmac(y_val, preds))