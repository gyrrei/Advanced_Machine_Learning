import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import data
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Reshape((512, 1)),
            # tf.keras.layers.Conv1D(filters=16, kernel_size=16, input_shape=(512, 1)),
            # tf.keras.layers.Conv1D(filters=32, kernel_size=8),
            # tf.keras.layers.MaxPool1D(2),
            # tf.keras.layers.Conv1D(filters=64, kernel_size=4),
            # tf.keras.layers.Conv1D(filters=128, kernel_size=2),
            # tf.keras.layers.Dropout(0.05),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Conv1D(64, 4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=3 * intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=2*intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=2 * intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            # tf.keras.layers.Dense(
            #     units=intermediate_dim,
            #     activation=tf.nn.relu,
            #     kernel_initializer='he_uniform'
            # )
        ]

        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layers[0](input_features)
        for i in range(1, len(self.hidden_layers)):
            activation = self.hidden_layers[i](activation)
        # activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units=intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=2 * intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=2 * intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            tf.keras.layers.Dense(
                units=3 * intermediate_dim,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            ),
            # tf.keras.layers.Reshape((intermediate_dim, 1)),
            # tf.keras.layers.Conv1D(128, 2),
            # tf.keras.layers.Conv1D(64, 4),
            # tf.keras.layers.UpSampling1D(2),
            # tf.keras.layers.Conv1D(32, 8),
            # tf.keras.layers.Conv1D(16, 16),
            # tf.keras.layers.Flatten()


            # tf.keras.layers.Dense(
            #     units=intermediate_dim,
            #     activation=tf.nn.relu,
            #     kernel_initializer='he_uniform'
            # )
        ]
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            # activation=tf.nn.sigmoid
        )

    def call(self, code):
        activation = self.hidden_layers[0](code)
        for i in range(1, len(self.hidden_layers)):
            activation = self.hidden_layers[i](activation)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


X_train = data.load_train_eeg2([0,1], do_filter=True)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = data.load_train_eeg2([2], do_filter=True)
X_val = scaler.transform(X_val)

autoencoder = Autoencoder(128, 512)
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=128, validation_data=(X_val, X_val))

preds = autoencoder.predict(X_val)
preds = scaler.inverse_transform(preds)
X_val = scaler.inverse_transform(X_val)

for i in range(20):
    plt.figure()
    plt.plot(np.arange(0, 512), X_val[i], label='original')
    plt.plot(np.arange(0, 512), preds[i], label='autoencoded')
    plt.legend()
    plt.show()


print(mean_squared_error(X_val, preds))


