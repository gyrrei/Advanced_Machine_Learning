from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from imblearn.over_sampling import SMOTE
import tensorflow as tf
from imblearn.combine import SMOTEENN
from imblearn.keras import BalancedBatchGenerator
from datetime import datetime
from tensorflow.keras.preprocessing import sequence
from custom_sampler import TSResampler

import matplotlib.pyplot as plt
import pandas as pd

import data
import numpy as np
import visualisation as vis


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Reshape((180,1)),
            tf.keras.layers.Conv1D(filters=100, kernel_size=3, input_shape=(180, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=intermediate_dim,
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


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)


np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 50
learning_rate = 1e-4
intermediate_dim = 64
original_dim = 180

def train(autoencoder):
    #heartbeats = data.load_train_heartbeats_mean()
    heartbeats = np.concatenate(data.load_train_heartbeats())
    y = data.load_labels()
    # X_train, X_val, y_train, y_val = train_test_split(heartbeats, y)

    X_train, X_val = train_test_split(heartbeats)


    # resampler = RandomUnderSampler(sampling_strategy='not majority')
    # X_train, y_train = resampler.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = X_train.astype('float32')

    training_dataset = tf.data.Dataset.from_tensor_slices((X_train
                                                           # + np.random.normal(0, 0.5, X_train.shape)
                                                           ).astype('float32'))
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(X_train.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    writer = tf.summary.create_file_writer('logs/autoencoder{0}'.format(datetime.now()))

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                for step, batch_features in enumerate(training_dataset):
                    train(loss, autoencoder, opt, batch_features)
                    loss_values = loss(autoencoder, batch_features)
                    # original = tf.reshape(batch_features, (batch_features.shape[0], 180, 1))
                    # reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 180, 1))
                    tf.summary.scalar('loss', loss_values, step=step)
                    # tf.summary.
                    # tf.summary.image('original', original, max_outputs=10, step=step)
                    # tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)


autoencoder = Autoencoder(
    intermediate_dim=intermediate_dim,
    original_dim=original_dim
)

# def evaluate():
    # # X_val_autoencoded = autoencoder(X_val)
    # # for i in range(50):
    # #     vis.plot_avg_heartbeat(X_val[i], X_val_autoencoded[i], ["Original", "Reconstructed"])
    # #     vis.plot_avg_heartbeat(scaler.inverse_transform(X_val[i]),
    # #                            scaler.inverse_transform(X_val_autoencoded[i]),
    # #                            ["Original(Inv)", "Reconstr.(Inv)"],
    # #                            # title="Class {0}".format(y_val[i])
    # #                            )
    #
    # print("Reconstruction Error={0}".format(mean_squared_error(scaler.inverse_transform(X_val), scaler.inverse_transform(X_val_autoencoded))))
# autoencoder.save_weights('autoencoder/Conv1D(100,3)-2*64')

def classify():
    autoencoder.load_weights('autoencoder/Conv1D(100,3)-2*64')

    scaler = StandardScaler()
    heartbeats = data.load_heartbeats_from_file()
    scaler.fit(np.concatenate(heartbeats))

    encoded = [autoencoder.encoder(scaler.transform(x)).numpy().mean(axis=0) for x in heartbeats]
    y = data.load_labels()

    clf = BalancedBaggingClassifier()
    X_train, X_val, y_train, y_val = train_test_split(encoded, y)
    clf.fit(X_train, y_train)

    print(f1_score(y_train, clf.predict(X_train), average='micro'))
    print(f1_score(y_val, clf.predict(X_val), average='micro'))


def classify_lstm():
    autoencoder.load_weights('autoencoder/Conv1D(100,3)-2*64')

    scaler = StandardScaler()
    heartbeats = data.load_heartbeats_from_file()
    minpeaks = [np.min(hb, axis=1) for hb in heartbeats]
    scaler.fit(np.concatenate(heartbeats))

    encoded = [autoencoder.encoder(scaler.transform(x)) for x in heartbeats]
    intervals = data.load_RR_interval_from_file()
    # peaks = data.load_r_peak_amplitude_train()


    # X = [np.concatenate([intervals[i].reshape((-1, 1)), encoded[i]], axis=1) for i in range(len(encoded))]
    X = []
    for i in range(len(intervals)):
        M = min(intervals[i].shape[0], heartbeats[i].shape[0])
        X.append(np.concatenate([
            intervals[i][:M].reshape(-1, 1),
            # peaks[i][:M].reshape(-1, 1),
            encoded[i][:M],
        ], axis=1))
    y = data.load_labels()
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    max_signal_length = 50
    X_train = sequence.pad_sequences(X_train, maxlen=max_signal_length, dtype='float64', value=-1)
    X_val = sequence.pad_sequences(X_val, maxlen=max_signal_length, dtype='float64', value=-1)


    y_train = data.one_hot(y_train)
    y_val = data.one_hot(y_val)

    # X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)

    clf = tf.keras.Sequential([
        tf.keras.layers.GRU(128, activation='relu',
                            input_shape=(max_signal_length,intermediate_dim+1),
                            # kernel_regularizer=tf.keras.regularizers.l1(0.01),
                            ),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    adam = tf.keras.optimizers.Adam(
        # learning_rate=0.01
    )
    clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model_name = 'autoenc-GRU128-128-64-64-(batch32)-maxlen=50'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/' + model_name + '-{0}'.format(datetime.now()),
                                                          histogram_freq=1,
                                                          profile_batch=0)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='gru/checkpoints/' + model_name + '-<{epoch}>.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1)

    # clf.fit_generator(
    #     generator=BalancedBatchGenerator(X_train, y_train, batch_size=32, sampler=TSResampler()),
    #     epochs=50,
    #     validation_data=(X_val, y_val),
    #     callbacks=[tensorboard_callback, checkpoint_callback],
    # )

    clf.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, checkpoint_callback],
        # class_weight={0: 1./8, 1: 3./8, 2: 2./8, 3: 2./8}
    )

    score = f1_score(np.argmax(y_val, axis=1), np.argmax(clf.predict(X_val), axis=1), average='micro')
    clf.save_weights('gru/' + model_name + '-{0}'.format(score))

    print(f1_score(np.argmax(y_train, axis=1), np.argmax(clf.predict(X_train), axis=1), average='micro'))
    print(score)
    print(classification_report(np.argmax(y_val, axis=1), np.argmax(clf.predict(X_val), axis=1)))
    print(confusion_matrix(np.argmax(y_val, axis=1), np.argmax(clf.predict(X_val), axis=1)))



def generate_output_rnn(filename):
    autoencoder.load_weights('autoencoder/Conv1D(100,3)-2*64')

    scaler = StandardScaler()
    heartbeats = data.load_heartbeats_from_file()
    scaler.fit(np.concatenate(heartbeats))

    test_hbs = data.load_test_heartbeats(False)

    encoded = [autoencoder.encoder(scaler.transform(x)) for x in test_hbs]
    intervals = data.load_RR_interval_test_from_file()

    X = []
    for i in range(len(encoded)):
        M = min(intervals[i].shape[0], heartbeats[i].shape[0])
        X.append(np.concatenate([
            intervals[i][:M].reshape(-1, 1),
            encoded[i][:M],
        ], axis=1))

    max_signal_length = 50
    X = sequence.pad_sequences(X, maxlen=max_signal_length, dtype='float64', value=-1)

    clf = tf.keras.Sequential([

        tf.keras.layers.GRU(128, activation='relu',
                            input_shape=(max_signal_length,intermediate_dim+1),
                            ),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    adam = tf.keras.optimizers.Adam(
        # learning_rate=0.01
    )
    clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    clf.load_weights('gru/autoenc-GRU128-128-64-64-(batch=32)-maxlen=50-nopeak-0.775')

    preds = np.argmax(clf.predict(X), axis=1)
    out = open('predictions/' + filename, 'w+')
    out.write("id,y\n")
    for i, y in enumerate(preds):
        out.write("{0},{1}\n".format(i, y))

    out.close()

if __name__ == "__main__":
    classify_lstm()
    # generate_output_rnn('autoenc-GRU128-128-64-64-(batch=32)-maxlen=50-nopeak-0.775.csv')