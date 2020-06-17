from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
# from imblearn.over_sampling import SMOTE
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses
import pandas as pd
import matplotlib.pyplot as plt

import data

import numpy as np
import csv





def build_model():
    model = keras.Sequential([
        layers.Dense(40, kernel_regularizer=keras.regularizers.l1(0.001),activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(20, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.5),
        # layers.Dense(10, activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer)
    return model




#Create an object of the classifier.



def testNN():

    X_train, X_val, y_train, y_val = data.undersampled_split()

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)


    y_train = tf.one_hot(y_train, 3)
    y_val = tf.one_hot(y_val, 3)

    X_train = tf.convert_to_tensor(X_train)
    X_val = tf.convert_to_tensor(X_val)

    model = build_model()
    EPOCHS = 50
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
        batch_size=32,
        callbacks=[early_stop],
        class_weight={0: 1, 1:1, 2:1}
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        # plt.figure()
        # plt.xlabel('Epoch')
        # plt.ylabel('R2')
        # plt.plot(hist['epoch'], hist['r2_keras'],
        #          label='Train Score')
        # plt.plot(hist['epoch'], hist['val_r2_keras'],
        #          label='Val Score')
        # plt.ylim([0, 1])
        # plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Categorical Crossentropy')
        plt.plot(hist['epoch'], hist['loss'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_loss'],
                 label='Val Error')
        min = np.min([np.min(hist['loss']), np.min(hist['val_loss'])])
        max = np.max([np.max(hist['loss']), np.max(hist['val_loss'])])
        # plt.ylim([0, 1])
        plt.legend()
        plt.show()

    plot_history(history)

    y_train_pred = model.predict(X_train)
    for i in range(len(y_train_pred)):
        print(y_train[i], y_train_pred[i])

    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    y_train_pred = np.argmax(y_train_pred, axis=1)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)

    BMAC_train = balanced_accuracy_score(y_train, y_train_pred)
    BMAC_val = balanced_accuracy_score(y_val, y_val_pred)

    print(BMAC_train)
    print(BMAC_val)


def make_output(filename):
    X, y, X_test = data.load()
    X = tf.convert_to_tensor(X)
    X_test = tf.convert_to_tensor(X_test)
    y = tf.one_hot(y, 3)
    model = build_model()
    EPOCHS = 50
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X, y,
        epochs=EPOCHS, validation_split=0.3, verbose=0,
        batch_size=32,
        callbacks=[early_stop],
        class_weight={0: 1, 1: 1, 2: 1}
    )

    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    out = open('out/' + filename, 'w+')
    out.write('id,y\n')

    for i in range(len(y_test_pred)):
        out.write("%d.0,%f\n" % (i, y_test_pred[i]))

    out.close()


make_output('vanilla-NN.csv')
