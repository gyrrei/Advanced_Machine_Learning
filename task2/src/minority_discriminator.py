import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import data
import numpy as np
import model_selection as ms
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
from preprocessing import pca
from sklearn.feature_selection import SelectPercentile
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt


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
    plt.title("Minority NN")
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Error')
    min = np.min([np.min(hist['loss']), np.min(hist['val_loss'])])
    max = np.max([np.max(hist['loss']), np.max(hist['val_loss'])])
    plt.ylim([0, 2])
    plt.legend()

    plt.figure()
    plt.title("Minority NN")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Val Error')
    plt.plot(hist['epoch'], np.repeat(0.76, len(hist['epoch'])))

    plt.ylim([0.5, 1])
    plt.legend()
    plt.show()

def cov_tensor(X):
    X_ = np.zeros((X.shape[0], X.shape[1], X.shape[1]))

    mu = np.mean(X, axis=0)

    for i in range(X.shape[0]):
        X_[i] = np.outer(X[i] - mu, X[i] - mu)

    return X_


def build_nn():
    model = keras.Sequential([
        layers.Dense(256, activation='relu',
                     # kernel_regularizer=keras.regularizers.l1(0.0001)
                     ),
        # layers.Dropout(0.1),
        layers.Dense(256,  activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.1),
        layers.Dense(150, activation='relu'),
        # layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        # layers.Dense(128, activation='relu'),


                                      # layers.Dropout(0.5),
        # layers.Dense(2, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(1, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(10, activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(0.00003)

    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                optimizer=optimizer)
    return model


models = {
    "SVM C=2": SVC(C=2, gamma='scale'),
    "SVM C=1": SVC(C=1, gamma='scale'),
    # "XGB eta=0.1, subs": XGBClassifier(learning_rate=0.1, subsample=0.8)
}

def trySVMS():

    X, y = data.load_minorities()
    # selector = SelectPercentile(percentile=50)
    # selector.fit(X, y)
    tomek = TomekLinks()

    for name, model in models.items():
        # ms.evaluate_minority_model(model, name + " Tomek", random_state=42, resampler=tomek)
        # ms.evaluate_minority_model(model, name, random_state=42, resampler=None)
        ms.evaluate_minority_model_kfold(model, name, random_state=99)

    for cont in [0.01]:
        X, y = data.load_minorities()
        f = IsolationForest(contamination=cont)
        f.fit(X, y)
        preds = f.predict(X)
        print (np.sum(preds == -1) / len(preds))
        X = X[preds == 1]
        y = y[preds == 1]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=99)
        for name, model in models.items():
            ms.evaluate_minority_model_kfold(model, name, k=5, sample_selection=(preds == 1))
            # print (name)
            # ms._evaluate(model, X_train, X_val, y_train, y_val)


def testNNkfold(k):
    datasets = data.minorities_kfold_split(k)
    scores = []
    for X_train, X_val, y_train, y_val in datasets:
        # X_train, X_val, y_train, y_val = data.minorities_split()
        y_train[y_train == 2] = 1
        y_val[y_val == 2] = 1

        # lda = LinearDiscriminantAnalysis()
        # X_train = lda.fit_transform(X_train, y_train)
        # X_val = lda.transform(X_val)
        normalizer = MinMaxScaler()
        X_train = normalizer.fit_transform(X_train)
        X_val = normalizer.transform(X_val)

        # y_train = data.one_hot(y_train)

        model = build_nn()
        EPOCHS = 80
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
            batch_size=100,
            callbacks=[],
            # class_weight={0: 1, 1:1, 2:1}
        )

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()

        # plot_history(history)

        y_train_pred = model.predict(X_train)
        # for i in range(len(y_train_pred)):
        #     print(y_train[i], y_train_pred[i])

        # y_train = np.argmax(y_train, axis=1)
        # y_val = np.argmax(y_val, axis=1)

        # y_train_pred = np.argmax(y_train_pred, axis=1)
        # y_val_pred = np.argmax(model.predict(X_val), axis=1)
        y_train_pred = np.round(y_train_pred)
        y_val_pred = np.round(model.predict(X_val))

        BMAC_train = balanced_accuracy_score(y_train, y_train_pred)
        BMAC_val = balanced_accuracy_score(y_val, y_val_pred)

        print(BMAC_train)
        print(BMAC_val)
        scores.append(BMAC_val)
    scores = np.array(scores)
    print("AVG: {0}".format(np.mean(scores)))



def testNN():
    X_train, X_val, y_train, y_val = data.minorities_split()
    y_train[y_train == 2] = 1
    y_val[y_val == 2] = 1

    # lda = LinearDiscriminantAnalysis()
    # X_train = lda.fit_transform(X_train, y_train)
    # X_val = lda.transform(X_val)
    normalizer = MinMaxScaler()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)

    # y_train = data.one_hot(y_train)

    model = build_nn()
    EPOCHS = 80
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
        batch_size=100,
        callbacks=[],
        # class_weight={0: 1, 1:1, 2:1}
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

    y_train_pred = model.predict(X_train)
    # for i in range(len(y_train_pred)):
    #     print(y_train[i], y_train_pred[i])

    # y_train = np.argmax(y_train, axis=1)
    # y_val = np.argmax(y_val, axis=1)

    # y_train_pred = np.argmax(y_train_pred, axis=1)
    # y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_train_pred = np.round(y_train_pred)
    y_val_pred = np.round(model.predict(X_val))

    BMAC_train = balanced_accuracy_score(y_train, y_train_pred)
    BMAC_val = balanced_accuracy_score(y_val, y_val_pred)

    print(BMAC_train)
    print(BMAC_val)

trySVMS()
# testNNkfold(5)