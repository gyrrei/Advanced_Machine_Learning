import numpy as np
import pandas as pd

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
import tensorflow.keras as keras
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
import model_selection as ms
import data

X_train, X_val, y_train, y_val = data.simple_split()
y_train[y_train == 2] = 0
y_val[y_val == 2] = 0


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
    plt.title("MinMaj NN")
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
    plt.show()

def testSVM():
    svm = SVC(class_weight={0: 3, 1: 1})
    svm.fit(X_train, y_train)

    ms._evaluate(svm, X_train, X_val, y_train, y_val)
    #
    # model = BalancedBaggingClassifier(SVC())
    # ms._evaluate(model, X_train, X_val, y_train, y_val)

    resampler =  RandomUnderSampler()
    X_t, y_t = resampler.fit_resample(X_train, y_train)

    for c in [0.5, 1.0, 1.5, 2, 3]:
        ms._evaluate(SVC(C=c, gamma='scale'), X_t, X_val, y_t, y_val)



def build_nn():
    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(256,  activation='relu'),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
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

    optimizer = tf.keras.optimizers.Adam(0.00002)

    model.compile(loss='binary_crossentropy',
                  metrics=[],
                optimizer=optimizer)
    return model


def testNN():
    global X_train, X_val, y_train, y_val

    resampler = RandomUnderSampler()
    X_train, y_train = resampler.fit_resample(X_train, y_train)

    normalizer = MinMaxScaler()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)

    # y_train = data.one_hot(y_train)

    model = build_nn()
    EPOCHS = 100
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, validation_data=(X_val, y_val), verbose=0,
        batch_size=16,
        callbacks=[],
        # class_weight={0: 1, 1:1, 2:1}
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

    preds = model.predict(X_val)
    print(balanced_accuracy_score(y_val, np.round(preds)))


if __name__ == "__main__":
    # testNN()
    testSVM()
