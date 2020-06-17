from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from imblearn.over_sampling import SMOTE
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

import data
import model_selection as ms

import numpy as np

def focal_loss(y_true, y_pred):

    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def balanced_recall(y_true, y_pred):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
    recall = true_positives / (possible_positives + K.epsilon())
    balanced_recall = K.mean(recall)
    return balanced_recall


def bmac(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred)
    y_true = tf.math.argmax(y_true)
    C = tf.math.confusion_matrix(y_true, y_pred)
    per_class = tf.linalg.diag(C) / K.sum(C, axis=1)
    return K.mean(per_class)


def build_base_model():
    nn_model = keras.Sequential([
        layers.Dense(25, activation='relu'),
        layers.Dense(25, activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    nn_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizer)


    return nn_model

def build_better_model():
    model = keras.Sequential([
        layers.Dense(40, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(40, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.5),
        # layers.Dense(10, activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam)
    return model


def gyris_model():
    model = keras.Sequential([
        layers.Dense(1000, activation='relu'),
        layers.Dense(512, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1024, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(1024, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1024, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(512, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        # layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        # layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(10, activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # # layers.Dense(25, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        # layers.Dense(200, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    # adam = Adam(lr=0.0005)  # orignial: 0.0001
    model.compile(loss='categorical_crossentropy',
                  # class_weights = {0: 6, 1: 1, 2: 6},
                  optimizer='adam')
    return model


def multi_sigmoid_model():
    model = keras.Sequential([
        layers.Dense(678, activation='relu'),
        layers.Dense(128, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, kernel_regularizer=keras.regularizers.l1(0.001), activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='sigmoid'),

        layers.Dense(3, activation='softmax')
    ])
    # adam = Adam(lr=0.0005)  # orignial: 0.0001
    model.compile(loss='categorical_crossentropy',
                  # class_weights = {0: 6, 1: 1, 2: 6},
                  optimizer='adam')
    return model


def new_model():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', ),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),

        # layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    adam = Adam(lr=0.00005)  # orignial: 0.0001
    model.compile(loss='categorical_crossentropy',
                  metrics=['acc'],
                  # class_weights = {0: 3, 1: 1, 2: 2.5},
                  optimizer=adam)
    return model


class CustomNN(KerasClassifier):
    def __init__(self, build_fn=build_base_model):
        super().__init__(build_fn=build_fn)
        # self.model = KerasClassifier(build_fn=build_base_model)

    def fit(self, X, y, validation_data=None):
        self.scaler = MinMaxScaler()
        # X = self.scaler.fit_transform(X)

        # X_ = tf.convert_to_tensor(X)
        y = tf.one_hot(y, 3)
        y = y.numpy()

        if validation_data is not None:
            (X_val, y_val) = validation_data
            X_val = self.scaler.transform(X_val)
            y_val = tf.one_hot(y_val, 3)
            y_val = y_val.numpy()
        # self.model.fit(X, y, )
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        if validation_data is None:
            h = super().fit(
                X, y,
                epochs=30, validation_split=0.2, verbose=0,
                batch_size=64,
                callbacks=[],
                )
        else:
            h = super().fit(
                X, y,
                epochs=50, validation_data=(X_val, y_val), verbose=0,
                batch_size=128,
                callbacks=[],
            )
        self.plot_history(h)

    def predict(self, X):
        # X = self.scaler.transform(X)
        # X_ = tf.convert_to_tensor(X)
        return super().predict(X)

    def get_params(self, deep):
        return super().get_params()

    def plot_history(self, history):
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
        plt.ylim([0, 2])
        plt.legend()
        # plt.figure()
        # plt.title("All classes NN")
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.plot(hist['epoch'], hist['acc'],
        #          label='Train Error')
        # plt.plot(hist['epoch'], hist['val_acc'],
        #          label='Val Error')
        # plt.plot(hist['epoch'], np.repeat(0.76, len(hist['epoch'])))
        #
        # plt.ylim([0.5, 1])
        # plt.legend()
        plt.show()

def testBaggedNN(build_fn=build_base_model):
    bagged_model = BalancedBaggingClassifier(base_estimator=CustomNN(build_fn),
                                             sampling_strategy='auto',
                                             replacement=True,
                                             n_estimators=10,
                                             random_state=0)

    ms.evaluate_model(bagged_model, "Bagged NNs")


# X_train, X_val, y_train, y_val = data.undersampled_split()
# f = IsolationForest(contamination=0.1)
# f.fit(X_train, y_train)
# X_train, y_train = X_train[f.predict(X_train) == 1], y_train[f.predict(X_train) == 1]
# testBaggedNN(build_better_model)

# pca = PCA(n_components=500)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_val = pca.transform(X_val)

def prep(X_train, X_val, y_train, y_val):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    model = CustomNN(new_model)
    ms.evaluate_model_downsampled(model, "NN new")
#
# model = CustomNN(gyris_model)
#
# model.fit(X_train, y_train, validation_data=(X_val, y_val))
# ms.print_scores(model, X_train, X_val, y_train, y_val)
#
# min_svm = SVC(C=2, gamma='scale')
# min_svm.fit(X_train[y_train != 1], y_train[y_train !=1])
# min_pred = min_svm.predict(X_val)
#
# y_mm = np.copy(y_train)
# y_mm[y_mm == 2] = 0
# maxmin_svm = SVC(C= 2, gamma='scale')
# maxmin_svm.fit(X_train, y_mm)
# maxmin_pred = maxmin_svm.predict(X_val)
#
# def format_maxmin(maxmin, min):
#     ret = ""
#     if maxmin == 1:
#         ret += "Maj"
#     else:
#         ret += "Min"
#     if min == 0:
#         ret += " 0"
#     elif min == 2:
#         ret += " 2"
#     return ret
#
# pred = model.predict_proba(X_val)
# for i in range(len(pred)):
#     if np.argmax(pred[i]) != y_val[i]:
#         confidence = np.max(pred[i]) / np.sum(pred[i])
#         print(pred[i], " (", confidence, ") --> ", y_val[i], " SVMs say:", format_maxmin(maxmin_pred[i], min_pred[i]))
#
# bmac_v = balanced_accuracy_score(y_val, np.argmax(pred, axis=1))
# print (bmac_v)
# if bmac_v > 0.71:
#     data.generate_output_wo_retrain("NN-{0}.csv".format(bmac_v), model, scaler=scaler, argmax=False)
#
# conf = np.max(pred, axis=1) / np.sum(pred, axis=1)
# print ("avg conf. correct", np.mean(conf[np.argmax(pred, axis=1) == y_val]))
# print ("avg conf. incorrect", np.mean(conf[np.argmax(pred, axis=1) != y_val]))
# #
# #
# # alternate_pred = np.copy(np.argmax(pred, axis=1))
# # for i in range(len(alternate_pred)):
# #     if conf[i] < 0.9:
# #         if maxmin_pred[i] == 1:
# #             alternate_pred[i] = 1
# #         else:
# #             alternate_pred[i] = min_pred[i]
# #
# #
# # print(balanced_accuracy_score(y_val, alternate_pred))
# # ms._evaluate(model, X_train, X_val, y_train, y_val)
# # ms.evaluate_model(model, "NN with multiout")
