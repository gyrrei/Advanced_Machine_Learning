from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf

import data
import numpy as np


def get_rr_feature_matrix(test=False):
    if test:
        intervals = data.load_RR_interval_test_from_file()
    else:
        intervals = data.load_RR_interval_from_file()

    features = np.array([[i.max(), i.min(), i.mean(), i.var()] for i in intervals])
    return features



if __name__ == "__main__":
    intervals = data.load_RR_interval_from_file()
    features = np.array([ [i.max(), i.min(), i.mean(), i.var()] for i in intervals ])
    y = data.load_labels()

    X_train, X_val, y_train, y_val = train_test_split(features, y)

    # scaler = MinMaxScaler()
    # scaler.fit(X_train)
    #
    # X_train = scaler.transform(X_train)
    # X_val = scaler.transform(X_val)

    clf = BalancedBaggingClassifier(n_estimators=100, max_samples=1.0, max_features=1.0)
    # clf = EasyEnsembleClassifier(n_estimators=10)
    clf.fit(X_train, y_train)

    print(f1_score(y_train, clf.predict(X_train), average='micro'))
    print(f1_score(y_val, clf.predict(X_val), average='micro'))

def gen_output(filename):
    test_intervals = data.load_RR_interval_test_from_file()
    test_features = np.array([ [i.max(), i.min(), i.mean(), i.var()] for i in test_intervals ])
    preds = clf.predict(test_features)

    out = open('predictions/' + filename, 'w+')
    out.write("id,y\n")
    for i, y in enumerate(preds):
        out.write("{0},{1}\n".format(i, y))

    out.close()
