import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bmac
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from sklearn.feature_selection import  SelectKBest, SelectPercentile
import data
import biosppy.signals.eeg as eeg

# calculated_stuff = {}

NB_EPOCHS = 21600


def get_wave_features2(signal, size=0.25):
    results = []
    for epoch in range(len(signal)):
        results.append(eeg.get_power_features(signal[epoch].reshape(-1, 1), sampling_rate=128, size=size))

    wave_features = np.array([np.concatenate((r['alpha_low'].reshape((-1,)),
                                              r['alpha_high'].reshape((-1,)),
                                              r['beta'].reshape((-1,)),
                                              r['gamma'].reshape((-1,)),
                                              r['theta'].reshape((-1,)))
                                             ) for r in results])

    return wave_features

if __name__ == "__main__":
    clf = SVC()

    s = 0.25
    X_train = get_wave_features2(data.load_train_eeg1(subjects=[0,2], do_filter=True),
                                 size=s)

    print(X_train.shape)

    y_train = data.load_train_labels(subjects=[0,2])

    X_val = get_wave_features2(data.load_train_eeg1(subjects=[1], do_filter=True),
                               size=s)
    y_val = data.load_train_labels([1])

    clf.fit(X_train, y_train)
    print("window size = {0}".format(s))
    print(bmac(y_val, clf.predict(X_val)))


    # X_train = get_wave_features2(data.load_train_eeg1(subjects=[0, 2], do_filter=True))
    # y_train = data.load_train_labels([0, 2])
    #
    #
    #
    # selector = SelectPercentile(percentile=25)
    # selector.fit(X_train, y_train)
    # # print(selector.scores_)
    #
    # # X_train = selector.transform(X_train)
    # # X_val = selector.transform(X_val)
    #
    # clf = SVC()
    # for p in [10, 25, 50, 75, 100]:
    #     sel = SelectPercentile(percentile=p)
    #     sel.fit(X_train, y_train)
    #     clf.fit(sel.transform(X_train), y_train)
    #     print("percentile of best features: {0}".format(p))
    #     print(bmac(y_val, clf.predict(sel.transform(X_val))))



    # X_train = get_wave_features(data.load_train_eeg1([0,2]))
    # y_train = data.load_train_labels([0,2])
    # X_val = get_wave_features(data.load_train_eeg1([1]))
    # y_val = data.load_train_labels([1])
    #
    # clfs = [SVC(), SVC(gamma='scale'), BalancedBaggingClassifier(), BalancedBaggingClassifier(n_estimators=100), EasyEnsembleClassifier()]
    # for clf in clfs:
    #     clf.fit(X_train, y_train)
    #     print(bmac(y_val, clf.predict(X_val)))

