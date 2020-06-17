import numpy as np
from itertools import repeat

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score as bmac
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.combine import SMOTEENN
from sklearn.neighbors import LocalOutlierFactor

# X_train, y_train, _ = data.load_signals()


def calculate_variance(class_label=None):
    # train_HB = data.load_train_heartbeats(class_label=class_label, copy=False)
    train_HB = data.load_heartbeats_from_file(False)
    print(len(train_HB))
    vars = []
    for hb in train_HB:
        vars.append(np.var(hb, axis=0))

    vars = np.array(vars)
    return vars


def calculate_lof(class_label=None):
    train_HB = data.load_train_heartbeats(class_label=class_label, copy=False)
    lofs = []
    for hb in train_HB:
        lof = LocalOutlierFactor()
        lof.fit(hb)
        features = [np.min(lof.negative_outlier_factor_), np.mean(lof.negative_outlier_factor_), np.max(lof.negative_outlier_factor_)]
        lofs.append(np.array(features))

    lofs = np.array(lofs)
    return lofs


def dissimilarity_data(strategy='var'):
    if strategy == 'var':
        vs = [calculate_variance(c) for c in [0,1,2,3]]

        X = np.concatenate(
            (vs[0], vs[1], vs[2], vs[3]),
            axis=0
        )

        y = np.concatenate(
            (np.repeat(0, vs[0].shape[0]), np.repeat(1, vs[1].shape[0]), np.repeat(2, vs[2].shape[0]), np.repeat(3, vs[3].shape[0]))
        )
    elif strategy == 'LOF':
        ls = [calculate_lof(c) for c in [0,1,2,3]]

        X = np.concatenate(
            (ls[0], ls[1], ls[2], ls[3]),
            axis=0
        )

        y = np.concatenate(
            (np.repeat(0, ls[0].shape[0]), np.repeat(1, ls[1].shape[0]), np.repeat(2, ls[2].shape[0]),
             np.repeat(3, ls[3].shape[0]))
        )


    return X, y


def dissimilarity_classifier_for_noise_class(strategy="var"):
    X = calculate_variance()
    y = data.load_labels()
    print(X.shape, y.shape)
    y[y == 1] = 0
    y[y == 2] = 0
    y[y == 3] = 1

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    resampler = RandomUnderSampler()

    X_train_undersampled, y_train_undersampled = resampler.fit_resample(X_train, y_train)



    # svm = SVC()
    # svm.fit(X_train_undersampled, y_train_undersampled)
    # print("Undersampled:")
    # # print(f1_score(y_train, svm.predict(X_train), average='micro'))
    # # print(f1_score(y_val, svm.predict(X_val), average='micro'))
    # print(bmac(y_val, svm.predict(X_val)))
    #
    # print("Mixture:")
    # smoteen = SMOTEENN()
    # X_train_ms, y_train_ms = smoteen.fit_resample(X_train, y_train)
    # svm = SVC()
    # svm.fit(X_train_ms, y_train_ms)
    # print(bmac(y_val, svm.predict(X_val)))

    print('adaboost')
    boost = AdaBoostClassifier()
    boost.fit(X_train, y_train)
    print(bmac(y_val, boost.predict(X_val)))

    print("Ensemble Balanced BAgging:")
    bagger = BalancedBaggingClassifier(n_estimators=100)
    bagger.fit(X_train, y_train)
    print(bmac(y_val, bagger.predict(X_val)))


# def heartbeat_classifier():
#     heartbeats = data.load_train_heartbeats()
#     y = data.load_labels()
#     y[y==1] = 0
#     y[y==2] = 0
#     y[y==3] = 1


def dissimilarity_classifier(strategy="var"):
    X, y = dissimilarity_data(strategy=strategy)

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    resampler = RandomUnderSampler()

    X_train_undersampled, y_train_undersampled = resampler.fit_resample(X_train, y_train)

    # svm = SVC()
    # svm.fit(X_train_undersampled, y_train_undersampled)
    # print("Undersampled:")
    # # print(f1_score(y_train, svm.predict(X_train), average='micro'))
    # # print(f1_score(y_val, svm.predict(X_val), average='micro'))
    # print(bmac(y_val, svm.predict(X_val)))
    #
    # print("Mixture:")
    # smoteen = SMOTEENN()
    # X_train_ms, y_train_ms = smoteen.fit_resample(X_train, y_train)
    # svm = SVC()
    # svm.fit(X_train_ms, y_train_ms)
    # print(bmac(y_val, svm.predict(X_val)))

    print("Ensemble Balanced Bagging:")
    bagger = BalancedBaggingClassifier(base_estimator=SVC(), sampling_strategy='not majority')
    bagger.fit(X_train, y_train)
    print(bmac(y_val, bagger.predict(X_val)))
    print(f1_score(y_val, bagger.predict(X_val), average='micro'))



if __name__ == "__main__":
    # print("VAR all classes")
    dissimilarity_classifier_for_noise_class(strategy="var")

    # dissimilarity_classifier("var")

    # X= calculate_variance()
    # y = data.load_labels()
    #
    # assert(len(X) == len(y))
    #
    # out = open('out/HB_Variance_train.csv', 'w+')
    # out.write("id,")
    # for f in range(X.shape[1] - 1):
    #     out.write("t{0},".format(f))
    # out.write("t{0}\n".format(X.shape[1] -1 ))
    #
    # for i, x in enumerate(X):
    #     out.write("{0},".format(i))
    #     for j in range(len(x) - 1):
    #         out.write("{0},".format(x[j]))
    #     out.write("{0}\n".format(x[-1]))
    #
    # out.close()
    # for c in [0,1,2,3]:
    #     vars = calculate_variance(class_label=c)
    #     print("AVG / STD of var in class {0}".format(c))
    #     print(np.mean(vars))
    #     print(np.std(vars))
    #     # print(vars.shape)
    #     # for signal in vars:
    #     #     print(np.mean(signal), np.max(signal), np.min(signal))
