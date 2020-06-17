import csv
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold


def map_(f, l):
    return list(map(f, l))


def one_hot(y):
    y_ = np.zeros((len(y), y.max() + 1))
    y_[np.arange(len(y)), y] = 1
    return y_


def load():
    trainFile = open('X_train.csv', 'rU')
    testFile = open('X_test.csv', 'rU')
    labelsFile = open('y_train.csv', 'rU')
    csvReader = csv.reader(trainFile)
    csvReader2 = csv.reader(testFile)
    csvReaderY = csv.reader(labelsFile)
    X = np.array(map_(lambda l: map_(lambda x: float(x) if x != "" else float('NaN'), l), list(csvReader)[1:]))[:, 1:]
    X_test = np.array(map_(lambda l: map_(lambda x: float(x) if x != "" else float('NaN'), l), list(csvReader2)[1:]))[:, 1:]
    y = np.array(map_(lambda l: map_(lambda x: int(x) if x != "" else float('NaN'), l), list(csvReaderY)[1:]))[:, 1:]
    y = y.reshape((len(y),))
    testFile.close()
    trainFile.close()
    labelsFile.close()

    return X, y, X_test


def load_minorities():
    X, y, _ = load()

    X0, y0 = X[y == 0], y[y == 0]
    X2, y2 = X[y == 2], y[y == 2]


    X = np.concatenate([X0, X2])
    # np.random.shuffle(X)
    y = np.concatenate([y0, y2])
    # y[y == 2] = 1
    # np.random.shuffle(y)

    p = np.random.permutation(len(X))


    return X[p], y[p]


def minorities_split(split=0.3, random_state=None):
    X, y = load_minorities()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=random_state)

    return X_train, X_val, y_train, y_val


def minorities_kfold_split(k=10, random_state=None, feature_selection=None, sample_selection=None):
    X, y = load_minorities()
    cv = KFold(n_splits=k, random_state=random_state)

    if feature_selection is None:
        feature_selection = np.ones(X.shape[1], dtype=bool)

    X = X[:, feature_selection]

    if sample_selection is not None:
        X = X[sample_selection]
        y = y[sample_selection]

    ret = []
    for train_index, test_index in cv.split(X):
        X_train, y_train = X[train_index], y[train_index]
        ret.append((X_train, X[test_index], y_train, y[test_index]))

    return ret


def undersampled(random_state=None):
    X, y, _ = load()
    resampler = RandomUnderSampler(random_state=random_state)
    X, y = resampler.fit_resample(X, y)
    return X, y

def undersampled_split(split=0.3, random_state=None):
    """
    Loads the data, splits it 70/30 into train and validation data and performs
    undersampling on the training data.
    The validation data is original.
    :param random_state: An optional argument that sets the seed.
    :return: X_train, X_val, y_train, y_val
    """
    X, y, _ = load()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=random_state)

    resampler = RandomUnderSampler(random_state=random_state)
    X_train, y_train = resampler.fit_resample(X_train, y_train)
    return X_train, X_val, y_train, y_val


def undersampled_kfold_split(k=10, random_state=None, feature_selection=None):
    X, y, _ = load()
    cv = KFold(n_splits=k, random_state=random_state)
    resampler = RandomUnderSampler(random_state=random_state)

    if feature_selection is None:
        feature_selection = np.ones(X.shape[1], dtype=bool)

    X = X[:, feature_selection]

    ret = []
    for train_index, test_index in cv.split(X):
        X_train, y_train = resampler.fit_resample(X[train_index], y[train_index])
        ret.append((X_train, X[test_index], y_train, y[test_index]))

    return ret


def oversampled_split(split=0.3, random_state=None):
    """
    Loads the data, splits it 70/30 into train and validation data and performs
    undersampling on the training data.
    The validation data is original.
    :param random_state: An optional argument that sets the seed.
    :return: X_train, X_val, y_train, y_val
    """
    X, y, _ = load()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=random_state)

    resampler = SMOTE(random_state=random_state)
    X_train, y_train = resampler.fit_resample(X_train, y_train)
    return X_train, X_val, y_train, y_val


def oversampled_kfold_split(k=10, random_state=None):
    X, y, _ = load()
    cv = KFold(n_splits=k, random_state=random_state)
    resampler = SMOTE(random_state=random_state)

    ret = []
    for train_index, test_index in cv.split(X):
        X_train, y_train = resampler.fit_resample(X[train_index], y[train_index])
        ret.append((X_train, X[test_index], y_train, y[test_index]))

    return ret


def generate_output(filename, model, preprocess=None):
    X, y, X_test = load()
    if preprocess:
        X, X_test, y, _ = preprocess(X, X_test, y, None)

    model.fit(X, y)

    y_test_pred = model.predict(X_test)
    out = open('out/' + filename, 'w+')
    out.write('id,y\n')

    for i in range(len(y_test_pred)):
        out.write("%d.0,%f\n" % (i, y_test_pred[i]))

    out.close()

def generate_output_wo_retrain(filename, model, argmax=False, scaler=None):
    _, _, X_test = load()
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_test = model.predict(X_test)
    if argmax:
        y_test = np.argmax(y_test, axis=1)
    out = open('out/' + filename, 'w+')
    out.write('id,y\n')

    for i in range(len(y_test)):
        out.write("%d.0,%f\n" % (i, y_test[i]))

    out.close()


def generate_output_undersampled(filename, model, preprocess=None, feature_selection=None):
    X, y, X_test = load()
    if feature_selection is not None:
        X = X[:, feature_selection]
        X_test = X_test[:, feature_selection]
    if preprocess:
        X, X_test, y, _ = preprocess(X, X_test, y, None)

    resampler = RandomUnderSampler()
    X, y = resampler.fit_resample(X, y)
    model.fit(X, y)

    y_test_pred = model.predict(X_test)
    out = open('out/' + filename, 'w+')
    out.write('id,y\n')

    for i in range(len(y_test_pred)):
        out.write("%d.0,%f\n" % (i, y_test_pred[i]))

    out.close()

def kfold_split(k=10, random_state=None):
    X, y, _ = load()
    cv = KFold(n_splits=k, random_state=random_state)

    return [(X[train_ind], X[val_ind], y[train_ind], y[val_ind]) for train_ind, val_ind in cv.split(X)]


def simple_split(split=0.3, random_state=None):
    """
    Loads the data, splits it 70/30 into train and validation data and performs
    undersampling on the training data.
    The validation data is original.
    :param random_state: An optional argument that sets the seed.
    :return: X_train, X_val, y_train, y_val
    """
    X, y, _ = load()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=random_state)
    return X_train, X_val, y_train, y_val
