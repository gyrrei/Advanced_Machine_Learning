import numpy as np

from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import KFold
import visualisation as vis

import data


def print_scores(model, X_train, X_val, y_train, y_val):
    y_train_pred = model.predict(X_train)
    BMAC_train = balanced_accuracy_score(y_train, y_train_pred)

    y_val_pred = model.predict(X_val)
    BMAC_val = balanced_accuracy_score(y_val, y_val_pred)

    print("Train Score: {0}".format(BMAC_train))
    print(classification_report(y_train, y_train_pred))
    print("-----------------------------------------------")
    print("Validation Score: {0}".format(BMAC_val))
    print(classification_report(y_val, y_val_pred))
    print("-----------------------------------------------")
    vis.plot_confusion_matrix(y_val, y_val_pred, np.unique(y_train), normalize=True)


def _evaluate(model, X_train, X_val, y_train, y_val, confusion_matrix=False, preprocess=None):
    if preprocess:
        X_train, X_val, y_train, y_val = preprocess(X_train, X_val, y_train, y_val)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    BMAC_train = balanced_accuracy_score(y_train, y_train_pred)

    y_val_pred = model.predict(X_val)
    BMAC_val = balanced_accuracy_score(y_val, y_val_pred)

    print("Train Score: {0}".format(BMAC_train))
    print(classification_report(y_train, y_train_pred))
    print("-----------------------------------------------")
    print("Validation Score: {0}".format(BMAC_val))
    print(classification_report(y_val, y_val_pred))
    print("-----------------------------------------------")

    if confusion_matrix:
        vis.plot_confusion_matrix(y_val, y_val_pred, np.unique(y_train), normalize=True)
    return BMAC_train, BMAC_val


def _evaluate_kfold(model, datasets, verbose=False, preprocess=None):
    train_values = []
    val_values = []
    k = 0
    for X_train, X_val, y_train, y_val in datasets:
        if preprocess:
            X_train, X_val, y_train, y_val = preprocess(X_train, X_val, y_train, y_val)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        train_score = balanced_accuracy_score(y_train, y_train_pred)
        train_values.append(train_score)

        y_val_pred = model.predict(X_val)
        val_score = balanced_accuracy_score(y_val, y_val_pred)
        val_values.append(val_score)

        if verbose:
            print("\t**** Fold {0} ****".format(k))
            print("Train Score: {0}".format(train_score))
            print(classification_report(y_train, y_train_pred))
            print("-----------------------------------------------")
            print("Validation Score: {0}".format(val_score))
            print(classification_report(y_val, y_val_pred))
            print("-----------------------------------------------")

        k += 1

    if verbose:
        print("===============================================")

    train_score = np.mean(np.array(train_values))
    val_score = np.mean(np.array(val_values))

    print("Train Score: {0}".format(train_score))
    print("-----------------------------------------------")
    print("Validation Score: {0}".format(val_score))
    print("-----------------------------------------------")
    return train_score, val_score


def evaluate_model_downsampled(model, name, split=0.3, random_state=None, confusion_matrix=False, preprocess=None):
    X_train, X_val, y_train, y_val = data.undersampled_split(split=split, random_state=random_state)
    print("============{0} SCORES =============".format(name))
    print("Simple {0} Split, training data downsampled".format(split))

    return _evaluate(model, X_train, X_val, y_train, y_val, confusion_matrix=confusion_matrix, preprocess=preprocess)

def evaluate_model_oversampled(model, name, split=0.3, random_state=None, confusion_matrix=False, preprocess=None):
    X_train, X_val, y_train, y_val = data.oversampled_split(split=split, random_state=random_state)
    print("============{0} SCORES =============".format(name))
    print("Simple {0} Split, training data upsampled".format(split))

    return _evaluate(model, X_train, X_val, y_train, y_val, confusion_matrix=confusion_matrix, preprocess=preprocess)


def evaluate_model_kfold_downsampled(model, name, k=10, random_state=None, preprocess=None, feature_selection=None):
    """

    :param model:
    :param name:
    :param k:
    :param random_state:
    :param preprocess:
    :return: the train and validation score (BMAC)
    """
    datasets = data.undersampled_kfold_split(k=k, random_state=random_state, feature_selection=feature_selection)
    assert len(datasets) == k

    print("============{0} SCORES =============".format(name))
    print("{0}-fold CV, training data downsampled".format(k))

    return _evaluate_kfold(model, datasets, preprocess=preprocess)


def evaluate_model_kfold_oversampled(model, name, k=10, random_state=None, preprocess=None):
    datasets = data.oversampled_kfold_split(k=k, random_state=random_state)
    assert len(datasets) == k

    print("============{0} SCORES =============".format(name))
    print("{0}-fold CV, training data oversampled".format(k))

    return _evaluate_kfold(model, datasets, preprocess=preprocess)


def evaluate_model_kfold(model, name, k=10, verbose=False, random_state=None, preprocess=None):
    """

    :param model:
    :param name:
    :param k:
    :param verbose:
    :param random_state:
    :param preprocess:
    :return: The train and validation BMAC score
    """
    datasets = data.kfold_split(k=k, random_state=random_state)
    assert len(datasets) == k

    print("============{0} SCORES =============".format(name))
    print("{0}-fold CV".format(k))

    return _evaluate_kfold(model, datasets, verbose=verbose, preprocess=preprocess)


def evaluate_model(model, name, split=0.3, random_state=None, resampler=None, confusion_matrix=False, feature_selection=None):
    X_train, X_val, y_train, y_val = data.simple_split(split=split, random_state=random_state)
    if resampler is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)
    if feature_selection is not None:
        X_train, X_val = X_train[:, feature_selection], X_val[:, feature_selection]
    print("============{0} SCORES =============".format(name))
    print("Simple {0} Split".format(split))

    return _evaluate(model, X_train, X_val, y_train, y_val, confusion_matrix=confusion_matrix)


def evaluate_minority_model(model, name,
                            split=0.3, preprocess=None, resampler=None,
                            random_state=None, confusion_matrix=False,
                            feature_selection=None,
                            sample_selection=None
                            ):
    X_train, X_val, y_train, y_val = data.minorities_split(split=split, random_state=random_state)
    print("============{0} SCORES =============".format(name))
    print("Simple {0} Split for telling minorities apart".format(split))
    if feature_selection is not None:
        X_train = X_train[:, feature_selection]
        X_val = X_val[:, feature_selection]
    if preprocess:
        X_train, X_val, y_train, y_val = preprocess(X_train, X_val, y_train, y_val)
    if resampler is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)

    return _evaluate(model, X_train, X_val, y_train, y_val, confusion_matrix=confusion_matrix)


def evaluate_minority_model_kfold(model, name, k=10,
                                  verbose=False, random_state=None,
                                  preprocess=None, sample_selection=None):
    """

    :param model:
    :param name:
    :param k:
    :param verbose:
    :param random_state:
    :param preprocess:
    :return: The train and validation BMAC score
    """
    datasets = data.minorities_kfold_split(k=k, random_state=random_state, sample_selection=sample_selection)
    assert len(datasets) == k

    print("============{0} SCORES =============".format(name))
    print("{0}-fold CV for telling minorities apart".format(k))

    return _evaluate_kfold(model, datasets, verbose=verbose, preprocess=preprocess)