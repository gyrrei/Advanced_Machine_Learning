import pandas as pd
import numpy as np

# Class Imbalance
from sklearn.utils import resample


import warnings
warnings.filterwarnings('ignore')


def get_general_features(X):
    """
        Takes a list of arrays X and computes features for each array. Set into a dataframe and normalize.
        :return panda dataframe of shape [len(X), ]
        """

    X_features = pd.DataFrame(0, index=np.arange(len(X)), columns=['mean', 'median', 'std', 'max',
                                                                   'min'])

    for i in range(0,len(X)):
        X_features.set_value(i, 'mean', X[i].mean())
        X_features.set_value(i, 'median', np.median(X[i]))
        X_features.set_value(i, 'std', X[i].std())
        X_features.set_value(i, 'max', X[i].max())
        X_features.set_value(i, 'min', X[i].min())


    return (X_features-X_features.mean(axis=0))/X_features.std(axis=0)

def get_heartbeat_features(X):
    """
        Takes a list of arrays of heartbeat templates shape [#heartbeats, 180] and computes features for the array.
        Set into a dataframe and normalize.
        :return panda dataframe of shape [len(X), ]
        """

    HB_features = pd.DataFrame(0, index=np.arange(len(X)), columns=['HB_mean', 'HB_median', 'HB_std', 'HB_max',
                                                                   'HB_min', 'rpeak1', 'rpeak2', 'rpeak3'])

    for i in range(0, len(X)):
        HB_features.set_value(i, 'HB_mean', np.median(X[i].mean(axis=0)))
        HB_features.set_value(i, 'HB_median', np.median(np.median(X[i], axis=0)))
        HB_features.set_value(i, 'HB_std', np.median(X[i].std(axis=0)))
        HB_features.set_value(i, 'HB_max', np.median(X[i].max(axis=0)))
        HB_features.set_value(i, 'HB_min', np.median(X[i].min(axis=0)))
        HB_features.set_value(i, 'rpeak1', np.median(X[i][:, 59]))
        HB_features.set_value(i, 'rpeak2', np.median(X[i][:, 60]))
        HB_features.set_value(i, 'rpeak3', np.median(X[i][:, 61]))

    return (HB_features - HB_features.mean(axis=0)) / HB_features.std(axis=0)

def resample_set (X, y, n_samples, i ):

    A = pd.concat([X, y], axis=1)
    # separate minority and majority classes
    class_0 = A[A.y == 0]
    class_1 = A[A.y == 1]
    class_2 = A[A.y == 2]
    class_3 = A[A.y == 3]

    # downsample majority
    class_0 = resample(class_0,
                       replace=True,  # sample with replacement
                       n_samples=n_samples,  # match number in majority class
                       random_state=i)  # reproducible results
    # downsample majority
    class_1 = resample(class_1,
                       replace=True,  # sample with replacement
                       n_samples=n_samples,  # match number in majority class
                       random_state=i)  # reproducible results
    # downsample majority
    class_2 = resample(class_2,
                       replace=True,  # sample with replacement
                       n_samples=n_samples,  # match number in majority class
                       random_state=i)  # reproducible results
    # downsample majority
    class_3 = resample(class_3,
                       replace=True,  # sample with replacement
                       n_samples=n_samples,  # match number in majority class
                       random_state=i)  # reproducible results

    # combine dowmsampled majority and minority
    downsampled = pd.concat([class_0, class_1, class_2, class_3])

    y_downsampled = downsampled.y
    X_downsampled = downsampled.drop('y', axis=1)


    return X_downsampled, y_downsampled

def multiple_models():
    resampled = []
    for i in range(0, 10):
        resampled.append(general_features.resample_set(X_train, y_train, y_train['y'].value_counts()[3], i))
    resampled[1][1]

    # Merge the classes to only be [0,1,2] vs. [3]
    y_train_init = np.where(y_train == 3, 1.0, 0.0)
    y_test_init = np.where(y_test == 3, 1.0, 0.0)

    # Extract the predictions for the merged classes - train a model on only them
    A = pd.concat([X_train, y_train], axis=1)
    # separate minority and majority classes
    class_0 = A[A.y == 0]
    class_1 = A[A.y == 1]
    class_2 = A[A.y == 2]

    # downsample majority
    class_0 = resample(class_0,
                       replace=True,  # sample with replacement
                       n_samples=len(class_0),  # match number in majority class
                       random_state=42)  # reproducible results
    # downsample majority
    class_1 = resample(class_1,
                       replace=True,  # sample with replacement
                       n_samples=len(class_0),  # match number in majority class
                       random_state=42)  # reproducible results
    # downsample majority
    class_2 = resample(class_2,
                       replace=True,  # sample with replacement
                       n_samples=len(class_0),  # match number in majority class
                       random_state=42)  # reproducible results

    # combine dowmsampled majority and minority
    downsampled = pd.concat([class_0, class_1, class_2])

    y_downsampled = downsampled.y
    X_downsampled = downsampled.drop('y', axis=1)

    # train the model
    clf = SVC(gamma='auto').fit(X_downsampled, y_downsampled)

    y_pred_init[y_pred_init == 1] = 3

    indeces_pred = np.where(y_pred_init != 3)

    print(indeces_pred[0])
    y_pred = clf.predict(X_test.iloc[indeces_pred[0]])

    for i in range(0, len(y_pred)):
        y_pred_init[indeces_pred[0][i]] = y_pred[i]

    return(None)