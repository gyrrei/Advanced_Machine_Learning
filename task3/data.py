import csv
import biosppy.signals.ecg as ecg
import numpy as np


_X_train = None
_X_test = None
_y_train = None

_HB_train = None
_HB_train_c = [None, None, None, None]

_X_f_train = None

_t_data = None


def _map(f, l):
    return list(map(f, l))


def load_X_train(copy=True):
    """
        Loads the data from the csv files and returns a list of ECG signals. Each signal in the list is
        represented as a numpy array.
        :return list<np.array>: A list of vectors, each representing one ECG signal
        """
    global _X_train
    if _X_train is None:
        train_file = open('data/X_train.csv', 'r')
        train_reader = csv.reader(train_file)

        _X_train = _map(lambda s: np.array(_map(lambda x: float(x), s[1:])), list(train_reader)[1:])
        train_file.close()
    if copy:
        return _X_train.copy()
    else:
        return _X_train


def load_X_test(copy=True):
    """
        Loads the data from the csv files and returns a list of ECG signals. Each signal in the list is
        represented as a numpy array.
        :return list<np.array>: A list of vectors, each representing one ECG signal
        """
    global _X_test
    if _X_test is None:
        test_file = open('data/X_test.csv', 'r')
        test_reader = csv.reader(test_file)

        _X_test = _map(lambda s: np.array(_map(lambda x: float(x), s[1:])), list(test_reader)[1:])
        test_file.close()
    if copy:
        return _X_test.copy()
    else:
        return _X_test


def load_labels(copy=True):
    """
        Loads the data from the csv files and returns a list of ECG signals. Each signal in the list is
        represented as a numpy array.
        :return list<np.array>: A list of vectors, each representing one ECG signal
        """
    global _y_train
    labels_file = open('data/y_train.csv', 'r')
    labels_reader = csv.reader(labels_file)

    _y_train = np.array(_map(lambda s: int(s[1]), list(labels_reader)[1:]))
    labels_file.close()
    if copy:
        return _y_train.copy()
    else:
        return _y_train


def load_signals():
    return load_X_train(), load_labels(), load_X_test()


def load_heartbeats(signal):
    global _t_data
    ret = ecg.ecg(signal=signal, sampling_rate=300, show=False)
    if _t_data is None:
        _t_data = ret['templates_ts']
    return ret['templates']


def load_filtered_signal(signal):
    return ecg.ecg(signal=signal, sampling_rate=300, show=False)['filtered']


def load_filtered_train(copy=True):
    global _X_f_train
    if _X_f_train is None:
        X_train = load_X_train(False)
        _X_f_train = []
        for x in X_train:
            _X_f_train.append(load_filtered_signal(x))

    if copy:
        return _X_f_train.copy()
    else:
        return _X_f_train

def load_train_heartbeats(class_label=None, copy=True):
    global _HB_train, _HB_train_c
    if _HB_train is None:
        X_train = load_X_train(False)

        _HB_train = []
        for i, x in enumerate(X_train):
            _HB_train.append(load_heartbeats(x))

    if class_label is not None:
        if _HB_train_c[class_label] is None:
            _HB_train_c[class_label] = []
            y_train = load_labels(False)
            for i, hb in enumerate(_HB_train):
                if y_train[i] == class_label:
                    _HB_train_c[class_label].append(hb)

        if copy:
            return _HB_train_c[class_label].copy()
        else:
            return _HB_train_c[class_label]

    else:
        if copy:
            return _HB_train.copy()
        else:
            return _HB_train


def load_heartbeats_from_file(copy=True):
    global _HB_train
    nb_files = 5116
    if _HB_train is None:
        _HB_train = []
        for i in range(nb_files + 1):
            file = open('heartbeats/beat-{0}.csv'.format(i), 'r')
            r = csv.reader(file)
            _HB_train.append(np.array(_map(lambda s: np.array(_map(lambda x: float(x), s)), list(r))))

    if copy:
        return _HB_train.copy()
    else:
        return _HB_train



def load_train_heartbeats_mean():
    HB_train = load_train_heartbeats(copy=False)
    means = [np.mean(sequence, axis=0) for sequence in HB_train]
    return np.array(means)


def one_hot(y):
    y_ = np.zeros((len(y), y.max() + 1))
    y_[np.arange(len(y)), y] = 1
    return y_


def generate_heartbeat_files():
    HBs = load_train_heartbeats()
    for i, hb in enumerate(HBs):
        outfile = open('heartbeats/beat-{0}.csv'.format(i), 'w+')
        for row in range((hb.shape[0] - 1)):
            for col in range((hb.shape[1] - 1)):
                outfile.write("{0},".format(hb[row, col]))
            outfile.write("{0}\n".format(hb[row][-1]))

        for col in range(hb.shape[1] - 1):
            outfile.write("{0},".format(hb[-1][col]))
        outfile.write("{0}\n".format(hb[-1][-1]))
        outfile.close()



if __name__ == "__main__":
    hb = load_heartbeats_from_file()
    print(hb)

