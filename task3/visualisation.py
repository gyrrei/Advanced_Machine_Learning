import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
import data
import numpy as np


def plot_signal(index, class_label=None):
    X_train = data.load_X_train()
    if class_label is not None:
        y_train = data.load_labels()
        new_X_train = []
        for i in range(len(X_train)):
            if y_train[i] == class_label:
                new_X_train.append(X_train[i])

        X_train = new_X_train



    if index == 'r':
        index = np.random.randint(0, len(X_train))

    signal = X_train[index]
    ecg.ecg(signal=signal, sampling_rate=300, show=True)


if __name__ == "__main__":
    for c in [0,1,2,3]:
        plot_signal('r', class_label=c)


def plot_avg_heartbeat(hb, comparison=None, titles=None, title=""):
    if titles is None:
        titles = ["Avg Hearbeat", "Comparison"]

    plt.figure()
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Average Heartbeat')
    plt.plot(data._t_data, hb, label=titles[0])
    if comparison is not None:
        plt.plot(data._t_data, comparison, label=titles[1])

    plt.legend()
    plt.show()
