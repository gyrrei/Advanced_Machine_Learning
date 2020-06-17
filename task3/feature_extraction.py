from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import data
import tsfresh
import pandas as pd
from itertools import repeat

signal = data.load_filtered_train()
y = data.load_labels()

indices = np.random.random_integers(0, len(signal), 2)

def extract_features(X, y, only_relevant=True):
    dic = {"id": [], "t": [], "mV": []}

    for i, x in enumerate(X):
        dic["id"].extend(repeat(i, len(x)))
        dic["t"].extend(range(len(x)))
        dic["mV"].extend(x)

    timeseries = pd.DataFrame.from_dict(dic)

    if only_relevant:
        return tsfresh.extract_relevant_features(timeseries,
                                                 pd.Series(y, index=range(len(X))),
                                                 column_id="id",
                                                 column_sort="t",
        )
    else:
        return tsfresh.extract_features(timeseries,
                                        column_id='id',
                                        column_sort='t',
                                        )

chosen_signals = []
for i in indices:
    chosen_signals.append(signal[i])

if __name__ == "__main__":
    X = extract_features(chosen_signals, None, only_relevant=False)
    # X = extract_features(chosen_signals, y[indices])

    out = open('out/ts-fresh-features-train.csv', 'w+')
    out.write("id,")
    for f in range(X.shape[1] - 1):
        out.write("x{0},".format(f))
    out.write("x{0}\n".format(X.shape[1] -1 ))

    for i, x in enumerate(X):
        out.write("{0},".format(i))
        for j in range(len(x) - 1):
            out.write("{0},".format(x[j]))
        out.write("{0}\n".format(x[-1]))

    out.close()

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    svm = SVC(gamma='scale')
    svm.fit(X_train, y_train)
    print(f1_score(y_val, svm.predict(X_val)))