from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import data
import numpy as np

X_train, X_val, y_train, y_val = data.simple_split(random_state=42)
classes = [0, 1, 2]
maxminc = [0, 1]
mins = [0, 2]
RESAMPLE = 0.5
FEATURES = 0.5

# def sampling_strategy(sample=0.8):
#     def strat(y):
#         cl = np.unique(y)
#         N = int(np.floor(sample * np.min(np.array([np.sum(y == c) for c in cl]))))
#         ret = dict()
#         for c in cl:
#             ret[c] = N
#         return c
#     return strat
N = int(np.floor(RESAMPLE * np.min(np.array([np.sum(y_train == c) for c in classes]))))
N = int(N)
strat = {
    0: N,
    1: N,
    2: N
}

# N = int(np.floor(2 * RESAMPLE * np.min(np.array([np.sum(y_train == c) for c in classes]))))
# N = int(N)

strat2 = {
    0: N,
    1: N,
    2: N
}

min_strat = {
    0: N,
    2: N
}

minmax_strat = {
    0: N,
    1: N
}

t_preds = []
v_preds = []



trees = []
for i in range(11):
    trees.append(DecisionTreeClassifier(max_depth=2))
    X_t, y_t = RandomUnderSampler(sampling_strategy=strat).fit_resample(X_train, y_train)
    features = np.random.choice(np.arange(X_t.shape[1]), int(np.floor(X_t.shape[1] * FEATURES)))
    X_t = X_t[:, features]

    trees[i].fit(X_t, y_t)
    for c in classes:
        t_preds.append(trees[i].predict_proba(X_train[:, features])[:, c])
        v_preds.append(trees[i].predict_proba(X_val[:, features])[:, c])
        # t_true.append(data.one_hot(y_train)[:, c])


svms = []
for i in range(11):
    svms.append(SVC())
    X_t, y_t = RandomUnderSampler(sampling_strategy=strat).fit_resample(X_train, y_train)
    features = np.random.choice(np.arange(X_t.shape[1]), int(np.floor(X_t.shape[1] * FEATURES)))
    X_t = X_t[:, features]

    svms[i].fit(X_t, y_t)
    for c in classes:
        t_preds.append(data.one_hot(svms[i].predict(X_train[:, features]))[:, c])
        v_preds.append(data.one_hot(svms[i].predict(X_val[:, features]))[:, c])
        # t_true.append(data.one_hot(y_train)[:, c])


minority_svms = []
for i in range(5):
    minority_svms.append(SVC())
    X_t = np.concatenate([X_train[y_train ==0], X_train[y_train == 2]])
    y_t = np.concatenate([y_train[y_train == 0], y_train[y_train == 2]])
    X_t, y_t = RandomUnderSampler(sampling_strategy=min_strat).fit_resample(X_t, y_t)
    features = np.random.choice(np.arange(X_t.shape[1]), int(np.floor(X_t.shape[1] * FEATURES * 2)))
    X_t = X_t[:, features]

    minority_svms[i].fit(X_t, y_t)
    for c in mins:
        t_preds.append(data.one_hot(minority_svms[i].predict(X_train[:, features]))[:, c])
        v_preds.append(data.one_hot(minority_svms[i].predict(X_val[:, features]))[:, c])
        # t_true.append(data.one_hot(y_train)[:, c])

maxmin_svms = []
for i in range(5):
    maxmin_svms.append(SVC())
    y_t = np.copy(y_train)
    y_t[y_train == 2] = 0
    X_t, y_t = RandomUnderSampler(sampling_strategy=minmax_strat).fit_resample(X_train, y_t)

    features = np.random.choice(np.arange(X_t.shape[1]), int(np.floor(X_t.shape[1] * FEATURES * 2)))
    X_t = X_t[:, features]

    maxmin_svms[i].fit(X_t, y_t)
    for c in maxminc:
        t_preds.append(data.one_hot(maxmin_svms[i].predict(X_train[:, features]))[:, c])
        v_preds.append(data.one_hot(maxmin_svms[i].predict(X_val[:, features]))[:, c])
        # y_true = np.copy(y_train)
        # y_true[y_train == 2] = 0
        # t_true.append(data.one_hot(y_true))

t_preds = np.array(t_preds).T
v_preds = np.array(v_preds).T

print ("================================================================")
for i in range(100):
    print(t_preds[i], " --> ", y_train[i])
print ("================================================================")
for i in range(100):
    print(v_preds[i], " --> ", y_val[i])


clf = SVC()
clf.fit(t_preds, y_train)

print(balanced_accuracy_score(y_train, clf.predict(t_preds)))
print(balanced_accuracy_score(y_val, clf.predict(v_preds)))
