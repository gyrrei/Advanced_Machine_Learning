import numpy as np
from sklearn.metrics import balanced_accuracy_score

import data
from sklearn.svm import SVC

X_train, X_val, y_train, y_val = data.undersampled_split()

svm = SVC(C=2, gamma='scale', probability=True)
svm.fit(X_train, y_train)

probas = svm.predict_proba(X_val)
preds = svm.predict(X_val)
conf = np.max(probas, axis=1)

for i in range(len(X_val)):
    print(probas[i], "-->", preds[i], " | Correct: ", y_val[i])


print ("avg conf. correct", np.mean(conf[preds == y_val]))
print ("avg conf. incorrect", np.mean(conf[preds != y_val]))
print("SCORE:", balanced_accuracy_score(y_val, preds))


print("TRAINING:")
t_probas =svm.predict_proba(X_train)
t_preds = svm.predict(X_train)
t_conf = np.max(t_probas, axis=1)

print ("avg train conf. correct", np.mean(t_conf[t_preds == y_train]))
print ("avg train conf. incorrect", np.mean(t_conf[t_preds != y_train]))

insecure_svm = SVC()
X_insecure = X_train[t_conf < 0.75]
y_insecure = y_train[t_conf < 0.75]

insecure_svm.fit(X_insecure, y_insecure)
val_insecure = insecure_svm.predict(X_val[conf < 0.75])
print("TRAIN Score:", balanced_accuracy_score(y_insecure, insecure_svm.predict(X_insecure)))
preds[conf < 0.75] = val_insecure
print("SCORE: ", balanced_accuracy_score(y_val, preds))