from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import data

heartbeats = data.load_train_heartbeats_mean()
y = data.load_labels()

X_train, X_val, y_train, y_val = train_test_split(heartbeats, y)

clf = BalancedBaggingClassifier(SVC())
clf.fit(X_train, y_train)
print(f1_score(y_val, clf.predict(X_val), average='micro'))