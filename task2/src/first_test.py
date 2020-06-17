from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from generative import GaussianGenerativeModel, DensityEstimationModel
import model_selection as ms

from xgboost import XGBClassifier

import numpy as np
import csv

from sklearn.model_selection import train_test_split


import data
# X, y, X_test = data.load()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

resampler = SMOTE()
#X, y = resampler.fit_resample(X, y)

def build_weighted_svm_model():
    return BalancedBaggingClassifier(base_estimator=SVC(kernel='rbf', gamma='scale'),
                                    sampling_strategy='auto',
                                    replacement=True,
                                    n_estimators=50,
                                    random_state=0)

def build_simple_svm_model():
    return BalancedBaggingClassifier(base_estimator=SVC(kernel='rbf', gamma='scale'),
                                sampling_strategy='auto',
                                replacement=True,
                                n_estimators=50,
                                random_state=0)

# 0.6984267128396707 cv-result


def build_generative_model():
    return BalancedBaggingClassifier(base_estimator=GaussianGenerativeModel(),
                                sampling_strategy='auto',
                                replacement=True,
                                n_estimators=50,
                                random_state=0)


def build_simple_knn_model():
    return BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(),
                                sampling_strategy='auto',
                                replacement=True,
                                n_estimators=10,
                                random_state=0)

clf = build_simple_knn_model()
ms.evaluate_model_kfold(clf, "Simple Bagged KNN", k=5)

# maxModel = None
# maxVal = -1
# for n in [10, 25, 50, 100]:
#     for b in [True, False]:
#         print("\n")
#         print("///////////////////////////////////////////////////////////////////")
#         model = BalancedBaggingClassifier(base_estimator=SVC(kernel='rbf', gamma='scale'),
#                                             sampling_strategy='auto',
#                                             replacement=b,
#                                             n_estimators=n,
#                                             random_state=0)
#         _, val_score = ms.evaluate_model_kfold(model, name="Balanced Bagging SVM, n={0}, b={1}".format(n, b))
#         if val_score > maxVal:
#             maxModel = model
#
#
# data.generate_output("balanced-bagging-svm-optimal-n.csv", maxModel)