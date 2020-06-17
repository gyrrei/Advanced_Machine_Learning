import numpy as np
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from preprocessing import pca
import model_selection as ms
import data

class TwoStepEstimatorSplit(BaseEstimator, ClassifierMixin):

    def __init__(self, maxmin=[SVC(), SVC(), SVC()]):
        self.maxmin_clfs = maxmin
        self.min_clf = SVC(C=2, gamma='scale')

    def fit(self, X, y):
        ymaxmin = np.copy(y)
        ymaxmin[y == 2] = 0 # 0 = minorities, 1 majority
        # resampler = RandomUnderSampler()
        # X_mm, y_mm = resampler.fit_resample(X, ymaxmin)

        X_majs = np.array_split(X[y==1], 3)
        y_majs = np.array_split(y[y==1], 3)
        X_min = X[y==0]
        y_min = y[y==0]

        for i in range(3):
            X_mm = np.concatenate([X_majs[i], X_min])
            y_mm = np.concatenate([y_majs[i], y_min])
            p = np.random.permutation(len(X_mm))

            self.maxmin_clfs[i].fit(X_mm[p], y_mm[p])

        X0, y0 = X[y == 0], y[y == 0]
        X2, y2 = X[y == 2], y[y == 2]

        X_ = np.concatenate([X0, X2])
        # np.random.shuffle(X)
        y_ = np.concatenate([y0, y2])

        self.min_clf.fit(X_, y_)

    def predict(self, X):
        preds = np.ones(len(X))
        for clf in self.maxmin_clfs:
            preds = np.logical_and(clf.predict(X), preds)

        # preds = self.maxmin_clf.predict(X)
        preds[preds == 0] = self.min_clf.predict(X[preds == 0])

        return preds


class TwoStepEstimatorUndersample(BaseEstimator, ClassifierMixin):

    def __init__(self, maxmin=SVC()):
        self.maxmin_clf = maxmin
        self.min_clf = SVC(C=2, gamma='scale')

    def fit(self, X, y):
        ymaxmin = np.copy(y)
        ymaxmin[y == 2] = 0 # 0 = minorities, 1 majority
        resampler = RandomUnderSampler()
        X_mm, y_mm = resampler.fit_resample(X, ymaxmin)

        self.maxmin_clf.fit(X_mm, y_mm)

        X0, y0 = X[y == 0], y[y == 0]
        X2, y2 = X[y == 2], y[y == 2]

        X_ = np.concatenate([X0, X2])
        # np.random.shuffle(X)
        y_ = np.concatenate([y0, y2])

        self.min_clf.fit(X_, y_)

    def predict(self, X):
        preds = self.maxmin_clf.predict(X)
        preds[preds == 0] = self.min_clf.predict(X[preds == 0])

        return preds



model = TwoStepEstimatorSplit()
ms.evaluate_model_kfold(model, "Two Step Estimator with split, pca", preprocess=pca)
ms.evaluate_model_kfold(model, "2-step split")

model = TwoStepEstimatorUndersample()
ms.evaluate_model_kfold(model, "2-step undersmaple pca", preprocess=pca)
ms.evaluate_model(model, '2-step undersample')

# data.generate_output("2-step-SVM-SVM-datasplit.csv", model)
