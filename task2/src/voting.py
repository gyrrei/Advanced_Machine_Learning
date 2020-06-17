import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import model_selection as ms
import visualisation as vis


def one_hot(y):
    y_ = np.zeros((len(y), y.max() + 1))
    y_[np.arange(len(y)), y] = 1
    return y_


class WeightedVotingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, class_weight=None):
        self.base_estimators = base_estimators
        self.class_weight = class_weight

    def fit(self, X, y):
        for e in self.base_estimators:
            e.fit(X, y)
        self.nb_classes = len(np.unique(y))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.nb_classes))
        for e in self.base_estimators:
            predictions += one_hot(e.predict(X))

        if self.class_weight is not None:
            predictions *= self.class_weight

        # TODO: think about tie-breaking
        return np.argmax(predictions, axis=1)


# for i in range(100):
#     # estimators.append(DecisionTreeClassifier(max_depth=8))
#     estimators.append(SVC())

# model = WeightedVotingEstimator(estimators, class_weight=np.array([2, 1, 1]))
# ms.evaluate_model_downsampled(model, "Weighted [2 1 2] Voting SVM")
#
# model = WeightedVotingEstimator(estimators)
# ms.evaluate_model_downsampled(model, "Unweighted Voting SVM")

# ms.evaluate_model(model, "Weighted Voting based on DT")

class MinorityVotingEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, minority_classes, minority_discriminator):
        self.base_estimators = base_estimators
        self.minorities = minority_classes
        self.discriminator = minority_discriminator

    def fit(self, X, y):
        for e in self.base_estimators:
            e.fit(X, y)
        self.nb_classes = len(np.unique(y))
        minority_indices = np.zeros(X.shape[0], dtype=bool)
        for c in self.minorities:
            minority_indices = np.logical_or(minority_indices, y == c)
        self.discriminator.fit(X[minority_indices], y[minority_indices])

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.nb_classes))
        for e in self.base_estimators:
            preds += one_hot(e.predict(X))

        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if np.sum(preds[i, self.minorities]) > 0:
                prediction[i] = self.discriminator.predict(X[i].reshape(1,-1)).reshape(1)
            else:
                prediction[i] = np.argmax(preds[i])
        return prediction


estimators = [SVC(decision_function_shape='ovo'), SVC(decision_function_shape='ovr'), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=5), LogisticRegression(), LogisticRegression(penalty='l2')]
model = MinorityVotingEstimator(estimators, [0,2], SVC(class_weight={0: 0.5, 2: 1}))
ms.evaluate_model(model, "Custom Voting", confusion_matrix=True)
