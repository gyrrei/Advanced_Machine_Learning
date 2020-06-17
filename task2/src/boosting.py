from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import model_selection as ms


# clf = AdaBoostClassifier(base_estimator=SVC(probability=True), n_estimators=10)
# ms.evaluate_model(clf, "AdaBoost whole DataSet")
# ms.evaluate_model_downsampled(clf, "AdaBoost downsampled")

clf = GradientBoostingClassifier(subsample=0.5, n_iter_no_change=10, learning_rate=0.05, max_features=0.9, n_estimators=200, max_depth=3, min_samples_split=2)
ms.evaluate_model_oversampled(clf, "GBC", )