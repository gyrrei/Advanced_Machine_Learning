import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile

import data
import model_selection as ms


def get_lasso_features():
    X, y = data.undersampled()
    model = LogisticRegression(penalty='l1', C=0.2)
    model.fit(X, y)
    return np.abs(np.sum(np.abs(model.coef_), axis=0)) > 0

def test_logistic_regression_features():
    model = LogisticRegression(penalty='l1', C=0.2)

    X_train, X_val, y_train, y_val = data.undersampled_split()
    model.fit(X_train, y_train)
    ms._evaluate(model, X_train, X_val, y_train, y_val)

    # print(model.coef_)
    print(model.coef_.shape)

    # print(np.sum(model.coef_ == 0))
    print(np.sum(np.sum(model.coef_, axis=0) == 0))

    relevant_features = np.abs(np.sum(np.abs(model.coef_), axis=0)) > 10e-5

    # features_mrmr = [142, 481, 600, 321, 827, 781, 376, 162, 616, 245, 853, 513, 845, 613, 151, 66, 129, 89, 574, 522, 921, 945, 67, 968, 30, 936, 860, 130, 745, 204, 494, 900, 561, 515, 754, 865, 969, 120, 413, 38, 180, 680, 378, 409, 993, 982, 749, 517, 560, 185]
    # relevant_features = np.zeros(1000, dtype=bool)
    # for i in range(1000):
    #     if i in features_mrmr:
    #         relevant_features[i] = True
    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with selected features", feature_selection=relevant_features)

    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with all features")


def get_percentile_features():
    selector = SelectPercentile(percentile=20)
    X, y = data.undersampled()
    selector.fit(X, y)
    return selector.get_support()


def test_percentile_features():
    features = get_lasso_features()
    print (features.shape)
    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with selected features", feature_selection=features)

    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with all features")

def log_vs_perc():
    features = get_lasso_features()
    print(np.sum(features))
    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with lasso selected features", feature_selection=features)

    features = get_percentile_features()
    print(np.sum(features))
    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with percentile selected features", feature_selection=features)

    svm = SVC(decision_function_shape='ovo')
    ms.evaluate_model_kfold_downsampled(svm, "SVM with all features")

def main():
    log_vs_perc()


if __name__ == "__main__":
    main()
