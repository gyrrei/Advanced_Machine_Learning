from sklearn.svm import SVC

import model_selection as ms
import data
from imblearn.ensemble import BalancedBaggingClassifier
from preprocessing import pca

max = 0
max_model = None
max_name = ""
for n in [10, 20, 50, 100]:
    for r in [True, False]:
        model = BalancedBaggingClassifier(base_estimator=SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo'),
                                          sampling_strategy='auto',
                                          replacement=r,
                                          n_estimators=n,
                                          random_state=0)
        name = "Bagged-PCA-SVM-{0}-estimators;replacement={1}".format(n, r)
        train_score, validation_score = ms.evaluate_model_kfold(model, name=name, random_state=0, preprocess=pca)
        if validation_score > max:
            max = validation_score
            max_model = model
            max_name = name

data.generate_output(max_name + "[" + str(max) + "].csv", max_model, preprocess=pca)
