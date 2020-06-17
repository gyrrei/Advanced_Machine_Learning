from sklearn.ensemble import BaggingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from xgboost import XGBClassifier

import model_selection as ms
from CustomNN import CustomNN, new_model

# models = [
#     ("SVM", SVC(C=2, gamma='scale', probability=True)),
#     ("Random Forest", RandomForestClassifier()),
#     ("LASSO", LogisticRegression(penalty='l1', C=0.5)),
#     ("NN", CustomNN(new_model)),
#     ("XGBR", XGBClassifier())
# ]
#
# # model = VotingClassifier(models)
# # ms.evaluate_model_kfold_downsampled(model, "hard voting")
# model = VotingClassifier(models, voting='hard', )
# ms.evaluate_model_kfold_downsampled(model, "hard voting")

model = AdaBoostClassifier(n_estimators=100, learning_rate=1.1)
ms.evaluate_model_kfold_downsampled(model,  "Boosted RF", k=5)