from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import model_selection as ms

model = XGBClassifier(max_depth=2)
ms.evaluate_model(model, "XGB Classifier")
