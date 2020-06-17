from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import SVC
import model_selection as ms

estimator = SVC()
model = OutputCodeClassifier(estimator, random_state=42)

ms.evaluate_model_kfold(model, "Simple ECOC with SVM base")