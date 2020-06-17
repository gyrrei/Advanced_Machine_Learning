import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.linear_model import LogisticRegression

import data
import model_selection as ms
from preprocessing import pca

# model = LogisticRegression(penalty='l1', C=0.1)

def pca200(X_train, X_val, y_train, y_val):
    return pca(X_train, X_val, y_train, y_val, 200)

# X_train, X_val, y_train, y_val = data.undersampled_split()
# model.fit(X_train, y_train)
# ms._evaluate(model, X_train, X_val, y_train, y_val)
#
# # print(model.coef_)
# print(model.coef_.shape)
#
# # print(np.sum(model.coef_ == 0))
# print(np.sum(np.sum(model.coef_, axis=0) == 0))
#
# relevant_features = np.abs(np.sum(np.abs(model.coef_), axis=0)) > 10e-5

kernel = 1.0 * WhiteKernel() + 0.2 * RBF(length_scale_bounds=[0.1,100]) ** 2
model = GaussianProcessClassifier(kernel=kernel)
ms.evaluate_model_downsampled(model, "Gaussian Process Classifier", preprocess=pca200)