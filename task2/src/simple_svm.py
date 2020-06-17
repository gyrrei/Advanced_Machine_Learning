import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalanceCascade
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import data
import matplotlib.pyplot as plt
from dimension_reduction import KernelDiscriminantAnalysis
import model_selection as ms
from preprocessing import pca
from feature_importance import get_lasso_features

# X_train, X_val, y_train, y_val = data.undersampled_split()

# model = SVC(class_weight={0: 2, 1: 1, 2: 2})
# ms.evaluate_model_downsampled(model, "Weighted SVM, Weights")
# ms.evaluate_model_kfold_downsampled(model, "Weighted SVM, Weights={2,1,2}")

# for w in [1, 1.5, 2, 4, 6]:
#     for c in [0.5, 1.0, 2.0, 5.0]:
#         for g in ['auto', 'scale']:
#             model = SVC(class_weight={0: w, 1:1, 2: w}, C=c, gamma=g)
#             ms.evaluate_model_downsampled(model, "SVM with w={0}, c={1} and g={2}".format(w, c, g))


# ms.evaluate_model_kfold_downsampled(SVC(gamma='scale', decision_function_shape='ovo'), "ovo SVM, PCA undersampled", preprocess=pca)
# features = get_lasso_features()
# for g in ['scale', 0.001, 0.002, 0.003]:
#     model = SVC(C=2, gamma=g)
#     np.set_printoptions(precision=3)
#     ms.evaluate_model_kfold_downsampled(model, "SVM gamma={0}".format(g))


def pca_(X_train, X_val, y_train, y_val):
    return pca(X_train, X_val, y_train, y_val, n_samples=1000)
svm = SVC(C=2, gamma='scale')
ms.evaluate_model_kfold_downsampled(svm, "SVM with 670 PCA", preprocess=pca_)

# for name, resampler in {
#     "None": None,
#     "Rand": RandomUnderSampler(),
#     "Casc": BalanceCascade()
#     # "NM-1": NearMiss(version=1),
#     # "NM-2": NearMiss(version=2),
#     # "NM-3": NearMiss(version=3),
#     # "ENN": EditedNearestNeighbours(),
#     # "IHS": InstanceHardnessThreshold(),
#     # "IHS2": InstanceHardnessThreshold(estimator=LogisticRegression(penalty='l1')),
#     # "SMOTEEN": SMOTEENN(),
#     # "SMOTE": SMOTE(),
#     # "ADASYN": ADASYN()
# }.items():
#     ms.evaluate_model(model, name, resampler=resampler)

# resampler = SMOTEENN()
# ms.evaluate_model(model, "SVM with SMOTEEN", resampler=resampler)


# _, _, X_test = data.load()
# X, X_val, y, y_val = data.simple_split(0.3)
# lda = KernelPCA(kernel='rbf', n_components=2)
#
#
# X = lda.fit_transform(X)
# X_val = lda.transform(X_val)
#
# model.fit(X, y)
# h = .02
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
# plt.axis('tight')
# plt.show()


# ms.evaluate_model_kfold_downsampled(SVC(gamma='scale', decision_function_shape='ovo'), "ovo SVM, selected features, undersampled", feature_selection=features)
# ms.evaluate_model_kfold_downsampled(model, "ovo SVM, selected features, undersampled", feature_selection=features)
# ms.evaluate_model_kfold_downsampled(model, "ovo SVM, selected features, PCA, undersampled", feature_selection=features, preprocess=pca)



# _, _, X_test = data.load()
# X_train, X_val, y_train, y_val = data.undersampled_split(0.3)
# X_train = X_train[:, features]
# X_val = X_val[:, features]
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_val = pca.transform(X_val)
# X_test = pca.transform(X_test[:, features])
# # X_train, X_val, y_train, y_val = pca(X_train, X_val, y_train, y_val)
# ms._evaluate(model, X_train, X_val, y_train, y_val)
#
#
#
# y_test_pred = model.predict(X_test)
# out = open('out/ovmSVMLASSOPCAundersampled-TEST2.csv', 'w+')
# out.write('id,y\n')
#
# for i in range(len(y_test_pred)):
#     out.write("%d.0,%f\n" % (i, y_test_pred[i]))
#
# out.close()
# data.generate_output_undersampled("ovm-SVM-LASSOfeatures-PCA-undersampled-corr.csv", model, preprocess=pca, feature_selection=features)
