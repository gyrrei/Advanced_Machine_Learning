import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import data
import model_selection as ms

# X_train, X_val, y_train, y_val = data.undersampled_split(random_state=42)
# svm = SVC(gamma='scale', decision_function_shape='ovo')

# lda = LinearDiscriminantAnalysis(n_components=2)
# X_train = lda.fit_transform(X_train, y_train)
# X_val = lda.transform(X_val)




def pca(X_train, X_val, y_train, y_val, n_samples=None):
    pca = PCA(n_components=n_samples)
    X_train_ = pca.fit_transform(X_train)
    X_val_ = pca.transform(X_val)
    return X_train_, X_val_, y_train, y_val


# ms.evaluate_model_kfold_downsampled(svm, "SVM + PCA", preprocess=pca)

def norm(X_train, X_val, y_train, y_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val

# ms.evaluate_model_kfold_downsampled(svm, "SVM + Norm", preprocess=norm)

def normprep(X_train, X_val, y_train, y_val):
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    return X_train, X_val, y_train, y_val

# ms.evaluate_model_kfold_downsampled(svm, "SVM + PCA +  Norm", preprocess=normprep)

# ms.evaluate_model_kfold_downsampled(svm, "SVM", preprocess=None)
# data.generate_output_undersampled("svm-ovo + pca.csv", svm, preprocess=prep)

if __name__ == "__main__":
    pca = PCA()
    X_train, _, _, _ = data.undersampled_split()
    pca.fit(X_train)
    np.set_printoptions(precision=3, suppress=True)
    print(pca.explained_variance_)
    print(np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.99))