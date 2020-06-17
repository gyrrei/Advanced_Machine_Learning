import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from imblearn.under_sampling import TomekLinks, RandomUnderSampler

import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import data

def visualise_2d(undersample=False):
    X, y, _ = data.load()
    if undersample:
        resampler = RandomUnderSampler()
        X, y = resampler.fit_resample(X,  y)

    np.random.seed(5)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    X = lda.transform(X)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def visualise_3d():
    resampler = RandomUnderSampler()
    X, y, X_test = data.load()

    X, y = resampler.fit_resample(X, y)

    print(y.shape)
    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]


    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=-170)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    print(X.shape)
    print((y == 1).shape)
    print(X)

    for name, label in [('Cat 0', 0), ('Cat 1', 1), ('Cat 2', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    # plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.show()




def visualise_minorities():
    X, X_, y, y_ = data.minorities_split()

    np.random.seed(5)

    pca = PCA(n_components=2)
    pca.fit(X, y)
    X = pca.transform(X)
    X_ = pca.transform(X_)

    plt.figure()
    plt.title("Trained PCA")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.scatter(X[:, 0], y)
    plt.figure()
    plt.title("Validation")
    plt.scatter(X_[:, 0], X_[:, 1], c=y_)
    # plt.scatter(X_[:, 0], y_)
    plt.show()


if __name__ == "__main__":
    visualise_minorities()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

