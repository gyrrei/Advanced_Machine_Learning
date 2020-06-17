import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity


def pdf(x, Mu, Sig, k):
    """
    First of all: I apologise for this monstrosity. I am very aware that I could have used the multivariate.pdf()
    method repeatedly, however I wanted to take on the challenge and solve it using matrix multiplication etc. in
    numpy.
    My (brief) tests have shown that this is in fact significantly faster than using the scipy function in a loop,
    and about twice as fast as using the numpy functions k-times in a loop, so at least in some perspective, it
    was worth it.
    :param x: the data matrix with the points to evaluate, shape: [m, nbFeatures]
    :param Mu: the data matrix with the centers of the gaussian, shape: [k, nbFeatures]
    :param Sig: a list of lenghth k with the covariance matrices of the gaussian, each of shape [nbFeatures, nbFeatures]
    :return: A matrix of shape [m, k], where the [i,j]-th entry corresponds to the probability of the i-th data point
             under the j-th centers and covariance matrix
    """
    nbFeatures = x.shape[1]
    m = x.shape[0]

    # The idea is to first make an (m x k) matrix where the [i,j]th entry is (x[i] - mu[j])^T @ sigma_j^-1 @ (x[i] - mu[j])
    # With that, we can then apply numpy's exp function component-wise

    # making an array out of sigma to use numpy's awesome POWERS and calculate the inverses of all the matrices
    # Since Sigma is now a 3D-array, this will be done component-wise
    Sigma = np.array(Sig)
    Sigma_inv = np.linalg.inv(Sigma)  # shape: [k, nbFeatures, nbFeatures], Sigma_inv[j] == Sigma[j]^-1

    # Here we build up a 3d-matrix where the [i,j]th entry is x[i] - Mu[j]. To that end, we first repeat the matrices
    xr = np.repeat(np.reshape(x, [1, m, nbFeatures]), k, axis=0)
    Mur = np.repeat(np.reshape(Mu, [k, 1, nbFeatures]), m, axis=1)
    Diff = xr - Mur  # shape: [k, m, nbFeatures]

    # We first perform the right part of the multplication to get an array where each [i,j]th entry is
    # sigma_j^-1 @ (x[i] - mu[j]). We reshape the array such that these entries are column vectors of the shape
    # [nbFeatures, 1]. Unfortunately, that means we have a 4d array now :D
    sigmaTimesDiff = np.matmul(Diff, Sigma_inv)
    sigmaTimesDiff = np.reshape(sigmaTimesDiff, [k, m, nbFeatures, 1])

    # We also reshape Diff such that its entries are row vectors of the shape [1, nbFeatures], since we want their
    # transpose in the original formula
    Diff = np.reshape(Diff, [k, m, 1, nbFeatures])

    # This array now has entries corresponding to (x[i] - mu[j])^T @ sigma_j^-1 @ (x[i] - mu[j])
    # Since its entries are scalars now, we reshape it into the sensible shape of [k,m]
    diffTimesSigmaTimesDiff = np.matmul(Diff, sigmaTimesDiff)
    diffTimesSigmaTimesDiff = np.reshape(diffTimesSigmaTimesDiff, [k, m])

    # Now we apply the exponential function with a scaling of -0.5 as per the definition of the gaussian pdf
    num = np.exp(-0.5 * diffTimesSigmaTimesDiff).T

    # The denominator of the pdf is this. numpy's det function works component-wise, so that this gives a vector
    # of length k and we can then divide component wise
    denom = np.sqrt((2 * np.pi) ** nbFeatures * np.linalg.det(Sigma))

    return num / denom  # shape [k,m]


class DensityEstimationModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.ratios = np.array([np.sum(y == c) / len(y) for c in self.classes])
        self.C = len(self.classes)

        self.selector = SelectKBest(k=100)
        X = self.selector.fit_transform(X, y)

        self.densities = []
        for c in self.classes:
            kde = KernelDensity()
            kde.fit(X)
            self.densities.append(kde)


    def predict(self, X):
        X = self.selector.transform(X)
        probs = np.array([np.exp(kde.score_samples(X)) for kde in self.densities])
        return np.argmax(probs, axis=0)
        # ret = []
        # for x in X:
        #     probs = [multivariate_normal.pdf(x, self.means[c], self.vars[c]) for c in self.classes]
        #     ret.append(np.argmax(np.array(probs)))
        # return ret


class GaussianGenerativeModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.ratios = np.array([np.sum(y == c) / len(y) for c in self.classes])
        self.C = len(self.classes)
        mus = []
        sigmas = []

        self.selector = SelectKBest(k=100)
        X = self.selector.fit_transform(X, y)

        for c in self.classes:
            mus.append(np.mean(X[y == c], axis=0))
            sigmas.append(np.cov(X[y == c].T))

        mus = np.array(mus)

        self.means = mus
        self.vars = sigmas

    def predict(self, X):
        X = self.selector.transform(X)
        probs = pdf(X, self.means, self.vars, self.C)
        return np.argmax(probs, axis=1)
        # ret = []
        # for x in X:
        #     probs = [multivariate_normal.pdf(x, self.means[c], self.vars[c]) for c in self.classes]
        #     ret.append(np.argmax(np.array(probs)))
        # return ret