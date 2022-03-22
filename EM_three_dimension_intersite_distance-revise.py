# encoding=utf-8
'''
 scikit-learn 0.22.1

'''

import numpy as np
from sklearn.mixture.base import BaseMixture, _check_shape
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import logsumexp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
from sklearn.mixture import GaussianMixture
from scipy.stats import norm



def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def intersite_3D(mu, ss, x):
    return (np.sqrt(2 / np.pi / ss)
            * x / mu
            * np.exp(-(mu ** 2 + x ** 2) / 2 / ss)
            * np.sinh(x * mu / ss))

def gaussian(mu, ss, x):
    return 1/(np.sqrt(2*np.pi)*ss)*np.exp(-(x-mu)*(x-mu)/(2*ss*ss))

    

class DistanceMixture(BaseMixture):
    # init_params can be set as either "kmeans" for K-means initialization, or "order" for "order" initialization.
    def __init__(self, p = intersite_3D, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='order',
                 minimize_method='L-BFGS-B',
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.p = p
        self.minimize_method = minimize_method
        self.weights_ = None

    def _initialize_parameters(self, X, random_state):
        nc = self.n_components
        sX = np.sort(X)
        step = len(X) // nc
        mX = sX[:nc * step].reshape(nc, step)
        self.means_ = np.mean(mX, axis=1)
        self.covariances_ = np.var(mX, axis=1)

        if self.init_params == 'order':
            self.weights_ = np.ones(nc) / nc
            _, log_resp = self._estimate_log_prob_resp(X)
            resp = np.exp(log_resp)
            self._initialize(X, resp)
        else:
            super()._initialize_parameters(X, random_state)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            self._estimate_parameters(X, np.exp(log_resp)))
        self.weights_ /= n_samples


    def _estimate_log_prob(self, X):
        prob = self.p(self.means_, self.covariances_, X)
        return np.log(prob)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_'])

    def _check_parameters(self, X):
        pass

    def _initialize(self, X, resp):
        """Initialization of the Distance_Mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_parameters(X, resp)
        weights /= n_samples

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _get_parameters(self):
        return self.weights_, self.means_, self.covariances_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covariances_ = params

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _estimate_parameters(self, X, resp):#, weight=None, x0=None):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        if self.weights_ is None:
            self.weights_ = nk / X.shape[0]
        nc = resp.shape[1]
        eps = 1e-6
        bounds = [(eps, None) for _ in range(nc * 2)]

        def Q(theta):
            means = theta[:nc]
            covariances = theta[nc:]
            return -(np.dot(nk, np.log(self.weights_))
                     + sum([np.dot(resp[:, j], np.log(self.p(means[j], covariances[j], X)))
                                 for j in range(nc)]))

        x0 = np.concatenate((self.means_, self.covariances_))
        res = minimize(Q, x0, method=self.minimize_method, bounds=bounds)
        means = res.x[:nc]
        covariances = res.x[nc:]

        return nk, means, covariances

    def plot(self, X, file_path ,x_label = '3D inter-site distance', y_label = 'Density'):
        plt.hist(X, density = True)
        t = np.arange(0, max(X), max(X)/ len(X))    
        total = np.zeros(len(t))
        for i in range(len(self.means_)):
            plt.plot(t, self.weights_[i]*self.p(self.means_[i], self.covariances_[i], t), 'r--')
            total += self.weights_[i]*self.p(self.means_[i], self.covariances_[i], t)
        plt.plot(t, total, 'k')
        plt.ylabel(y_label, fontsize=20)
        plt.xlabel(x_label, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(file_path,bbox_inches = 'tight')
        plt.show()

    def score(self, X):
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1).mean()

    def _n_parameters(self):
        return self.n_components

    def bic(self, X):
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def summary(self,X):
        d_e = np.sort(np.exp(self.means_))
        cov = np.sort(self.covariances_)
        result = {}
        result['mu'] = d_e
        result['bic'] = self.bic(X)
        result['aic'] = self.aic(X)
        result['weights'] = self.weights_
        result['variances'] = cov
        result['score'] = self.score(X)
        return result

