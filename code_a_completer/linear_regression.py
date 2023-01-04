#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np
from algorithms import mp, omp, ridge_regression, normalize_dictionary


class LinearRegression:
    """
    Generic class for linear regression

    Two attributes ``w`` and ``b`` are used for a linear regression of the form
    f(x) = w.x + b

    All linear regression method should inherit from :class:`LinearRegression`,
    and implement the ``fit`` method to learn ``w`` and ``b``. Method
    ``predict`` implemented in :class:`LinearRegression` will be inherited
    without needed to be reimplemented.
    """
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        This method should be used to learn w and b in any subclass. Here,
        it is just checking the parameters' properties (call this parent's
        method in any subclass).

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        assert X.ndim == 2
        n_samples, n_features = X.shape
        assert y.ndim == 1
        assert y.size == n_samples

        self.w = None
        self.b = None

    def predict(self, X):
        """
        Predict labels from feature vectors via a linear function.

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n feature vectors with size d

        Returns
        -------
        y : np.ndarray [n]
            Vector of n predicted labels for X
        """
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[1] == self.w.shape[0]

        return X @ self.w + self.b


class LinearRegressionLeastSquares(LinearRegression):
    """
    Linear regression using a least-squares method
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn parameters using a least-square method

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter
        X_tild = np.hstack([X, np.ones((X.shape[0], 1))])
        X_pinv = np.linalg.pinv(X_tild)

        # Compute the dot product of Xpinv and y using np.dot
        w_tild = np.dot(X_pinv, y)
        w = w_tild[:-1]
        b = w_tild[-1]

        self.w=w
        self.b=b

class LinearRegressionMean(LinearRegression):
    """
    Constant-valued regression function equal to the mean of the training
    labels
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the mean of the training labels

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter
        w = np.zeros(X.shape[1])
        b = np.mean(y)
        self.w = w
        self.b = b


class LinearRegressionMedian(LinearRegression):
    """
    Constant-valued regression function equal to the median of the training
    labels
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the median of the training labels

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter
        w = np.zeros(X.shape[1])
        b = np.median(y)
        self.w = w
        self.b = b


class LinearRegressionMajority(LinearRegression):
    """
    Constant-valued regression function equal to the majority training label
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the majority training label

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        h, bins = np.histogram(y, bins=np.arange(np.min(y), np.max(y) + 2))

        # TODO À compléter
        w = np.zeros(X.shape[1])
        b, a = np.histogram(y,bins=np.arange(np.min(y), np.max(y) + 2))
        b = a[np.argmax(b)]
        self.w = w
        self.b = b



class LinearRegressionRidge(LinearRegression):
    """
    Linear regression based on ridge regression

    Parameters
    ----------
    lambda_ridge : float
        Non-negative penalty coefficient
    """
    def __init__(self, lambda_ridge):
        LinearRegression.__init__(self)
        self.lambda_ridge = lambda_ridge

    def fit(self, X, y):
        """
        Learn linear function using ridge regression

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using a feature matrix
        normalized with l2-norm columns

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter
        b = sum(y) / len(y)

        # Pré-traitement des données
        y_centered = y - b
        X_normalized, alpha = normalize_dictionary(X)

        # Appliquer l'algorithme MP, OMP ou ridge
        w_tilde = ridge_regression(X_normalized, y_centered, self.lambda_ridge)

        # Post-traitement du résultat
        #w = [w_tilde[m]/alpha[m] for m in range(X.shape[1])]
        w = np.array([w_tilde[m] / alpha[m] for m in range(X.shape[1])])

        self.w = w
        self.b = b


class LinearRegressionMp(LinearRegression):
    """
    Linear regression based on Matching Pursuit

    Parameters
    ----------
    n_iter : int
        Number of iterations
    """
    def __init__(self, n_iter):
        LinearRegression.__init__(self)
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Learn linear function using Matching Pursuit (MP)

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using MP,
        after normalizing the dictionary ``X``

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d (may not be
            normalized)
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter


class LinearRegressionOmp(LinearRegression):
    """
    Linear regression based on Matching Pursuit

    Parameters
    ----------
    n_iter : int
        Number of iterations
    """
    def __init__(self, n_iter):
        LinearRegression.__init__(self)
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Learn linear function using Orthogonal Matching Pursuit (MP)

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using MP,
        after normalizing the dictionary ``X``

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d (may not be
            normalized)
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        # TODO À compléter
