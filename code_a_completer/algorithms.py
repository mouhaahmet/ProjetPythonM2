#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np


def normalize_dictionary(X):
    """
    Normalize matrix to have unit l2-norm columns

    Parameters
    ----------
    X : np.ndarray [n, d]
        Matrix to be normalized

    Returns
    -------
    X_normalized : np.ndarray [n, d]
        Normalized matrix
    norm_coefs : np.ndarray [d]
        Normalization coefficients (i.e., l2-norm of each column of ``X``)
    """
    # Check arguments
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    # TODO À compléter

    alpha = np.linalg.norm(X, axis=0)
    X_normalized = X / alpha
    return X_normalized, alpha





def ridge_regression(X, y, lambda_ridge):
    """
    Ridge regression estimation

    Minimize $\left\| X w - y\right\|_2^2 + \lambda \left\|w\right\|_2^2$
    with respect to vector $w$, for $\lambda > 0$ given a matrix $X$ and a
    vector $y$.

    Note that no constant term is added.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Data matrix composed of ``n`` training examples in dimension ``d``
    y : np.ndarray [n]
        Labels of the ``n`` training examples
    lambda_ridge : float
        Non-negative penalty coefficient

    Returns
    -------
    w : np.ndarray [d]
        Estimated weight vector
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter

    # Calculate the weight vector using the formula
    # w = (XT X + lambda_ridge IM)^-1 XT y
    XT = X.T
    w = np.linalg.inv(XT @ X + lambda_ridge * np.eye(X.shape[1])) @ XT @ y
    return w





def mp(X, y, n_iter):
    """
    Matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter
    # Initialiser le vecteur de poids comme un vecteur de zéros de taille M
    w = np.zeros(X.shape[1])
    error_norm = np.zeros(n_iter+1)
    # Initialiser le résidu à y
    r = y
    # Initialiser le vecteur error_norm avec l'énergie du résidu au début de l'algorithme
    # Répéter pendant un maximum de n_iter itérations
    error_norm[0] = (np.linalg.norm(r) )
    for k in range(1, n_iter+1):
        # Calculer les corrélations entre le résidu et chaque caractéristique dans X
        correlations = [np.dot(X[:, m], r) for m in range(X.shape[1])]
        # Sélectionner la caractéristique avec la plus forte corrélation
        m_hat = np.argmax(np.abs(correlations))
        # Mettre à jour le vecteur de poids
        w[m_hat] += correlations[m_hat]
        # Mettre à jour le résidu
        r -= correlations[m_hat] * X[:, m_hat]
        error_norm[k]=(np.linalg.norm(r))
        # Ajouter l'énergie du résidu à la fin de l'itération au vecteur error_norm
    # Renvoyer le vecteur de poids parcimonieux et le vecteur error_norm
    return w, error_norm



def omp(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter

    w = np.zeros(X.shape[1])
    # Initialiser le résidu à y
    r = y
    error_norm = np.zeros(n_iter+1)
    # Initialize the support
    support = set()
    error_norm[0] = (np.linalg.norm(r) ** 2)
    for k in range(1, n_iter+1):
        correlations = [np.dot(X[:, m], r) for m in range(X.shape[1])]
        #correlations = X.T@r
        # Sélectionner la caractéristique avec la plus forte corrélation
        m_hat = np.argmax(np.abs(correlations))
        # Update the support
        support.add(m_hat)
        # Update the decomposition
        w[list(support)] = np.linalg.pinv(X[:, list(support)]) @ y

        # Update the residual signal
        r = y - np.dot(X[:, list(support)], w[list(support)])
        error_norm[k]=(np.linalg.norm(r) ** 2)
    return w,error_norm



