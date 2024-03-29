import sys

import numpy
import numpy as np
from os import listdir
import pandas as pd

def load_data(filename):
    """
    Load data examples from a npz file
    Parameters
    ----------
    filename : str
        Name of the npz file containing the data to be loaded
    Returns
    -------
    X_labeled : np.ndarray [n, d]
        Array of n feature vectors with size d
    y_labeled : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    X_unlabeled :
    """
    # TODO À compléter (à la place de l'instruction pass ci-dessous)

    #recuperer tous les fichiers qui sont dans notre répertoire et qui se termine par npy
    #fichier=[file for file in listdir(filename) if file.endswith(".npy")]

    #Verifier le nombre de fichier npy que nous disposons , il doit en avoir 3
    #if (len(fichier)!=3):
    #    raise ValueError("Le nombre de parametre doit etre egale à 3")

    #Charger les données
    try:
        data=np.load('data/'+filename)
        X_labeled = data["X_labeled.npy"]
        X_unlabeled = data["X_unlabeled.npy"]
        y_labeled = data["y_labeled.npy"]
    except OSError:
        print("Could not open/read file:")
        sys.exit()

    return X_labeled,y_labeled,X_unlabeled

# Randomly permute the examples in the labeled set (X, y)
def randomize_data(X, y):
    """
    Randomly permute the examples in the labeled set (X, y), i.e. the rows
    of X and the elements of y, simultaneously.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors

    Returns
    -------
    Xr : np.ndarray [n, d]
        Permuted version of X
    yr : np.ndarray [n]
        Permuted version of y

    Raises
    ------
    ValueError
        If the number of rows in X differs from the number of elements in y.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError('Number of rows in X ({}) differs from the number of '
                         'elements in y'
                         .format(X.shape[0], y.shape[0]))
    # TODO À compléter
    # Stack X and y horizontally and permute the stacked array
    Xy = np.column_stack((X, y))
    Xy_permut = numpy.random.permutation(Xy)

    # Split the permuted array back into X and y
    X_r = Xy_permut[:, :-1]
    y_r = Xy_permut[:, -1]

    return X_r,y_r


# Split a set of labeled examples into two subsets as a random partition
def split_data(X, y, ratio):
    """
    Split a set of n labeled examples into two subsets as a random partition.

    split_data(X, y, ratio) returns a tuple (X1, y1, X2, y2). The n input
    labeled examples (X,y) are randomly permuted and split as a partition
    {(X1, y1), (X2, y2)}. The respective size n1 and n2 is such that
    n1/n approximately equals the input argument `ratio` and n1+n2 = n.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    ratio : float
        Ratio of data to be extracted into (X1, y1)

    Returns
    -------
    X1 : np.ndarray [n1, d]
        Array of n1 feature vectors
    y1 : np.ndarray [n1]
        Vector of n1 label
    X2 : np.ndarray [n2, d]
        Array of n2 feature vectors
    y2 : np.ndarray [n2]
        Vector of n2 labels selected

    """
    # TODO À compléter (à la place de l'instruction pass ci-dessous)
    # Randomly permute the examples
    X_r, y_r = randomize_data(X, y)
    Xy_r =  np.column_stack((X_r, y_r))
    X1y1_r = Xy_r[:int(ratio*len(Xy_r))]
    X2y2_r = Xy_r[int(ratio*len(Xy_r)):]


    # Extract the data
    X1 = X1y1_r[:, :-1]
    y1 = X1y1_r[:, -1]
    X2 = X2y2_r[:, :-1]
    y2 = X2y2_r[:, -1]

    return X1,y1,X2,y2