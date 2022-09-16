import numpy as np
from data_utils import load_data


""" Load training data and print dimensions as well as a few coefficients
in the first and last places and at random locations.
"""
# TODO À compléter

X_labeled,y_labeled,X_unlabeled=load_data('/home/lamine/Master2/ProgrammationPython/ProjetPythonM2/code_a_completer/data/YearPredictionMSD_100')
for i in (X_labeled,y_labeled,X_unlabeled):
    print("type:"+str(type(i)))
    print("nombre de dimension:"+str(i.ndim))
    print("shape:"+str(i.shape))

for i in (X_labeled, X_unlabeled):
    #2 premiers coef premier ligne et de la derniere ligne
    print(i[0, 0:2])
    print(i[-1, 0:2])
    #dernier coef premier ligne et dernier ligne
    print(i[0, -1])
    print(i[-1, -1])

#5 premiers et 5 derniers elements de y_labelled
y_labeled[:5]
y_labeled[-5:]

# affichez les 2 premiers coefficients de la première ligne et de la dernière ligne de chaque matrice




