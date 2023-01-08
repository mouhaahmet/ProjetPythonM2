import pandas as pd

from data_utils import load_data, split_data
from linear_regression import LinearRegression,LinearRegressionLeastSquares,LinearRegressionMedian,\
    LinearRegressionMajority,LinearRegressionMean
import numpy as np
import time



#LinearRegressionLeastSquares

# Charger les données et séparer en données d'entraînement et de validation
X_labeled, y_labeled = load_data('YearPredictionMSD_100.npz')[0:2]
X0_train, Y0_train, X0_valid, Y0_valid = split_data(X_labeled, y_labeled, 2/3)
# Créer la liste de tailles de données d'entraînement à tester
N = [2**p for p in range(5, 11+1)]
train_errore = valid_errore = np.zeros(len(N))

model = LinearRegressionLeastSquares()
train_error_LeastSquare = valid_errore_leastSquare = [[] for _ in range(1)]
for i, size in enumerate(N):
    # Sélectionner les données d'entraînement , les donnees de validation restent les memes
    X_train = X0_train[:size]
    Y_train = Y0_train[:size]
    # Appliquer le modèle de régression linéaire aux données d'entraînement
    model.fit(X_train, Y_train)

    # Prédire les étiquettes de validation à l'aide du modèle entraîné
    y_pred_va = model.predict(X0_valid)
    valid_error_temp = np.sum((Y0_valid - y_pred_va) ** 2) / len(Y0_valid)
    print(valid_error_temp)

    # Prédire les étiquettes d'entraînement à l'aide du modèle entraîné
    y_pred = model.predict(X_train)
    train_error_temp = np.sum((Y_train - y_pred) ** 2) / len(Y_train)


    # Mettre à jour la valeur dans la matrice avec l'erreur d'entraînement
    train_errore[i] = train_error_temp
    #print(size)
    #print(train_error_temp)


    valid_errore[i] = valid_error_temp
    print(valid_error_temp)
#train_error_LeastSquare[0].append(f"train_errore")
#train_error_LeastSquare[0].append(eval(train_error_LeastSquare[0][0]))



    #np.savez(f'errorLeastSquare' + str(size), valid_error=valid_error, train_error=train_error, N=N)
print(valid_errore)


