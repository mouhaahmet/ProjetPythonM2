import pandas as pd

from data_utils import load_data, split_data
from linear_regression import LinearRegression, LinearRegressionLeastSquares, LinearRegressionMedian, \
    LinearRegressionMajority, LinearRegressionMean
import numpy as np
import time

# Charger les données et séparer en données d'entraînement et de validation
X_labeled, y_labeled = load_data('data/YearPredictionMSD_100')[0:2]
X0_train, Y0_train, X0_valid, Y0_valid = split_data(X_labeled, y_labeled, 2 / 3)
# Créer la liste de tailles de données d'entraînement à tester
N = [2 ** p for p in range(5, 11 + 1)]

methode_constant = [LinearRegressionLeastSquares, LinearRegressionMedian,
                    LinearRegressionMajority, LinearRegressionMean]
met = ['LinearRegressionLeastSquares', 'LinearRegressionMedian',
       'LinearRegressionMajority', 'LinearRegressionMean']

# train_error = valid_error = learning_time = {methode_constant[i]: ['col{}_list'.format(i+1)] for i in range(len(methode_constant))}
train_error = []
valid_error = []
learning_time = []
#
# train_error.append(methode_constant)
for j, methode in enumerate(methode_constant):
    model = methode()
    N = [2 ** p for p in range(5, 11 + 1)]
    # train_errore  = difftime = np.zeros_like(N, dtype=list)
    train_errore = []
    difftime = []
    valid_errore = []
    for i, size in enumerate(N):
        # Sélectionner les données d'entraînement , les donnees de validation restent les memes
        X_train = X0_train[:size]
        Y_train = Y0_train[:size]
        # Appliquer le modèle de régression linéaire aux données d'entraînement

        start_time = time.perf_counter()
        model.fit(X_train, Y_train)
        end_time = time.perf_counter()
        difftime.append(end_time - start_time)

        # Prédire les étiquettes de validation à l'aide du modèle entraîné

        y_pred_va = model.predict(X0_valid)
        valid_error_temp = np.sum((Y0_valid - y_pred_va) ** 2) / len(Y0_valid)
        valid_errore.append(valid_error_temp)
        # print("valid error temp {}.".format(valid_error_temp))
        # print(valid_errore[i])

        # Prédire les étiquettes d'entraînement à l'aide du modèle entraîné
        y_pred = model.predict(X_train)
        train_error_temp = np.sum((Y_train - y_pred) ** 2) / len(Y_train)

        # Mettre à jour la valeur dans la matrice avec l'erreur d'entraînement
        train_errore.append(train_error_temp)

    # print(train_errore)
    train_error.append(train_errore)
    valid_error.append(valid_errore)
    # print(valid_error[methode_constant[j]])

    learning_time.append(difftime)

# print(typevalid_error)
# print(pd.DataFrame.from_dict(valid_error, columns=met))


valid_error = pd.DataFrame(data=np.array(valid_error),
                           index=met)
valid_error = valid_error.T
print(valid_error)


train_error = pd.DataFrame(data=np.array(train_error),
                           index=met)
train_error = train_error.T
# print(train_error)


learning_time = pd.DataFrame(data=np.array(learning_time),
                             index=met)
learning_time = learning_time.T

# print(learning_time)


np.savez(f'ErrorAndTime.npz', valid_error=valid_error, train_error=train_error, learning_time=learning_time, N=N,
         methode=met)
