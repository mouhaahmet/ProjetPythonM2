import time

from data_utils import split_data,load_data
from linear_regression import LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp
import numpy as np
import pandas as pd

def learn_all_with_ridge(X, y):
        # generate a set of possible values for the hyperparameter n_iter
        #lambda_iter_values = np.logspace(-5, 5, 20)
        lambda_iter_values = np.logspace(-5, 5, 20)

        # create a list to store the model objects
        models = []

        # divise data
        X_train, y_train, X_valid, y_valid = split_data(X, y, 2 / 3)
        # train a model for each value of n_iter
        for n_iter in lambda_iter_values:
            model = LinearRegressionRidge(n_iter)
            model.fit(X_train, y_train)
            models.append(model)

        # select the optimal value of n_iter based on the performance on the validation set
        best_n_iter = select_optimal_hyperparameter(models, X_valid, y_valid)
        # model final
        model_final = LinearRegressionRidge(best_n_iter)
        model_final.fit(X, y)

        # return the model object initialized with the optimal value of n_iter
        return model_final


def learn_all_with_mp(X, y):
        # generate a set of possible values for the hyperparameter n_iter
        lambda_iter_values = np.arange(0, 20, dtype=int)
        # create a list to store the model objects
        models = []

        # divise data
        X_train, y_train, X_valid, y_valid = split_data(X, y, 2 / 3)
        # train a model for each value of n_iter
        for n_iter in lambda_iter_values:
            model = LinearRegressionMp(n_iter)
            model.fit(X_train, y_train)
            models.append(model)

        # select the optimal value of n_iter based on the performance on the validation set
        best_n_iter = select_optimal_hyperparameter(models, X_valid, y_valid)
        # model final
        model_final = LinearRegressionMp(best_n_iter)
        model_final.fit(X, y)

        # return the model object initialized with the optimal value of n_iter
        return model_final


def learn_all_with_omp(X, y):
        # generate a set of possible values for the hyperparameter n_iter
        lambda_iter_values = np.arange(0, 20, dtype=int)

        # create a list to store the model objects
        models = []

        # divise data
        X_train, y_train, X_valid, y_valid = split_data(X, y, 2 / 3)
        # train a model for each value of n_iter
        for n_iter in lambda_iter_values:
            model = LinearRegressionOmp(n_iter)
            model.fit(X_train, y_train)
            models.append(model)

        # select the optimal value of n_iter based on the performance on the validation set
        best_n_iter = select_optimal_hyperparameter(models, X_valid, y_valid)
        # model final
        model_final = LinearRegressionOmp(best_n_iter)
        model_final.fit(X, y)

        # return the model object initialized with the optimal value of n_iter
        return model_final


def select_optimal_hyperparameter(models, X, y):
    # initialize a dictionary to store the performance of each model
    performance = {}

    # evaluate the performance of each model on the validation set
    for model in models:
        print(str(type(model)))
        prediction = model.predict(X)
        error = np.mean((y - prediction) ** 2)
        if isinstance(model, LinearRegressionRidge):
            performance[model.lambda_ridge] = error
        else:
            performance[model.n_iter] = error



    # return the hyperparameter value associated with the best performing model
    return min(performance, key=performance.get)

# Charger les données et séparer en données d'entraînement et de validation
X_labeled, y_labeled = load_data('data/YearPredictionMSD_100')[0:2]
X0_train, Y0_train, X0_valid, Y0_valid = split_data(X_labeled, y_labeled, 2 / 3)
# Créer la liste de tailles de données d'entraînement à tester
N = [2 ** p for p in range(5, 11 + 1)]


#model_ridge = learn_all_with_ridge(X0_train, Y0_train)
#model_mp = learn_all_with_mp(X0_train, Y0_train)
#model_omp = learn_all_with_omp(X0_train, Y0_train)


methode_regularized = [learn_all_with_ridge,learn_all_with_mp,learn_all_with_omp]
#methode_regularized = [learn_all_with_ridge]

met = ["LinearRegressionRidge","LinearRegressionMp","LinearRegressionOmp"]

train_error = []
valid_error = []
learning_time = []
#
# train_error.append(methode_constant)
for j, methode in enumerate(methode_regularized):
    model = methode(X0_train, Y0_train)
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


valid_error = pd.DataFrame(data=np.array(valid_error),index=met)
valid_error = valid_error.T
print(valid_error)
""""""
""""""""
train_error = pd.DataFrame(data=np.array(train_error),
                           index=met)
train_error = train_error.T
# print(train_error)


learning_time = pd.DataFrame(data=np.array(learning_time),
                             index=met)
learning_time = learning_time.T

# print(learning_time)


np.savez(f'ErrorAndTime_Regularized.npz', valid_error=valid_error, train_error=train_error, learning_time=learning_time, N=N,
         methode=met)








