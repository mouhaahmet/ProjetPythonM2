import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data, split_data, randomize_data
from linear_regression import LinearRegressionRidge

# Séparation des données en un ensemble d'apprentissage et un ensemble de validation
X, y = load_data('YearPredictionMSD_100.npz')[0:2]

X_labeled = X[:500]
y_labeled = y[:500]
X0_train, y0_train, X0_valid, y0_valid = split_data(X_labeled, y_labeled, 2 / 3)
# Définition de l'ensemble de valeurs à tester pour l'hyperparamètre de régression d'arrête (coefficient de pénalisation lambda)
lambdas = np.logspace(-4, 4, 9)

# Tableaux pour stocker les erreurs sur l'ensemble d'apprentissage et sur l'ensemble de validation
train_errors = []
val_errors = []

# Pour chaque valeur de lambda possible
for lambda_ridge in lambdas:
    # Appliquer l'algorithme de régression rifge en utilisant cette valeur de lambda
    model = LinearRegressionRidge(lambda_ridge=lambda_ridge)

    model.fit(X0_train, y0_train)

    # Calcul de l'erreur sur l'ensemble d'apprentissage et sur l'ensemble de validation
    train_pred = model.predict(X0_train)
    val_pred = model.predict(X0_valid)
    train_errors.append(np.mean((y0_train - train_pred)**2))
    val_errors.append(np.mean((y0_valid - val_pred)**2))

# Tracer les erreurs en fonction de la valeur de lambda
plt.plot(lambdas, train_errors, label="Erreur sur l'ensemble d'apprentissage")
plt.plot(lambdas, val_errors, label="Erreur sur l'ensemble de validation")
plt.xlabel("Valeur de lambda")
plt.ylabel("Erreur")
plt.legend()
plt.savefig('SortieFigure/Erreur_lambdas')
plt.show()

