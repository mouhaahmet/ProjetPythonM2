from data_utils import load_data
from linear_regression import LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp
from train_error_regularized import learn_all_with_ridge,learn_all_with_mp,learn_all_with_omp
import numpy as np
def learn_best_predictor_and_predict_test_data():
    X, y, X_test = load_data('YearPredictionMSD_100.npz')
    X_train = X[:500]
    y_train = y[:500]
    X_valid = X[500:]
    y_valid = y[500:]

    # Learn the model parameters using the training set
    model = learn_all_with_ridge(X_train, y_train)
    # Calculate and print the validation error
    y_pred = model.predict(X_valid)
    validation_error = np.mean((y_valid - y_pred) ** 2)
    print(f'Validation error: {validation_error}')

    # Predict the labels for the test data
    y_test = model.predict(X_test)

    # Save the predicted labels to a file
    np.save('Error/test_prediction_results.npy', y_test)
learn_best_predictor_and_predict_test_data()

