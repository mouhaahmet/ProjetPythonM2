from data_utils import load_data
from linear_regression import LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp

def learn_best_predictor_and_predict_test_data(X, y, X_test, algo):
    X, y, X_test = load_data('data/YearPredictionMSD_100')
    X_train = X[:500]
    y_train = y[:500]
    X_valid = X[500:]
    y_valid = y[500:]

    # Learn the model parameters using the training set
    model = learn_all_with_A(X_train, y_train, algo)

    # Calculate and print the validation error
    validation_prediction = model.predict(X_valid)
    validation_error = np.mean((y_valid - validation_prediction) ** 2)
    print(f'Validation error: {validation_error}')

    # Load the test data
    X_test, _, _, _ = load_data('YearPredictionMSD_100.npz', True)

    # Predict the labels for the test data
    y_test = model.predict(X_test)

    # Save the predicted labels to a file
    np.save('test_prediction_results.npy', y_test)
