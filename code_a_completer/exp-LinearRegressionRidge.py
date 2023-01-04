import numpy as np
from linear_regression import LinearRegressionRidge
import unittest
class TestLinearRegressionRidge(unittest.TestCase):
    def test_fit(self):
        # Initialize a LinearRegressionRidge model
        model = LinearRegressionRidge(lambda_ridge=2)

        # Generate synthetic training data
        X = np.random.rand(100, 20)
        w_true = np.random.rand(20)
        y = X.dot(w_true) + np.random.randn(100)

        # Fit the model on the training data
        w = model.fit(X, y)

        # Check that the shape of w is correct
        self.assertEqual(w.shape, (20,))

        # Check that the model has learned a good approximation of w_true
        self.assertLess(np.linalg.norm(w - w_true), 1e-2)

