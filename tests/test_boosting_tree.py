import unittest
import numpy as np
import pandas as pd
import os
from boosting_trees.boosting_tree import GradientBoostingClassifier

class TestGradientBoostingClassifier(unittest.TestCase):
    def _load_data(self, case, n_samples):
        """Helper method to load X and y from CSV files in the data folder."""
        data_dir = "data"
        X_path = os.path.join(data_dir, f"X_{case}_{n_samples}.csv")
        y_path = os.path.join(data_dir, f"y_{case}_{n_samples}.csv")
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Data files for case '{case}' with {n_samples} samples not found in {data_dir}")
        
        X = pd.read_csv(X_path).to_numpy()
        y = pd.read_csv(y_path)['label'].to_numpy()
        return X, y

    def test_fit_predict_linear(self):
        X, y = self._load_data(case='linear', n_samples=100)
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        self.assertGreater(accuracy, 0.9, "Accuracy on linear data should be high")

    def test_fit_predict_circle(self):
        X, y = self._load_data(case='circle', n_samples=100)
        model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, max_depth=4)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        self.assertGreater(accuracy, 0.8, "Accuracy on circle data should be reasonable")

    def test_predict_proba_range(self):
        X, y = self._load_data(case='imbalanced', n_samples=50)
        model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        probs = model.predict_proba(X)
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]")

    def test_small_dataset(self):
        X, y = self._load_data(case='linear', n_samples=10)
        model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y), "Prediction length should match input")

if __name__ == '__main__':
    unittest.main()