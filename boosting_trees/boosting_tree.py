import numpy as np
from .decision_tree import DecisionTree

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_log_odds = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss_gradient(self, y, p):
        """Compute residuals (negative gradient of log-loss)."""
        return y - p

    def fit(self, X, y):
        """Fit the gradient-boosting model."""
        # Initialize with log-odds
        p = np.mean(y)
        self.initial_log_odds = np.log(p / (1 - p + 1e-10))  # Avoid division by zero
        predictions = np.full_like(y, self.initial_log_odds, dtype=float)

        # Boosting iterations
        for _ in range(self.n_estimators):
            probs = self._sigmoid(predictions)
            residuals = self._log_loss_gradient(y, probs)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        predictions = np.full(X.shape[0], self.initial_log_odds)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        probs = self._sigmoid(predictions)
        return np.vstack((1 - probs, probs)).T

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)