import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 or np.var(y) == 0:
            return np.mean(y)  # Leaf node: mean of residuals
        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return np.mean(y)
        mask = X[:, feature] <= threshold
        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return (feature, threshold, left, right)

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                if np.sum(mask) < 1 or np.sum(~mask) < 1:
                    continue
                mse = self._compute_mse(y[mask], y[~mask])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _compute_mse(self, left_y, right_y):
        left_var = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
        right_var = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
        total_count = len(left_y) + len(right_y)
        return (left_var + right_var) / total_count if total_count > 0 else float('inf')

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self._predict_one(x, left)
        return self._predict_one(x, right)