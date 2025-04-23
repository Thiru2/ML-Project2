import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boosting_trees.boosting_tree import GradientBoostingClassifier

# Load data from CSV files
def load_data(case, n_samples):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_path = os.path.join(data_dir, f"X_{case}_{n_samples}.csv")
    y_path = os.path.join(data_dir, f"y_{case}_{n_samples}.csv")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files for case '{case}' with {n_samples} samples not found in {data_dir}")
    
    X = pd.read_csv(X_path).to_numpy()
    y = pd.read_csv(y_path)['label'].to_numpy()
    return X, y

# Load circle data with 100 samples
X, y = load_data(case='circle', n_samples=100)

# Train model
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=4)
model.fit(X, y)

# Predict
preds = model.predict(X)
probs = model.predict_proba(X)

# Print results
accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy:.3f}")
print(f"Sample probabilities (first 5):\n{probs[:5]}")