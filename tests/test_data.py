import numpy as np
import os
import pandas as pd

def generate_synthetic_data(n_samples=100, case='linear', random_seed=42):
    np.random.seed(random_seed)
    if case == 'linear':
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    elif case == 'circle':
        radius = np.random.uniform(0, 1.5, n_samples)
        angle = np.random.uniform(0, 2 * np.pi, n_samples)
        X = np.vstack((radius * np.cos(angle), radius * np.sin(angle))).T
        y = (radius > 0.7).astype(int)
    elif case == 'imbalanced':
        X = np.random.randn(n_samples, 2)
        y = np.zeros(n_samples, dtype=int)
        n_positive = int(0.2 * n_samples)
        indices = np.random.choice(n_samples, n_positive, replace=False)
        y[indices] = 1
        X[indices] += np.array([2, 2])  
    else:
        raise ValueError("Unknown case")
    return X, y

os.makedirs('data', exist_ok=True)

# Define test cases based on your unittest
cases = [
    ('linear', 100),    
    ('circle', 100),    
    ('imbalanced', 50), 
    ('linear', 10),     
]

# Generate and save data for each test case
for case, n_samples in cases:
    X, y = generate_synthetic_data(n_samples=n_samples, case=case, random_seed=42)
    
    # Convert X to DataFrame with column names 'feature_0' and 'feature_1'
    X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
    # Convert y to DataFrame with column name 'label'
    y_df = pd.DataFrame(y, columns=['label'])
    
    # Define file paths
    X_path = os.path.join('data', f'X_{case}_{n_samples}.csv')
    y_path = os.path.join('data', f'y_{case}_{n_samples}.csv')
    
    # Save to CSV
    X_df.to_csv(X_path, index=False)
    y_df.to_csv(y_path, index=False)
    
    print(f"Saved data for case '{case}' with {n_samples} samples to {X_path} and {y_path}")