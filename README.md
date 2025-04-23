#              Team Members
#    THIRUMALESH KURUKUNDHA  -- A20561775
#    MAHENDRA REDDY KADIRE   -- A20549975
#    KARTHIK KALLURI         -- A20552245


# Gradient-Boosting Tree Classifier

This project implements a binary gradient-boosting tree classifier from first principles, following Sections 10.9–10.10 of *Elements of Statistical Learning* (2nd Edition). It uses a custom decision tree regressor to fit residuals and supports a fit-predict interface. No external machine learning libraries (e.g., scikit-learn) are used.

## Questions

### 1. What does the model do, and when should it be used?

The model builds an ensemble of decision trees to predict binary class labels or probabilities. Each tree fits the negative gradient of the log-loss, and predictions are combined with a learning rate to produce final outputs via a sigmoid function.

**Use cases**:
- Classification tasks with tabular data, especially for non-linear relationships.
- Robust prediction in noisy or imbalanced datasets (e.g., fraud detection, medical diagnosis).
- Scenarios prioritizing accuracy over interpretability.

### 2. How did you test your model to determine if it is working reasonably correctly?

The model was tested with synthetic datasets:
- **Linearly separable data**: Points separated by a line to test basic classification.
- **Circular data**: Non-linear boundaries to verify complex pattern learning.
- **Imbalanced data**: Uneven class distribution to check robustness.

Tests in `tests/test_boosting_tree.py` verify:
- Accuracy (>90% for linear, >80% for circular data).
- Probabilities are in [0, 1].
- Correct handling of small datasets.

Manual checks confirmed reasonable decision boundaries on sample data.

### 3. What parameters have you exposed to users to tune performance?

The `GradientBoostingClassifier` supports:
- `n_estimators`: Number of trees (default: 100).
- `learning_rate`: Step size for updates (default: 0.1).
- `max_depth`: Maximum tree depth (default: 3).

**Example** (see `demo/demo.py`):

```python
from boosting_tree import GradientBoostingClassifier
import numpy as np

X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=4)
model.fit(X, y)
preds = model.predict(X)
probs = model.predict_proba(X)
print(f"Accuracy: {np.mean(preds == y):.3f}")
print("Sample probabilities (first 5):")
print(probs[:5])
```

#### 4. Are there specific inputs that your implementation has trouble with?

**Challenges:**

- High-dimensional data: Slow split search in decision trees.

- Small datasets: Risk of overfitting with deep trees.

- Noisy data: Outliers can skew residual fitting.

**Improvements to be done:**

- Feature subsampling for high-dimensional data.

- Regularization.

- Early stopping based on validation loss.

- Optimized split search.

### Setup and Running the Project

1. Clone the repository:

- git clone <your-repo-url>
- cd Project2

2. Create a virtual environment (optional but recommended):

- python3 -m venv venv

- source venv/bin/activate

3. Install dependencies

- pip3 install -r requirements.txt

4. Run tests

- bash python3 -m unittest discover tests 

**Expected output:**
```
Saved data for case 'linear' with 100 samples to data/X_linear_100.csv and data/y_linear_100.csv
Saved data for case 'circle' with 100 samples to data/X_circle_100.csv and data/y_circle_100.csv
Saved data for case 'imbalanced' with 50 samples to data/X_imbalanced_50.csv and data/y_imbalanced_50.csv
Saved data for case 'linear' with 10 samples to data/X_linear_10.csv and data/y_linear_10.csv
....
Ran 4 tests in ~1.2s
OK
```

**Run the demo:**

- python3 demo/demo.py

**Expected output:**

Accuracy: 0.95
Sample probabilities (first 5):
[[0.71992303 0.28007697]
 [0.28007697 0.71992303]
 [0.28007697 0.71992303]
 [0.28007697 0.71992303]
 [0.71992303 0.28007697]]

