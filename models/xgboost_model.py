import numpy as np
from models.decision_tree import DecisionTreeRegressor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── How XGBoost works (interview explanation) ──────────────────────────────────
# 1. Start with a base prediction (mean of y)
# 2. Compute residuals = actual - predicted
# 3. Fit a tree to the residuals (not the target directly)
# 4. Add tree predictions × learning_rate to current predictions
# 5. Repeat n_estimators times
# Each tree corrects the errors of all previous trees combined.
# This is called "gradient boosting" because residuals = negative gradient of MSE loss.

class XGBoostScratch:
    """
    Gradient Boosted Tree Regressor built from scratch.
    Uses our DecisionTreeRegressor as the base learner.
    
    Loss function: MSE = (1/n) * sum((y - y_pred)^2)
    Gradient of MSE w.r.t y_pred = -2*(y - y_pred) = -2*residual
    So we fit each tree to residuals (the negative gradient).
    
    learning_rate (eta): shrinks each tree's contribution
    — prevents overfitting by not trusting any single tree too much.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 4, min_samples_split: int = 10):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.trees             = []
        self.base_prediction   = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Step 1: base prediction = mean(y)
        # This is the simplest possible starting point
        self.base_prediction = np.mean(y)
        y_pred = np.full(len(y), self.base_prediction, dtype=float)

        for i in range(self.n_estimators):
            # Step 2: residuals = what our current model gets wrong
            residuals = y - y_pred

            # Step 3: fit a tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)

            # Step 4: update predictions by adding scaled tree output
            update = self.learning_rate * tree.predict(X)
            y_pred = y_pred + update

            self.trees.append(tree)

            if (i+1) % 10 == 0:
                mse = np.mean((y - y_pred)**2)
                logger.info("Round %d | MSE: %.6f", i+1, mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Start from base prediction then add each tree's contribution
        y_pred = np.full(X.shape[0], self.base_prediction, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred