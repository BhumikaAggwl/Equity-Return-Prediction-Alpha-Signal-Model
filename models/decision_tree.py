import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Why build from scratch? ────────────────────────────────────────────────────
# XGBoost is an ensemble of decision trees.
# If you understand how a single tree splits data, you understand
# how XGBoost works at its core.
# Interview point: "I implemented the tree myself so I understand
# what happens at each node before wrapping it in a boosting framework."

class Node:
    """Represents a single node in the decision tree."""
    def __init__(self):
        self.feature    = None   # which feature to split on
        self.threshold  = None   # value to split at
        self.left       = None   # left child node
        self.right      = None   # right child node
        self.value      = None   # prediction if this is a leaf


class DecisionTreeRegressor:
    """
    Decision Tree Regressor built from scratch using NumPy.

    Splitting criterion: Variance Reduction (equivalent to MSE reduction)
    
    At each node we find the feature and threshold that produces the
    greatest reduction in variance of the target variable.

    Variance Reduction = Var(parent) - weighted_avg(Var(left), Var(right))
    
    We stop splitting when:
    - max_depth is reached
    - node has fewer than min_samples_split samples
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.root              = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()

        # Stopping conditions → make this a leaf node
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node.value = np.mean(y)   # leaf predicts mean of samples
            return node

        feature, threshold = self._best_split(X, y)

        if feature is None:           # no split improved variance
            node.value = np.mean(y)
            return node

        # Split data into left (<=threshold) and right (>threshold)
        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.feature   = feature
        node.threshold = threshold
        node.left      = self._grow(X[left_mask],  y[left_mask],  depth+1)
        node.right     = self._grow(X[right_mask], y[right_mask], depth+1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 50):
        best_var_red   = 0
        best_feature   = None
        best_threshold = None
        parent_var     = np.var(y) * len(y)

        for feature in range(X.shape[1]):
            col = X[:, feature]
            # Use percentile-based candidate thresholds instead of every unique value
            # Massively faster on large datasets, negligible accuracy loss
            candidates = np.unique(np.percentile(col, np.linspace(1, 99, n_thresholds)))

            for threshold in candidates:
                left_mask  = col <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                var_after = np.var(y_left)*len(y_left) + np.var(y_right)*len(y_right)
                var_red   = parent_var - var_after
                if var_red > best_var_red:
                    best_var_red   = var_red
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x: np.ndarray, node: Node) -> float:
        if node.value is not None:     # leaf node
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)