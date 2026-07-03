import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from features.builder import build_features, get_feature_cols
from src.utils.logger import get_logger

logger = get_logger(__name__)
PLOT_DIR = Path("reports/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Why SHAP? ──────────────────────────────────────────────────────────────────
# XGBoost is a black box. A PM cannot act on "the model says BUY"
# without understanding why. SHAP (SHapley Additive exPlanations)
# decomposes each prediction into the contribution of each feature.
#
# SHAP is based on Shapley values from cooperative game theory:
# each feature gets credit proportional to its marginal contribution
# across all possible subsets of features.
#
# Interview line:
# "I used SHAP to make every prediction auditable —
#  a compliance requirement in institutional wealth management."

def load_model_and_data():
    model = joblib.load("models/saved/xgboost_library.pkl")
    cols  = joblib.load("models/saved/feature_cols.pkl")
    df    = build_features()
    X     = df[cols].values
    return model, X, cols, df


def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """
    TreeExplainer is optimised for tree-based models.
    Much faster than KernelExplainer for XGBoost.
    Returns shap_values array of shape (n_samples, n_features).
    Each value = how much that feature pushed prediction
    above or below the base (mean) prediction.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    logger.info("SHAP values computed: shape %s", str(shap_values.shape))
    return shap_values, explainer


def plot_summary(shap_values: np.ndarray, X: np.ndarray, cols: list):
    """
    Summary plot: shows which features matter most overall
    and whether high values push predictions up or down.
    Red = high feature value, Blue = low feature value.
    """
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=cols, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    logger.info("SHAP summary plot saved")


def plot_waterfall(explainer, shap_values: np.ndarray,
                   X: np.ndarray, cols: list, idx: int = 0):
    """
    Waterfall plot for a single prediction.
    Shows step by step how each feature moved the prediction
    from the base value to the final predicted return.
    Perfect for explaining a single BUY/SELL call to a PM.
    """
    explanation = shap.Explanation(
        values    = shap_values[idx],
        base_values = explainer.expected_value,
        data      = X[idx],
        feature_names = cols
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"shap_waterfall_{idx}.png", dpi=150)
    plt.close()
    logger.info("Waterfall plot saved for index %d", idx)


def plot_feature_importance(shap_values: np.ndarray, cols: list):
    """
    Bar chart of mean absolute SHAP values.
    This is more reliable than XGBoost's built-in feature importance
    because it accounts for feature interactions.
    mean(|SHAP|) = average impact on prediction magnitude.
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_shap, index=cols).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    importance.head(15).plot(kind="bar", color="steelblue")
    plt.title("Top 15 Features by Mean |SHAP| Value")
    plt.ylabel("Mean |SHAP Value|")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_importance.png", dpi=150)
    plt.close()
    logger.info("Feature importance plot saved")
    return importance


def run_shap_analysis():
    model, X, cols, df = load_model_and_data()
    shap_values, explainer = compute_shap_values(model, X)
    plot_summary(shap_values, X, cols)
    plot_waterfall(explainer, shap_values, X, cols, idx=0)
    importance = plot_feature_importance(shap_values, cols)

    print("\n── Top 10 Most Influential Features ──────────────────")
    print(importance.head(10).to_string())
    print("\nPlots saved to reports/plots/")
    return shap_values, importance


if __name__ == "__main__":
    run_shap_analysis()