import numpy as np
from scipy.stats import spearmanr
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    MAE  = mean(|actual - predicted|)          — average error magnitude
    RMSE = sqrt(mean((actual - predicted)^2))  — penalises large errors more
    R2   = 1 - SS_res/SS_tot                  — % variance explained by model
    """
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2   = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

    logger.info("MAE:%.4f RMSE:%.4f R2:%.4f", mae, rmse, r2)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Information Coefficient (IC) = Spearman rank correlation between
    predicted returns and actual returns.

    IC = 1  → perfect rank prediction
    IC = 0  → no predictive power
    IC > 0.05 is considered meaningful in quant research.

    We use Spearman (not Pearson) because we care about whether we
    ranked stocks correctly, not the exact magnitude of returns.
    This is exactly how quant PMs evaluate signals at BlackRock.
    """
    ic, _ = spearmanr(y_true, y_pred)
    logger.info("Information Coefficient: %.4f", ic)
    return float(ic)


def sharpe_of_signal(y_true: np.ndarray, y_pred: np.ndarray,
                     periods: int = 252) -> float:
    """
    Treat predicted return as a trading signal:
    - If predicted > 0 → go long (return = actual)
    - If predicted < 0 → go short (return = -actual)
    Then compute annualised Sharpe of that strategy.

    Sharpe = mean(strategy_returns) / std(strategy_returns) * sqrt(periods)

    This tells us: does following the model's signal make money
    on a risk-adjusted basis?
    """
    signal_returns = np.where(y_pred > 0, y_true, -y_true)
    mean_r = np.mean(signal_returns)
    std_r  = np.std(signal_returns)
    sharpe = (mean_r / std_r * np.sqrt(periods)) if std_r != 0 else 0.0
    logger.info("Signal Sharpe: %.4f", sharpe)
    return float(sharpe)