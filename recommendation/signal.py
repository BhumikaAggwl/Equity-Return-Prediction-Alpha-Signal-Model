from config import BUY_THRESHOLD, SELL_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_signal(predicted_return: float) -> str:
    """
    Converts a predicted return into a BUY / HOLD / SELL signal.

    Logic:
    predicted_return > +3%  → BUY  (meaningful upside expected)
    predicted_return < -3%  → SELL (meaningful downside expected)
    in between              → HOLD (insufficient conviction)

    Thresholds are configurable in config.py.
    In practice quant desks calibrate these to hit a
    target number of signals per rebalance period.

    Args:
        predicted_return: float, e.g. 0.045 means +4.5% predicted

    Returns:
        "BUY", "HOLD", or "SELL"
    """
    if predicted_return >= BUY_THRESHOLD:
        return "BUY"
    elif predicted_return <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def generate_signal_batch(predicted_returns: list) -> list:
    """Apply generate_signal to a list of predictions."""
    return [generate_signal(r) for r in predicted_returns]


def signal_summary(signals: list) -> dict:
    """
    Returns count of BUY/HOLD/SELL in a batch.
    Useful for dashboard summary cards.
    """
    return {
        "BUY" : signals.count("BUY"),
        "HOLD": signals.count("HOLD"),
        "SELL": signals.count("SELL"),
    }