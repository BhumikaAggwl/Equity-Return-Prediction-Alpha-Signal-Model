import pandas as pd
import numpy as np
from data.storage import get_conn


def compute_var(confidence: float = 0.95) -> dict:
    """
    Historical VaR on the current BUY-signal basket.
    VaR = the loss level such that we expect to lose more than this
    only (1-confidence)% of the time.
    """
    conn = get_conn()
    preds = pd.read_sql("SELECT * FROM predictions WHERE signal='BUY'", conn)
    conn.close()

    if preds.empty:
        return {"var": None, "cvar": None, "n": 0}

    returns = preds["actual_return"].dropna()
    var  = -np.percentile(returns, (1 - confidence) * 100)
    cvar = -returns[returns <= -var].mean()  # expected shortfall

    return {
        "var": round(var, 4),
        "cvar": round(cvar, 4),
        "confidence": confidence,
        "n": len(returns)
    }


if __name__ == "__main__":
    result = compute_var()
    print(f"95% VaR: {result['var']*100:.2f}% | CVaR: {result['cvar']*100:.2f}% | n={result['n']}")