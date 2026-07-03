import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from evaluation.metrics import compute_metrics, information_coefficient, sharpe_of_signal
from features.builder import build_features, get_feature_cols
from recommendation.signal import generate_signal_batch, signal_summary
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Why Walk-Forward? ──────────────────────────────────────────────────────────
# A simple train/test split on financial data is dangerous.
# If we train on 2018-2024 and test on a random 20% slice,
# we may accidentally use future data to predict the past (lookahead bias).
#
# Walk-forward validation mimics real trading:
# - Train on everything up to year T
# - Predict year T+1 (never seen during training)
# - Roll forward one year and repeat
# This gives us honest out-of-sample performance across multiple regimes.


def walk_forward_backtest(
    n_splits: int = 3,
    min_train_years: int = 2
) -> pd.DataFrame:
    """
    Performs walk-forward validation on the full feature matrix.

    Example with n_splits=3:
    Fold 1: Train 2018-2020 → Test 2021
    Fold 2: Train 2018-2021 → Test 2022
    Fold 3: Train 2018-2022 → Test 2023

    Args:
        n_splits: number of out-of-sample periods to test
        min_train_years: minimum years of data before first test period

    Returns:
        DataFrame with per-fold metrics and aggregated summary
    """
    df   = build_features()
    cols = get_feature_cols(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Get unique years available
    years      = sorted(df["date"].dt.year.unique())
    test_years = years[min_train_years:]   # first min_train_years used for initial training

    if len(test_years) < n_splits:
        n_splits = len(test_years)
        logger.warning("Reducing n_splits to %d due to limited data", n_splits)

    test_years = test_years[-n_splits:]    # take the last n_splits years as test periods
    fold_results = []

    for test_year in test_years:
        # ── Define train and test masks ────────────────────────────────────────
        train_mask = df["date"].dt.year < test_year
        test_mask  = df["date"].dt.year == test_year

        X_train = df.loc[train_mask, cols].values
        y_train = df.loc[train_mask, "forward_return"].values
        X_test  = df.loc[test_mask,  cols].values
        y_test  = df.loc[test_mask,  "forward_return"].values
        meta    = df.loc[test_mask,  ["ticker","date"]].reset_index(drop=True)

        if len(X_train) < 100 or len(X_test) < 10:
            logger.warning("Skipping fold %d — insufficient data", test_year)
            continue

        # ── Train XGBoost on expanding window ─────────────────────────────────
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ── Compute metrics for this fold ──────────────────────────────────────
        m       = compute_metrics(y_test, preds)
        ic      = information_coefficient(y_test, preds)
        sharpe  = sharpe_of_signal(y_test, preds)
        signals = generate_signal_batch(preds.tolist())
        summary = signal_summary(signals)

        fold_results.append({
            "test_year"  : test_year,
            "train_size" : len(X_train),
            "test_size"  : len(X_test),
            "mae"        : round(m["mae"],  4),
            "rmse"       : round(m["rmse"], 4),
            "r2"         : round(m["r2"],   4),
            "ic"         : round(ic,        4),
            "sharpe"     : round(sharpe,    4),
            "buy_count"  : summary["BUY"],
            "hold_count" : summary["HOLD"],
            "sell_count" : summary["SELL"],
        })

        logger.info("Fold %d | IC:%.4f | Sharpe:%.4f | MAE:%.4f",
                    test_year, ic, sharpe, m["mae"])

    results_df = pd.DataFrame(fold_results)

    # ── Aggregate summary across all folds ─────────────────────────────────────
    print("\n── Walk-Forward Backtest Results ─────────────────────────────")
    print(results_df.to_string(index=False))
    print("\n── Average Across Folds ──────────────────────────────────────")
    print(f"  Mean IC     : {results_df['ic'].mean():.4f}  (>0.05 is meaningful)")
    print(f"  Mean Sharpe : {results_df['sharpe'].mean():.4f}  (>0.5 is acceptable)")
    print(f"  Mean MAE    : {results_df['mae'].mean():.4f}")
    print(f"  Mean R2     : {results_df['r2'].mean():.4f}")

    return results_df


if __name__ == "__main__":
    walk_forward_backtest(n_splits=3, min_train_years=2)