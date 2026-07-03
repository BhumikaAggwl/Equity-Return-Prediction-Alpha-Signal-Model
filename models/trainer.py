import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from models.xgboost_model import XGBoostScratch
from models.decision_tree import DecisionTreeRegressor
from features.builder import build_features, get_feature_cols
from evaluation.metrics import compute_metrics, information_coefficient
from data.storage import get_conn
from src.utils.logger import get_logger
from config import TEST_SIZE, RANDOM_STATE

logger = get_logger(__name__)
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data():
    """Load features, split into train/test, return arrays."""
    df   = build_features()
    cols = get_feature_cols(df)
    X    = df[cols].values
    y = df["excess_return"].values
    meta = df[["ticker","date"]].reset_index(drop=True)

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    meta_test       = meta.iloc[split_idx:].reset_index(drop=True)

    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, meta_test, cols


def train_all():
    """Trains 5 models, compares by IC, saves best to disk."""
    X_train, X_test, y_train, y_test, meta_test, cols = prepare_data()
    results = {}

    logger.info("Training Decision Tree (scratch)...")
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=20)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    results["decision_tree_scratch"] = compute_metrics(y_test, preds_dt)
    results["decision_tree_scratch"]["ic"] = information_coefficient(y_test, preds_dt)

    logger.info("Training XGBoost (scratch)...")
    xgb_s = XGBoostScratch(n_estimators=100, learning_rate=0.05, max_depth=4)
    xgb_s.fit(X_train, y_train)
    preds_xgb_s = xgb_s.predict(X_test)
    results["xgboost_scratch"] = compute_metrics(y_test, preds_xgb_s)
    results["xgboost_scratch"]["ic"] = information_coefficient(y_test, preds_xgb_s)

    logger.info("Training XGBoost (library)...")
    xgb_lib = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0
    )
    xgb_lib.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_xgb = xgb_lib.predict(X_test)
    results["xgboost_library"] = compute_metrics(y_test, preds_xgb)
    results["xgboost_library"]["ic"] = information_coefficient(y_test, preds_xgb)

    logger.info("Training LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    preds_lgbm = lgbm.predict(X_test)
    results["lightgbm"] = compute_metrics(y_test, preds_lgbm)
    results["lightgbm"]["ic"] = information_coefficient(y_test, preds_lgbm)

    logger.info("Training CatBoost...")
    cat = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=4,
        random_seed=RANDOM_STATE, verbose=0
    )
    cat.fit(X_train, y_train)
    preds_cat = cat.predict(X_test)
    results["catboost"] = compute_metrics(y_test, preds_cat)
    results["catboost"]["ic"] = information_coefficient(y_test, preds_cat)

    best_name = max(results, key=lambda k: results[k]["ic"])
    logger.info("Best model by IC: %s", best_name)

    joblib.dump(xgb_lib, MODEL_DIR / "xgboost_library.pkl")
    joblib.dump(lgbm,    MODEL_DIR / "lightgbm.pkl")
    joblib.dump(cat,     MODEL_DIR / "catboost.pkl")
    joblib.dump(cols,    MODEL_DIR / "feature_cols.pkl")

    conn = get_conn()
    now  = datetime.datetime.now().isoformat()
    for model_name, m in results.items():
        conn.execute("""
            INSERT INTO model_metrics (model, mae, rmse, r2, ic, trained_at)
            VALUES (?,?,?,?,?,?)
        """, (model_name, m["mae"], m["rmse"], m["r2"], m["ic"], now))
    conn.commit()
    conn.close()

    save_predictions(meta_test, preds_xgb, y_test, "xgboost_library")

    print("\n── Model Comparison ──────────────────────────────────")
    for name, m in results.items():
        print(f"{name:30s} | MAE:{m['mae']:.4f} RMSE:{m['rmse']:.4f} "
              f"R2:{m['r2']:.4f} IC:{m['ic']:.4f}")

    return results, xgb_lib, cols


def save_predictions(meta: pd.DataFrame, preds: np.ndarray,
                      actuals: np.ndarray, model_name: str):
    from recommendation.signal import generate_signal
    conn = get_conn()
    for i, row in meta.iterrows():
        signal = generate_signal(preds[i])
        conn.execute("""
            INSERT OR REPLACE INTO predictions
            (ticker, date, predicted_return, actual_return, signal, model)
            VALUES (?,?,?,?,?,?)
        """, (row["ticker"], str(row["date"]), float(preds[i]),
              float(actuals[i]), signal, model_name))
    conn.commit()
    conn.close()
    logger.info("Predictions saved for %d rows", len(meta))


if __name__ == "__main__":
    train_all()