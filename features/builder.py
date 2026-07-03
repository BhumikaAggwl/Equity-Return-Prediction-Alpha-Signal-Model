import pandas as pd
import numpy as np
from data.storage import get_conn
from features.fundamental import compute_ratios
from features.valuation import compute_valuation
from features.technical import compute_technicals
from features.sentiment import compute_sentiment_features
from src.utils.logger import get_logger
from config import FORWARD_DAYS

logger = get_logger(__name__)

def build_features() -> pd.DataFrame:
    """
    Merges fundamental ratios, valuation metrics, technical indicators
    and sentiment into a single model-ready dataframe with target attached.

    Target: Forward 21-day return = (price_t+21 - price_t) / price_t
    """
    conn = get_conn()
    prices = pd.read_sql("SELECT ticker, date, close FROM prices", conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker","date"])

    prices["future_close"]   = prices.groupby("ticker")["close"].shift(-FORWARD_DAYS)
    prices["forward_return"] = (prices["future_close"] - prices["close"]) / prices["close"]
    prices = prices.dropna(subset=["forward_return"])

    spy = prices[prices["ticker"]=="SPY"][["date","forward_return"]]\
              .rename(columns={"forward_return":"spy_return"})
    prices = prices.merge(spy, on="date", how="left")
    prices["excess_return"] = prices["forward_return"] - prices["spy_return"].fillna(0)
    
    prices = prices[prices["ticker"] != "SPY"]
    tech = compute_technicals()
    tech["date"] = pd.to_datetime(tech["date"])
    df = prices.merge(tech, on=["ticker","date"], how="left")

    ratios = compute_ratios()
    ratios["period"] = pd.to_datetime(ratios["period"])
    ratios = ratios.sort_values(["ticker","period"])
    latest_ratios = ratios.groupby("ticker").last().reset_index()
    latest_ratios = latest_ratios.drop(columns=["period"])
    df = df.merge(latest_ratios, on="ticker", how="left")

    val = compute_valuation()
    df = df.merge(val, on="ticker", how="left")

    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [c for c in df.columns if c not in ["ticker","date","close",
                    "future_close","forward_return"]]
    thresh = int(len(feature_cols) * 0.5)
    df = df.dropna(thresh=thresh)

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── Sentiment ──────────────────────────────────────────────────────────
    try:
        sent = compute_sentiment_features()
        if not sent.empty:
            sent["date"] = pd.to_datetime(sent["date"])
            df = df.merge(sent[["ticker","date","sentiment_score","sentiment_5d",
                                 "headline_count"]], on=["ticker","date"], how="left")
            df["sentiment_score"] = df["sentiment_score"].fillna(0)
            df["sentiment_5d"]    = df["sentiment_5d"].fillna(0)
            df["headline_count"]  = df["headline_count"].fillna(0)
        else:
            df["sentiment_score"] = 0
            df["sentiment_5d"]    = 0
            df["headline_count"]  = 0
    except Exception as e:
        logger.warning("Sentiment merge skipped: %s", e)
        df["sentiment_score"] = 0
        df["sentiment_5d"]    = 0
        df["headline_count"]  = 0

    logger.info("Feature matrix built: %d rows x %d cols", len(df), len(df.columns))
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Returns list of feature column names, excluding metadata and target."""
    exclude = ["ticker","date","close","future_close","forward_return","market_cap"]
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df[["ticker","date","forward_return"]].head())