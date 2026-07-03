import pandas as pd
import numpy as np
from transformers import pipeline
from data.storage import get_conn
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Why FinBERT? ───────────────────────────────────────────────────────────────
# General BERT doesn't understand financial language well.
# "Earnings beat expectations" is positive — general BERT may miss nuance.
# FinBERT was fine-tuned on financial news, earnings calls, analyst reports.
# It outputs: positive / negative / neutral with a confidence score.
#
# We aggregate daily headlines per ticker into a single sentiment score:
# daily_sentiment = mean(score) where score = +1 (pos), -1 (neg), 0 (neutral)
#
# Interview line:
# "I used FinBERT rather than a lexicon approach like Loughran-McDonald
#  because it captures context, not just individual word polarity."

# Load FinBERT once — this downloads ~400MB model on first run
# ProsusAI/finbert is the standard financial sentiment model on HuggingFace
def load_finbert():
    logger.info("Loading FinBERT model...")
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )


def score_sentiment(headlines: list, pipe) -> float:
    """
    Scores a list of headlines and returns aggregate sentiment score.
    
    Score mapping:
    positive →  +1
    neutral  →   0
    negative →  -1
    
    Final score = mean across all headlines for that ticker-date.
    Range: -1 (fully bearish) to +1 (fully bullish)
    """
    if not headlines:
        return 0.0

    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores    = []

    for headline in headlines:
        try:
            result = pipe(headline[:512])[0]   # truncate to model max length
            score  = label_map.get(result["label"].lower(), 0.0)
            # Weight by confidence — high confidence negative is worse than uncertain negative
            scores.append(score * result["score"])
        except Exception as e:
            logger.warning("Sentiment scoring failed: %s", e)
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def compute_sentiment_features() -> pd.DataFrame:
    """
    For each ticker-date combination in the news table,
    computes an aggregate FinBERT sentiment score.
    
    Returns DataFrame with columns:
    ticker | date | sentiment_score | headline_count
    """
    conn = get_conn()

    try:
        news = pd.read_sql("SELECT * FROM news", conn)
    except Exception:
        logger.warning("No news table found. Run fetch_news() first.")
        return pd.DataFrame()
    finally:
        conn.close()

    if news.empty:
        logger.warning("News table is empty.")
        return pd.DataFrame()

    pipe    = load_finbert()
    results = []

    for (ticker, date), group in news.groupby(["ticker","date"]):
        headlines = group["headline"].tolist()
        score     = score_sentiment(headlines, pipe)

        results.append({
            "ticker"         : ticker,
            "date"           : date,
            "sentiment_score": score,
            "headline_count" : len(headlines)
        })

        logger.info("Sentiment scored: %s %s | score=%.3f | n=%d",
                    ticker, date, score, len(headlines))

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])

    # Rolling 5-day average sentiment — smooths noise from single articles
    df = df.sort_values(["ticker","date"])
    df["sentiment_5d"] = df.groupby("ticker")["sentiment_score"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    logger.info("Sentiment features computed: %d rows", len(df))
    return df