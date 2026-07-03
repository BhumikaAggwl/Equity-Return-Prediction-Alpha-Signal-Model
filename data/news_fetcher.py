import yfinance as yf
import pandas as pd
from data.storage import get_conn
from src.utils.logger import get_logger
from config import TICKERS

logger = get_logger(__name__)

# yfinance provides recent news headlines for free
# Each headline gets scored by FinBERT in sentiment.py
# We store ticker + date + headline in SQLite
def fetch_news() -> pd.DataFrame:
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            ticker   TEXT,
            date     TEXT,
            headline TEXT,
            PRIMARY KEY (ticker, date, headline)
        )
    """)
    conn.commit()

    all_news = []
    for ticker in TICKERS:
        try:
            t    = yf.Ticker(ticker)
            news = t.news
            if not news:
                logger.warning("No news for %s", ticker)
                continue

            for item in news:
                # Handle both old flat schema and new nested "content" schema
                content = item.get("content", item)
                headline = content.get("title", "")
                pub_date = content.get("pubDate") or item.get("providerPublishTime")

                if not headline:
                    continue

                try:
                    if isinstance(pub_date, (int, float)):
                        date = pd.to_datetime(pub_date, unit="s")
                    else:
                        date = pd.to_datetime(pub_date)
                except Exception:
                    date = pd.Timestamp.today()

                all_news.append({
                    "ticker"  : ticker,
                    "date"    : date.strftime("%Y-%m-%d"),
                    "headline": headline
                })

            logger.info("News fetched: %s (%d articles)", ticker, len(news))
        except Exception as e:
            logger.error("News fetch failed %s: %s", ticker, e)

    if not all_news:
        logger.warning("No news fetched for any ticker")
        return pd.DataFrame()

    df = pd.DataFrame(all_news).drop_duplicates()
    df.to_sql("news", conn, if_exists="append", index=False)
    conn.close()
    logger.info("Total news headlines stored: %d", len(df))
    return df