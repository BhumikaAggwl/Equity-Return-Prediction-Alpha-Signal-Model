import pandas as pd
import numpy as np
from data.storage import get_conn
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_technicals() -> pd.DataFrame:
    conn = get_conn()
    prices = pd.read_sql("SELECT ticker, date, close, high, low FROM prices", conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker","date"])
    out = []

    for ticker, df in prices.groupby("ticker"):
        df = df.copy()
        c  = df["close"]

        # SMA — average price over window, smooths noise
        df["sma_20"] = c.rolling(20).mean()
        df["sma_50"] = c.rolling(50).mean()

        # EMA — exponentially weighted, recent prices matter more
        df["ema_20"] = c.ewm(span=20, adjust=False).mean()
        df["ema_50"] = c.ewm(span=50, adjust=False).mean()

        # RSI — Relative Strength Index (Wilder 1978)
        # RSI = 100 - 100/(1 + RS) where RS = avg gain / avg loss
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD = EMA12 - EMA26, Signal = EMA9 of MACD
        ema12       = c.ewm(span=12, adjust=False).mean()
        ema26       = c.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # ATR — Average True Range, measures volatility
        # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
        hl  = df["high"] - df["low"]
        hcp = (df["high"] - c.shift()).abs()
        lcp = (df["low"]  - c.shift()).abs()
        df["atr"] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

        # Momentum = price today - price 10 days ago
        df["momentum"] = c.diff(10)

        # Rolling 21-day return and volatility
        df["roll_return"] = c.pct_change(21)
        df["roll_vol"]    = c.pct_change().rolling(21).std()

        out.append(df)

    result = pd.concat(out, ignore_index=True)
    logger.info("Technicals computed: %d rows", len(result))
    return result[["ticker","date","sma_20","sma_50","ema_20","ema_50",
                   "rsi","macd","macd_signal","atr","momentum","roll_return","roll_vol"]]