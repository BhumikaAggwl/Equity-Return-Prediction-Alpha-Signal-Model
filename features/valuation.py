import pandas as pd
from data.storage import get_conn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_valuation() -> pd.DataFrame:
    """
    Computes valuation ratios from already-fetched price + fundamental data.
    Avoids live yfinance calls entirely — no rate limit risk.
    """
    conn = get_conn()
    prices = pd.read_sql("SELECT ticker, date, close FROM prices", conn)
    inc    = pd.read_sql("SELECT * FROM income_statement", conn)
    bal    = pd.read_sql("SELECT * FROM balance_sheet", conn)
    conn.close()

    prices["date"] = pd.to_datetime(prices["date"])
    latest_price = prices.sort_values("date").groupby("ticker").last().reset_index()
    latest_price = latest_price[["ticker","close"]]

    latest_inc = inc.sort_values("period").groupby("ticker").last().reset_index()
    latest_bal = bal.sort_values("period").groupby("ticker").last().reset_index()

    df = latest_price.merge(latest_inc[["ticker","eps","ebitda"]], on="ticker", how="left")
    df = df.merge(latest_bal[["ticker","total_equity","total_debt","cash"]], on="ticker", how="left")

    df["pe"]        = df["close"] / df["eps"].replace(0, pd.NA)
    df["pb"]        = df["close"] / (df["total_equity"] / 1e9).replace(0, pd.NA)  # rough proxy, no shares data
    df["ev_ebitda"] = (df["total_equity"] + df["total_debt"] - df["cash"]) / df["ebitda"].replace(0, pd.NA)

    # peg, dividend_yield, beta not derivable from cached data — fill neutral
    df["peg"]            = pd.NA
    df["dividend_yield"] = pd.NA
    df["beta"]           = pd.NA
    df["market_cap"]     = df["total_equity"] + df["total_debt"] - df["cash"]  # proxy

    df = df[["ticker","pe","pb","ev_ebitda","peg","dividend_yield","beta","market_cap"]]
    logger.info("Valuation computed locally: %d rows", len(df))
    return df