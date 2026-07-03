import yfinance as yf
import pandas as pd
from data.storage import get_conn, init_db
from src.utils.logger import get_logger
from config import TICKERS, START_DATE, END_DATE

logger = get_logger(__name__)

import time
import random

def fetch_prices():
    conn = get_conn()
    logger.info("Batch downloading %d tickers...", len(TICKERS))

    data = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                        auto_adjust=True, group_by="ticker", progress=False)

    for ticker in TICKERS:
        try:
            df = data[ticker].copy()
            if df.empty:
                raise ValueError("empty dataframe")

            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            df["ticker"] = ticker
            df["adj_close"] = df["close"]
            df = df.dropna(subset=["close"])

            df[["ticker","date","open","high","low","close","volume","adj_close"]]\
                .to_sql("prices", conn, if_exists="append", index=False)
            logger.info("Prices saved: %s", ticker)
        except Exception as e:
            logger.error("Price fetch failed %s: %s", ticker, e)

    conn.close()

def fetch_fundamentals():
    conn = get_conn()
    for ticker in TICKERS:
        try:
            t = yf.Ticker(ticker)

            # Income Statement
            inc = t.financials.T.reset_index()
            inc.columns = [c.lower().replace(" ","_") for c in inc.columns]
            inc["ticker"] = ticker
            inc["period"] = inc["index"].astype(str).str[:10]
            mapping = {
                "total_revenue":"revenue","gross_profit":"gross_profit",
                "ebit":"operating_income","net_income":"net_income",
                "ebitda":"ebitda","basic_eps":"eps",
                "interest_expense":"interest_expense"
            }
            for src, dst in mapping.items():
                if src not in inc.columns: inc[dst] = None
                else: inc[dst] = inc[src]
            inc[["ticker","period","revenue","gross_profit","operating_income",
                 "net_income","ebitda","eps","interest_expense"]]\
                .to_sql("income_statement", conn, if_exists="append", index=False)

            # Balance Sheet
            bal = t.balance_sheet.T.reset_index()
            bal.columns = [c.lower().replace(" ","_") for c in bal.columns]
            bal["ticker"] = ticker
            bal["period"] = bal["index"].astype(str).str[:10]
            bmap = {
                "total_assets":"total_assets","total_liabilities_net_minority_interest":"total_liabilities",
                "stockholders_equity":"total_equity","current_assets":"current_assets",
                "current_liabilities":"current_liabilities","cash_and_cash_equivalents":"cash",
                "total_debt":"total_debt","inventory":"inventory"
            }
            for src, dst in bmap.items():
                if src not in bal.columns: bal[dst] = None
                else: bal[dst] = bal[src]
            bal[["ticker","period","total_assets","total_liabilities","total_equity",
                 "current_assets","current_liabilities","cash","total_debt","inventory"]]\
                .to_sql("balance_sheet", conn, if_exists="append", index=False)

            # Cash Flow
            cf = t.cashflow.T.reset_index()
            cf.columns = [c.lower().replace(" ","_") for c in cf.columns]
            cf["ticker"] = ticker
            cf["period"] = cf["index"].astype(str).str[:10]
            if "operating_cash_flow" not in cf.columns: cf["operating_cf"] = None
            else: cf["operating_cf"] = cf["operating_cash_flow"]
            if "capital_expenditure" not in cf.columns: cf["capex"] = None
            else: cf["capex"] = cf["capital_expenditure"]
            cf["free_cf"] = cf["operating_cf"] - cf["capex"].fillna(0)
            cf[["ticker","period","operating_cf","capex","free_cf"]]\
                .to_sql("cash_flow", conn, if_exists="append", index=False)

            logger.info("Fundamentals saved: %s", ticker)
        except Exception as e:
            logger.error("Fundamentals fetch failed %s: %s", ticker, e)
    conn.close()

if __name__ == "__main__":
    init_db()
    fetch_prices()
    fetch_fundamentals()