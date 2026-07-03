import pandas as pd
import numpy as np
from data.storage import get_conn
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_ratios() -> pd.DataFrame:
    conn = get_conn()
    inc = pd.read_sql("SELECT * FROM income_statement", conn)
    bal = pd.read_sql("SELECT * FROM balance_sheet", conn)
    cf  = pd.read_sql("SELECT * FROM cash_flow", conn)
    conn.close()

    df = inc.merge(bal, on=["ticker","period"]).merge(cf, on=["ticker","period"])

    # Profitability
    # ROE = Net Income / Equity — how much profit per rupee of equity
    df["roe"] = df["net_income"] / df["total_equity"]

    # ROA = Net Income / Assets — how efficiently assets generate profit
    df["roa"] = df["net_income"] / df["total_assets"]

    # Margins — what % of revenue survives each stage
    df["gross_margin"]     = df["gross_profit"]      / df["revenue"]
    df["operating_margin"] = df["operating_income"]  / df["revenue"]
    df["net_margin"]       = df["net_income"]         / df["revenue"]

    # Liquidity
    # Current Ratio > 1 means company can cover short term obligations
    df["current_ratio"] = df["current_assets"] / df["current_liabilities"]

    # Quick Ratio excludes inventory (less liquid)
    df["quick_ratio"] = (df["current_assets"] - df["inventory"].fillna(0)) / df["current_liabilities"]

    # Leverage
    df["debt_equity"] = df["total_debt"] / df["total_equity"]

    # Interest Coverage = EBIT / Interest — can company pay its interest?
    df["interest_coverage"] = df["operating_income"] / df["interest_expense"].replace(0, np.nan)

    # Efficiency
    df["asset_turnover"] = df["revenue"] / df["total_assets"]

    # Growth — YoY change sorted within each ticker
    df = df.sort_values(["ticker","period"])
    df["revenue_growth"] = df.groupby("ticker")["revenue"].pct_change()
    df["eps_growth"]     = df.groupby("ticker")["eps"].pct_change()
    df["fcf_growth"]     = df.groupby("ticker")["free_cf"].pct_change()

    # Altman Z-Score — bankruptcy prediction model (Altman 1968)
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Assets
    # X2 = Retained Earnings / Assets (approx equity - debt / assets)
    # X3 = EBIT / Assets
    # X4 = Equity / Liabilities
    # X5 = Revenue / Assets
    X1 = (df["current_assets"] - df["current_liabilities"]) / df["total_assets"]
    X2 = df["total_equity"] / df["total_assets"]
    X3 = df["operating_income"] / df["total_assets"]
    X4 = df["total_equity"] / df["total_liabilities"].replace(0, np.nan)
    X5 = df["revenue"] / df["total_assets"]
    df["altman_z"] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    logger.info("Fundamental ratios computed for %d rows", len(df))
    return df[["ticker","period","roe","roa","gross_margin","operating_margin",
               "net_margin","current_ratio","quick_ratio","debt_equity",
               "interest_coverage","asset_turnover","revenue_growth",
               "eps_growth","fcf_growth","altman_z"]]