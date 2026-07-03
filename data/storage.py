import sqlite3
from config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
    close REAL, volume REAL, adj_close REAL,
    PRIMARY KEY (ticker, date));

CREATE TABLE IF NOT EXISTS income_statement (
    ticker TEXT, period TEXT, revenue REAL, gross_profit REAL,
    operating_income REAL, net_income REAL, ebitda REAL,
    eps REAL, interest_expense REAL,
    PRIMARY KEY (ticker, period));

CREATE TABLE IF NOT EXISTS balance_sheet (
    ticker TEXT, period TEXT, total_assets REAL, total_liabilities REAL,
    total_equity REAL, current_assets REAL, current_liabilities REAL,
    cash REAL, total_debt REAL, inventory REAL,
    PRIMARY KEY (ticker, period));

CREATE TABLE IF NOT EXISTS cash_flow (
    ticker TEXT, period TEXT, operating_cf REAL, capex REAL,
    free_cf REAL, PRIMARY KEY (ticker, period));

CREATE TABLE IF NOT EXISTS financial_ratios (
    ticker TEXT, period TEXT, roe REAL, roa REAL, current_ratio REAL,
    quick_ratio REAL, debt_equity REAL, gross_margin REAL,
    operating_margin REAL, net_margin REAL, interest_coverage REAL,
    asset_turnover REAL, revenue_growth REAL, eps_growth REAL,
    fcf_growth REAL, pe REAL, pb REAL, ev_ebitda REAL,
    dividend_yield REAL, beta REAL, volatility REAL, altman_z REAL,
    PRIMARY KEY (ticker, period));

CREATE TABLE IF NOT EXISTS predictions (
    ticker TEXT, date TEXT, predicted_return REAL, actual_return REAL,
    signal TEXT, model TEXT, PRIMARY KEY (ticker, date, model));

CREATE TABLE IF NOT EXISTS model_metrics (
    model TEXT, mae REAL, rmse REAL, r2 REAL, ic REAL, trained_at TEXT);
"""

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip(): conn.execute(stmt)
    logger.info("DB initialised at %s", DB_PATH)

def get_conn(): return sqlite3.connect(DB_PATH)