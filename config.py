from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "data" / "equity_research.db"

# Expanded to S&P 100 for cross-sectional credibility
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","JNJ","V",
    "UNH","XOM","WMT","MA","PG","CVX","HD","MRK","ABBV","PEP",
    "KO","AVGO","COST","MCD","TMO","ACN","CSCO","ABT","DHR","NEE",
    "LIN","TXN","PM","ORCL","MS","RTX","SPGI","HON","AMGN","IBM",
    "CAT","GE","INTU","AMAT","BKNG","AXP","GS","BLK","SYK","ISRG",
    "GILD","ADI","VRTX","PLD","CI","MDLZ","REGN","ZTS","CB","BDX",
    "MMC","SO","DUK","CL","EOG","SLB","MO","BMY","ITW","ELV",
    "AON","CME","PNC","USB","TGT","ETN","NOC","FDX","EMR","PSA",
    "APD","MCO","TFC","NSC","AIG","HUM","FCX","MCK","WM","ECL",
    "D","F","GM","BAC","C","WFC","T","VZ","DIS","NFLX","SPY"
]

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

FORWARD_DAYS   = 21
BUY_THRESHOLD  =  0.03
SELL_THRESHOLD = -0.03

TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

LOG_LEVEL = "INFO"