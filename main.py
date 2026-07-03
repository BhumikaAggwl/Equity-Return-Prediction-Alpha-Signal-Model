from data.storage import init_db
from data.fetcher import fetch_prices, fetch_fundamentals
from data.news_fetcher import fetch_news
from models.trainer import train_all
from backtest.walkforward import walk_forward_backtest
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("═══ Equity Research Platform Starting ═══")

    logger.info("Step 1: Initialising database...")
    init_db()

    logger.info("Step 2: Fetching prices (100 stocks)...")
    fetch_prices()

    logger.info("Step 3: Fetching fundamentals...")
    fetch_fundamentals()

    logger.info("Step 4: Fetching news headlines...")
    fetch_news()

    logger.info("Step 5: Training and comparing 5 models...")
    train_all()

    logger.info("Step 6: Walk-forward backtest...")
    walk_forward_backtest(n_splits=3, min_train_years=2)

    logger.info("═══ Pipeline Complete ═══")
    print("\nRun dashboard:  streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()