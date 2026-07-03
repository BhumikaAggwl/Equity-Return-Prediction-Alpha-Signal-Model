# Equity Return Prediction & Alpha Signal Model

ML pipeline predicting 21-day excess stock returns vs S&P 500 across 100 large-cap equities. Compares Decision Tree, XGBoost (built from scratch and via library), LightGBM, and CatBoost using walk-forward validation and Information Coefficient. Generates BUY/HOLD/SELL signals with SHAP explainability and FinBERT-driven news sentiment as an alpha factor.

## Overview

- **Universe**: 100 S&P large-cap stocks + SPY benchmark
- **Target**: 21-day forward excess return (stock return − SPY return)
- **Features**: 32 total — fundamental ratios, technical indicators, valuation metrics, FinBERT sentiment
- **Validation**: Walk-forward (expanding window, no lookahead bias)
- **Explainability**: SHAP feature attribution per prediction
- **Risk**: Historical VaR / CVaR on signal basket

## Architecture

| Folder | Purpose |
|---|---|
| `data/` | yfinance ingestion (prices, fundamentals, news) → SQLite |
| `features/` | fundamental, technical, valuation, FinBERT sentiment |
| `models/` | Decision Tree (scratch), XGBoost (scratch), XGBoost/LightGBM/CatBoost (library) |
| `evaluation/` | IC, Sharpe, VaR/CVaR metrics |
| `backtest/` | walk-forward validation |
| `explainability/` | SHAP analysis |
| `recommendation/` | BUY/HOLD/SELL signal generation |
| `dashboard/` | Streamlit interactive dashboard |
## Methodology

1. Fetch daily prices, fundamentals, and news headlines for 100 equities + SPY
2. Engineer 32 features across fundamental, technical, valuation, and sentiment categories
3. Compute target as **excess return** (stock return minus SPY return) — isolates stock-specific signal from market movement
4. Train and compare 5 models, selecting the best by **Information Coefficient**, not just RMSE
5. Validate via walk-forward backtesting across multiple market regimes (2018–2024)
6. Explain every prediction with SHAP; convert predictions into BUY/HOLD/SELL signals
7. Assess risk on the BUY-signal basket via historical VaR/CVaR

## Results

| Model | MAE | RMSE | R² | IC |
|---|---|---|---|---|
| Decision Tree (scratch) | 0.0631 | 0.0915 | -0.0177 | 0.0990 |
| XGBoost (scratch) | 0.0616 | 0.0893 | 0.0308 | 0.1396 |
| XGBoost (library) | 0.0631 | 0.0918 | -0.0245 | 0.1377 |
| LightGBM | 0.0631 | 0.0911 | -0.0087 | 0.1375 |
| CatBoost | 0.0622 | 0.0900 | 0.0150 | 0.1362 |

All models achieve IC > 0.05, generally considered a meaningful signal threshold in systematic equity research.

## Setup

```bash
git clone https://github.com/BhumikaAggwl/Equity-Return-Prediction-Alpha-Signal-Model.git
cd Equity-Return-Prediction-Alpha-Signal-Model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, FinBERT (transformers), yfinance, SQLite, Streamlit

## Challenges & Fixes

- **yfinance rate limiting**: Hit Yahoo's per-request rate limit fetching 100 tickers individually. Switched price ingestion to a single batched `yf.download()` call instead of per-ticker loops, cutting network requests from 100+ to 1.
- **No batch API for fundamentals/valuation**: `yfinance.Ticker().info` has no batch equivalent, so valuation metrics were re-derived locally from cached price + fundamental data (PE, PB, EV/EBITDA computed from stored EPS, equity, debt, cash) instead of live API calls — eliminating rate-limit risk entirely for this stage.
- **Slow scratch Decision Tree**: Initial implementation brute-forced every unique feature value as a split threshold, making it impractical on 170K+ rows. Fixed by sampling percentile-based candidate thresholds (~50 per feature) instead of all unique values.
- **News schema change**: yfinance's news API silently changed its JSON structure (`title`/`providerPublishTime` → nested `content` object), causing headlines to be silently dropped. Fixed by handling both schemas defensively.
- **Return framing**: Initially modeled raw forward returns, which mostly reflect broad market movement rather than stock-specific skill. Switched target to **excess return vs. SPY** to isolate alpha — the correct framing for systematic equity research (not beta-adjusted / not Jensen's alpha, noted as a limitation).
  
## Limitations

- Transaction costs and slippage not modeled
- Fundamental data is annual/quarterly — frequency mismatch with daily prices
- Sentiment coverage limited by free news API availability
- Excess return is not beta-adjusted (not Jensen's alpha) — future work could incorporate CAPM-adjusted alpha

## Author

Bhumika Aggarwal
