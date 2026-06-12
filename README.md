# AI-Powered Investment Intelligence Platform

Decision-support platform for NIFTY-50 historical market data (2000–2021). Combines technical feature engineering, ML-based forward-return prediction, risk analytics, and profile-based portfolio optimization — all computed deterministically from OHLCV data.

## Project Structure

```
├── archive/               # NIFTY-50 CSV dataset (not committed if large)
│   ├── RELIANCE.csv
│   ├── stock_metadata.csv
│   └── ...
├── data_pipeline.py       # Module 1: Data loading & feature engineering
├── predictor.py           # Module 2: ML predictive engine + XAI
├── portfolio_engine.py    # Module 3: Risk metrics & portfolio optimizer
├── app.py                 # Module 4: Streamlit dashboard
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Reproducing Results

### 1. Data Pipeline

```bash
python data_pipeline.py
```

Loads all stock CSVs from `archive/`, engineers SMA, RSI, MACD, volatility, and `Forward_21_Return` target.

### 2. Predictive Engine

```bash
python predictor.py
```

Trains a Random Forest regressor with chronological hold-out validation. Prints RMSE, MAE, R², directional accuracy, and feature importances.

### 3. Portfolio Optimizer

```bash
python portfolio_engine.py
```

Generates Conservative, Balanced, and Aggressive portfolio allocations with Sharpe ratio and max drawdown metrics.

### 4. EDA & Technical Report Assets

```bash
python eda.py
```

Generates PDF-ready figures (`output/figures/`), summary tables (`output/tables/`), and narrative content (`output/eda_report.md`) for the technical report.

## Modules Overview

| Module | File | Key Capabilities |
|--------|------|------------------|
| 1 | `data_pipeline.py` | CSV loading, NaN handling, SMA/RSI/MACD/volatility, forward return target |
| 2 | `predictor.py` | Random Forest / XGBoost pipeline, time-series split, evaluation, XAI importances |
| 3 | `portfolio_engine.py` | Sharpe, Sortino, max drawdown, profile-based allocation (MVO, ERC, momentum) |
| 4 | `app.py` | Streamlit dashboard with profile selector, allocation charts, risk analytics |

## Constraints

- **No external APIs** — all features derived from provided historical data only.
- **Decision support focus** — portfolio construction and risk analytics, not just price forecasting.
- **Time-series aware validation** — no random shuffling of financial data.

## Assumptions & Limitations

- Risk-free rate default: 6% annualized (configurable in `portfolio_engine.py`).
- Forward 21-day return ≈ one trading month horizon.
- Past performance does not guarantee future results; this is a hackathon prototype.
# portfolio-optimization
# portfolio-optimization
# portfolio-optimization
# portfolio-optimization
# portfolio-optimization-cult
