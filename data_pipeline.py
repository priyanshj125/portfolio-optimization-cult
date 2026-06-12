"""
Module 1: Data Pipeline & Technical Feature Engineering
========================================================
Loads NIFTY-50 historical OHLCV data, engineers technical indicators,
and creates the forward-return target for predictive modeling.

All indicators are computed deterministically from price/volume data only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns expected in individual stock CSV files (Kaggle NIFTY-50 format)
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Turnover"]
OPTIONAL_COLUMNS = ["Symbol", "Series", "Prev Close", "Last", "VWAP", "Trades"]

# Feature columns produced by the pipeline (used downstream by predictor)
FEATURE_COLUMNS = [
    "SMA_50",
    "SMA_200",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Volatility_20_Ann",
    "Daily_Return",
    "Volume_Change",
    "Price_to_SMA50",
    "Price_to_SMA200",
]

TARGET_COLUMN = "Forward_21_Return"


# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_stock_csv(
    filepath: str | Path,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single stock CSV, parse dates, sort chronologically, and standardize columns.

    Parameters
    ----------
    filepath : str | Path
        Path to an individual stock CSV (e.g. archive/RELIANCE.csv).
    symbol : str, optional
        Override symbol name; if None, inferred from filename stem.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by Date with a Symbol column.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Stock file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    inferred_symbol = symbol or path.stem.upper()

    # Keep only columns that exist; warn on missing required ones
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")

    keep_cols = ["Date"] + [c for c in df.columns if c != "Date"]
    df = df[keep_cols].copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df["Symbol"] = inferred_symbol

    # Coerce numeric columns; invalid values become NaN
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Turnover"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("Date")
    return df


def load_all_stocks(
    data_dir: str | Path,
    symbols: Optional[Iterable[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all individual stock CSVs from a directory (excludes NIFTY50_all.csv and metadata).

    Parameters
    ----------
    data_dir : str | Path
        Directory containing per-stock CSV files (e.g. 'archive/').
    symbols : iterable of str, optional
        If provided, load only these symbols.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol -> cleaned DataFrame.
    """
    data_dir = Path(data_dir)
    exclude = {"NIFTY50_all", "stock_metadata"}
    csv_files = sorted(
        f for f in data_dir.glob("*.csv")
        if f.stem not in exclude
    )

    if symbols is not None:
        symbol_set = {s.upper() for s in symbols}
        csv_files = [f for f in csv_files if f.stem.upper() in symbol_set]

    stocks: dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        try:
            stocks[fp.stem.upper()] = load_stock_csv(fp)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fp.name, exc)

    logger.info("Loaded %d stock files from %s", len(stocks), data_dir)
    return stocks


def handle_missing_values(
    df: pd.DataFrame,
    price_cols: Optional[list[str]] = None,
    impute_volume: bool = True,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values in OHLCV data.

    Strategy
    --------
    1. Drop rows where Close is NaN (cannot compute returns).
    2. Forward-fill then backward-fill price columns (common for thin trading days).
    3. Impute Volume/Turnover with 0 or rolling median.
    4. Drop rows where >drop_threshold fraction of price cols are still NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input stock DataFrame (Date-indexed).
    price_cols : list[str], optional
        Price columns to impute. Defaults to OHLC.
    impute_volume : bool
        If True, fill missing Volume with 0 and Turnover with 0.
    drop_threshold : float
        Drop row if this fraction of price columns are NaN after imputation.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    out = df.copy()
    price_cols = price_cols or ["Open", "High", "Low", "Close"]

    out = out.dropna(subset=["Close"])

    for col in price_cols:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()

    if impute_volume:
        for col in ["Volume", "Turnover"]:
            if col in out.columns:
                out[col] = out[col].fillna(0)

    # Drop rows that still have too many missing price values
    if price_cols:
        nan_frac = out[price_cols].isna().mean(axis=1)
        out = out[nan_frac <= drop_threshold]

    return out


# ---------------------------------------------------------------------------
# Technical Indicators (manual implementations)
# ---------------------------------------------------------------------------

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average. Returns NaN for the first (window-1) observations."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with standard span parameter."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    14-day Relative Strength Index (Wilder's smoothing).

    RSI = 100 - (100 / (1 + RS)),  RS = avg_gain / avg_loss
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's EMA: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is 0 and avg_gain > 0, RSI = 100
    rsi = rsi.where(avg_loss > 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    MACD       = EMA(fast) - EMA(slow)
    Signal     = EMA(MACD, signal)
    Histogram  = MACD - Signal
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": histogram},
        index=close.index,
    )


def compute_annualized_volatility(
    close: pd.Series,
    window: int = 20,
    trading_days: int = 252,
) -> pd.Series:
    """
    Rolling annualized volatility from daily log returns.

    Vol = std(daily_returns, window) * sqrt(trading_days)
    """
    daily_returns = close.pct_change()
    rolling_std = daily_returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(trading_days)


def compute_forward_return(close: pd.Series, horizon: int = 21) -> pd.Series:
    """
    Forward percentage return over `horizon` trading days.

    Forward_21_Return = (Close[t+21] / Close[t] - 1) * 100
    """
    future_close = close.shift(-horizon)
    forward_return = (future_close / close - 1.0) * 100.0
    return forward_return


# ---------------------------------------------------------------------------
# Full Feature Engineering Pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    forward_horizon: int = 21,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """
    Apply full technical feature engineering to a single stock DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV data (Date-indexed, must contain 'Close').
    forward_horizon : int
        Days ahead for target variable (default 21 ~ 1 trading month).
    drop_warmup : bool
        If True, drop rows with NaN in any feature or target (post warm-up).

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns + engineered features + target.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    out = df.copy()
    close = out["Close"]

    # --- Moving averages ---
    out["SMA_50"] = compute_sma(close, 50)
    out["SMA_200"] = compute_sma(close, 200)

    # --- Momentum / trend ---
    out["RSI_14"] = compute_rsi(close, period=14)
    macd_df = compute_macd(close)
    out = out.join(macd_df)

    # --- Volatility ---
    out["Volatility_20_Ann"] = compute_annualized_volatility(close, window=20)
    out["Daily_Return"] = close.pct_change()

    # --- Volume-derived ---
    if "Volume" in out.columns:
        out["Volume_Change"] = out["Volume"].pct_change()
        # Replace inf from zero-volume days
        out["Volume_Change"] = out["Volume_Change"].replace([np.inf, -np.inf], np.nan)
    else:
        out["Volume_Change"] = np.nan

    # --- Relative price features (regime indicators) ---
    out["Price_to_SMA50"] = close / out["SMA_50"]
    out["Price_to_SMA200"] = close / out["SMA_200"]

    # --- Target variable ---
    out[TARGET_COLUMN] = compute_forward_return(close, horizon=forward_horizon)

    if drop_warmup:
        feature_and_target = FEATURE_COLUMNS + [TARGET_COLUMN]
        out = out.dropna(subset=feature_and_target)

    return out


def build_master_dataset(
    data_dir: str | Path,
    symbols: Optional[Iterable[str]] = None,
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """
    End-to-end pipeline: load all stocks, clean, engineer features, concatenate.

    Returns
    -------
    pd.DataFrame
        Combined multi-stock dataset with Symbol column and Date index.
    """
    stocks = load_all_stocks(data_dir, symbols=symbols)
    frames: list[pd.DataFrame] = []

    for symbol, raw_df in stocks.items():
        cleaned = handle_missing_values(raw_df)
        featured = engineer_features(cleaned, forward_horizon=forward_horizon)
        if featured.empty:
            logger.warning("No usable rows for %s after feature engineering.", symbol)
            continue
        featured["Symbol"] = symbol
        frames.append(featured)

    if not frames:
        raise RuntimeError("No stock data could be processed. Check data_dir and file formats.")

    master = pd.concat(frames, axis=0).sort_index()
    master.index.name = "Date"
    logger.info(
        "Master dataset: %d rows, %d symbols, %s to %s",
        len(master),
        master["Symbol"].nunique(),
        master.index.min().date(),
        master.index.max().date(),
    )
    return master


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load stock_metadata.csv for sector/industry enrichment."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# CLI entry point for quick validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    DATA_DIR = Path(__file__).parent / "archive"
    master_df = build_master_dataset(DATA_DIR)

    print("\n=== Data Pipeline Summary ===")
    print(f"Shape: {master_df.shape}")
    print(f"Symbols: {master_df['Symbol'].nunique()}")
    print(f"Date range: {master_df.index.min().date()} → {master_df.index.max().date()}")
    print(f"\nFeature columns sample:\n{master_df[FEATURE_COLUMNS].describe().T}")
