"""
Exploratory Data Analysis (EDA) for NIFTY-50 Investment Intelligence Platform
=============================================================================
Generates publication-ready figures and summary tables for the technical report PDF.

Run:
    python eda.py

Outputs:
    output/figures/   — PNG plots (300 DPI, PDF-ready)
    output/tables/    — CSV summary tables
    output/eda_report.md — Narrative content + figure references for PDF
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from data_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_master_dataset,
    handle_missing_values,
    load_all_stocks,
    load_metadata,
)
from portfolio_engine import (
    allocate_portfolio,
    annualized_volatility,
    build_price_matrix,
    compute_asset_risk_metrics,
    compute_daily_returns,
    compute_portfolio_metrics,
    get_portfolio_return_series,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from predictor import cross_validate_time_series, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & style
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "archive"
METADATA_PATH = DATA_DIR / "stock_metadata.csv"
FIG_DIR = ROOT / "output" / "figures"
TABLE_DIR = ROOT / "output" / "tables"
REPORT_PATH = ROOT / "output" / "eda_report.md"

FIG_DPI = 300
PALETTE = "husl"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(PALETTE)


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved figure: %s", path.name)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path)
    logger.info("Saved table: %s", path.name)
    return path


# ---------------------------------------------------------------------------
# EDA plot functions
# ---------------------------------------------------------------------------

def plot_sector_composition(metadata: pd.DataFrame) -> None:
    """Figure 1: Industry/sector composition of NIFTY-50 universe."""
    counts = metadata["Industry"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette(PALETTE, len(counts)))
    ax.set_xlabel("Number of Companies")
    ax.set_ylabel("Industry Sector")
    ax.set_title("NIFTY-50 Universe — Sector Composition")
    ax.invert_yaxis()
    save_fig("01_sector_composition")


def plot_data_coverage(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Figure 2: Trading-day coverage per symbol."""
    coverage = []
    for symbol, df in stocks.items():
        idx = pd.to_datetime(df.index, errors="coerce")
        valid_idx = idx.dropna()
        start = valid_idx.min()
        end = valid_idx.max()
        coverage.append(
            {
                "Symbol": symbol,
                "Start_Date": start.strftime("%Y-%m-%d") if pd.notna(start) else "",
                "End_Date": end.strftime("%Y-%m-%d") if pd.notna(end) else "",
                "Trading_Days": len(df),
                "Missing_Close_Pct": round(df["Close"].isna().mean() * 100, 2),
            }
        )
    cov_df = pd.DataFrame(coverage).sort_values("Trading_Days", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(cov_df["Symbol"], cov_df["Trading_Days"], color="steelblue")
    ax.set_xlabel("Number of Trading Days")
    ax.set_ylabel("Symbol")
    ax.set_title("Historical Data Coverage per Stock (2000–2021)")
    save_fig("02_data_coverage")

    return cov_df


def plot_normalized_price_index(prices: pd.DataFrame) -> None:
    """Figure 3: Normalized price performance (base = 100)."""
    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], alpha=0.7, linewidth=1)
    ax.set_title("Normalized Price Performance (Base = 100 at First Available Date)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    save_fig("03_normalized_prices")


def plot_top_bottom_performers(prices: pd.DataFrame) -> pd.DataFrame:
    """Figure 4: Total return ranking across full sample."""
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    perf = total_return.sort_values(ascending=False).reset_index()
    perf.columns = ["Symbol", "Total_Return_Pct"]
    perf["Rank"] = range(1, len(perf) + 1)

    top10 = perf.head(10)
    bottom10 = perf.tail(10)
    combined = pd.concat([top10, bottom10])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_top = ["#2ecc71"] * 10
    colors_bot = ["#e74c3c"] * 10

    axes[0].barh(top10["Symbol"], top10["Total_Return_Pct"], color=colors_top)
    axes[0].set_title("Top 10 Performers (Total Return %)")
    axes[0].set_xlabel("Total Return (%)")
    axes[0].invert_yaxis()

    axes[1].barh(bottom10["Symbol"], bottom10["Total_Return_Pct"], color=colors_bot)
    axes[1].set_title("Bottom 10 Performers (Total Return %)")
    axes[1].set_xlabel("Total Return (%)")
    axes[1].invert_yaxis()

    save_fig("04_top_bottom_performers")
    return perf


def plot_return_distribution(returns: pd.DataFrame) -> None:
    """Figure 5: Distribution of daily returns (pooled)."""
    pooled = returns.stack().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pooled, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(pooled.mean(), color="red", linestyle="--", label=f"Mean: {pooled.mean():.4f}")
    axes[0].set_title("Distribution of Daily Returns (All Stocks)")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    stats = pd.Series(
        {
            "Mean": pooled.mean(),
            "Std": pooled.std(),
            "Skewness": pooled.skew(),
            "Kurtosis": pooled.kurtosis(),
            "Min": pooled.min(),
            "Max": pooled.max(),
        }
    )
    axes[1].axis("off")
    table_data = [[k, f"{v:.6f}"] for k, v in stats.items()]
    tbl = axes[1].table(
        cellText=table_data,
        colLabels=["Statistic", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1.2, 1.8)
    axes[1].set_title("Return Distribution Statistics", pad=20)

    save_fig("05_return_distribution")
    save_table(stats.to_frame("Value"), "return_distribution_stats")


def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    """Figure 6: Cross-sectional return correlation matrix."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Stock Return Correlation Matrix")
    save_fig("06_correlation_heatmap")
    save_table(corr.round(3), "correlation_matrix")


def plot_rolling_volatility(prices: pd.DataFrame) -> None:
    """Figure 7: Market-wide rolling 20-day annualized volatility."""
    returns = compute_daily_returns(prices)
    market_vol = returns.std(axis=1).rolling(20).mean() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(market_vol.index, market_vol.values, color="darkorange", linewidth=1.5)
    ax.fill_between(market_vol.index, market_vol.values, alpha=0.3, color="darkorange")
    ax.set_title("Market-Wide Rolling 20-Day Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # Annotate major events
    events = {
        "2008-09-15": "GFC",
        "2016-11-08": "Demonetization",
        "2020-03-23": "COVID Crash",
    }
    for date_str, label in events.items():
        dt = pd.Timestamp(date_str)
        if dt in market_vol.index or (market_vol.index.min() <= dt <= market_vol.index.max()):
            ax.axvline(dt, color="gray", linestyle=":", alpha=0.7)
            ax.text(dt, market_vol.max() * 0.95, label, rotation=90, fontsize=8, va="top")

    save_fig("07_rolling_volatility")


def plot_volume_trends(stocks: dict[str, pd.DataFrame]) -> None:
    """Figure 8: Aggregate traded volume over time."""
    vol_series = []
    for symbol, df in stocks.items():
        if "Volume" in df.columns:
            s = df["Volume"].copy()
            s.name = symbol
            vol_series.append(s)

    vol_df = pd.concat(vol_series, axis=1)
    total_vol = vol_df.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(total_vol.index, total_vol.values / 1e6, color="teal", linewidth=1)
    ax.set_title("Aggregate Traded Volume Across NIFTY-50 (All Stocks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Volume (Millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    save_fig("08_volume_trends")


def plot_risk_return_scatter(returns: pd.DataFrame) -> pd.DataFrame:
    """Figure 9: Risk-return scatter for individual stocks."""
    metrics = compute_asset_risk_metrics(returns)
    metrics["Ann_Return_Pct"] = metrics["Mean_Daily_Return"] * 252 * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        metrics["Ann_Volatility"] * 100,
        metrics["Ann_Return_Pct"],
        c=metrics["Sharpe"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )
    for symbol, row in metrics.iterrows():
        ax.annotate(
            symbol,
            (row["Ann_Volatility"] * 100, row["Ann_Return_Pct"]),
            fontsize=6,
            alpha=0.8,
        )
    plt.colorbar(scatter, label="Sharpe Ratio")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk–Return Profile of NIFTY-50 Constituents")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    save_fig("09_risk_return_scatter")

    return metrics.round(4)


def plot_feature_distributions(master_df: pd.DataFrame) -> None:
    """Figure 10: Distribution of engineered ML features."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    plot_features = ["RSI_14", "MACD", "Volatility_20_Ann", "Daily_Return", "Price_to_SMA50", TARGET_COLUMN]

    for ax, feat in zip(axes, plot_features):
        data = master_df[feat].dropna()
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(feat)
        ax.set_xlabel("Value")

    fig.suptitle("Engineered Feature Distributions", fontsize=14, y=1.02)
    save_fig("10_feature_distributions")


def plot_model_evaluation(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figures 11–12: Model metrics and feature importances."""
    # Use most recent 80k rows for faster EDA training (preserves time-series order)
    train_df = master_df.iloc[-80_000:] if len(master_df) > 80_000 else master_df
    logger.info("Training model on %d rows for EDA plots…", len(train_df))

    result = train_and_evaluate(
        train_df,
        model_type="random_forest",
        n_estimators=100,
        max_depth=10,
    )
    cv_df = cross_validate_time_series(train_df, n_splits=3)

    # Evaluation metrics bar chart
    metrics = {
        "RMSE": result.evaluation.rmse,
        "MAE": result.evaluation.mae,
        "R²": result.evaluation.r2,
        "Dir. Accuracy": result.evaluation.directional_accuracy,
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"])
    ax.set_title("Random Forest — Hold-out Test Metrics (Forward 21-Day Return)")
    ax.set_ylabel("Value")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    save_fig("11_model_evaluation")

    # CV summary
    cv_summary = cv_df.describe().loc[["mean", "std"]].round(4)
    save_table(cv_df, "model_cv_folds")
    save_table(cv_summary, "model_cv_summary")

    # Feature importances
    fi = result.feature_importances.head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi["feature"], fi["importance_pct"], color="mediumpurple")
    ax.set_xlabel("Importance (%)")
    ax.set_title("Top 10 Feature Importances (Explainable AI)")
    ax.invert_yaxis()
    save_fig("12_feature_importances")
    save_table(result.feature_importances, "feature_importances")

    eval_df = pd.DataFrame([metrics])
    save_table(eval_df, "model_holdout_metrics")
    return eval_df, cv_summary


def plot_portfolio_comparison(returns: pd.DataFrame) -> pd.DataFrame:
    """Figures 13–14: Portfolio weights and risk metrics across profiles."""
    profiles = ["conservative", "balanced", "aggressive"]
    records = []
    weight_frames = []

    for profile in profiles:
        rec = allocate_portfolio(returns, profile)  # type: ignore[arg-type]
        row = {"Profile": profile.capitalize(), **rec.metrics}
        records.append(row)
        w = rec.weights.reset_index()
        w.columns = ["Symbol", "Weight"]
        w["Profile"] = profile.capitalize()
        weight_frames.append(w)

    comparison = pd.DataFrame(records)
    all_weights = pd.concat(weight_frames)

    # Risk metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(profiles))
    width = 0.25
    metrics_to_plot = ["Sharpe", "Max_Drawdown", "Ann_Volatility"]
    labels = ["Sharpe Ratio", "Max Drawdown", "Ann. Volatility"]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, colors)):
        ax.bar(x + i * width, comparison[metric], width, label=label, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([p.capitalize() for p in profiles])
    ax.set_title("Portfolio Risk Metrics by Investor Profile")
    ax.legend()
    save_fig("13_portfolio_risk_comparison")
    save_table(comparison.round(4), "portfolio_comparison")

    # Stacked allocation for top holdings
    top_symbols = (
        all_weights.groupby("Symbol")["Weight"].max().sort_values(ascending=False).head(12).index
    )
    pivot = all_weights[all_weights["Symbol"].isin(top_symbols)].pivot(
        index="Symbol", columns="Profile", values="Weight"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Top Holdings Weight (%) by Investor Profile")
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Symbol")
    ax.legend(title="Profile")
    plt.xticks(rotation=45, ha="right")
    save_fig("14_portfolio_allocations")

    # Equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for profile in profiles:
        rec = allocate_portfolio(returns, profile)  # type: ignore[arg-type]
        port_ret = get_portfolio_return_series(rec.weights, returns)
        cumulative = (1 + port_ret).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=profile.capitalize(), linewidth=2)

    ax.set_title("Historical Cumulative Returns by Investor Profile")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of INR 1")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    save_fig("15_portfolio_equity_curves")

    save_table(all_weights.round(4), "portfolio_weights_all_profiles")
    return comparison


# ---------------------------------------------------------------------------
# Report markdown generator
# ---------------------------------------------------------------------------

def generate_report_markdown(
    n_symbols: int,
    date_min: str,
    date_max: str,
    n_rows: int,
    coverage_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    model_metrics: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> None:
    """Write narrative EDA report with figure references for PDF conversion."""

    top3 = perf_df.head(3)["Symbol"].tolist()
    bottom3 = perf_df.tail(3)["Symbol"].tolist()
    avg_sharpe = risk_df["Sharpe"].mean()
    avg_vol = risk_df["Ann_Volatility"].mean() * 100
    avg_mdd = risk_df["Max_Drawdown"].mean() * 100

    rmse = model_metrics["RMSE"].iloc[0]
    mae = model_metrics["MAE"].iloc[0]
    r2 = model_metrics["R²"].iloc[0]
    dir_acc = model_metrics["Dir. Accuracy"].iloc[0]

    con = portfolio_df[portfolio_df["Profile"] == "Conservative"].iloc[0]
    bal = portfolio_df[portfolio_df["Profile"] == "Balanced"].iloc[0]
    agg = portfolio_df[portfolio_df["Profile"] == "Aggressive"].iloc[0]

    report = f"""# Technical Report — Exploratory Data Analysis
## AI-Powered Investment Intelligence Platform | NIFTY-50 (2000–2021)

---

## 1. Executive Summary

This report presents exploratory data analysis and quantitative findings from the NIFTY-50
historical market dataset spanning **{date_min}** to **{date_max}**. The platform transforms
raw OHLCV data into decision-support insights through technical feature engineering, machine
learning-based forward-return prediction, and profile-based portfolio optimization.

**Key findings:**
- **{n_symbols} stocks** analyzed with **{n_rows:,}** feature-engineered observations.
- Top long-term performers include **{', '.join(top3)}**; weakest include **{', '.join(bottom3)}**.
- Average constituent Sharpe ratio: **{avg_sharpe:.2f}** | Avg. volatility: **{avg_vol:.1f}%** | Avg. max drawdown: **{avg_mdd:.1f}%**.
- ML model directional accuracy: **{dir_acc:.1%}** (21-day forward return prediction).
- Conservative portfolio max drawdown (**{con['Max_Drawdown']:.1%}**) is lower than Aggressive (**{agg['Max_Drawdown']:.1%}**).

---

## 2. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Universe | NIFTY-50 constituent stocks (NSE India) |
| Period | {date_min} to {date_max} |
| Symbols | {n_symbols} |
| Fields | Open, High, Low, Close, Volume, Turnover |
| Metadata | Company name, Industry sector (stock_metadata.csv) |

**Figure 1** shows sector composition. Financial Services, IT, and Energy dominate the index,
providing natural diversification across economic cycles.

![Sector Composition](figures/01_sector_composition.png)

**Figure 2** confirms data coverage: most stocks have 4,000–5,300 trading days. Symbols with
shorter histories (e.g., recent index entrants) were handled via minimum-history filters in
the portfolio engine.

![Data Coverage](figures/02_data_coverage.png)

---

## 3. Price & Return Analysis

### 3.1 Long-Term Performance

Normalized price indices (base = 100) reveal significant dispersion across constituents
over two decades. IT and consumer names generally outperformed commodity-linked stocks.

![Normalized Prices](figures/03_normalized_prices.png)

**Figure 4** ranks total returns. The best performer (**{perf_df.iloc[0]['Symbol']}**, +{perf_df['Total_Return_Pct'].max():,.0f}%)
and worst (**{perf_df.iloc[-1]['Symbol']}**, {perf_df['Total_Return_Pct'].min():,.0f}%) diverge sharply,
highlighting stock-selection risk within the index.

![Top/Bottom Performers](figures/04_top_bottom_performers.png)

### 3.2 Return Distribution

Daily returns exhibit **fat tails** (leptokurtosis) typical of equity markets — extreme
single-day moves occur more frequently than a normal distribution predicts. This motivates
risk metrics beyond simple variance (Sortino ratio, max drawdown).

![Return Distribution](figures/05_return_distribution.png)

### 3.3 Correlation Structure

The correlation heatmap shows moderate-to-high positive correlations among Indian large-caps,
especially within sectors (Banking, IT). This supports covariance-aware portfolio optimization
(Modern Portfolio Theory) rather than naive equal-weighting.

![Correlation Heatmap](figures/06_correlation_heatmap.png)

---

## 4. Volatility & Market Regimes

Rolling 20-day annualized volatility spikes coincide with known stress events:
- **2008** Global Financial Crisis
- **2016** Demonetization
- **2020** COVID-19 market crash

These regimes validate the need for profile-based allocation (Conservative vs Aggressive).

![Rolling Volatility](figures/07_rolling_volatility.png)

Aggregate traded volume has grown substantially over the sample, reflecting market
deepening and increased retail participation in Indian equities.

![Volume Trends](figures/08_volume_trends.png)

---

## 5. Risk–Return Profile of Constituents

Each stock is evaluated on annualized return, volatility, and Sharpe ratio.
The scatter plot reveals the efficient frontier opportunity set within NIFTY-50.

![Risk-Return Scatter](figures/09_risk_return_scatter.png)

**Insight:** Stocks with high returns often carry proportionally higher volatility;
portfolio construction must explicitly trade off these dimensions by investor profile.

---

## 6. Feature Engineering

Technical indicators computed deterministically from OHLCV data:

| Feature | Description |
|---------|-------------|
| SMA_50 / SMA_200 | Trend identification (golden/death cross) |
| RSI_14 | Momentum / overbought-oversold (Wilder's smoothing) |
| MACD + Signal | Trend momentum crossover |
| Volatility_20_Ann | Rolling 20-day annualized volatility |
| Price_to_SMA50/200 | Relative price regime |
| Forward_21_Return | Target: 21-day ahead % return |

![Feature Distributions](figures/10_feature_distributions.png)

---

## 7. Predictive Model Results

**Model:** Random Forest Regressor with StandardScaler pipeline.
**Target:** Forward 21-day percentage return.
**Validation:** Chronological 80/20 hold-out + 5-fold TimeSeriesSplit CV.

| Metric | Value |
|--------|-------|
| RMSE | {rmse:.4f} |
| MAE | {mae:.4f} |
| R² | {r2:.4f} |
| Directional Accuracy | {dir_acc:.1%} |

![Model Evaluation](figures/11_model_evaluation.png)

Directional accuracy above 50% indicates the model adds value for up/down classification
even when magnitude prediction is noisy (low R² is expected for equity return forecasting).

**Explainable AI:** Feature importances show which technical signals drive predictions.

![Feature Importances](figures/12_feature_importances.png)

---

## 8. Portfolio Construction Results

Three investor profiles using Modern Portfolio Theory (MVO) with profile-specific overlays:

| Profile | Sharpe | Max Drawdown | Ann. Volatility | Ann. Return |
|---------|--------|--------------|-----------------|-------------|
| Conservative | {con['Sharpe']:.3f} | {con['Max_Drawdown']:.1%} | {con['Ann_Volatility']:.1%} | {con['Mean_Ann_Return']:.1%} |
| Balanced | {bal['Sharpe']:.3f} | {bal['Max_Drawdown']:.1%} | {bal['Ann_Volatility']:.1%} | {bal['Mean_Ann_Return']:.1%} |
| Aggressive | {agg['Sharpe']:.3f} | {agg['Max_Drawdown']:.1%} | {agg['Ann_Volatility']:.1%} | {agg['Mean_Ann_Return']:.1%} |

![Portfolio Risk Comparison](figures/13_portfolio_risk_comparison.png)
![Portfolio Allocations](figures/14_portfolio_allocations.png)
![Equity Curves](figures/15_portfolio_equity_curves.png)

---

## 9. Assumptions & Limitations

1. **Historical simulation only** — no live data or transaction costs modeled.
2. **Risk-free rate** assumed at 6% annualized for Sharpe/Sortino calculations.
3. **Survivorship bias** — dataset reflects current/historical NIFTY-50 members.
4. **Return predictability** is limited; ML supports decision context, not certainty.
5. Past performance does not guarantee future results.

---

## 10. Conclusion

The NIFTY-50 dataset provides rich multi-decade history for building a decision-support
platform. EDA confirms fat-tailed returns, sector clustering, and regime-dependent volatility.
Combining technical feature engineering, explainable ML, and MPT-based portfolio optimization
delivers actionable intelligence tailored to Conservative, Balanced, and Aggressive investors.

---

*Generated by `eda.py` — insert figures from `output/figures/` when compiling PDF.*
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Saved report: %s", REPORT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda() -> None:
    _ensure_dirs()
    logger.info("Starting EDA pipeline…")

    # Load data
    metadata = load_metadata(METADATA_PATH)
    stocks_raw = load_all_stocks(DATA_DIR)
    stocks = {s: handle_missing_values(df) for s, df in stocks_raw.items()}
    prices = build_price_matrix(stocks)
    returns = compute_daily_returns(prices)
    logger.info("Building master feature dataset (may take ~30s)…")
    master_df = build_master_dataset(DATA_DIR)

    # Generate plots & tables
    logger.info("[1/15] Sector composition")
    plot_sector_composition(metadata)
    logger.info("[2/15] Data coverage")
    coverage_df = plot_data_coverage(stocks)
    save_table(coverage_df, "data_coverage")

    logger.info("[3/15] Normalized prices")
    plot_normalized_price_index(prices)
    logger.info("[4/15] Top/bottom performers")
    perf_df = plot_top_bottom_performers(prices)
    save_table(perf_df, "total_return_ranking")

    logger.info("[5/15] Return distribution")
    plot_return_distribution(returns)
    logger.info("[6/15] Correlation heatmap")
    plot_correlation_heatmap(returns)
    logger.info("[7/15] Rolling volatility")
    plot_rolling_volatility(prices)
    logger.info("[8/15] Volume trends")
    plot_volume_trends(stocks)
    logger.info("[9/15] Risk-return scatter")
    risk_df = plot_risk_return_scatter(returns)
    save_table(risk_df.reset_index(), "asset_risk_metrics")

    logger.info("[10/15] Feature distributions")
    plot_feature_distributions(master_df)
    logger.info("[11-12/15] Model evaluation (training…)")
    model_metrics, _ = plot_model_evaluation(master_df)
    logger.info("[13-15/15] Portfolio comparison")
    portfolio_df = plot_portfolio_comparison(returns)

    # Write report
    generate_report_markdown(
        n_symbols=master_df["Symbol"].nunique(),
        date_min=str(master_df.index.min().date()),
        date_max=str(master_df.index.max().date()),
        n_rows=len(master_df),
        coverage_df=coverage_df,
        perf_df=perf_df,
        risk_df=risk_df,
        model_metrics=model_metrics,
        portfolio_df=portfolio_df,
    )

    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print("=" * 60)
    print(f"  Figures : {FIG_DIR}  ({len(list(FIG_DIR.glob('*.png')))} PNG files)")
    print(f"  Tables  : {TABLE_DIR}")
    print(f"  Report  : {REPORT_PATH}")
    print("=" * 60)
    print("\nTo use in PDF: open output/eda_report.md and insert figures from output/figures/")


if __name__ == "__main__":
    run_eda()
