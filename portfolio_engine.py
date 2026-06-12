"""
Module 3: Risk Analytics & Portfolio Optimizer
===============================================
Computes risk metrics (Sharpe, Sortino, Max Drawdown) and constructs
portfolios for Conservative, Balanced, and Aggressive investor profiles.

All analytics are derived deterministically from historical daily returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06  # ~6% annualized (India historical avg; configurable)

InvestorProfile = Literal["conservative", "balanced", "aggressive"]


# ---------------------------------------------------------------------------
# Return series utilities
# ---------------------------------------------------------------------------

def compute_daily_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute daily returns from a wide price DataFrame (Date index, symbols as columns).

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, one column per symbol.
    method : str
        'simple' for pct_change, 'log' for log returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna(how="all")


def build_price_matrix(
    stock_dfs: dict[str, pd.DataFrame],
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Build a wide Close-price matrix from per-stock DataFrames.

    Drops symbols with fewer than min_history observations.
    """
    series_map: dict[str, pd.Series] = {}
    for symbol, df in stock_dfs.items():
        if "Close" not in df.columns or len(df) < min_history:
            continue
        series_map[symbol] = df["Close"]

    if not series_map:
        raise ValueError("No symbols with sufficient history.")

    prices = pd.DataFrame(series_map)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Forward-fill short gaps, then drop symbols still mostly NaN
    prices = prices.ffill()
    valid_frac = prices.notna().mean()
    keep = valid_frac[valid_frac >= 0.8].index
    prices = prices[keep].dropna(how="any")

    return prices


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS,
) -> float:
    """
    Annualized Sharpe Ratio.

    Sharpe = (mean(excess_return) / std(return)) * sqrt(periods)
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = r - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS,
) -> float:
    """
    Annualized Sortino Ratio (penalizes only downside volatility).
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = r - daily_rf
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0

    downside_std = np.sqrt((downside ** 2).mean())
    return float(excess.mean() / downside_std * np.sqrt(periods))


def maximum_drawdown(returns: pd.Series | np.ndarray) -> float:
    """
    Maximum Drawdown as a positive fraction (e.g. 0.35 = 35% peak-to-trough loss).
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0

    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(abs(drawdown.min()))


def annualized_volatility(
    returns: pd.Series | np.ndarray,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualized standard deviation of daily returns."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(periods))


def compute_asset_risk_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Compute risk metrics for each asset (column) in a returns DataFrame.
    """
    records: list[dict] = []
    for symbol in returns.columns:
        r = returns[symbol].dropna()
        records.append(
            {
                "Symbol": symbol,
                "Ann_Volatility": annualized_volatility(r),
                "Sharpe": sharpe_ratio(r, risk_free_rate),
                "Sortino": sortino_ratio(r, risk_free_rate),
                "Max_Drawdown": maximum_drawdown(r),
                "Mean_Daily_Return": r.mean(),
            }
        )
    return pd.DataFrame(records).set_index("Symbol")


def compute_portfolio_metrics(
    weights: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """
    Compute portfolio-level risk metrics given weights and asset return series.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()  # normalize

    port_returns = returns.dot(w)
    return {
        "Ann_Volatility": annualized_volatility(port_returns),
        "Sharpe": sharpe_ratio(port_returns, risk_free_rate),
        "Sortino": sortino_ratio(port_returns, risk_free_rate),
        "Max_Drawdown": maximum_drawdown(port_returns),
        "Mean_Ann_Return": float(port_returns.mean() * TRADING_DAYS),
    }


# ---------------------------------------------------------------------------
# Portfolio optimization helpers
# ---------------------------------------------------------------------------

def _portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def _neg_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> float:
    port_return = weights @ mean_returns
    port_vol = _portfolio_vol(weights, cov)
    if port_vol == 0:
        return 0.0
    return -((port_return - risk_free_rate) / port_vol)


def mean_variance_max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = 0.25,
) -> pd.Series:
    """
    Mean-Variance Optimization: maximize Sharpe ratio subject to:
      - weights sum to 1
      - 0 <= w_i <= max_weight (diversification constraint)
    """
    n = returns.shape[1]
    mean_ret = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS

    # Regularize covariance for numerical stability
    cov += np.eye(n) * 1e-8

    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        _neg_sharpe,
        x0,
        args=(mean_ret, cov, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        logger.warning("MVO did not converge: %s. Using equal weights.", result.message)
        weights = x0
    else:
        weights = result.x

    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()

    return pd.Series(weights, index=returns.columns, name="weight")


def min_volatility_portfolio(
    returns: pd.DataFrame,
    max_weight: float = 0.20,
) -> pd.Series:
    """Minimum-variance portfolio (defensive allocation)."""
    n = returns.shape[1]
    cov = returns.cov().values * TRADING_DAYS
    cov += np.eye(n) * 1e-8

    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        lambda w, c: _portfolio_vol(w, c),
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    weights = result.x if result.success else x0
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns, name="weight")


def min_drawdown_portfolio(
    returns: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    top_n: int = 15,
) -> pd.Series:
    """
    Conservative heuristic: equal-weight the lowest max-drawdown assets.

    Selects top_n assets by lowest historical max drawdown, then equal-weights.
    """
    ranked = risk_metrics.sort_values("Max_Drawdown").head(top_n)
    selected = ranked.index.tolist()
    w = 1.0 / len(selected)
    weights = pd.Series(0.0, index=returns.columns, name="weight")
    weights[selected] = w
    return weights


def momentum_score_portfolio(
    returns: pd.DataFrame,
    lookback: int = 126,
    top_n: int = 10,
) -> pd.Series:
    """
    Aggressive momentum scoring: rank by trailing 6-month total return,
    allocate proportionally to positive momentum stocks.
    """
    if len(returns) < lookback:
        lookback = len(returns) // 2

    trailing = (1 + returns.tail(lookback)).prod() - 1
    trailing = trailing[trailing > 0].sort_values(ascending=False).head(top_n)

    if trailing.empty:
        # Fallback: equal weight all assets
        n = returns.shape[1]
        return pd.Series(1.0 / n, index=returns.columns, name="weight")

    weights = trailing / trailing.sum()
    full = pd.Series(0.0, index=returns.columns, name="weight")
    full[weights.index] = weights.values
    return full


def equal_risk_contribution(
    returns: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> pd.Series:
    """
    Equal Risk Contribution (ERC) portfolio via iterative risk-budgeting.

    Each asset contributes equally to total portfolio volatility.
    """
    n = returns.shape[1]
    cov = returns.cov().values * TRADING_DAYS
    cov += np.eye(n) * 1e-8

    w = np.ones(n) / n
    for _ in range(max_iter):
        port_vol = _portfolio_vol(w, cov)
        if port_vol == 0:
            break

        marginal = cov @ w
        risk_contrib = w * marginal / port_vol
        target = port_vol / n

        # Multiplicative update toward equal risk contribution
        w = w * (target / (risk_contrib + 1e-12))
        w = np.clip(w, 0, None)
        w = w / w.sum()

        if np.max(np.abs(risk_contrib - target)) < tol:
            break

    return pd.Series(w, index=returns.columns, name="weight")


def balanced_portfolio(
    returns: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    defensive_weight: float = 0.5,
) -> pd.Series:
    """
    Balanced profile: blend defensive (min-vol) and aggressive (max-Sharpe) sleeves.

    Default 50/50 split between the two optimized sub-portfolios.
    """
    defensive_weight = np.clip(defensive_weight, 0.0, 1.0)
    aggressive_weight = 1.0 - defensive_weight

    w_def = min_volatility_portfolio(returns)
    w_agg = mean_variance_max_sharpe(returns)

    combined = defensive_weight * w_def + aggressive_weight * w_agg
    combined = combined / combined.sum()
    return combined


# ---------------------------------------------------------------------------
# Profile-based allocation
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRecommendation:
    """Structured output for a recommended portfolio."""

    profile: str
    weights: pd.Series
    metrics: dict[str, float]
    asset_metrics: pd.DataFrame
    rationale: str


def allocate_portfolio(
    returns: pd.DataFrame,
    profile: InvestorProfile,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PortfolioRecommendation:
    """
    Generate a portfolio allocation for the given investor profile.

    Profiles
    --------
    conservative : Minimize drawdown & volatility (low-vol + low-drawdown bias).
    aggressive   : Maximize Sharpe via MVO, with momentum tilt as secondary signal.
    balanced     : Equal-risk contribution blend of defensive and aggressive sleeves.
    """
    profile = profile.lower()  # type: ignore[assignment]
    asset_metrics = compute_asset_risk_metrics(returns, risk_free_rate)

    if profile == "conservative":
        # Blend min-vol (60%) and min-drawdown equal-weight (40%)
        w_vol = min_volatility_portfolio(returns, max_weight=0.15)
        w_dd = min_drawdown_portfolio(returns, asset_metrics, top_n=12)
        weights = 0.6 * w_vol + 0.4 * w_dd
        weights = weights / weights.sum()
        rationale = (
            "Conservative: 60% minimum-volatility optimization + 40% equal-weight "
            "in lowest max-drawdown assets. Prioritizes capital preservation."
        )

    elif profile == "aggressive":
        w_mvo = mean_variance_max_sharpe(returns, risk_free_rate, max_weight=0.25)
        w_mom = momentum_score_portfolio(returns, lookback=126, top_n=10)
        weights = 0.7 * w_mvo + 0.3 * w_mom
        weights = weights / weights.sum()
        rationale = (
            "Aggressive: 70% mean-variance max-Sharpe + 30% 6-month momentum tilt. "
            "Seeks highest risk-adjusted return with concentration in trending names."
        )

    elif profile == "balanced":
        weights = balanced_portfolio(returns, asset_metrics, defensive_weight=0.5)
        rationale = (
            "Balanced: 50/50 blend of minimum-volatility and max-Sharpe portfolios. "
            "Provides moderate risk-return trade-off across market regimes."
        )

    else:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose: conservative, balanced, aggressive."
        )

    # Zero out negligible weights (< 0.5%) for cleaner display
    weights[weights < 0.005] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum()

    port_metrics = compute_portfolio_metrics(weights, returns, risk_free_rate)

    return PortfolioRecommendation(
        profile=profile,
        weights=weights[weights > 0].sort_values(ascending=False),
        metrics=port_metrics,
        asset_metrics=asset_metrics,
        rationale=rationale,
    )


def get_portfolio_return_series(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """Historical daily return series for a weighted portfolio."""
    aligned_weights = weights.reindex(returns.columns, fill_value=0.0)
    aligned_weights = aligned_weights / aligned_weights.sum()
    return returns.dot(aligned_weights)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from data_pipeline import load_all_stocks, handle_missing_values

    DATA_DIR = __import__("pathlib").Path(__file__).parent / "archive"
    stocks = load_all_stocks(DATA_DIR)
    cleaned = {s: handle_missing_values(df) for s, df in stocks.items()}
    prices = build_price_matrix(cleaned)
    returns = compute_daily_returns(prices)

    for profile in ["conservative", "balanced", "aggressive"]:
        rec = allocate_portfolio(returns, profile)  # type: ignore[arg-type]
        print(f"\n{'=' * 50}")
        print(f"  {profile.upper()} Portfolio")
        print(f"{'=' * 50}")
        print(f"Rationale: {rec.rationale}\n")
        print("Top Holdings:")
        for sym, w in rec.weights.head(10).items():
            print(f"  {sym:<15} {w:.1%}")
        print(f"\nPortfolio Metrics:")
        for k, v in rec.metrics.items():
            print(f"  {k:<20} {v:.4f}")
