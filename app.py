"""
Module 4: Streamlit Dashboard
==============================
Interactive decision-support dashboard tying together data pipeline,
predictive engine, and portfolio optimizer modules.

Run:  streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (
    build_master_dataset,
    engineer_features,
    handle_missing_values,
    load_all_stocks,
    load_metadata,
)
from portfolio_engine import (
    allocate_portfolio,
    build_price_matrix,
    compute_daily_returns,
    compute_portfolio_metrics,
    get_portfolio_return_series,
    maximum_drawdown,
    sharpe_ratio,
)
from predictor import train_and_evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "archive"
METADATA_PATH = DATA_DIR / "stock_metadata.csv"

PROFILE_OPTIONS = {
    "Conservative": "conservative",
    "Balanced": "balanced",
    "Aggressive": "aggressive",
}

PROFILE_DESCRIPTIONS = {
    "Conservative": (
        "Prioritizes capital preservation via minimum-volatility optimization "
        "and low max-drawdown stock selection."
    ),
    "Balanced": (
        "Blends defensive (min-vol) and aggressive (max-Sharpe) sleeves "
        "for a moderate risk-return profile."
    ),
    "Aggressive": (
        "Maximizes risk-adjusted returns using mean-variance optimization "
        "with a momentum tilt toward trending stocks."
    ),
}


# ---------------------------------------------------------------------------
# Cached data loaders (Streamlit performance)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and engineering features…")
def load_master_data() -> pd.DataFrame:
    return build_master_dataset(DATA_DIR)


@st.cache_data(show_spinner="Building price matrix…")
def load_prices_and_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks = load_all_stocks(DATA_DIR)
    cleaned = {s: handle_missing_values(df) for s, df in stocks.items()}
    prices = build_price_matrix(cleaned)
    returns = compute_daily_returns(prices)
    return prices, returns


@st.cache_resource(show_spinner="Training predictive model…")
def get_trained_model(_master_df: pd.DataFrame):
    """Cache the fitted model (resource, not data)."""
    return train_and_evaluate(_master_df, model_type="random_forest")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_allocation_chart(weights: pd.Series) -> None:
    """Pie chart of portfolio weights."""
    fig = px.pie(
        values=weights.values,
        names=weights.index,
        title="Recommended Portfolio Allocation",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=450)
    st.plotly_chart(fig, use_container_width=True)


def render_allocation_table(weights: pd.Series, metadata: pd.DataFrame | None) -> None:
    """Table of holdings with sector info when metadata is available."""
    table = pd.DataFrame({"Weight (%)": (weights * 100).round(2)})
    table.index.name = "Symbol"

    if metadata is not None and "Symbol" in metadata.columns:
        meta = metadata.set_index("Symbol")
        table = table.join(meta[["Industry"]], how="left")

    table = table.sort_values("Weight (%)", ascending=False)
    st.dataframe(table, use_container_width=True)


def render_risk_metrics(metrics: dict[str, float]) -> None:
    """Display key portfolio risk metrics as metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
    col2.metric("Max Drawdown", f"{metrics['Max_Drawdown']:.1%}")
    col3.metric("Ann. Volatility", f"{metrics['Ann_Volatility']:.1%}")
    col4.metric("Ann. Return", f"{metrics['Mean_Ann_Return']:.1%}")


def render_equity_curve(port_returns: pd.Series, title: str) -> None:
    """Cumulative return chart for the portfolio."""
    cumulative = (1 + port_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of ₹1",
        height=400,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(port_returns: pd.Series) -> None:
    """Underwater (drawdown) chart."""
    cumulative = (1 + port_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="crimson"),
        )
    )
    fig.update_layout(
        title="Portfolio Drawdown History",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NIFTY-50 Investment Intelligence",
        page_icon="📈",
        layout="wide",
    )

    st.title("AI-Powered Investment Intelligence Platform")
    st.caption(
        "Decision-support system built on NIFTY-50 historical data (2000–2021). "
        "All analytics computed deterministically from OHLCV — no external APIs."
    )

    # --- Sidebar ---
    st.sidebar.header("Investor Profile")
    selected_profile = st.sidebar.selectbox(
        "Select your risk profile",
        options=list(PROFILE_OPTIONS.keys()),
        index=1,
    )
    profile_key = PROFILE_OPTIONS[selected_profile]
    st.sidebar.info(PROFILE_DESCRIPTIONS[selected_profile])

    show_model = st.sidebar.checkbox("Show ML Predictor Insights", value=False)
    show_stock_explorer = st.sidebar.checkbox("Stock Explorer", value=False)

    # --- Load data ---
    try:
        master_df = load_master_data()
        prices, returns = load_prices_and_returns()
        metadata = load_metadata(METADATA_PATH) if METADATA_PATH.exists() else None
    except Exception as exc:
        st.error(f"Failed to load data from `{DATA_DIR}`: {exc}")
        st.stop()

    # --- Portfolio recommendation ---
    st.header(f"Portfolio Recommendation — {selected_profile}")

    with st.spinner("Optimizing portfolio allocation…"):
        recommendation = allocate_portfolio(returns, profile_key)  # type: ignore[arg-type]

    st.markdown(f"**Strategy:** {recommendation.rationale}")

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        render_allocation_chart(recommendation.weights)
    with col_table:
        st.subheader("Holdings")
        render_allocation_table(recommendation.weights, metadata)

    # --- Risk analytics ---
    st.header("Portfolio Risk Analytics")
    render_risk_metrics(recommendation.metrics)

    port_returns = get_portfolio_return_series(recommendation.weights, returns)
    render_equity_curve(port_returns, f"{selected_profile} Portfolio — Historical Performance")
    render_drawdown_chart(port_returns)

    # --- ML Predictor section (optional) ---
    if show_model:
        st.header("Predictive Engine — Forward 21-Day Return")
        st.markdown(
            "Random Forest model trained on technical indicators. "
            "Directional accuracy indicates how often the model correctly "
            "predicts positive vs negative forward returns."
        )

        with st.spinner("Training model…"):
            result = get_trained_model(master_df)

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("RMSE", f"{result.evaluation.rmse:.3f}")
        mcol2.metric("MAE", f"{result.evaluation.mae:.3f}")
        mcol3.metric("R²", f"{result.evaluation.r2:.3f}")
        mcol4.metric("Directional Acc.", f"{result.evaluation.directional_accuracy:.1%}")

        st.subheader("Feature Importances (Explainable AI)")
        fi = result.feature_importances.copy()
        fi_fig = px.bar(
            fi.head(10),
            x="importance_pct",
            y="feature",
            orientation="h",
            title="Top 10 Features Driving Predictions",
            labels={"importance_pct": "Importance (%)", "feature": "Feature"},
        )
        fi_fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=400)
        st.plotly_chart(fi_fig, use_container_width=True)

    # --- Stock explorer (optional) ---
    if show_stock_explorer:
        st.header("Stock Explorer")
        symbols = sorted(master_df["Symbol"].unique())
        chosen = st.selectbox("Select a stock", symbols)

        stock_raw = load_all_stocks(DATA_DIR, symbols=[chosen])[chosen.upper()]
        stock_clean = handle_missing_values(stock_raw)
        stock_feat = engineer_features(stock_clean, drop_warmup=False)

        tab1, tab2, tab3 = st.tabs(["Price & SMAs", "RSI & MACD", "Risk Stats"])

        with tab1:
            fig_price = go.Figure()
            recent = stock_feat.tail(500)
            fig_price.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name="Close"))
            fig_price.add_trace(go.Scatter(x=recent.index, y=recent["SMA_50"], name="SMA 50"))
            fig_price.add_trace(go.Scatter(x=recent.index, y=recent["SMA_200"], name="SMA 200"))
            fig_price.update_layout(title=f"{chosen} — Price & Moving Averages", height=400)
            st.plotly_chart(fig_price, use_container_width=True)

        with tab2:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=recent.index, y=recent["RSI_14"], name="RSI 14"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title=f"{chosen} — RSI", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with tab3:
            daily_ret = stock_clean["Close"].pct_change().dropna()
            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric("Sharpe", f"{sharpe_ratio(daily_ret):.2f}")
            rcol2.metric("Max Drawdown", f"{maximum_drawdown(daily_ret):.1%}")
            rcol3.metric(
                "Ann. Volatility",
                f"{daily_ret.std() * (252 ** 0.5):.1%}",
            )

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Dataset:** {master_df['Symbol'].nunique()} symbols  \n"
        f"**Period:** {master_df.index.min().date()} → {master_df.index.max().date()}  \n"
        f"**Rows:** {len(master_df):,}"
    )


if __name__ == "__main__":
    main()
