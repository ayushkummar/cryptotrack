from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BACKTEST_DIR = PROJECT_ROOT / "backtests"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

ASSETS = ["BTC", "ETH", "XRP", "ADA", "DOGE", "DOT", "LTC"]

FEATURE_SETS = {
    "Market + Google Trends": "market_trends",
    "Market Only": "market_only",
    "Market + Twitter": "market_twitter",
    "All Sources": "market_twitter_trends",
}

TRANSACTION_COST = 0.001
PHI = 0.50
COVARIANCE_WINDOW = 90

st.set_page_config(
    page_title="CryptoTrack",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.20);
    }

    .metric-card {
        padding: 16px;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        background: rgba(128,128,128,.05);
    }

    .small-note {
        color: #777;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 12px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        for col in ["timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return None


@st.cache_data
def read_parquet(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )
        return df
    except Exception:
        return None


def pct(x, decimals=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def num(x, decimals=3):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def safe_value(row, column, default=np.nan):
    try:
        if row is None or column not in row.index:
            return default
        value = row[column]
        return default if pd.isna(value) else value
    except Exception:
        return default


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(
        weights, nan=0.0, posinf=0.0, neginf=0.0
    )
    weights = np.maximum(weights, 0.0)

    total = weights.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)

    return weights / total


def show_missing(message):
    st.warning(message)
    st.caption(
        "Make sure the generated CSV/Parquet files are committed "
        "to GitHub and included in the Streamlit deployment."
    )


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

def probability_gmv_weights(covariance, probabilities, phi=0.50):
    """
    Same probability-constrained minimum-variance idea used
    by the CryptoTrack research backtest.

    Higher model probabilities impose stronger minimum weights,
    while covariance determines the risk-minimizing allocation.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    n = len(probabilities)
    if n == 0:
        return np.array([])

    total_probability = probabilities.sum()
    if total_probability <= 0:
        return np.ones(n) / n

    normalized_probability = probabilities / total_probability
    lower_bounds = np.minimum(
        phi * normalized_probability, 1.0
    )

    try:
        from scipy.optimize import minimize

        cov = np.asarray(covariance, dtype=float)
        cov = np.nan_to_num(
            cov, nan=0.0, posinf=0.0, neginf=0.0
        )
        cov = (cov + cov.T) / 2.0
        cov += np.eye(n) * 1e-8

        x0 = normalize_weights(normalized_probability)
        x0 = np.maximum(x0, lower_bounds)
        x0 = normalize_weights(x0)

        objective = lambda w: float(w.T @ cov @ w)

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            }
        ]

        bounds = [
            (float(lower_bounds[i]), 1.0)
            for i in range(n)
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if result.success:
            return normalize_weights(result.x)

    except Exception:
        pass

    return normalize_weights(normalized_probability)


def load_predictions(feature_suffix):
    frames = []

    for asset in ASSETS:
        path = (
            EXPERIMENT_DIR
            / f"{asset}_{feature_suffix}_predictions_1d.csv"
        )

        df = read_csv(path)

        if df is None or df.empty:
            continue

        df = df.copy()
        df["asset"] = asset
        frames.append(df)

    if not frames:
        return None

    predictions = pd.concat(frames, ignore_index=True)

    if "timestamp" not in predictions.columns:
        return None

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], errors="coerce"
    )

    return (
        predictions.dropna(subset=["timestamp"])
        .sort_values("timestamp")
    )


def load_prices():
    frames = []

    for asset in ASSETS:
        path = PROCESSED_DIR / f"{asset}_features.parquet"
        df = read_parquet(path)

        if df is None or df.empty:
            continue

        if "timestamp" not in df.columns or "close" not in df.columns:
            continue

        temp = df[["timestamp", "close"]].copy()
        temp = temp.rename(columns={"close": asset})
        frames.append(temp)

    if not frames:
        return None

    prices = frames[0]

    for frame in frames[1:]:
        prices = prices.merge(
            frame,
            on="timestamp",
            how="outer",
        )

    prices["timestamp"] = pd.to_datetime(
        prices["timestamp"], errors="coerce"
    )

    prices = (
        prices.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
        .reindex(columns=ASSETS)
        .ffill()
    )

    return prices


# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================

st.sidebar.title("₿ CryptoTrack")
st.sidebar.caption(
    "Probabilistic cryptocurrency prediction & portfolio analytics"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Crypto Prediction",
        "Portfolio Allocation",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Research setup")
st.sidebar.write("Assets: BTC, ETH, XRP, ADA, DOGE, DOT, LTC")
st.sidebar.write("Validation: Walk-forward")
st.sidebar.write("Transaction cost: 0.1%")
st.sidebar.write("Rebalance: 1 / 7 / 30 days")

st.sidebar.divider()
st.sidebar.caption("Twitter coverage ends 2023-06-12")

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("₿ CryptoTrack")
    st.subheader(
        "Probabilistic Multi-Source Cryptocurrency "
        "Prediction & Portfolio Optimization"
    )

    st.write(
        """
        CryptoTrack extends a binary cryptocurrency movement
        prediction framework by producing probability forecasts
        and using those forecasts inside a transaction-cost-aware
        portfolio allocation framework.
        """
    )

    summary = read_csv(
        BACKTEST_DIR / "multi_source_portfolio_summary.csv"
    )

    if (
        summary is not None
        and not summary.empty
        and "sharpe" in summary.columns
    ):
        best = summary.sort_values(
            "sharpe", ascending=False
        ).iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Best Sharpe",
            num(best.get("sharpe")),
        )
        c2.metric(
            "Annualized Return",
            pct(best.get("annualized_return")),
        )
        c3.metric(
            "Max Drawdown",
            pct(best.get("max_drawdown")),
        )
        c4.metric(
            "Turnover",
            pct(best.get("average_turnover"), 3),
        )

        st.markdown("### Best observed configuration")

        details = pd.DataFrame(
            {
                "Metric": [
                    "Feature Set",
                    "Strategy",
                    "Rebalance",
                    "Annualized Return",
                    "Volatility",
                    "Sharpe",
                    "Sortino",
                    "Maximum Drawdown",
                    "Transaction Costs",
                ],
                "Value": [
                    best.get("feature_set", "—"),
                    best.get("strategy", "—"),
                    (
                        f"{int(best['rebalance_days'])} days"
                        if "rebalance_days" in best
                        else "—"
                    ),
                    pct(best.get("annualized_return")),
                    pct(best.get("annualized_volatility")),
                    num(best.get("sharpe")),
                    num(best.get("sortino")),
                    pct(best.get("max_drawdown")),
                    pct(best.get("total_transaction_cost"), 3),
                ],
            }
        )

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

        # Main research chart
        st.markdown("### Sharpe Ratio by Rebalance Frequency")

        if {
            "feature_set",
            "strategy",
            "rebalance_days",
            "sharpe",
        }.issubset(summary.columns):

            chart_df = summary.copy()
            chart_df["Configuration"] = (
                chart_df["feature_set"].astype(str)
                + " · "
                + chart_df["strategy"].astype(str)
            )

            fig = px.line(
                chart_df,
                x="rebalance_days",
                y="sharpe",
                color="Configuration",
                markers=True,
                labels={
                    "rebalance_days": "Rebalance frequency (days)",
                    "sharpe": "Sharpe ratio",
                },
            )

            fig.update_layout(
                height=480,
                margin=dict(l=20, r=20, t=40, b=20),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:
        show_missing(
            "Could not find "
            "backtests/multi_source_portfolio_summary.csv."
        )

    st.markdown("### Research Pipeline")

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "Market data",
                "Google Trends",
                "Twitter sentiment",
                "Feature engineering",
                "Walk-forward SVM",
                "Probability calibration",
                "Portfolio optimization",
                "Transaction-cost backtest",
                "Regime robustness",
            ],
            "Status": ["✓"] * 9,
        }
    )

    st.dataframe(
        pipeline,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        """
        Research limitation: the Twitter dataset used by CryptoTrack
        is influencer-focused and ends on 2023-06-12. It is not an
        exact replication of the original paper's Twitter corpus.
        """
    )

# ============================================================
# CRYPTO PREDICTION
# ============================================================

elif page == "Crypto Prediction":

    st.title("🔮 Crypto Prediction")
    st.write(
        """
        View the model's out-of-sample probability forecast for
        the next-day cryptocurrency price movement.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        asset = st.selectbox(
            "Cryptocurrency",
            ASSETS,
        )

    with c2:
        feature_label = st.selectbox(
            "Feature Set",
            list(FEATURE_SETS.keys()),
            index=0,
        )

    feature_suffix = FEATURE_SETS[feature_label]

    predictions = read_csv(
        EXPERIMENT_DIR
        / f"{asset}_{feature_suffix}_predictions_1d.csv"
    )

    if predictions is None or predictions.empty:
        show_missing(
            f"Prediction file not found for {asset}: "
            f"{feature_suffix}_predictions_1d.csv"
        )
    else:

        predictions = predictions.dropna(
            subset=["timestamp"]
        ).sort_values("timestamp")

        dates = list(
            predictions["timestamp"].unique()
        )

        selected_date = st.selectbox(
            "Prediction Date",
            dates,
            index=len(dates) - 1,
            format_func=lambda x:
                pd.Timestamp(x).strftime("%Y-%m-%d"),
        )

        selected_date = pd.Timestamp(selected_date)

        rows = predictions[
            predictions["timestamp"] == selected_date
        ]

        if rows.empty:
            st.warning(
                "No prediction available for this date."
            )
        else:
            row = rows.iloc[0]

            probability = float(
                np.clip(
                    safe_value(
                        row,
                        "prob_positive",
                        0.5,
                    ),
                    0,
                    1,
                )
            )

            prediction = int(
                safe_value(
                    row,
                    "prediction",
                    int(probability >= 0.5),
                )
            )

            confidence = float(
                safe_value(
                    row,
                    "confidence",
                    abs(probability - 0.5) * 2,
                )
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Positive Return Probability",
                pct(probability, 1),
            )
            c2.metric(
                "Prediction",
                "↑ Positive" if prediction == 1
                else "↓ Negative",
            )
            c3.metric(
                "Confidence",
                pct(confidence, 1),
            )
            c4.metric(
                "Date",
                selected_date.strftime("%Y-%m-%d"),
            )

            # Probability gauge
            st.markdown("### Probability Forecast")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={
                        "text":
                        "Probability of Positive Movement"
                    },
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": 50,
                        },
                    },
                )
            )

            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.markdown("### Model Information")

            info = pd.DataFrame(
                [
                    ["Asset", asset],
                    ["Feature Set", feature_label],
                    ["Prediction Horizon", "1 day"],
                    ["Probability", pct(probability, 2)],
                    [
                        "Prediction",
                        "Positive"
                        if prediction == 1
                        else "Negative",
                    ],
                    ["Confidence", pct(confidence, 2)],
                ],
                columns=["Metric", "Value"],
            )

            st.dataframe(
                info,
                hide_index=True,
                use_container_width=True,
            )

            if feature_suffix in {
                "market_twitter",
                "market_twitter_trends",
            }:
                st.warning(
                    """
                    Twitter-derived features are limited to the
                    available dataset period. The Twitter dataset
                    ends on 2023-06-12. Missing observations are
                    treated as unavailable, not neutral sentiment.
                    """
                )

# ============================================================
# PORTFOLIO ALLOCATION
# ============================================================

elif page == "Portfolio Allocation":

    st.title("💼 Portfolio Allocation")
    st.subheader(
        "Probability-driven cryptocurrency portfolio"
    )

    st.write(
        """
        CryptoTrack converts out-of-sample probability forecasts
        into risk-aware portfolio weights. The allocation uses
        model probabilities together with historical covariance
        in a constrained minimum-variance optimization.
        """
    )

    st.info(
        """
        Research/backtesting interface only. These allocations are
        generated from historical model data and are not investment
        advice.
        """
    )

    feature_label = st.selectbox(
        "Prediction Feature Set",
        list(FEATURE_SETS.keys()),
        index=0,
    )

    rebalance_days = st.selectbox(
        "Rebalance Frequency",
        [1, 7, 30],
        index=2,
    )

    feature_suffix = FEATURE_SETS[feature_label]

    if feature_suffix in {
        "market_twitter",
        "market_twitter_trends",
    }:
        st.warning(
            """
            Twitter coverage is limited. The available Twitter
            dataset ends on 2023-06-12. Dates after this point are
            explicitly marked unavailable rather than interpreted
            as neutral sentiment.
            """
        )

    predictions = load_predictions(feature_suffix)

    if predictions is None or predictions.empty:
        show_missing(
            "No prediction files were found for the selected "
            "feature set."
        )
        st.stop()

    dates = sorted(
        predictions["timestamp"].unique()
    )

    selected_date = st.selectbox(
        "Prediction Date",
        dates,
        index=len(dates) - 1,
        format_func=lambda x:
            pd.Timestamp(x).strftime("%Y-%m-%d"),
    )

    selected_date = pd.Timestamp(selected_date)

    daily = predictions[
        predictions["timestamp"] == selected_date
    ].copy()

    # --------------------------------------------------------
    # Historical prices
    # --------------------------------------------------------

    prices = load_prices()

    if prices is not None:

        historical_returns = (
            prices.pct_change()
            .loc[lambda x: x.index <= selected_date]
            .tail(COVARIANCE_WINDOW)
            .reindex(columns=ASSETS)
        )

        covariance = historical_returns.cov().values

    else:
        covariance = np.eye(len(ASSETS))

    covariance = np.nan_to_num(
        covariance,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    covariance = (
        covariance + covariance.T
    ) / 2.0

    covariance += (
        np.eye(len(ASSETS)) * 1e-8
    )

    # --------------------------------------------------------
    # Probability table
    # --------------------------------------------------------

    rows = []

    for asset in ASSETS:

        asset_data = daily[
            daily["asset"] == asset
        ]

        if asset_data.empty:
            probability = 0.5
            prediction = 1
            available = False
        else:
            row = asset_data.iloc[0]

            probability = float(
                np.clip(
                    safe_value(
                        row,
                        "prob_positive",
                        0.5,
                    ),
                    0,
                    1,
                )
            )

            prediction = int(
                safe_value(
                    row,
                    "prediction",
                    int(probability >= 0.5),
                )
            )

            available = True

        rows.append(
            {
                "Asset": asset,
                "Probability": probability,
                "Prediction": prediction,
                "Available": available,
            }
        )

    allocation = pd.DataFrame(rows)

    probabilities = allocation[
        "Probability"
    ].values

    weights = probability_gmv_weights(
        covariance,
        probabilities,
        phi=PHI,
    )

    allocation["Weight"] = weights

    allocation["Signal"] = np.where(
        allocation["Probability"] >= 0.5,
        "Bullish",
        "Bearish",
    )

    allocation = (
        allocation
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Model Date",
        selected_date.strftime("%Y-%m-%d"),
    )

    c2.metric(
        "Feature Set",
        feature_label,
    )

    largest = allocation.iloc[0]

    c3.metric(
        "Largest Allocation",
        largest["Asset"],
        pct(largest["Weight"], 1),
    )

    c4.metric(
        "Average Probability",
        pct(
            allocation["Probability"].mean(),
            1,
        ),
    )

    # --------------------------------------------------------
    # Allocation chart
    # --------------------------------------------------------

    st.markdown("## 📊 Portfolio Allocation")

    chart_col, table_col = st.columns(
        [1.35, 1]
    )

    with chart_col:

        fig = px.bar(
            allocation,
            x="Asset",
            y="Weight",
            color="Probability",
            text="Weight",
            color_continuous_scale="Blues",
        )

        fig.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[
                0,
                max(
                    0.35,
                    float(
                        allocation["Weight"].max()
                    ) * 1.25,
                ),
            ],
        )

        fig.update_layout(
            height=500,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
            coloraxis_colorbar_title="Probability",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with table_col:

        display = allocation.copy()

        display["Probability"] = display[
            "Probability"
        ].map(lambda x: f"{x:.1%}")

        display["Weight"] = display[
            "Weight"
        ].map(lambda x: f"{x:.1%}")

        display["Prediction"] = display[
            "Prediction"
        ].map(
            lambda x:
                "↑ Positive"
                if x == 1
                else "↓ Negative"
        )

        display = display[
            [
                "Asset",
                "Probability",
                "Prediction",
                "Signal",
                "Weight",
            ]
        ]

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Research backtest reference
    # --------------------------------------------------------

    st.markdown("## 📈 Research Backtest Reference")

    summary = read_csv(
        BACKTEST_DIR
        / "multi_source_portfolio_summary.csv"
    )

    if summary is not None and not summary.empty:

        probability_results = summary[
            summary["strategy"] == "probability"
        ].copy()

        matching = probability_results[
            probability_results["rebalance_days"]
            == rebalance_days
        ]

        if matching.empty:
            matching = (
                probability_results
                .sort_values(
                    "sharpe",
                    ascending=False,
                )
                .head(1)
            )

        if not matching.empty:

            result = matching.iloc[0]

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Annualized Return",
                pct(
                    result[
                        "annualized_return"
                    ]
                ),
            )

            c2.metric(
                "Volatility",
                pct(
                    result[
                        "annualized_volatility"
                    ]
                ),
            )

            c3.metric(
                "Sharpe",
                num(result["sharpe"]),
            )

            c4.metric(
                "Sortino",
                num(result["sortino"]),
            )

            c5.metric(
                "Max Drawdown",
                pct(result["max_drawdown"]),
            )

    st.markdown("## 🧠 How CryptoTrack Allocates")

    st.write(
        """
        The probability strategy does not simply select the
        cryptocurrency with the highest predicted probability.

        Instead, the probabilities create allocation requirements
        while historical covariance is used to construct a
        minimum-variance portfolio. This allows the model to use
        both directional information and portfolio risk information.
        """
    )

    st.caption(
        f"Current configuration: {feature_label} · "
        f"{rebalance_days}-day rebalance · "
        f"{TRANSACTION_COST:.1%} transaction cost · "
        f"φ = {PHI:.2f}"
    )

# ============================================================
# END
# ============================================================
