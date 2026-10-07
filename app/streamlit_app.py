import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BACKTEST_DIR = PROJECT_ROOT / "backtests"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "ADA",
    "DOGE",
    "DOT",
    "LTC",
]

FEATURE_SETS = {
    "Market Only": "market_only",
    "Market + Google Trends": "market_trends",
    "Market + Twitter": "market_twitter",
    "All Sources": "market_twitter_trends",
}

STRATEGIES = [
    "equal_weight",
    "minimum_variance",
    "paper_binary",
    "paper_binary_equal",
    "probability",
]

TRANSACTION_COST = 0.001
PHI = 0.50
COVARIANCE_WINDOW = 90


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="CryptoTrack",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        background: rgba(128,128,128,.05);
    }

    .small-note {
        color: #777;
        font-size: .85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

@st.cache_data
def read_csv(path):
    """
    Safely read a CSV file.
    """
    p = Path(path)

    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)

        for col in ["timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

        return df

    except Exception:
        return None


@st.cache_data
def read_parquet(path):
    """
    Safely read a parquet file.
    """
    p = Path(path)

    if not p.exists():
        return None

    try:
        df = pd.read_parquet(p)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

        return df

    except Exception:
        return None


def pct(x, decimals=2):
    """
    Convert decimal value to percentage string.
    """
    try:
        if pd.isna(x):
            return "—"

        return f"{float(x) * 100:.{decimals}f}%"

    except Exception:
        return "—"


def num(x, decimals=3):
    """
    Format numerical value.
    """
    try:
        if pd.isna(x):
            return "—"

        return f"{float(x):.{decimals}f}"

    except Exception:
        return "—"


def safe_value(row, column, default=np.nan):
    """
    Safely retrieve a value from a dataframe row.
    """
    try:

        if row is None:
            return default

        if column not in row.index:
            return default

        value = row[column]

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default


def find_experiment(name):
    """
    Read an experiment CSV.
    """
    return read_csv(
        EXPERIMENT_DIR / name
    )


def find_backtest(name):
    """
    Read a backtest CSV.

    This function fixes the NameError that appeared
    in the previous version of the dashboard.
    """
    return read_csv(
        BACKTEST_DIR / name
    )


def show_missing(message):
    """
    Display a standard missing-file message.
    """
    st.warning(message)

    st.caption(
        "Run the CryptoTrack data, prediction and backtest "
        "pipelines first, or place the generated CSV files "
        "in the expected folders."
    )


def normalize_weights(weights):
    """
    Make weights valid and fully invested.
    """
    weights = np.asarray(
        weights,
        dtype=float
    )

    weights = np.nan_to_num(
        weights,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    weights = np.maximum(
        weights,
        0.0
    )

    total = weights.sum()

    if total <= 0:
        return np.ones(
            len(weights)
        ) / len(weights)

    return weights / total


# ============================================================
# PORTFOLIO WEIGHT FUNCTIONS
# ============================================================

def equal_weight(n_assets):
    """
    Standard 1/N portfolio.
    """
    if n_assets <= 0:
        return np.array([])

    return np.ones(
        n_assets
    ) / n_assets


def minimum_variance_weights(
    returns,
    covariance_window=90
):
    """
    Same inverse-variance approximation used
    by the research backtest implementation.
    """

    if returns is None:
        return None

    if returns.empty:
        return None

    historical = returns.tail(
        covariance_window
    )

    variances = historical.var()

    variances = variances.replace(
        [np.inf, -np.inf],
        np.nan
    )

    variances = variances.fillna(
        variances.median()
    )

    variances = variances.replace(
        0,
        np.nan
    )

    if variances.isna().all():
        return equal_weight(
            len(returns.columns)
        )

    variances = variances.fillna(
        variances.max()
    )

    inverse_variance = (
        1.0 / variances
    )

    inverse_variance = (
        inverse_variance
        .replace(
            [np.inf, -np.inf],
            0
        )
        .fillna(0)
    )

    return normalize_weights(
        inverse_variance.values
    )


def probability_weights(
    covariance,
    probabilities,
    phi=0.50
):
    """
    Probability-based allocation.

    This follows the research backtest idea:
    probabilities influence minimum portfolio weights,
    while covariance is used for risk-aware optimization.

    Uses scipy SLSQP when available.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    n = len(
        probabilities
    )

    if n == 0:
        return np.array([])

    probability_sum = probabilities.sum()

    if probability_sum <= 0:
        return equal_weight(n)

    normalized_probability = (
        probabilities /
        probability_sum
    )

    lower_bounds = (
        phi *
        normalized_probability
    )

    # Ensure the lower bounds do not exceed 1.
    lower_bounds = np.minimum(
        lower_bounds,
        1.0
    )

    # --------------------------------------------------------
    # Try scipy optimization
    # --------------------------------------------------------

    try:

        from scipy.optimize import minimize

        cov = np.asarray(
            covariance,
            dtype=float
        )

        cov = np.nan_to_num(
            cov,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cov = (
            cov + cov.T
        ) / 2.0

        cov += (
            np.eye(n) * 1e-8
        )

        x0 = normalize_weights(
            normalized_probability
        )

        # Make x0 satisfy the lower bounds.
        x0 = np.maximum(
            x0,
            lower_bounds
        )

        x0 = normalize_weights(
            x0
        )

        objective = lambda w: float(
            w.T @ cov @ w
        )

        constraints = [
            {
                "type": "eq",
                "fun": lambda w:
                    np.sum(w) - 1.0,
            }
        ]

        bounds = [
            (
                float(lower_bounds[i]),
                1.0
            )
            for i in range(n)
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 500,
                "ftol": 1e-10,
            },
        )

        if result.success:

            return normalize_weights(
                result.x
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return normalize_weights(
        normalized_probability
    )


def binary_weights(
    covariance,
    predictions,
    phi=0.50
):
    """
    Binary paper-inspired allocation.
    """

    predictions = np.asarray(
        predictions,
        dtype=int
    )

    n = len(
        predictions
    )

    if n == 0:
        return np.array([])

    positive = (
        predictions == 1
    )

    n_positive = positive.sum()

    if n_positive == 0:

        return np.zeros(n)

    try:

        from scipy.optimize import minimize

        cov = np.asarray(
            covariance,
            dtype=float
        )

        cov = np.nan_to_num(
            cov,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cov = (
            cov + cov.T
        ) / 2.0

        cov += (
            np.eye(n) * 1e-8
        )

        lower = np.zeros(n)

        lower[
            positive
        ] = (
            phi /
            n_positive
        )

        x0 = np.zeros(n)

        x0[
            positive
        ] = (
            1.0 /
            n_positive
        )

        objective = lambda w: float(
            w.T @ cov @ w
        )

        constraints = [
            {
                "type": "eq",
                "fun": lambda w:
                    np.sum(w) - 1.0,
            }
        ]

        bounds = [
            (
                float(lower[i]),
                1.0
            )
            for i in range(n)
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 500,
                "ftol": 1e-10,
            },
        )

        if result.success:

            return normalize_weights(
                result.x
            )

    except Exception:
        pass

    fallback = np.zeros(n)

    fallback[
        positive
    ] = (
        1.0 /
        n_positive
    )

    return fallback


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "₿ CryptoTrack"
)

st.sidebar.caption(
    "Probabilistic multi-source crypto portfolio analytics"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Crypto Prediction",
        "Portfolio Allocation",
        "Prediction Analysis",
        "Portfolio Comparison",
        "Regime Robustness",
        "Research Evidence",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Research configuration"
)

st.sidebar.write(
    "Assets: BTC, ETH, XRP, ADA, DOGE, DOT, LTC"
)

st.sidebar.write(
    "Rebalance: 1 / 7 / 30 days"
)

st.sidebar.write(
    "Transaction cost: 0.1%"
)

st.sidebar.write(
    "Validation: walk-forward"
)

st.sidebar.divider()

st.sidebar.caption(
    "Twitter data coverage ends 2023-06-12"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title(
        "₿ CryptoTrack"
    )

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

    # --------------------------------------------------------
    # Key research result
    # --------------------------------------------------------

    summary = find_backtest(
        "multi_source_portfolio_summary.csv"
    )

    if (
        summary is not None
        and not summary.empty
        and "sharpe" in summary.columns
    ):

        best = summary.sort_values(
            "sharpe",
            ascending=False
        ).iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Best Sharpe",
                num(
                    best["sharpe"]
                )
            )

        with c2:

            st.metric(
                "Annualized Return",
                pct(
                    best["annualized_return"]
                )
            )

        with c3:

            st.metric(
                "Max Drawdown",
                pct(
                    best["max_drawdown"]
                )
            )

        with c4:

            st.metric(
                "Turnover",
                pct(
                    best["average_turnover"],
                    3
                )
            )

        st.markdown(
            "### Best observed configuration"
        )

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
                    best.get(
                        "feature_set",
                        "—"
                    ),
                    best.get(
                        "strategy",
                        "—"
                    ),
                    (
                        f"{int(best['rebalance_days'])} days"
                        if "rebalance_days"
                        in best
                        else "—"
                    ),
                    pct(
                        best.get(
                            "annualized_return",
                            np.nan
                        )
                    ),
                    pct(
                        best.get(
                            "annualized_volatility",
                            np.nan
                        )
                    ),
                    num(
                        best.get(
                            "sharpe",
                            np.nan
                        )
                    ),
                    num(
                        best.get(
                            "sortino",
                            np.nan
                        )
                    ),
                    pct(
                        best.get(
                            "max_drawdown",
                            np.nan
                        )
                    ),
                    pct(
                        best.get(
                            "total_transaction_cost",
                            np.nan
                        ),
                        3
                    ),
                ],
            }
        )

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Sharpe chart
        # ----------------------------------------------------

        if (
            PLOTLY_OK
            and "feature_set" in summary.columns
            and "strategy" in summary.columns
            and "rebalance_days" in summary.columns
        ):

            st.markdown(
                "### Sharpe by rebalance frequency"
            )

            chart_df = summary.copy()

            chart_df[
                "Configuration"
            ] = (
                chart_df[
                    "feature_set"
                ].astype(str)
                + " · "
                + chart_df[
                    "strategy"
                ].astype(str)
            )

            fig = px.line(
                chart_df,
                x="rebalance_days",
                y="sharpe",
                color="Configuration",
                markers=True,
                labels={
                    "rebalance_days":
                        "Rebalance frequency (days)",
                    "sharpe":
                        "Sharpe ratio",
                },
            )

            fig.update_layout(
                legend_title_text=""
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

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.markdown(
        "### Research Pipeline"
    )

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
            "Status": [
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
                "✓",
            ],
        }
    )

    st.dataframe(
        pipeline,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        """
        Research note: the Twitter dataset used by CryptoTrack
        is influencer-focused and ends on 2023-06-12. It is not
        an exact replication of the original paper's Twitter corpus.
        """
    )


# ============================================================
# CRYPTO PREDICTION
# ============================================================

elif page == "Crypto Prediction":

    st.title(
        "🔮 Crypto Prediction"
    )

    st.write(
        """
        View the model's out-of-sample probability forecast
        for each cryptocurrency.
        """
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        asset = st.selectbox(
            "Cryptocurrency",
            ASSETS
        )

    with c2:

        feature_label = st.selectbox(
            "Feature Set",
            list(
                FEATURE_SETS.keys()
            ),
            index=1
        )

    feature_suffix = FEATURE_SETS[
        feature_label
    ]

    prediction_path = (
        EXPERIMENT_DIR
        / f"{asset}_{feature_suffix}_predictions_1d.csv"
    )

    predictions = read_csv(
        prediction_path
    )

    if predictions is None or predictions.empty:

        show_missing(
            f"Prediction file not found: "
            f"{prediction_path.name}"
        )

    else:

        if (
            "timestamp"
            not in predictions.columns
        ):

            st.error(
                "Prediction file does not contain a timestamp column."
            )

            st.stop()

        predictions = predictions.dropna(
            subset=["timestamp"]
        )

        predictions = predictions.sort_values(
            "timestamp"
        )

        dates = list(
            predictions[
                "timestamp"
            ].unique()
        )

        selected_date = st.selectbox(
            "Prediction Date",
            dates,
            index=len(dates) - 1,
            format_func=lambda x:
                pd.Timestamp(x).strftime(
                    "%Y-%m-%d"
                )
        )

        selected_date = pd.Timestamp(
            selected_date
        )

        rows = predictions[
            predictions[
                "timestamp"
            ] == selected_date
        ]

        if rows.empty:

            st.warning(
                "No prediction available for this date."
            )

        else:

            row = rows.iloc[0]

            probability = float(
                safe_value(
                    row,
                    "prob_positive",
                    0.5
                )
            )

            probability = np.clip(
                probability,
                0,
                1
            )

            prediction = int(
                safe_value(
                    row,
                    "prediction",
                    int(
                        probability >= 0.5
                    )
                )
            )

            confidence = float(
                safe_value(
                    row,
                    "confidence",
                    abs(
                        probability -
                        0.5
                    ) * 2
                )
            )

            # ------------------------------------------------
            # Prediction metrics
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Probability of Positive Return",
                    pct(
                        probability,
                        1
                    )
                )

            with c2:

                st.metric(
                    "Prediction",
                    (
                        "↑ Positive"
                        if prediction == 1
                        else "↓ Negative"
                    )
                )

            with c3:

                st.metric(
                    "Confidence",
                    pct(
                        confidence,
                        1
                    )
                )

            with c4:

                st.metric(
                    "Date",
                    selected_date.strftime(
                        "%Y-%m-%d"
                    )
                )

            # ------------------------------------------------
            # Probability gauge
            # ------------------------------------------------

            if PLOTLY_OK:

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={
                            "text":
                                "Positive Movement Probability"
                        },
                        number={
                            "suffix": "%"
                        },
                        gauge={
                            "axis": {
                                "range": [
                                    0,
                                    100
                                ]
                            }
                        },
                    )
                )

                fig.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ------------------------------------------------
            # Model metrics
            # ------------------------------------------------

            st.markdown(
                "### Model Information"
            )

            info = {
                "Asset": asset,
                "Feature Set": feature_label,
                "Prediction Horizon": "1 day",
                "Probability": pct(
                    probability,
                    2
                ),
                "Prediction": (
                    "Positive"
                    if prediction == 1
                    else "Negative"
                ),
                "Confidence": pct(
                    confidence,
                    2
                ),
            }

            st.dataframe(
                pd.DataFrame(
                    info.items(),
                    columns=[
                        "Metric",
                        "Value"
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Twitter coverage
            # ------------------------------------------------

            if feature_suffix in [
                "market_twitter",
                "market_twitter_trends",
            ]:

                st.warning(
                    """
                    Twitter-derived features are limited to
                    the available dataset period. The Twitter
                    dataset ends on 2023-06-12. Missing Twitter
                    observations are treated as unavailable,
                    not as neutral sentiment.
                    """
                )


# ============================================================
# PORTFOLIO ALLOCATION
# ============================================================

elif page == "Portfolio Allocation":

    st.title(
        "💼 Portfolio Allocation"
    )

    st.subheader(
        "Probability-driven cryptocurrency portfolio"
    )

    st.write(
        """
        CryptoTrack converts the model's out-of-sample
        probability forecasts into portfolio weights.
        The probability strategy combines model probabilities
        with historical covariance to construct a risk-aware
        allocation.
        """
    )

    st.info(
        """
        Research/backtesting interface only.
        These allocations are generated from historical
        model data and are not investment advice.
        """
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        feature_label = st.selectbox(
            "Prediction Feature Set",
            list(
                FEATURE_SETS.keys()
            ),
            index=1,
        )

    with c2:

        rebalance_days = st.selectbox(
            "Rebalance Frequency",
            [1, 7, 30],
            index=2,
        )

    feature_suffix = FEATURE_SETS[
        feature_label
    ]

    # --------------------------------------------------------
    # Twitter warning
    # --------------------------------------------------------

    if feature_suffix in [
        "market_twitter",
        "market_twitter_trends",
    ]:

        st.warning(
            """
            Twitter coverage is limited.
            The available Twitter dataset ends on 2023-06-12.
            Dates after this point are explicitly marked as
            unavailable rather than interpreted as neutral sentiment.
            """
        )

    # --------------------------------------------------------
    # Load prediction files
    # --------------------------------------------------------

    prediction_frames = []

    for current_asset in ASSETS:

        prediction_path = (
            EXPERIMENT_DIR
            / (
                f"{current_asset}_"
                f"{feature_suffix}_"
                f"predictions_1d.csv"
            )
        )

        prediction_df = read_csv(
            prediction_path
        )

        if (
            prediction_df is None
            or prediction_df.empty
        ):
            continue

        prediction_df = prediction_df.copy()

        prediction_df[
            "asset"
        ] = current_asset

        prediction_frames.append(
            prediction_df
        )

    if not prediction_frames:

        show_missing(
            "No prediction files were found."
        )

    else:

        predictions = pd.concat(
            prediction_frames,
            ignore_index=True
        )

        predictions = predictions.dropna(
            subset=["timestamp"]
        )

        predictions = predictions.sort_values(
            "timestamp"
        )

        dates = sorted(
            predictions[
                "timestamp"
            ].unique()
        )

        selected_date = st.selectbox(
            "Prediction Date",
            dates,
            index=len(dates) - 1,
            format_func=lambda x:
                pd.Timestamp(x).strftime(
                    "%Y-%m-%d"
                )
        )

        selected_date = pd.Timestamp(
            selected_date
        )

        daily = predictions[
            predictions[
                "timestamp"
            ] == selected_date
        ]

        # ----------------------------------------------------
        # Load historical prices
        # ----------------------------------------------------

        price_frames = []

        for current_asset in ASSETS:

            price_path = (
                PROCESSED_DIR
                / f"{current_asset}_features.parquet"
            )

            price_df = read_parquet(
                price_path
            )

            if (
                price_df is None
                or price_df.empty
            ):
                continue

            if (
                "timestamp"
                not in price_df.columns
                or "close"
                not in price_df.columns
            ):
                continue

            price_df = price_df[
                [
                    "timestamp",
                    "close"
                ]
            ].copy()

            price_df = price_df.rename(
                columns={
                    "close":
                        current_asset
                }
            )

            price_frames.append(
                price_df
            )

        if price_frames:

            prices = price_frames[0]

            for frame in price_frames[1:]:

                prices = prices.merge(
                    frame,
                    on="timestamp",
                    how="outer"
                )

            prices = prices.sort_values(
                "timestamp"
            )

            prices = prices.set_index(
                "timestamp"
            )

            prices = prices[
                ASSETS
            ].ffill()

            returns = prices.pct_change()

        else:

            prices = None
            returns = None

        # ----------------------------------------------------
        # Build probability table
        # ----------------------------------------------------

        rows = []

        for current_asset in ASSETS:

            asset_data = daily[
                daily["asset"]
                == current_asset
            ]

            if asset_data.empty:

                rows.append(
                    {
                        "Asset":
                            current_asset,
                        "Probability":
                            0.5,
                        "Prediction":
                            1,
                        "Available":
                            False,
                    }
                )

            else:

                row = asset_data.iloc[0]

                probability = float(
                    safe_value(
                        row,
                        "prob_positive",
                        0.5
                    )
                )

                probability = np.clip(
                    probability,
                    0,
                    1
                )

                prediction = int(
                    safe_value(
                        row,
                        "prediction",
                        int(
                            probability >= 0.5
                        )
                    )
                )

                rows.append(
                    {
                        "Asset":
                            current_asset,
                        "Probability":
                            probability,
                        "Prediction":
                            prediction,
                        "Available":
                            True,
                    }
                )

        allocation = pd.DataFrame(
            rows
        )

        # ----------------------------------------------------
        # Covariance
        # ----------------------------------------------------

        if returns is not None:

            historical_returns = (
                returns.loc[
                    returns.index
                    <= selected_date
                ]
                .tail(
                    COVARIANCE_WINDOW
                )
            )

            historical_returns = (
                historical_returns
                .reindex(
                    columns=ASSETS
                )
            )

            covariance = (
                historical_returns
                .cov()
                .values
            )

        else:

            covariance = np.eye(
                len(ASSETS)
            )

        covariance = np.nan_to_num(
            covariance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        covariance = (
            covariance +
            covariance.T
        ) / 2.0

        covariance += (
            np.eye(
                len(ASSETS)
            ) * 1e-8
        )

        # ----------------------------------------------------
        # Probability allocation
        # ----------------------------------------------------

        probabilities = allocation[
            "Probability"
        ].values

        weights = probability_weights(
            covariance,
            probabilities,
            phi=PHI,
        )

        allocation[
            "Weight"
        ] = weights

        allocation[
            "Signal"
        ] = np.where(
            allocation[
                "Probability"
            ] >= 0.5,
            "Bullish",
            "Bearish",
        )

        allocation = allocation.sort_values(
            "Weight",
            ascending=False
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Key metrics
        # ----------------------------------------------------

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Model Date",
                selected_date.strftime(
                    "%Y-%m-%d"
                )
            )

        with c2:

            st.metric(
                "Feature Set",
                feature_label
            )

        with c3:

            largest = allocation.iloc[0]

            st.metric(
                "Largest Allocation",
                largest["Asset"],
                pct(
                    largest["Weight"],
                    1
                )
            )

        with c4:

            st.metric(
                "Average Probability",
                pct(
                    allocation[
                        "Probability"
                    ].mean(),
                    1
                )
            )

        # ----------------------------------------------------
        # Allocation chart
        # ----------------------------------------------------

        st.markdown(
            "## 📊 Portfolio Allocation"
        )

        chart_col, table_col = st.columns(
            [1.35, 1]
        )

        with chart_col:

            if PLOTLY_OK:

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
                )

                fig.update_layout(
                    height=450,
                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=20,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with table_col:

            display = allocation.copy()

            display[
                "Probability"
            ] = display[
                "Probability"
            ].map(
                lambda x:
                    f"{x:.1%}"
            )

            display[
                "Weight"
            ] = display[
                "Weight"
            ].map(
                lambda x:
                    f"{x:.1%}"
            )

            display[
                "Prediction"
            ] = display[
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

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        st.markdown(
            "## 🧠 How CryptoTrack Allocates"
        )

        st.write(
            """
            The probability strategy does not simply choose
            the cryptocurrency with the highest predicted
            probability.

            Instead, the model probabilities are incorporated
            into a constrained minimum-variance optimization.
            Assets with higher predicted probabilities receive
            stronger allocation requirements while historical
            covariance is used to account for portfolio risk.
            """
        )

        # ----------------------------------------------------
        # Research backtest reference
        # ----------------------------------------------------

        st.markdown(
            "## 📈 Research Backtest Reference"
        )

        summary = find_backtest(
            "multi_source_portfolio_summary.csv"
        )

        if (
            summary is not None
            and not summary.empty
        ):

            probability_results = summary[
                summary["strategy"]
                == "probability"
            ].copy()

            if not probability_results.empty:

                matching = probability_results[
                    probability_results[
                        "rebalance_days"
                    ]
                    == rebalance_days
                ]

                if matching.empty:

                    matching = (
                        probability_results
                        .sort_values(
                            "sharpe",
                            ascending=False
                        )
                        .head(1)
                    )

                result = matching.iloc[0]

                c1, c2, c3, c4, c5 = (
                    st.columns(5)
                )

                with c1:

                    st.metric(
                        "Annualized Return",
                        pct(
                            result[
                                "annualized_return"
                            ]
                        )
                    )

                with c2:

                    st.metric(
                        "Volatility",
                        pct(
                            result[
                                "annualized_volatility"
                            ]
                        )
                    )

                with c3:

                    st.metric(
                        "Sharpe",
                        num(
                            result[
                                "sharpe"
                            ]
                        )
                    )

                with c4:

                    st.metric(
                        "Sortino",
                        num(
                            result[
                                "sortino"
                            ]
                        )
                    )

                with c5:

                    st.metric(
                        "Max Drawdown",
                        pct(
                            result[
                                "max_drawdown"
                            ]
                        )
                    )

        # ----------------------------------------------------
        # Data coverage
        # ----------------------------------------------------

        st.markdown(
            "## 🗂️ Data Coverage"
        )

        coverage = pd.DataFrame(
            {
                "Source": [
                    "Market Data",
                    "Google Trends",
                    "Twitter",
                ],
                "Coverage": [
                    "2021 – 2026",
                    "Historical research period",
                    "2021 – 2023-06-12",
                ],
                "Status": [
                    "Available",
                    "Available",
                    "Limited",
                ],
            }
        )

        st.dataframe(
            coverage,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

elif page == "Prediction Analysis":

    st.title(
        "📊 Prediction Analysis"
    )

    pred_results = find_experiment(
        "prediction_experiment_results.csv"
    )

    if (
        pred_results is None
        or pred_results.empty
    ):

        show_missing(
            "Could not find "
            "experiments/prediction_experiment_results.csv."
        )

    else:

        st.write(
            "Out-of-sample prediction quality "
            "across feature sets."
        )

        metric_options = [
            c
            for c in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier",
                "log_loss",
                "ece",
            ]
            if c in pred_results.columns
        ]

        feature_col = (
            "feature_set"
            if "feature_set"
            in pred_results.columns
            else (
                "experiment"
                if "experiment"
                in pred_results.columns
                else None
            )
        )

        if (
            metric_options
            and feature_col
        ):

            selected_metric = st.selectbox(
                "Metric",
                metric_options,
                index=0,
            )

            lower_is_better = (
                selected_metric
                in [
                    "brier",
                    "log_loss",
                    "ece",
                ]
            )

            agg = (
                pred_results
                .groupby(feature_col)[
                    selected_metric
                ]
                .mean()
                .reset_index()
                .sort_values(
                    selected_metric,
                    ascending=lower_is_better
                )
            )

            if PLOTLY_OK:

                fig = px.bar(
                    agg,
                    x=feature_col,
                    y=selected_metric,
                    text_auto=".3f",
                    labels={
                        feature_col:
                            "Feature set",
                        selected_metric:
                            selected_metric,
                    },
                )

                fig.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.markdown(
                "### Out-of-sample results"
            )

            st.dataframe(
                pred_results.sort_values(
                    selected_metric,
                    ascending=lower_is_better
                ),
                hide_index=True,
                use_container_width=True,
            )

        else:

            st.warning(
                "Prediction results were found, but "
                "the expected columns were not detected."
            )

            st.write(
                "Detected columns:",
                list(
                    pred_results.columns
                )
            )

        st.markdown(
            "### Interpretation"
        )

        st.info(
            """
            The probabilistic formulation should be evaluated
            using both discrimination metrics such as ROC-AUC
            and probability-quality metrics such as Brier score,
            log loss and calibration error, rather than accuracy alone.
            """
        )


# ============================================================
# PORTFOLIO COMPARISON
# ============================================================

elif page == "Portfolio Comparison":

    st.title(
        "⚖️ Portfolio Comparison"
    )

    summary = find_backtest(
        "multi_source_portfolio_summary.csv"
    )

    if (
        summary is None
        or summary.empty
    ):

        show_missing(
            "Could not find "
            "backtests/multi_source_portfolio_summary.csv."
        )

    else:

        st.write(
            """
            Compare portfolio strategies, feature sets
            and rebalance frequencies.
            """
        )

        feature_values = (
            sorted(
                summary[
                    "feature_set"
                ]
                .dropna()
                .unique()
            )
            if "feature_set"
            in summary.columns
            else []
        )

        strategy_values = (
            sorted(
                summary[
                    "strategy"
                ]
                .dropna()
                .unique()
            )
            if "strategy"
            in summary.columns
            else []
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            feature_filter = st.multiselect(
                "Feature Set",
                feature_values,
                default=feature_values,
            )

        with c2:

            strategy_filter = st.multiselect(
                "Strategy",
                strategy_values,
                default=strategy_values,
            )

        with c3:

            metric = st.selectbox(
                "Rank By",
                [
                    "sharpe",
                    "annualized_return",
                    "sortino",
                    "max_drawdown",
                    "average_turnover",
                ],
            )

        df = summary.copy()

        if feature_filter:

            df = df[
                df[
                    "feature_set"
                ].isin(
                    feature_filter
                )
            ]

        if strategy_filter:

            df = df[
                df[
                    "strategy"
                ].isin(
                    strategy_filter
                )
            ]

        ascending = metric in [
            "max_drawdown",
            "average_turnover",
        ]

        df = df.sort_values(
            metric,
            ascending=ascending
        )

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )

        if (
            PLOTLY_OK
            and not df.empty
        ):

            st.markdown(
                "### Risk-return view"
            )

            plot_df = df.copy()

            plot_df[
                "label"
            ] = (
                plot_df[
                    "feature_set"
                ].astype(str)
                + " · "
                + plot_df[
                    "strategy"
                ].astype(str)
                + " · "
                + plot_df[
                    "rebalance_days"
                ].astype(str)
                + "d"
            )

            fig = px.scatter(
                plot_df,
                x="annualized_volatility",
                y="annualized_return",
                color="strategy",
                hover_name="label",
                size=np.maximum(
                    plot_df[
                        "sharpe"
                    ].abs(),
                    0.05
                ),
                labels={
                    "annualized_volatility":
                        "Annualized volatility",
                    "annualized_return":
                        "Annualized return",
                },
            )

            fig.update_xaxes(
                tickformat=".0%"
            )

            fig.update_yaxes(
                tickformat=".0%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# REGIME ROBUSTNESS
# ============================================================

elif page == "Regime Robustness":

    st.title(
        "📅 Regime Robustness"
    )

    st.write(
        """
        Fixed-strategy comparison across common out-of-sample
        calendar years. No strategy is selected separately
        for each year.
        """
    )

    df = find_experiment(
        "final_regime_robustness.csv"
    )

    if (
        df is None
        or df.empty
    ):

        show_missing(
            "Could not find "
            "experiments/final_regime_robustness.csv."
        )

    else:

        if (
            "year" in df.columns
            and "configuration"
            in df.columns
        ):

            sharpe = df.pivot(
                index="year",
                columns="configuration",
                values="sharpe",
            )

            returns = df.pivot(
                index="year",
                columns="configuration",
                values="annualized_return",
            )

            st.markdown(
                "### Sharpe Ratio by Year"
            )

            if PLOTLY_OK:

                plot_sharpe = (
                    sharpe
                    .reset_index()
                    .melt(
                        id_vars="year",
                        var_name="configuration",
                        value_name="sharpe",
                    )
                )

                fig = px.line(
                    plot_sharpe,
                    x="year",
                    y="sharpe",
                    color="configuration",
                    markers=True,
                    labels={
                        "sharpe":
                            "Sharpe ratio"
                    },
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.dataframe(
                sharpe.round(4),
                use_container_width=True
            )

            st.markdown(
                "### Annualized Return by Year"
            )

            if PLOTLY_OK:

                plot_ret = (
                    returns
                    .reset_index()
                    .melt(
                        id_vars="year",
                        var_name="configuration",
                        value_name="annualized_return",
                    )
                )

                fig = px.bar(
                    plot_ret,
                    x="year",
                    y="annualized_return",
                    color="configuration",
                    barmode="group",
                    labels={
                        "annualized_return":
                            "Annualized return"
                    },
                )

                fig.update_yaxes(
                    tickformat=".0%"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.dataframe(
                (
                    returns * 100
                ).round(2),
                use_container_width=True
            )

            if {
                "CryptoTrack_Probability",
                "Binary_Baseline",
            }.issubset(
                sharpe.columns
            ):

                diff = (
                    sharpe[
                        "CryptoTrack_Probability"
                    ]
                    -
                    sharpe[
                        "Binary_Baseline"
                    ]
                )

                wins = int(
                    (
                        diff > 0
                    ).sum()
                )

                st.success(
                    f"CryptoTrack Probability has the "
                    f"higher Sharpe in {wins}/"
                    f"{len(diff)} evaluated calendar years."
                )

                comparison = pd.DataFrame(
                    {
                        "CryptoTrack Sharpe":
                            sharpe[
                                "CryptoTrack_Probability"
                            ],
                        "Binary Sharpe":
                            sharpe[
                                "Binary_Baseline"
                            ],
                        "Difference":
                            diff,
                        "CryptoTrack Wins":
                            diff > 0,
                    }
                )

                st.dataframe(
                    comparison.round(4),
                    use_container_width=True
                )


# ============================================================
# RESEARCH EVIDENCE
# ============================================================

elif page == "Research Evidence":

    st.title(
        "📚 Research Evidence"
    )

    st.write(
        """
        A compact evidence panel for the research paper,
        project report and presentation.
        """
    )

    files = {
        "Prediction experiments":
            EXPERIMENT_DIR
            / "prediction_experiment_results.csv",

        "Portfolio experiments":
            BACKTEST_DIR
            / "multi_source_portfolio_summary.csv",

        "Regime robustness":
            EXPERIMENT_DIR
            / "final_regime_robustness.csv",

        "Regime Sharpe table":
            EXPERIMENT_DIR
            / "final_regime_sharpe_table.csv",

        "Transaction-cost sensitivity":
            EXPERIMENT_DIR
            / "transaction_cost_sensitivity.csv",

        "Phi sensitivity":
            EXPERIMENT_DIR
            / "phi_sensitivity.csv",
    }

    rows = []

    for label, path in files.items():

        rows.append(
            {
                "Evidence":
                    label,

                "Status":
                    (
                        "Available"
                        if path.exists()
                        else "Missing"
                    ),

                "Path":
                    (
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        )
                        if path.exists()
                        else "—"
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Current research findings
    # --------------------------------------------------------

    st.markdown(
        "### Current Research Findings"
    )

    st.markdown(
        """
        **1. Probabilistic allocation**

        The probability-based portfolio strategy provides
        a continuous alternative to the binary 0/1 forecast
        used by the baseline methodology.

        **2. Feature-source ablation**

        Market data, Google Trends and Twitter-derived
        features can be evaluated separately and in combination
        to determine their incremental contribution.

        **3. Transaction costs**

        Probability-based allocation generally produces
        substantially lower turnover than the binary strategy,
        making its performance less sensitive to transaction costs.

        **4. Walk-forward validation**

        Model hyperparameters are selected using chronological
        validation while keeping later observations for
        out-of-sample evaluation.

        **5. Regime robustness**

        The probability strategy is not assumed to dominate
        every individual market regime. Calendar-year analysis
        is used to evaluate whether the advantage is consistent.

        **6. Dataset limitation**

        The Twitter dataset used by CryptoTrack is smaller and
        differently constructed from the original paper's Twitter
        corpus. Therefore, the project should be presented as
        an extension rather than an exact replication.
        """
    )

    st.warning(
        """
        CryptoTrack is a research and portfolio analytics
        prototype. It is not a live trading system and does
        not provide investment advice.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "CryptoTrack • Research Prototype"
)