from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import pandas as pd


# ============================================================
# BASIC HELPERS
# ============================================================

def calculate_transaction_costs(
    old_weights,
    new_weights,
    transaction_cost,
):
    """
    Transaction cost based on portfolio turnover.

    turnover = sum(abs(new_weights - old_weights))

    cost = turnover * transaction_cost
    """

    if old_weights is None:
        return 0.0

    turnover = np.sum(
        np.abs(new_weights - old_weights)
    )

    return float(turnover * transaction_cost)


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def portfolio_metrics(
    portfolio_returns,
    turnovers,
    transaction_costs,
):
    """
    Calculate portfolio performance metrics.

    portfolio_returns:
        Daily net portfolio returns after transaction costs.

    turnovers:
        Daily turnover.

    transaction_costs:
        Daily transaction costs.
    """

    portfolio_returns = pd.Series(
        portfolio_returns
    ).dropna()

    if len(portfolio_returns) == 0:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "average_turnover": np.nan,
            "total_transaction_cost": np.nan,
        }

    # --------------------------------------------------------
    # Cumulative return
    # --------------------------------------------------------

    cumulative_return = (
        (1 + portfolio_returns).prod() - 1
    )

    # --------------------------------------------------------
    # Annualized return
    # Crypto trades 365 days/year
    # --------------------------------------------------------

    n_days = len(portfolio_returns)

    annualized_return = (
        (1 + cumulative_return) ** (365 / n_days)
    ) - 1

    # --------------------------------------------------------
    # Annualized volatility
    # --------------------------------------------------------

    annualized_volatility = (
        portfolio_returns.std(ddof=1) * np.sqrt(365)
    )

    # --------------------------------------------------------
    # Sharpe ratio
    # Risk-free rate = 0
    # --------------------------------------------------------

    if annualized_volatility > 0:
        sharpe = (
            annualized_return /
            annualized_volatility
        )
    else:
        sharpe = np.nan

    # --------------------------------------------------------
    # Sortino ratio
    # --------------------------------------------------------

    downside_returns = portfolio_returns[
        portfolio_returns < 0
    ]

    if len(downside_returns) > 1:

        downside_deviation = (
            downside_returns.std(ddof=1)
            * np.sqrt(365)
        )

        if downside_deviation > 0:
            sortino = (
                annualized_return /
                downside_deviation
            )
        else:
            sortino = np.nan

    else:
        sortino = np.nan

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    wealth = (
        1 + portfolio_returns
    ).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max - 1
    )

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Turnover / transaction costs
    # --------------------------------------------------------

    turnovers = pd.Series(turnovers).reindex(
        portfolio_returns.index,
        fill_value=0.0,
    )

    transaction_costs = pd.Series(
        transaction_costs
    ).reindex(
        portfolio_returns.index,
        fill_value=0.0,
    )

    average_turnover = turnovers.mean()

    total_transaction_cost = (
        transaction_costs.sum()
    )

    return {
        "cumulative_return": float(
            cumulative_return
        ),
        "annualized_return": float(
            annualized_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "average_turnover": float(
            average_turnover
        ),
        "total_transaction_cost": float(
            total_transaction_cost
        ),
    }


# ============================================================
# PORTFOLIO WEIGHTS
# ============================================================

def equal_weight(n_assets):
    """
    Standard 1/N portfolio.
    """

    if n_assets <= 0:
        return np.array([])

    return np.ones(n_assets) / n_assets


def minimum_variance_weights(
    returns,
    covariance_window=90,
):
    """
    Long-only minimum variance portfolio.

    Uses inverse variance approximation.

    This is kept as the existing lightweight benchmark.
    """

    returns = returns.tail(
        covariance_window
    )

    if len(returns) < 20:
        return equal_weight(
            returns.shape[1]
        )

    variances = returns.var().values

    variances = np.where(
        np.isfinite(variances)
        & (variances > 1e-12),
        variances,
        1e-12,
    )

    inverse_variance = 1.0 / variances

    weights = (
        inverse_variance /
        inverse_variance.sum()
    )

    return weights


# ============================================================
# PAPER-INSPIRED BINARY GMV
# ============================================================

def paper_binary_gmv_weights(
    cov,
    forecast,
    phi=0.50,
):
    """
    Exact constrained minimum-variance optimization
    inspired by Model (11)/(12) of Zhou et al.

    forecast:
        0/1 vector.

        1 = predicted price will not fall
        0 = predicted price will fall

    Constraints:

        sum(w) = 1

        w_i >= 0

        For assets predicted not to fall:

            w_i >= phi / n_positive

    If every asset is predicted to fall,
    return zero weights.
    """

    cov = np.asarray(
        cov,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    n = len(forecast)

    # --------------------------------------------------------
    # Number of positive forecasts
    # --------------------------------------------------------

    n_positive = int(
        np.sum(forecast)
    )

    # If all assets are predicted to fall,
    # paper model allows zero investment.
    if n_positive == 0:
        return np.zeros(n)

    # --------------------------------------------------------
    # Clean covariance matrix
    # --------------------------------------------------------

    cov = np.nan_to_num(
        cov,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cov = (
        cov + cov.T
    ) / 2.0

    # Small diagonal regularization
    cov = (
        cov +
        np.eye(n) * 1e-8
    )

    # --------------------------------------------------------
    # Paper-inspired lower bounds
    # --------------------------------------------------------

    min_positive_weight = (
        phi / n_positive
    )

    lower_bounds = np.zeros(n)

    for i in range(n):

        if forecast[i] == 1:
            lower_bounds[i] = (
                min_positive_weight
            )

    # --------------------------------------------------------
    # Objective
    # Minimum portfolio variance
    # --------------------------------------------------------

    def objective(w):

        return float(
            w.T @ cov @ w
        )

    # --------------------------------------------------------
    # Constraint
    # Fully invested portfolio
    # --------------------------------------------------------

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        }
    ]

    # --------------------------------------------------------
    # Feasible starting point
    # --------------------------------------------------------

    remaining = (
        1.0 -
        np.sum(lower_bounds)
    )

    if remaining < -1e-10:

        raise ValueError(
            "Infeasible portfolio "
            "constraints. Check phi "
            "and forecast vector."
        )

    w0 = (
        lower_bounds +
        remaining / n
    )

    # --------------------------------------------------------
    # SLSQP optimization
    # --------------------------------------------------------

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[
            (
                lower_bounds[i],
                1.0,
            )
            for i in range(n)
        ],
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 500,
            "disp": False,
        },
    )

    # --------------------------------------------------------
    # Optimization failure fallback
    # --------------------------------------------------------

    if not result.success:

        print(
            "Warning: constrained binary "
            "optimizer failed:",
            result.message,
        )

        return (
            w0 /
            np.sum(w0)
        )

    # --------------------------------------------------------
    # Clean final weights
    # --------------------------------------------------------

    weights = np.asarray(
        result.x,
        dtype=float,
    )

    weights[weights < 0] = 0

    weight_sum = weights.sum()

    if weight_sum <= 1e-12:
        return np.zeros(n)

    weights = (
        weights /
        weight_sum
    )

    return weights


# ============================================================
# PAPER-INSPIRED 1/N
# ============================================================

def paper_binary_equal_weight(
    binary_predictions,
):
    """
    Forecast-informed 1/N strategy.

    Equal weight only among assets predicted
    not to fall.

    If none are predicted positive,
    remain fully invested using equal weight.
    """

    f = np.asarray(
        binary_predictions,
        dtype=float,
    )

    positive = f > 0

    n_positive = positive.sum()

    if n_positive == 0:

        return equal_weight(
            len(f)
        )

    weights = np.zeros(
        len(f)
    )

    weights[positive] = (
        1.0 / n_positive
    )

    return weights


# ============================================================
# PROBABILITY-BASED GMV
# ============================================================

def probability_gmv_weights(
    cov,
    probabilities,
    phi=0.50,
):
    """
    Probabilistic extension of the paper's
    forecast-constrained minimum-variance portfolio.

    probabilities:
        Probability that each cryptocurrency
        will NOT fall.

    The probability vector determines
    minimum allocations:

        w_i >= phi * p_i

    where:

        p_i = normalized probability

    The remaining capital is allocated by
    minimizing portfolio variance.

    This is the proposed CryptoTrack extension,
    not a direct equation from the original paper.
    """

    cov = np.asarray(
        cov,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    n = len(probabilities)

    # --------------------------------------------------------
    # Clean probabilities
    # --------------------------------------------------------

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    total_probability = (
        probabilities.sum()
    )

    # If probability mass is invalid,
    # return zero weights.
    if total_probability <= 1e-12:
        return np.zeros(n)

    # --------------------------------------------------------
    # Normalize probabilities
    # --------------------------------------------------------

    p = (
        probabilities /
        total_probability
    )

    # --------------------------------------------------------
    # Clean covariance matrix
    # --------------------------------------------------------

    cov = np.nan_to_num(
        cov,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cov = (
        cov + cov.T
    ) / 2.0

    cov = (
        cov +
        np.eye(n) * 1e-8
    )

    # --------------------------------------------------------
    # Probability-informed lower bounds
    # --------------------------------------------------------

    lower_bounds = (
        phi * p
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(w):

        return float(
            w.T @ cov @ w
        )

    # --------------------------------------------------------
    # Fully invested constraint
    # --------------------------------------------------------

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        }
    ]

    # --------------------------------------------------------
    # Feasibility
    # --------------------------------------------------------

    remaining = (
        1.0 -
        np.sum(lower_bounds)
    )

    if remaining < -1e-10:

        raise ValueError(
            "Infeasible probability "
            "portfolio constraints."
        )

    # --------------------------------------------------------
    # Feasible initial point
    # --------------------------------------------------------

    w0 = (
        lower_bounds +
        remaining / n
    )

    # --------------------------------------------------------
    # SLSQP optimization
    # --------------------------------------------------------

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[
            (
                lower_bounds[i],
                1.0,
            )
            for i in range(n)
        ],
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 500,
            "disp": False,
        },
    )

    # --------------------------------------------------------
    # Optimization failure
    # --------------------------------------------------------

    if not result.success:

        print(
            "Warning: probability "
            "optimizer failed:",
            result.message,
        )

        return (
            w0 /
            np.sum(w0)
        )

    # --------------------------------------------------------
    # Clean final weights
    # --------------------------------------------------------

    weights = np.asarray(
        result.x,
        dtype=float,
    )

    weights[weights < 0] = 0

    weight_sum = weights.sum()

    if weight_sum <= 1e-12:
        return np.zeros(n)

    weights = (
        weights /
        weight_sum
    )

    return weights


# ============================================================
# PREDICTION HELPERS
# ============================================================

def get_prediction_probabilities(
    predictions,
    date,
    assets,
    probability_column="prob_positive",
):
    """
    Get probability forecasts for all assets
    on a date.
    """

    daily = predictions[
        predictions["timestamp"] == date
    ]

    probabilities = []

    for asset in assets:

        row = daily[
            daily["asset"] == asset
        ]

        if len(row) == 0:

            probabilities.append(
                0.5
            )

        else:

            probabilities.append(
                float(
                    row.iloc[0][
                        probability_column
                    ]
                )
            )

    return np.asarray(
        probabilities
    )


def get_binary_predictions(
    predictions,
    date,
    assets,
):
    """
    Get binary forecasts for all assets
    on a date.
    """

    daily = predictions[
        predictions["timestamp"] == date
    ]

    binary = []

    for asset in assets:

        row = daily[
            daily["asset"] == asset
        ]

        if len(row) == 0:

            binary.append(0)

        else:

            binary.append(
                int(
                    row.iloc[0][
                        "prediction"
                    ]
                )
            )

    return np.asarray(
        binary
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    prices,
    predictions=None,
    strategy="equal_weight",
    transaction_cost=0.001,
    rebalance_frequency=30,
    covariance_window=90,
    probability_column="prob_positive",
    phi=0.50,
):
    """
    Run daily portfolio backtest.

    IMPORTANT TIMING:

        Prediction at day t
                    ↓
        portfolio weights determined
                    ↓
        hold portfolio
                    ↓
        return from t -> t+1

    Rebalance frequency determines when
    weights are changed.

    It does NOT remove daily returns
    from the backtest.
    """

    prices = prices.copy()

    prices.index = pd.to_datetime(
        prices.index
    )

    prices = prices.sort_index()

    assets = list(
        prices.columns
    )

    # --------------------------------------------------------
    # Daily asset returns
    # --------------------------------------------------------

    asset_returns = (
        prices.pct_change()
    )

    # --------------------------------------------------------
    # Prediction dates
    # --------------------------------------------------------

    if predictions is None:

        raise ValueError(
            "Predictions are required "
            "for this backtest."
        )

    predictions = predictions.copy()

    predictions["timestamp"] = (
        pd.to_datetime(
            predictions["timestamp"]
        )
    )

    prediction_dates = sorted(
        set(
            predictions["timestamp"]
        )
        &
        set(prices.index)
    )

    if len(prediction_dates) == 0:

        raise ValueError(
            "No prediction dates "
            "overlap with price data."
        )

    prediction_date_set = set(
        prediction_dates
    )

    # --------------------------------------------------------
    # Start at first OOS prediction
    # --------------------------------------------------------

    first_prediction_date = (
        prediction_dates[0]
    )

    first_position = (
        prices.index.get_loc(
            first_prediction_date
        )
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    portfolio_returns = []

    return_dates = []

    turnovers = []

    transaction_costs = []

    weights_history = []

    current_weights = None

    last_rebalance_position = None

    # --------------------------------------------------------
    # Main daily loop
    # --------------------------------------------------------

    for i in range(
        first_position,
        len(prices.index) - 1,
    ):

        current_date = (
            prices.index[i]
        )

        next_date = (
            prices.index[i + 1]
        )

        # ----------------------------------------------------
        # Determine whether we rebalance
        # ----------------------------------------------------

        should_rebalance = False

        if current_date in prediction_date_set:

            if current_weights is None:

                should_rebalance = True

            else:

                days_since_rebalance = (
                    i -
                    last_rebalance_position
                )

                if (
                    days_since_rebalance
                    >= rebalance_frequency
                ):

                    should_rebalance = True

        # ----------------------------------------------------
        # Calculate new weights
        # ----------------------------------------------------

        daily_turnover = 0.0

        daily_transaction_cost = 0.0

        if should_rebalance:

            # Historical returns available
            # only up to current date.
            historical_returns = (
                asset_returns.loc[
                    :current_date
                ].dropna()
            )

            # ------------------------------------------------
            # Not enough history
            # ------------------------------------------------

            if len(historical_returns) < 20:

                new_weights = equal_weight(
                    len(assets)
                )

            # ------------------------------------------------
            # Equal weight
            # ------------------------------------------------

            elif strategy == "equal_weight":

                new_weights = equal_weight(
                    len(assets)
                )

            # ------------------------------------------------
            # Minimum variance
            # ------------------------------------------------

            elif strategy == "minimum_variance":

                new_weights = (
                    minimum_variance_weights(
                        historical_returns,
                        covariance_window=(
                            covariance_window
                        ),
                    )
                )

            # ------------------------------------------------
            # Paper binary GMV
            # ------------------------------------------------

            elif strategy == "paper_binary":

                binary = (
                    get_binary_predictions(
                        predictions,
                        current_date,
                        assets,
                    )
                )

                # IMPORTANT:
                # The optimizer requires a covariance
                # matrix, not the raw return DataFrame.

                covariance_returns = (
                    historical_returns.tail(
                        covariance_window
                    )
                )

                cov = (
                    covariance_returns.cov()
                )

                new_weights = (
                    paper_binary_gmv_weights(
                        cov,
                        binary,
                        phi=phi,
                    )
                )

            # ------------------------------------------------
            # Paper binary 1/N
            # ------------------------------------------------

            elif strategy == "paper_binary_equal":

                binary = (
                    get_binary_predictions(
                        predictions,
                        current_date,
                        assets,
                    )
                )

                new_weights = (
                    paper_binary_equal_weight(
                        binary
                    )
                )

            # ------------------------------------------------
            # Probability GMV
            # ------------------------------------------------

            elif strategy == "probability":

                probabilities = (
                    get_prediction_probabilities(
                        predictions,
                        current_date,
                        assets,
                        probability_column=(
                            probability_column
                        ),
                    )
                )

                # IMPORTANT:
                # The optimizer requires a covariance
                # matrix.

                covariance_returns = (
                    historical_returns.tail(
                        covariance_window
                    )
                )

                cov = (
                    covariance_returns.cov()
                )

                new_weights = (
                    probability_gmv_weights(
                        cov,
                        probabilities,
                        phi=phi,
                    )
                )

            else:

                raise ValueError(
                    f"Unknown strategy: "
                    f"{strategy}"
                )

            # ------------------------------------------------
            # Clean weights
            # ------------------------------------------------

            new_weights = np.asarray(
                new_weights,
                dtype=float,
            )

            new_weights = np.nan_to_num(
                new_weights,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            new_weights = np.maximum(
                new_weights,
                0,
            )

            # ------------------------------------------------
            # Handle zero-investment case
            # ------------------------------------------------
            #
            # The paper-inspired binary GMV strategy
            # can legitimately return zero weights when
            # every asset is predicted to fall.
            #
            # We DO NOT automatically convert this to
            # equal weight for paper_binary.
            #
            # For other strategies, zero weights are treated
            # as an invalid portfolio and replaced by
            # equal weight.
            # ------------------------------------------------

            if new_weights.sum() <= 1e-12:

                if strategy == "paper_binary":

                    new_weights = np.zeros(
                        len(assets)
                    )

                else:

                    new_weights = equal_weight(
                        len(assets)
                    )

            else:

                new_weights = (
                    new_weights /
                    new_weights.sum()
                )

            # ------------------------------------------------
            # Transaction cost
            # ------------------------------------------------

            if current_weights is not None:

                daily_turnover = np.sum(
                    np.abs(
                        new_weights -
                        current_weights
                    )
                )

                daily_transaction_cost = (
                    daily_turnover *
                    transaction_cost
                )

            current_weights = (
                new_weights
            )

            last_rebalance_position = i

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if current_weights is None:

            continue

        # ----------------------------------------------------
        # Return from t -> t+1
        # ----------------------------------------------------

        next_day_returns = (
            asset_returns.loc[
                next_date
            ].reindex(assets)
        )

        next_day_returns = (
            next_day_returns.fillna(0)
        )

        # ----------------------------------------------------
        # Gross portfolio return
        # ----------------------------------------------------

        gross_portfolio_return = np.sum(
            current_weights *
            next_day_returns.values
        )

        # ----------------------------------------------------
        # Net return
        # ----------------------------------------------------

        net_portfolio_return = (
            gross_portfolio_return -
            daily_transaction_cost
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        return_dates.append(
            next_date
        )

        portfolio_returns.append(
            net_portfolio_return
        )

        turnovers.append(
            daily_turnover
        )

        transaction_costs.append(
            daily_transaction_cost
        )

        weights_history.append(
            current_weights.copy()
        )

    # ========================================================
    # Convert results
    # ========================================================

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=pd.to_datetime(
            return_dates
        ),
        name="portfolio_return",
    )

    turnovers = pd.Series(
        turnovers,
        index=portfolio_returns.index,
        name="turnover",
    )

    transaction_costs = pd.Series(
        transaction_costs,
        index=portfolio_returns.index,
        name="transaction_cost",
    )

    weights_history = pd.DataFrame(
        weights_history,
        index=portfolio_returns.index,
        columns=assets,
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = portfolio_metrics(
        portfolio_returns,
        turnovers,
        transaction_costs,
    )

    return {
        "portfolio_returns": portfolio_returns,
        "turnovers": turnovers,
        "transaction_costs": transaction_costs,
        "weights": weights_history,
        "metrics": metrics,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_backtest_results(
    results,
    output_path,
):
    """
    Save detailed backtest results.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame({

        "timestamp":
            results[
                "portfolio_returns"
            ].index,

        "portfolio_return":
            results[
                "portfolio_returns"
            ].values,

        "turnover":
            results[
                "turnovers"
            ].values,

        "transaction_cost":
            results[
                "transaction_costs"
            ].values,
    })

    df["cumulative_wealth"] = (
        1 +
        df["portfolio_return"]
    ).cumprod()

    weights = results[
        "weights"
    ].copy()

    for column in weights.columns:

        df[
            f"weight_{column}"
        ] = weights[
            column
        ].values

    df.to_csv(
        output_path,
        index=False,
    )