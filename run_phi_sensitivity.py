from pathlib import Path

import pandas as pd

from src.portfolio.backtest import run_backtest


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "experiments"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "ADA",
    "DOGE",
    "DOT",
    "LTC",
]


# Transaction cost used in the main experiment
TRANSACTION_COST = 0.001

# Same covariance window as the main experiment
COVARIANCE_WINDOW = 90

# Phi values to test
PHI_VALUES = [
    0.25,
    0.50,
    0.75,
]

# Only the two strategies that actually use phi
STRATEGIES = [
    "paper_binary",
    "probability",
]

# Rebalance frequencies
REBALANCE_FREQUENCIES = [
    7,
    30,
]


# ============================================================
# LOAD PRICES
# ============================================================

def load_prices():

    price_data = {}

    for asset in ASSETS:

        path = (
            DATA_DIR /
            f"{asset}_features.parquet"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing price file: {path}"
            )

        df = pd.read_parquet(path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )

        df = df.set_index(
            "timestamp"
        )

        price_data[asset] = df["close"]

    prices = pd.concat(
        price_data,
        axis=1,
    )

    prices.columns = ASSETS

    prices = prices.sort_index()

    return prices


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    prediction_frames = []

    for asset in ASSETS:

        path = (
            PRED_DIR /
            f"{asset}_market_trends_predictions_1d.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing prediction file: {path}"
            )

        df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df["asset"] = asset

        required = [
            "timestamp",
            "asset",
            "prediction",
            "prob_positive",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{asset}: missing columns {missing}"
            )

        df = df[
            [
                "timestamp",
                "asset",
                "prediction",
                "prob_positive",
            ]
        ]

        prediction_frames.append(df)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions = predictions.sort_values(
        [
            "timestamp",
            "asset",
        ]
    )

    predictions = predictions.reset_index(
        drop=True
    )

    return predictions


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    strategy,
    phi,
    rebalance,
    metrics,
):

    print("\n" + "-" * 70)

    print(
        f"Strategy: {strategy}"
    )

    print(
        f"Phi: {phi:.2f}"
    )

    print(
        f"Rebalance: every {rebalance} day(s)"
    )

    print(
        f"Cumulative return: "
        f"{metrics['cumulative_return']:.2%}"
    )

    print(
        f"Annualized return: "
        f"{metrics['annualized_return']:.2%}"
    )

    print(
        f"Volatility: "
        f"{metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe: "
        f"{metrics['sharpe']:.4f}"
    )

    print(
        f"Sortino: "
        f"{metrics['sortino']:.4f}"
    )

    print(
        f"Max drawdown: "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Average turnover: "
        f"{metrics['average_turnover']:.4f}"
    )

    print(
        f"Transaction costs: "
        f"{metrics['total_transaction_cost']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CRYPTOTRACK PHI SENSITIVITY EXPERIMENT"
    )

    print("=" * 70)

    print(
        "\nThis experiment tests whether portfolio results "
        "depend strongly on the choice of phi."
    )

    print(
        f"Phi values: {PHI_VALUES}"
    )

    print(
        f"Strategies: {STRATEGIES}"
    )

    print(
        f"Rebalance frequencies: "
        f"{REBALANCE_FREQUENCIES}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST:.4f}"
    )

    print(
        f"Covariance window: "
        f"{COVARIANCE_WINDOW}"
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading prices...")

    prices = load_prices()

    print(
        f"Price data: "
        f"{len(prices)} dates × "
        f"{len(prices.columns)} assets"
    )


    print("\nLoading predictions...")

    predictions = load_predictions()

    print(
        f"Prediction rows: "
        f"{len(predictions)}"
    )


    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    all_results = []

    total_experiments = (
        len(PHI_VALUES)
        * len(STRATEGIES)
        * len(REBALANCE_FREQUENCIES)
    )

    experiment_number = 0


    for phi in PHI_VALUES:

        for rebalance in REBALANCE_FREQUENCIES:

            for strategy in STRATEGIES:

                experiment_number += 1

                print("\n")
                print("#" * 70)

                print(
                    f"EXPERIMENT "
                    f"{experiment_number}/"
                    f"{total_experiments}"
                )

                print("#" * 70)

                print(
                    f"Strategy: {strategy}"
                )

                print(
                    f"Phi: {phi:.2f}"
                )

                print(
                    f"Rebalance: "
                    f"every {rebalance} day(s)"
                )


                # ------------------------------------------------
                # Run backtest
                # ------------------------------------------------

                results = run_backtest(

                    prices=prices,

                    predictions=predictions,

                    strategy=strategy,

                    transaction_cost=(
                        TRANSACTION_COST
                    ),

                    rebalance_frequency=(
                        rebalance
                    ),

                    covariance_window=(
                        COVARIANCE_WINDOW
                    ),

                    probability_column=(
                        "prob_positive"
                    ),

                    phi=phi,
                )


                metrics = results[
                    "metrics"
                ].copy()


                # Add experiment information
                metrics["strategy"] = strategy

                metrics["phi"] = phi

                metrics[
                    "rebalance_frequency"
                ] = rebalance


                all_results.append(
                    metrics
                )


                print_result(
                    strategy,
                    phi,
                    rebalance,
                    metrics,
                )


    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        all_results
    )


    columns = [
        "strategy",
        "phi",
        "rebalance_frequency",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "average_turnover",
        "total_transaction_cost",
    ]


    summary = summary[
        columns
    ]


    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    summary = summary.sort_values(
        [
            "strategy",
            "rebalance_frequency",
            "phi",
        ]
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = (
        OUT_DIR /
        "phi_sensitivity.csv"
    )


    summary.to_csv(
        output_path,
        index=False,
    )


    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print("\n\n")

    print("=" * 70)

    print(
        "PHI SENSITIVITY SUMMARY"
    )

    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )


    print("\n")

    print(
        f"Saved results to:"
    )

    print(
        output_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()