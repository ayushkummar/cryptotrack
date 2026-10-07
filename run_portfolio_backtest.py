from pathlib import Path

import sys

import pandas as pd

from numbers import Number

# ============================================================

# PATHS

# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"

PRED_DIR = ROOT / "experiments"

OUT_DIR = ROOT / "backtests"

OUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

sys.path.insert(

    0,

    str(ROOT / "src"),

)

from portfolio.backtest import run_backtest

# ============================================================

# CONFIGURATION

# ============================================================

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

    "market_only": "market_only",

    "market_trends": "market_trends",

    "market_twitter": "market_twitter",

    "market_twitter_trends": "market_twitter_trends",

}

STRATEGIES = [

    "equal_weight",

    "minimum_variance",

    "paper_binary",

    "paper_binary_equal",

    "probability",

]

REBALANCE_FREQUENCIES = [

    1,

    7,

    30,

]

TRANSACTION_COST = 0.001

COVARIANCE_WINDOW = 90

PHI = 0.50

# ============================================================

# LOAD PRICES

# ============================================================

def load_prices():

    price_frames = []

    for asset in ASSETS:

        path = (

            DATA_DIR /

            f"{asset}_features.parquet"

        )

        if not path.exists():

            raise FileNotFoundError(

                f"Missing feature file: {path}"

            )

        df = pd.read_parquet(path)

        df["timestamp"] = pd.to_datetime(

            df["timestamp"]

        )

        df = (

            df

            .sort_values("timestamp")

            .drop_duplicates(

                "timestamp"

            )

        )

        price = df[

            [

                "timestamp",

                "close",

            ]

        ].copy()

        price = price.rename(

            columns={

                "close": asset

            }

        )

        price_frames.append(price)

    prices = price_frames[0]

    for frame in price_frames[1:]:

        prices = prices.merge(

            frame,

            on="timestamp",

            how="inner",

        )

    prices = (

        prices

        .sort_values("timestamp")

        .dropna()

        .set_index("timestamp")

    )

    return prices

# ============================================================

# LOAD LONG-FORMAT PREDICTIONS

# ============================================================

def load_predictions(feature_suffix):

    frames = []

    for asset in ASSETS:

        path = (

            PRED_DIR /

            f"{asset}_{feature_suffix}_predictions_1d.csv"

        )

        if not path.exists():

            raise FileNotFoundError(

                f"Missing prediction file: {path}"

            )

        df = pd.read_csv(path)

        # ----------------------------------------------------

        # Required columns

        # ----------------------------------------------------

        required = [

            "timestamp",

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

                f"{path.name} is missing "

                f"columns: {missing}"

            )

        df = df[

            required

        ].copy()

        df["timestamp"] = pd.to_datetime(

            df["timestamp"]

        )

        df["asset"] = asset

        frames.append(df)

    predictions = pd.concat(

        frames,

        ignore_index=True,

    )

    predictions = (

        predictions

        .sort_values(

            [

                "timestamp",

                "asset",

            ]

        )

        .reset_index(drop=True)

    )

    return predictions

# ============================================================

# RUN ONE FEATURE SET

# ============================================================

def run_feature_set(

    prices,

    feature_name,

    feature_suffix,

):

    print("\n" + "=" * 80)

    print(

        f"FEATURE SET: {feature_name}"

    )

    print("=" * 80)

    predictions = load_predictions(

        feature_suffix

    )

    # --------------------------------------------------------

    # Keep only dates available in both datasets

    # --------------------------------------------------------

    common_dates = (

        set(prices.index)

        &

        set(predictions["timestamp"])

    )

    predictions = predictions[

        predictions["timestamp"].isin(

            common_dates

        )

    ].copy()

    feature_prices = prices.loc[

        sorted(common_dates)

    ].copy()

    print(

        f"Prediction rows: "

        f"{len(predictions)}"

    )

    print(

        f"Common dates: "

        f"{len(common_dates)}"

    )

    results = []

    daily_results = []

    # --------------------------------------------------------

    # Portfolio experiments

    # --------------------------------------------------------

    for strategy in STRATEGIES:

        for rebalance in REBALANCE_FREQUENCIES:

            print(

                f"\nRunning: "

                f"{feature_name} | "

                f"{strategy} | "

                f"{rebalance}d"

            )

            try:

                result = run_backtest(

                    prices=feature_prices,

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

                    phi=PHI,

                )

                # ------------------------------------------------

                # Summary result

                # ------------------------------------------------

                summary = {

                    "feature_set":

                        feature_name,

                    "strategy":

                        strategy,

                    "rebalance_days":

                        rebalance,

                    "transaction_cost":

                        TRANSACTION_COST,

                    "phi":

                        PHI,

                }
                # Portfolio metrics are returned inside result["metrics"].
                metrics = result.get("metrics", {})

                if not isinstance(metrics, dict):
                    raise TypeError(
                        "Expected result['metrics'] to be a dictionary, "
                        f"got {type(metrics).__name__}"
                    )

                for key, value in metrics.items():
                    if isinstance(value, Number):
                        summary[key] = float(value)

                results.append(summary)

                # ------------------------------------------------

                # Daily returns

                # ------------------------------------------------

                daily_returns = result.get(

                    "daily_returns"

                )

                if daily_returns is not None:

                    daily = pd.Series(

                        daily_returns

                    )

                    daily_df = (

                        daily

                        .rename(

                            "daily_return"

                        )

                        .reset_index()

                    )

                    daily_df = (

                        daily_df

                        .rename(

                            columns={

                                "index":

                                    "timestamp"

                            }

                        )

                    )

                    daily_df[

                        "feature_set"

                    ] = feature_name

                    daily_df[

                        "strategy"

                    ] = strategy

                    daily_df[

                        "rebalance_days"

                    ] = rebalance

                    daily_results.append(

                        daily_df

                    )

                print(

                    "  Completed"

                )

            except Exception as exc:

                print(

                    f"  ERROR: {exc}"

                )

    return results, daily_results

# ============================================================

# MAIN

# ============================================================

def main():

    print("=" * 80)

    print(

        "CryptoTrack: Multi-Source "

        "Portfolio Experiment"

    )

    print("=" * 80)

    # --------------------------------------------------------

    # Load prices

    # --------------------------------------------------------

    print(

        "\nLoading prices..."

    )

    prices = load_prices()

    print(

        f"Price data: "

        f"{prices.shape[0]} dates x "

        f"{prices.shape[1]} assets"

    )

    all_results = []

    all_daily_results = []

    # --------------------------------------------------------

    # Run all feature sets

    # --------------------------------------------------------

    for feature_name, feature_suffix in (

        FEATURE_SETS.items()

    ):

        results, daily_results = (

            run_feature_set(

                prices,

                feature_name,

                feature_suffix,

            )

        )

        all_results.extend(

            results

        )

        all_daily_results.extend(

            daily_results

        )

    # --------------------------------------------------------

    # Create result tables

    # --------------------------------------------------------

    summary_df = pd.DataFrame(

        all_results

    )

    if all_daily_results:

        daily_df = pd.concat(

            all_daily_results,

            ignore_index=True,

        )

    else:

        daily_df = pd.DataFrame()

    # --------------------------------------------------------

    # Save summary

    # --------------------------------------------------------

    summary_path = (

        OUT_DIR /

        "multi_source_portfolio_summary.csv"

    )

    summary_df.to_csv(

        summary_path,

        index=False,

    )

    # --------------------------------------------------------

    # Save daily results

    # --------------------------------------------------------

    daily_path = (

        OUT_DIR /

        "multi_source_portfolio_daily_returns.csv"

    )

    if not daily_df.empty:

        daily_df.to_csv(

            daily_path,

            index=False,

        )

    # --------------------------------------------------------

    # Final output

    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print(

        "EXPERIMENT COMPLETE"

    )

    print("=" * 80)

    print(

        f"\nSuccessful experiments: "

        f"{len(summary_df)}"

    )

    print(

        f"\nSummary saved to:\n"

        f"{summary_path}"

    )

    if not daily_df.empty:

        print(

            f"\nDaily returns saved to:\n"

            f"{daily_path}"

        )

    # --------------------------------------------------------

    # Display results

    if not summary_df.empty:

        columns = [
            "feature_set",
            "strategy",
            "rebalance_days",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "average_turnover",
            "total_transaction_cost",
        ]

        columns = [
            c for c in columns
            if c in summary_df.columns
        ]

        if "sharpe" in summary_df.columns:
            print("\nTop 20 results by Sharpe:")
            print(
                summary_df[columns]
                .sort_values("sharpe", ascending=False)
                .head(20)
                .to_string(index=False)
            )
        else:
            print("\nWARNING: 'sharpe' metric was not found.")
            print("Available columns:")
            print(summary_df.columns.tolist())


if __name__ == "__main__":

    main()