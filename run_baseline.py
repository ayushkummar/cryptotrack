from pathlib import Path

import pandas as pd

from src.models.walk_forward_svm import (
    walk_forward,
    evaluate,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "experiments"

OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Feature groups
# ---------------------------------------------------------

# Market / technical indicators
MARKET_FEATURES = [
    "return_1d",
    "return_3d",
    "return_7d",
    "volatility_7d",
    "volatility_14d",
    "volatility_30d",
    "sma_ratio_7",
    "sma_ratio_30",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
]


# Google Trends
TRENDS_FEATURES = [
    "google_trends",
    "google_trends_change",
    "google_trends_3m_avg",
]


# Twitter sentiment / activity
TWITTER_FEATURES = [
    "tweet_count",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sentiment_mean",
    "sentiment_std",
    "sentiment_weighted",
    "B_sentiment",
    "B_star_sentiment",
    "B_double_star_sentiment",
    "engagement_mean",
    "engagement_total",
    "twitter_available",
    "twitter_sentiment_7d_avg",
    "twitter_sentiment_7d_change",
    "tweet_count_7d_avg",
]


# ---------------------------------------------------------
# Experiments
# ---------------------------------------------------------

experiments = {

    # E1: Market-only baseline
    "market_only":
        MARKET_FEATURES,

    # E2: Market + Google Trends
    "market_trends":
        MARKET_FEATURES
        + TRENDS_FEATURES,

    # E3: Market + Twitter
    "market_twitter":
        MARKET_FEATURES
        + TWITTER_FEATURES,

    # E4: Market + Twitter + Google Trends
    "market_twitter_trends":
        MARKET_FEATURES
        + TWITTER_FEATURES
        + TRENDS_FEATURES,
}


# ---------------------------------------------------------
# Run experiments
# ---------------------------------------------------------

all_results = []


for experiment_name, features in experiments.items():

    print("\n" + "=" * 70)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print("=" * 70)

    for path in sorted(
        PROCESSED.glob(
            "*_features.parquet"
        )
    ):

        symbol = path.stem.replace(
            "_features",
            ""
        )

        print(
            f"\nRunning {symbol}..."
        )

        df = pd.read_parquet(path)

        # -------------------------------------------------
        # Check required features
        # -------------------------------------------------

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:

            print(
                f"SKIPPING {symbol}: "
                f"missing {missing}"
            )

            continue

        # -------------------------------------------------
        # Select model data
        # -------------------------------------------------

        model_df = df[
            [
                "timestamp",
                "close",
                "future_return_1d",
                "target_1d",
            ]
            + features
        ].copy()

        # -------------------------------------------------
        # Walk-forward prediction
        # -------------------------------------------------

        pred = walk_forward(
            model_df,
            feature_cols=features,
            target_col="target_1d",
        )

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------

        metrics = evaluate(pred)

        print(
            "\nMetrics:"
        )

        for metric, value in metrics.items():

            print(
                f"  {metric}: "
                f"{value:.6f}"
            )

        # -------------------------------------------------
        # Save predictions
        # -------------------------------------------------

        output_file = (
            OUT
            / f"{symbol}_{experiment_name}"
              f"_predictions_1d.csv"
        )

        pred.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nSaved: {output_file}"
        )

        # -------------------------------------------------
        # Store experiment results
        # -------------------------------------------------

        all_results.append({
            "experiment": experiment_name,
            "asset": symbol,
            **metrics,
        })


# ---------------------------------------------------------
# Save all results
# ---------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)


summary_file = (
    OUT
    / "prediction_experiment_results.csv"
)


results_df.to_csv(
    summary_file,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)


print("\nIndividual results:")

print(
    results_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Average performance
# ---------------------------------------------------------

print(
    "\n"
    + "=" * 70
)

print(
    "AVERAGE PERFORMANCE BY EXPERIMENT"
)

print(
    "=" * 70
)


average_results = (
    results_df
    .groupby("experiment")
    .mean(numeric_only=True)
)


print(
    average_results.to_string()
)


# ---------------------------------------------------------
# Best experiment by metric
# ---------------------------------------------------------

if not average_results.empty:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST EXPERIMENTS"
    )

    print(
        "=" * 70
    )

    for metric in [
        "accuracy",
        "roc_auc",
        "brier",
        "log_loss",
        "ece",
    ]:

        if metric not in average_results.columns:
            continue

        if metric in [
            "brier",
            "log_loss",
            "ece",
        ]:

            best = (
                average_results[metric]
                .idxmin()
            )

        else:

            best = (
                average_results[metric]
                .idxmax()
            )

        value = (
            average_results
            .loc[best, metric]
        )

        print(
            f"{metric}: "
            f"{best} "
            f"({value:.6f})"
        )


print(
    f"\nSaved: {summary_file}"
)