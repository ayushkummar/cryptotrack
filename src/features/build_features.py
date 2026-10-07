from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
TRENDS_DIR = RAW_DIR / "google_trends"
PROCESSED_DIR = Path("data/processed")

TWITTER_PATH = PROCESSED_DIR / "twitter_daily_sentiment.parquet"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "ADA",
    "DOGE",
    "DOT",
    "LTC",
]


# ---------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_features(df):
    df = df.copy()

    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_7d"] = df["close"].pct_change(7)

    df["volatility_7d"] = (
        df["return_1d"].rolling(7).std()
    )

    df["volatility_14d"] = (
        df["return_1d"].rolling(14).std()
    )

    df["volatility_30d"] = (
        df["return_1d"].rolling(30).std()
    )

    df["sma_7"] = df["close"].rolling(7).mean()
    df["sma_30"] = df["close"].rolling(30).mean()

    df["sma_ratio_7"] = (
        df["close"] / df["sma_7"]
    )

    df["sma_ratio_30"] = (
        df["close"] / df["sma_30"]
    )

    df["volume_change"] = (
        df["volume"].pct_change()
    )

    df["rsi_14"] = calculate_rsi(df["close"])

    ema_12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    return df


# ---------------------------------------------------------
# Google Trends
# ---------------------------------------------------------

def load_google_trends(asset):

    path = TRENDS_DIR / f"{asset}_trends.csv"

    if not path.exists():
        print(f"WARNING: Trends file not found: {path}")
        return None

    trends = pd.read_csv(path)

    trends["timestamp"] = pd.to_datetime(
        trends["timestamp"]
    )

    # Use 2021–2026
    trends = trends[
        trends["timestamp"] >= "2021-01-01"
    ].copy()

    trends = trends.sort_values("timestamp")

    trends["google_trends_change"] = (
        trends["google_trends"].pct_change()
    )

    trends["google_trends_3m_avg"] = (
        trends["google_trends"]
        .rolling(3)
        .mean()
    )

    return trends


def merge_trends(market_df, trends_df):

    if trends_df is None:
        return market_df

    market_df = market_df.sort_values("timestamp").copy()
    trends_df = trends_df.sort_values("timestamp").copy()

    merged = pd.merge_asof(
        market_df,
        trends_df,
        on="timestamp",
        direction="backward"
    )

    return merged


# ---------------------------------------------------------
# Twitter sentiment
# ---------------------------------------------------------

def load_twitter():

    if not TWITTER_PATH.exists():

        print(
            f"WARNING: Twitter sentiment file not found: "
            f"{TWITTER_PATH}"
        )

        return None

    twitter = pd.read_parquet(TWITTER_PATH)

    twitter["timestamp"] = pd.to_datetime(
        twitter["timestamp"]
    )

    # Keep only the project period
    twitter = twitter[
        twitter["timestamp"] >= "2021-01-01"
    ].copy()

    # Keep only our seven cryptocurrencies
    twitter = twitter[
        twitter["asset"].isin(ASSETS)
    ].copy()

    twitter = twitter.sort_values(
        ["asset", "timestamp"]
    )

    return twitter


def merge_twitter(market_df, twitter_df, asset):

    if twitter_df is None:
        return market_df

    # Select only the asset being processed
    asset_twitter = twitter_df[
        twitter_df["asset"] == asset
    ].copy()

    if asset_twitter.empty:

        print(
            f"WARNING: No Twitter data available for {asset}"
        )

        market_df["twitter_available"] = 0

        return market_df

    # -----------------------------------------------------
    # Core Twitter features
    # -----------------------------------------------------

    twitter_features = [
        "tweet_count",
        "positive_count",
        "negative_count",
        "neutral_count",
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
    ]

    # Only keep columns that actually exist
    available_features = [
        col
        for col in twitter_features
        if col in asset_twitter.columns
    ]

    asset_twitter = asset_twitter[
        ["timestamp"] + available_features
    ].copy()

    # Make sure one row exists per asset/date
    asset_twitter = (
        asset_twitter
        .groupby("timestamp", as_index=False)
        .mean(numeric_only=True)
    )

    # -----------------------------------------------------
    # Exact-date merge
    # -----------------------------------------------------

    market_df = market_df.copy()

    market_df = market_df.merge(
        asset_twitter,
        on="timestamp",
        how="left"
    )

    # -----------------------------------------------------
    # Missing Twitter observations
    # -----------------------------------------------------

    # No tweets on a date does NOT mean negative sentiment.
    # We therefore use zero for aggregate count/sentiment
    # features and explicitly record availability.

    market_df["twitter_available"] = (
        market_df["tweet_count"]
        .notna()
        .astype(int)
    )

    numeric_twitter_features = [
        col
        for col in available_features
        if col != "sentiment_std"
    ]

    for col in numeric_twitter_features:

        market_df[col] = (
            market_df[col]
            .fillna(0)
        )

    # A missing standard deviation means that there was
    # no Twitter observation for that day.
    if "sentiment_std" in market_df.columns:

        market_df["sentiment_std"] = (
            market_df["sentiment_std"]
            .fillna(0)
        )

    return market_df


# ---------------------------------------------------------
# Calendar-based Twitter rolling features
# ---------------------------------------------------------

def add_twitter_rolling_features(df):

    df = df.copy()

    if "twitter_available" not in df.columns:
        return df

    # -----------------------------------------------------
    # 7-calendar-day rolling features
    # -----------------------------------------------------

    if "sentiment_weighted" in df.columns:

        df["twitter_sentiment_7d_avg"] = (
            df["sentiment_weighted"]
            .rolling(7, min_periods=1)
            .mean()
        )

        df["twitter_sentiment_7d_change"] = (
            df["sentiment_weighted"]
            - df["sentiment_weighted"].shift(7)
        )

    if "tweet_count" in df.columns:

        df["tweet_count_7d_avg"] = (
            df["tweet_count"]
            .rolling(7, min_periods=1)
            .mean()
        )

    return df


# ---------------------------------------------------------
# Targets
# ---------------------------------------------------------

def create_targets(df):

    df["future_return_1d"] = (
        df["close"].shift(-1) / df["close"] - 1
    )

    df["future_return_3d"] = (
        df["close"].shift(-3) / df["close"] - 1
    )

    df["future_return_5d"] = (
        df["close"].shift(-5) / df["close"] - 1
    )

    df["target_1d"] = (
        df["future_return_1d"] > 0
    ).astype(int)

    df["target_3d"] = (
        df["future_return_3d"] > 0
    ).astype(int)

    df["target_5d"] = (
        df["future_return_5d"] > 0
    ).astype(int)

    return df


# ---------------------------------------------------------
# Process one asset
# ---------------------------------------------------------

def process_asset(asset):

    market_path = RAW_DIR / f"{asset}.csv"

    if not market_path.exists():

        print(
            f"WARNING: Market file not found: {market_path}"
        )

        return

    print(f"\nProcessing {asset}...")

    df = pd.read_csv(market_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values("timestamp")

    # -----------------------------------------------------
    # Restrict experiment to 2021–2026
    # -----------------------------------------------------

    df = df[
        df["timestamp"] >= "2021-01-01"
    ].copy()

    # -----------------------------------------------------
    # Technical features
    # -----------------------------------------------------

    df = calculate_features(df)

    # -----------------------------------------------------
    # Google Trends
    # -----------------------------------------------------

    trends = load_google_trends(asset)

    df = merge_trends(
        df,
        trends
    )

    # -----------------------------------------------------
    # Twitter
    # -----------------------------------------------------

    twitter = load_twitter()

    df = merge_twitter(
        df,
        twitter,
        asset
    )

    # -----------------------------------------------------
    # Twitter rolling features
    # -----------------------------------------------------

    df = add_twitter_rolling_features(df)

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------

    df = create_targets(df)

    # -----------------------------------------------------
    # Remove rows where required market features
    # are unavailable
    # -----------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        f"{asset}_features.parquet"
    )

    df.to_parquet(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    print("\nTwitter coverage:")

    if "twitter_available" in df.columns:

        print(
            df["twitter_available"]
            .value_counts()
            .sort_index()
        )

        coverage = (
            df["twitter_available"].mean() * 100
        )

        print(
            f"Twitter coverage: {coverage:.2f}%"
        )

    print("\nFeature sample:")

    sample_columns = [
        "timestamp",
        "close",
        "google_trends",
        "sentiment_mean",
        "sentiment_weighted",
        "tweet_count",
        "twitter_available",
    ]

    sample_columns = [
        col
        for col in sample_columns
        if col in df.columns
    ]

    print(
        df[sample_columns].head(10)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    for asset in ASSETS:
        process_asset(asset)

    print("\nFeature generation complete.")