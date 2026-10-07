from pathlib import Path
import re
import math
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

# Change this only if your CSV has a different name
INPUT_FILE = ROOT / "data" / "raw" / "twitter" / (
    "dataset_52-person-from-2021-02-05_2023-06-12_21-34-17-266_with_sentiment.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "twitter_daily_sentiment.parquet"
SUMMARY_FILE = OUTPUT_DIR / "twitter_coverage_summary.csv"

ASSETS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "DOT": ["polkadot", "dot"],
    "LTC": ["litecoin", "ltc"],
}


# ============================================================
# HELPERS
# ============================================================

def contains_alias(coin_string, aliases):
    """
    Check whether the cryptocurrency appears in the dataset's
    new_coins field.

    new_coins examples:
        (bitcoin)
        (bitcoin,btc)
        (eth,btc)
    """

    if pd.isna(coin_string):
        return False

    text = str(coin_string).lower()

    # Remove brackets/parentheses so matching is easier
    text = re.sub(r"[\(\)\[\]]", " ", text)

    # Split comma-separated values
    tokens = [
        token.strip()
        for token in text.split(",")
    ]

    return any(
        token in aliases
        for token in tokens
    )


def detect_english(text):
    """
    Original paper filters non-English tweets.
    We apply a lightweight language filter.
    """

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 3:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
    except Exception:
        return False


def clean_tweet(text):
    """
    Similar preprocessing concept to the original paper:
    remove URLs, usernames, RT markers, hashtags and extra spaces.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove @usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove RT markers
    text = re.sub(
        r"\brt\b",
        " ",
        text
    )

    # Remove hashtag symbol but preserve the word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def sentiment_category(compound):
    """
    VADER thresholds used by the paper:
        positive >= 0.05
        negative <= -0.05
        neutral otherwise
    """

    if compound >= 0.05:
        return "positive"

    if compound <= -0.05:
        return "negative"

    return "neutral"


# ============================================================
# LOAD DATA
# ============================================================

def load_tweets():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nTwitter dataset not found:\n{INPUT_FILE}\n\n"
            "Put the CSV inside:\n"
            "data/raw/twitter/\n"
        )

    print("Loading Twitter dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} tweets."
    )

    print("\nColumns:")
    print(df.columns.tolist())

    required = [
        "created_at",
        "full_text",
        "new_coins",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# PROCESS TWEETS
# ============================================================

def process_tweets(df):

    print("\nProcessing tweets...")

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    ).dt.normalize()

    df = df.dropna(
        subset=["date"]
    ).copy()

    # Keep only the period that overlaps our crypto project
    df = df[
        (df["date"] >= "2021-01-01")
        &
        (df["date"] <= "2023-06-12")
    ].copy()

    print(
        f"After date filtering: {len(df):,} tweets"
    )

    # --------------------------------------------------------
    # English filtering
    # --------------------------------------------------------

    print(
        "\nDetecting English tweets..."
    )

    english_mask = df["full_text"].apply(
        detect_english
    )

    df = df[english_mask].copy()

    print(
        f"After English filtering: {len(df):,} tweets"
    )

    # --------------------------------------------------------
    # Tweet cleaning
    # --------------------------------------------------------

    df["processed_text"] = df["full_text"].apply(
        clean_tweet
    )

    df = df[
        df["processed_text"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # VADER
    # --------------------------------------------------------

    print(
        "\nRunning VADER sentiment analysis..."
    )

    analyzer = SentimentIntensityAnalyzer()

    vader_scores = df["processed_text"].apply(
        analyzer.polarity_scores
    )

    df["vader_negative"] = vader_scores.apply(
        lambda x: x["neg"]
    )

    df["vader_neutral"] = vader_scores.apply(
        lambda x: x["neu"]
    )

    df["vader_positive"] = vader_scores.apply(
        lambda x: x["pos"]
    )

    df["vader_compound"] = vader_scores.apply(
        lambda x: x["compound"]
    )

    df["vader_sentiment"] = df[
        "vader_compound"
    ].apply(
        sentiment_category
    )

    # --------------------------------------------------------
    # Engagement
    # --------------------------------------------------------

    for col in [
        "retweet_count",
        "favorite_count",
        "reply_count",
    ]:

        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    df["engagement"] = (
        df["retweet_count"]
        + df["favorite_count"]
        + df["reply_count"]
    )

    # --------------------------------------------------------
    # Assign cryptocurrencies
    # --------------------------------------------------------

    print(
        "\nAssigning tweets to cryptocurrencies..."
    )

    asset_frames = []

    for asset, aliases in ASSETS.items():

        mask = df["new_coins"].apply(
            lambda x: contains_alias(
                x,
                aliases
            )
        )

        asset_df = df[mask].copy()

        asset_df["asset"] = asset

        asset_frames.append(
            asset_df
        )

        print(
            f"{asset}: {len(asset_df):,} tweets"
        )

    tweets = pd.concat(
        asset_frames,
        ignore_index=True
    )

    return tweets


# ============================================================
# DAILY SENTIMENT INDICATORS
# ============================================================

def build_daily_features(tweets):

    print(
        "\nBuilding daily sentiment indicators..."
    )

    records = []

    grouped = tweets.groupby(
        ["asset", "date"],
        sort=True
    )

    for (asset, date), group in grouped:

        positive = (
            group["vader_sentiment"]
            == "positive"
        ).sum()

        negative = (
            group["vader_sentiment"]
            == "negative"
        ).sum()

        neutral = (
            group["vader_sentiment"]
            == "neutral"
        ).sum()

        total = (
            positive
            + negative
            + neutral
        )

        # ----------------------------------------------------
        # Paper's first sentiment indicator:
        #
        # B_t =
        # (M_pos - M_neg) /
        # (M_pos + M_neg)
        # ----------------------------------------------------

        denominator = positive + negative

        if denominator > 0:
            B = (
                positive - negative
            ) / denominator
        else:
            B = 0.0

        # ----------------------------------------------------
        # Paper's second indicator:
        #
        # B*_t =
        # ln((M_pos + 1) / (M_neg + 1))
        # ----------------------------------------------------

        B_star = math.log(
            (positive + 1)
            /
            (negative + 1)
        )

        # ----------------------------------------------------
        # Paper's third indicator:
        #
        # B**_t =
        # B_t * ln(1 + M_t)
        # ----------------------------------------------------

        B_double_star = (
            B * math.log(1 + total)
        )

        # ----------------------------------------------------
        # Additional features for CryptoTrack
        # ----------------------------------------------------

        sentiment_mean = group[
            "vader_compound"
        ].mean()

        sentiment_std = group[
            "vader_compound"
        ].std()

        if pd.isna(sentiment_std):
            sentiment_std = 0.0

        positive_ratio = positive / total
        negative_ratio = negative / total
        neutral_ratio = neutral / total

        engagement_mean = group[
            "engagement"
        ].mean()

        # Engagement-weighted sentiment
        engagement_total = group[
            "engagement"
        ].sum()

        if engagement_total > 0:
            weighted_sentiment = (
                (
                    group["vader_compound"]
                    * group["engagement"]
                ).sum()
                /
                engagement_total
            )
        else:
            weighted_sentiment = sentiment_mean

        records.append({
            "timestamp": date,
            "asset": asset,

            "tweet_count": total,

            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,

            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,

            "sentiment_mean": sentiment_mean,
            "sentiment_std": sentiment_std,

            "sentiment_weighted": weighted_sentiment,

            "B_sentiment": B,
            "B_star_sentiment": B_star,
            "B_double_star_sentiment": B_double_star,

            "engagement_mean": engagement_mean,
            "engagement_total": engagement_total,
        })

    daily = pd.DataFrame(records)

    daily = daily.sort_values(
        ["asset", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    daily["sentiment_7d_avg"] = (
        daily
        .groupby("asset")["sentiment_mean"]
        .transform(
            lambda x: x.rolling(
                7,
                min_periods=1
            ).mean()
        )
    )

    daily["sentiment_7d_change"] = (
        daily
        .groupby("asset")["sentiment_mean"]
        .transform(
            lambda x: x.diff(7)
        )
    )

    daily["tweet_count_7d_avg"] = (
        daily
        .groupby("asset")["tweet_count"]
        .transform(
            lambda x: x.rolling(
                7,
                min_periods=1
            ).mean()
        )
    )

    daily["sentiment_7d_change"] = (
        daily["sentiment_7d_change"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    return daily


# ============================================================
# COVERAGE SUMMARY
# ============================================================

def create_summary(daily):

    summary = (
        daily
        .groupby("asset")
        .agg(
            first_date=("timestamp", "min"),
            last_date=("timestamp", "max"),
            days=("timestamp", "nunique"),
            total_tweet_days=("timestamp", "count"),
            total_tweets=("tweet_count", "sum"),
            average_daily_tweets=("tweet_count", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nSaved coverage summary:\n{SUMMARY_FILE}"
    )

    print("\nCoverage:")
    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTOTRACK - TWITTER SENTIMENT PIPELINE")
    print("=" * 70)

    df = load_tweets()

    tweets = process_tweets(df)

    daily = build_daily_features(
        tweets
    )

    create_summary(
        daily
    )

    print("\n" + "=" * 70)
    print("TWITTER FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "\nNext files:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()