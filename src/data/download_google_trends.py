from pathlib import Path
import time

import pandas as pd
from pytrends.request import TrendReq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_DATE = "2019-01-01"
END_DATE = "2026-09-30"

OUTPUT_DIR = Path("data/raw/google_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "LTC": "Litecoin",
}


# ---------------------------------------------------------
# Google Trends downloader
# ---------------------------------------------------------

def download_trend(keyword, asset):
    print(f"\nDownloading Google Trends: {asset} ({keyword})")

    pytrends = TrendReq(
    hl="en-US",
    tz=330,
    timeout=(10, 30)
)

    timeframe = f"{START_DATE} {END_DATE}"

    pytrends.build_payload(
        kw_list=[keyword],
        timeframe=timeframe,
        geo="",
        gprop=""
    )

    df = pytrends.interest_over_time()

    if df.empty:
        raise RuntimeError(
            f"No Google Trends data returned for {asset}"
        )

    # Remove Google's helper column
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index()

    # Standardize names
    df = df.rename(
        columns={
            "date": "timestamp",
            keyword: "google_trends"
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Keep only the required columns
    df = df[["timestamp", "google_trends"]]

    # Save
    output_file = OUTPUT_DIR / f"{asset}_trends.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df)}")
    print(df.head())

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    for asset, keyword in ASSETS.items():

        try:
            download_trend(keyword, asset)

        except Exception as e:
            print(f"ERROR for {asset}: {e}")

        # Avoid hammering Google Trends
        time.sleep(5)

    print("\nGoogle Trends download complete.")