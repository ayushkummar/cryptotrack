from pathlib import Path
import time
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "DOT": "DOT-USD",
    "LTC": "LTC-USD",
}

START = "2019-01-01"

for symbol, ticker in TICKERS.items():
    print(f"Downloading {symbol}...")
    df = yf.download(ticker, start=START, interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"date": "timestamp"})
    required = ["timestamp","open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{symbol}: missing {missing}")
    df[required].to_csv(RAW/f"{symbol}.csv", index=False)
    print(f"Saved {len(df):,} rows")
    time.sleep(1)
