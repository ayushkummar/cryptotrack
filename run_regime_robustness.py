from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CryptoTrack — Final Regime Robustness Analysis
# ============================================================
# Uses ONLY four pre-selected 30-day backtests.
# No yearly winner selection. No sensitivity-file search.
#
# Primary comparison:
#   CryptoTrack probability vs paper-inspired binary baseline
#
# Fixed settings:
#   rebalance = 30 days
#   transaction cost = 0.001 (0.10%)
#   phi = 0.50 for forecast-guided strategies
# ============================================================

ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "backtests"
OUTPUT_DIR = ROOT / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "CryptoTrack_Probability":
        "probability_cost_0.0010_rebalance_30d.csv",
    "Binary_Baseline":
        "paper_binary_cost_0.0010_rebalance_30d.csv",
    "Minimum_Variance":
        "minimum_variance_rebalance_30d.csv",
    "Equal_Weight":
        "equal_weight_rebalance_30d.csv",
}

EXPECTED_COST_RATE = 0.001
MIN_DAYS_PER_YEAR = 30


def calculate_metrics(df):
    r = pd.to_numeric(df["portfolio_return"], errors="coerce").dropna()

    if len(r) == 0:
        return None

    cumulative_return = (1.0 + r).prod() - 1.0
    n = len(r)

    # Same convention used by the project's backtest module:
    # crypto is annualized with 365 days.
    annualized_return = (1.0 + cumulative_return) ** (365.0 / n) - 1.0
    annualized_volatility = r.std(ddof=1) * np.sqrt(365.0)

    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0 else np.nan
    )

    downside = r[r < 0]
    if len(downside) > 1:
        downside_deviation = downside.std(ddof=1) * np.sqrt(365.0)
        sortino = (
            annualized_return / downside_deviation
            if downside_deviation > 0 else np.nan
        )
    else:
        sortino = np.nan

    wealth = (1.0 + r).cumprod()
    max_drawdown = (wealth / wealth.cummax() - 1.0).min()

    turnover = (
        pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)
        if "turnover" in df.columns else pd.Series(dtype=float)
    )
    costs = (
        pd.to_numeric(df["transaction_cost"], errors="coerce").fillna(0.0)
        if "transaction_cost" in df.columns else pd.Series(dtype=float)
    )

    return {
        "days": int(n),
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "average_turnover": (
            float(turnover.mean()) if len(turnover) else np.nan
        ),
        "total_transaction_cost": (
            float(costs.sum()) if len(costs) else np.nan
        ),
    }


def validate_file(label, path, df):
    required = {
        "timestamp",
        "portfolio_return",
        "turnover",
        "transaction_cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    # For the two cost-sensitivity filenames, verify mathematically that
    # transaction_cost ≈ turnover * 0.001 on nonzero-turnover rows.
    # This avoids relying only on the filename.
    if label in {"CryptoTrack_Probability", "Binary_Baseline"}:
        turnover = pd.to_numeric(df["turnover"], errors="coerce")
        cost = pd.to_numeric(df["transaction_cost"], errors="coerce")

        mask = turnover.notna() & cost.notna() & (turnover.abs() > 1e-12)
        if mask.any():
            implied_rate = (cost[mask] / turnover[mask]).median()
            if not np.isclose(
                implied_rate, EXPECTED_COST_RATE, rtol=1e-4, atol=1e-8
            ):
                raise ValueError(
                    f"{path.name}: implied transaction-cost rate is "
                    f"{implied_rate:.8f}, expected {EXPECTED_COST_RATE:.4f}."
                )
            print(
                f"  Verified {label}: implied cost rate "
                f"{implied_rate:.4%}"
            )
        else:
            raise ValueError(
                f"{path.name}: cannot verify transaction-cost rate "
                "because no nonzero-turnover rows were found."
            )


def main():
    print("=" * 78)
    print("CryptoTrack — Final Regime Robustness Analysis")
    print("=" * 78)

    frames = {}

    for label, filename in FILES.items():
        path = BACKTEST_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}\n\n"
                "Do not substitute another sensitivity file."
            )

        df = pd.read_csv(path)
        validate_file(label, path, df)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["portfolio_return"] = pd.to_numeric(
            df["portfolio_return"], errors="coerce"
        )
        df = (
            df.dropna(subset=["timestamp", "portfolio_return"])
              .sort_values("timestamp")
              .drop_duplicates("timestamp")
              .reset_index(drop=True)
        )

        if len(df) < MIN_DAYS_PER_YEAR:
            raise ValueError(f"{filename} contains too few usable rows.")

        frames[label] = df
        print(
            f"  Loaded {label}: {filename} "
            f"({df['timestamp'].min().date()} to "
            f"{df['timestamp'].max().date()}, {len(df)} rows)"
        )

    # Fair comparison: use only dates common to ALL four strategies.
    common_dates = None
    for df in frames.values():
        dates = set(df["timestamp"])
        common_dates = dates if common_dates is None else common_dates & dates

    common_dates = sorted(common_dates)

    if len(common_dates) < MIN_DAYS_PER_YEAR:
        raise RuntimeError("Too few dates common to all four configurations.")

    print(f"\nCommon OOS dates across all strategies: {len(common_dates)}")
    print(f"Common range: {common_dates[0].date()} to {common_dates[-1].date()}")

    rows = []

    for label, df in frames.items():
        df = df[df["timestamp"].isin(common_dates)].copy()
        df["year"] = df["timestamp"].dt.year

        for year, sub in df.groupby("year"):
            if len(sub) < MIN_DAYS_PER_YEAR:
                print(
                    f"  Skipping {label} / {year}: only {len(sub)} days"
                )
                continue

            m = calculate_metrics(sub)
            if m is not None:
                rows.append({
                    "configuration": label,
                    "year": int(year),
                    **m,
                })

    results = pd.DataFrame(rows)

    # Retain only years represented by all four configurations.
    counts = results.groupby("year")["configuration"].nunique()
    valid_years = counts[counts == len(FILES)].index.tolist()
    results = results[results["year"].isin(valid_years)].copy()

    if results.empty:
        raise RuntimeError(
            "No calendar year has enough common observations "
            "for all four configurations."
        )

    results = results.sort_values(["year", "configuration"])

    # Main paper-friendly tables.
    sharpe = results.pivot(
        index="year", columns="configuration", values="sharpe"
    )
    annual_return = results.pivot(
        index="year", columns="configuration", values="annualized_return"
    )
    max_dd = results.pivot(
        index="year", columns="configuration", values="max_drawdown"
    )

    # Direct CryptoTrack vs Binary comparison.
    comparison = pd.DataFrame(index=sharpe.index)
    comparison["cryptotrack_sharpe"] = sharpe["CryptoTrack_Probability"]
    comparison["binary_sharpe"] = sharpe["Binary_Baseline"]
    comparison["sharpe_difference"] = (
        comparison["cryptotrack_sharpe"]
        - comparison["binary_sharpe"]
    )
    comparison["cryptotrack_wins"] = (
        comparison["sharpe_difference"] > 0
    )

    # Save.
    results_path = OUTPUT_DIR / "final_regime_robustness.csv"
    sharpe_path = OUTPUT_DIR / "final_regime_sharpe_table.csv"
    return_path = OUTPUT_DIR / "final_regime_return_table.csv"
    dd_path = OUTPUT_DIR / "final_regime_drawdown_table.csv"
    comparison_path = OUTPUT_DIR / "final_regime_probability_vs_binary.csv"

    results.to_csv(results_path, index=False)
    sharpe.to_csv(sharpe_path)
    annual_return.to_csv(return_path)
    max_dd.to_csv(dd_path)
    comparison.to_csv(comparison_path)

    print("\n" + "=" * 78)
    print("SHARPE RATIO BY YEAR")
    print("=" * 78)
    print(sharpe.round(4).to_string())

    print("\n" + "=" * 78)
    print("ANNUALIZED RETURN BY YEAR")
    print("=" * 78)
    print((annual_return * 100).round(2).to_string() + "\n(values in %)")

    print("\n" + "=" * 78)
    print("CRYPTOTRACK PROBABILITY vs BINARY BASELINE")
    print("=" * 78)
    print(comparison.round(4).to_string())

    wins = int(comparison["cryptotrack_wins"].sum())
    total = len(comparison)
    print(
        f"\nCryptoTrack has the higher Sharpe in "
        f"{wins}/{total} evaluated calendar years."
    )

    print("\nDetailed results:")
    display_cols = [
        "configuration", "year", "days",
        "annualized_return", "annualized_volatility",
        "sharpe", "sortino", "max_drawdown",
        "average_turnover", "total_transaction_cost",
    ]
    print(results[display_cols].round(6).to_string(index=False))

    print("\nSaved:")
    for p in [
        results_path, sharpe_path, return_path,
        dd_path, comparison_path
    ]:
        print(f"  {p}")

    print("\nNo yearly strategy selection was performed.")


if __name__ == "__main__":
    main()
