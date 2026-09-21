"""
CLI entry point.

    python main.py
    python main.py --trades data/sample_trades.csv --output alerts.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from surveillance import run_surveillance  # noqa: E402


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Run rule-based and statistical trade surveillance.")
    parser.add_argument("--trades", default="data/sample_trades.csv")
    parser.add_argument("--output", default="alerts.csv")
    parser.add_argument("--contamination", type=float, default=0.03,
                         help="Expected proportion of anomalous trades for the Isolation Forest model.")
    args = parser.parse_args()

    df = load_trades(args.trades)
    alerts, _population_stats = run_surveillance(df, contamination=args.contamination)
    display_alerts = alerts.drop(columns="evidence")

    display_alerts.to_csv(args.output, index=False)

    print(f"{len(df)} trades analyzed, {len(alerts)} alerts generated.\n")
    if not alerts.empty:
        print(display_alerts["rule"].value_counts().to_string())
        print("\nTop alerts:")
        print(display_alerts.head(10).to_string(index=False))
    print(f"\nFull alert queue written to {args.output}")


if __name__ == "__main__":
    main()
