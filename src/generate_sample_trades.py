"""
Generates a synthetic order/trade dataset for the surveillance engine to
analyze: a base population of ordinary trades across several accounts, plus
a handful of deliberately crafted suspicious patterns so the detectors have
something real to catch:

  - a scalping burst on one account (many trades held for a few seconds)
  - an oversized position on another account (far larger than its own history)
  - a wash-trade pair between two related accounts (opposite sides, same
    symbol, same volume, seconds apart)

Reproducible via a fixed seed.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
ACCOUNTS = [f"ACC-{i}" for i in range(1, 11)]


def generate_normal_trades(rng: random.Random, n: int, base_time: datetime) -> list:
    rows = []
    for i in range(n):
        account = rng.choice(ACCOUNTS)
        symbol = rng.choice(SYMBOLS)
        side = rng.choice(["BUY", "SELL"])
        volume = round(rng.uniform(0.1, 2.0), 2)
        open_time = base_time + timedelta(seconds=rng.randint(0, 6 * 3600))
        hold_seconds = rng.randint(30, 4 * 3600)  # ordinary holding times: 30s to 4h
        rows.append({
            "trade_id": f"N{1000 + i}",
            "account_id": account,
            "related_group": account,  # normal trades: no shared beneficial-owner grouping
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "open_time": open_time,
            "close_time": open_time + timedelta(seconds=hold_seconds),
        })
    return rows


def generate_scalping_burst(rng: random.Random, base_time: datetime, account: str) -> list:
    rows = []
    t = base_time + timedelta(hours=1)
    for i in range(12):
        hold = rng.uniform(1, 4)  # 1-4 seconds: a scalping candidate under the 5s rule
        rows.append({
            "trade_id": f"SCALP{i}",
            "account_id": account,
            "related_group": account,
            "symbol": "EURUSD",
            "side": rng.choice(["BUY", "SELL"]),
            "volume": round(rng.uniform(0.1, 0.5), 2),
            "open_time": t,
            "close_time": t + timedelta(seconds=hold),
        })
        t += timedelta(seconds=rng.uniform(10, 40))
    return rows


def generate_oversized_position(base_time: datetime, account: str) -> list:
    # this account's normal trades are ~0.1-2.0 lots; this one is 10x that
    return [{
        "trade_id": "OVERSIZED1",
        "account_id": account,
        "related_group": account,
        "symbol": "GBPUSD",
        "side": "BUY",
        "volume": 15.0,
        "open_time": base_time + timedelta(hours=2),
        "close_time": base_time + timedelta(hours=2, minutes=20),
    }]


def generate_wash_trade_pair(base_time: datetime, account_a: str, account_b: str) -> list:
    t = base_time + timedelta(hours=3)
    return [
        {
            "trade_id": "WASH_A",
            "account_id": account_a,
            "related_group": "GROUP-X",
            "symbol": "USDJPY",
            "side": "BUY",
            "volume": 3.0,
            "open_time": t,
            "close_time": t + timedelta(minutes=5),
        },
        {
            "trade_id": "WASH_B",
            "account_id": account_b,
            "related_group": "GROUP-X",
            "symbol": "USDJPY",
            "side": "SELL",
            "volume": 3.0,
            "open_time": t + timedelta(seconds=8),
            "close_time": t + timedelta(minutes=5, seconds=8),
        },
    ]


def main():
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 9, 18, 7, 0, 0)

    rows = generate_normal_trades(rng, 400, base_time)
    rows += generate_scalping_burst(rng, base_time, account="ACC-3")
    rows += generate_oversized_position(base_time, account="ACC-7")
    rows += generate_wash_trade_pair(base_time, account_a="ACC-9", account_b="ACC-10")

    df = pd.DataFrame(rows)
    df["hold_seconds"] = (df["close_time"] - df["open_time"]).dt.total_seconds()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

    out_path = DATA_DIR / "sample_trades.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} trades to {out_path}")


if __name__ == "__main__":
    main()
