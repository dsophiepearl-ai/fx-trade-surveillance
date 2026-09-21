import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rules_engine import detect_oversized_positions, detect_scalping, detect_wash_trades

COLUMNS = ["trade_id", "account_id", "related_group", "symbol", "side", "volume",
           "open_time", "close_time", "hold_seconds"]


def make_df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    return df


def test_isolated_fast_trade_is_low_severity_not_burst():
    rows = [{"trade_id": "T1", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
             "side": "BUY", "volume": 1.0, "open_time": "2026-09-18T07:00:00",
             "close_time": "2026-09-18T07:00:02", "hold_seconds": 2}]
    df = make_df(rows)

    flags = detect_scalping(df, max_hold_seconds=5)

    assert len(flags) == 1
    assert flags.iloc[0]["rule"] == "fast_trade"
    assert flags.iloc[0]["severity"] == "low"


def test_multiple_fast_trades_close_together_are_flagged_as_burst():
    rows = []
    base = pd.Timestamp("2026-09-18T07:00:00")
    for i in range(5):
        t = base + pd.Timedelta(seconds=30 * i)
        rows.append({"trade_id": f"T{i}", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
                     "side": "BUY", "volume": 0.5, "open_time": t, "close_time": t + pd.Timedelta(seconds=2),
                     "hold_seconds": 2})
    df = make_df(rows)

    flags = detect_scalping(df, max_hold_seconds=5, min_count_for_burst=3, burst_window_minutes=10)

    assert len(flags) == 5
    assert (flags["rule"] == "scalping_burst").all()
    assert (flags["severity"] == "medium").all()


def test_normal_hold_time_is_not_flagged():
    rows = [{"trade_id": "T1", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
             "side": "BUY", "volume": 1.0, "open_time": "2026-09-18T07:00:00",
             "close_time": "2026-09-18T09:00:00", "hold_seconds": 7200}]
    df = make_df(rows)

    flags = detect_scalping(df, max_hold_seconds=5)

    assert flags.empty


def test_oversized_position_is_flagged():
    rows = []
    base = pd.Timestamp("2026-09-18T07:00:00")
    for i in range(5):
        rows.append({"trade_id": f"T{i}", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
                     "side": "BUY", "volume": 1.0, "open_time": base, "close_time": base, "hold_seconds": 60})
    rows.append({"trade_id": "BIG", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
                 "side": "BUY", "volume": 10.0, "open_time": base, "close_time": base, "hold_seconds": 60})
    df = make_df(rows)

    flags = detect_oversized_positions(df, multiplier=3.0)

    assert len(flags) == 1
    assert flags.iloc[0]["trade_id"] == "BIG"
    assert flags.iloc[0]["rule"] == "oversized_position"


def test_typical_volume_is_not_flagged_as_oversized():
    rows = [{"trade_id": f"T{i}", "account_id": "A1", "related_group": "A1", "symbol": "EURUSD",
             "side": "BUY", "volume": 1.0 + 0.1 * i, "open_time": "2026-09-18T07:00:00",
             "close_time": "2026-09-18T07:01:00", "hold_seconds": 60} for i in range(5)]
    df = make_df(rows)

    flags = detect_oversized_positions(df, multiplier=3.0)

    assert flags.empty


def test_offsetting_trades_between_related_accounts_are_flagged_as_wash_trade():
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = [
        {"trade_id": "A", "account_id": "ACC-1", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "BUY", "volume": 3.0, "open_time": base, "close_time": base + pd.Timedelta(minutes=5),
         "hold_seconds": 300},
        {"trade_id": "B", "account_id": "ACC-2", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "SELL", "volume": 3.0, "open_time": base + pd.Timedelta(seconds=10),
         "close_time": base + pd.Timedelta(minutes=5, seconds=10), "hold_seconds": 300},
    ]
    df = make_df(rows)

    flags = detect_wash_trades(df, time_window_seconds=60, volume_tolerance=0.05)

    assert len(flags) == 2
    assert set(flags["trade_id"]) == {"A", "B"}
    assert (flags["rule"] == "wash_trade_pair").all()


def test_unrelated_accounts_taking_opposite_sides_are_not_flagged():
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = [
        {"trade_id": "A", "account_id": "ACC-1", "related_group": "ACC-1", "symbol": "USDJPY",
         "side": "BUY", "volume": 3.0, "open_time": base, "close_time": base, "hold_seconds": 300},
        {"trade_id": "B", "account_id": "ACC-2", "related_group": "ACC-2", "symbol": "USDJPY",
         "side": "SELL", "volume": 3.0, "open_time": base + pd.Timedelta(seconds=10),
         "close_time": base, "hold_seconds": 300},
    ]
    df = make_df(rows)

    flags = detect_wash_trades(df, time_window_seconds=60, volume_tolerance=0.05)

    assert flags.empty


def test_offsetting_trades_too_far_apart_in_time_are_not_flagged():
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = [
        {"trade_id": "A", "account_id": "ACC-1", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "BUY", "volume": 3.0, "open_time": base, "close_time": base, "hold_seconds": 300},
        {"trade_id": "B", "account_id": "ACC-2", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "SELL", "volume": 3.0, "open_time": base + pd.Timedelta(hours=2),
         "close_time": base, "hold_seconds": 300},
    ]
    df = make_df(rows)

    flags = detect_wash_trades(df, time_window_seconds=60, volume_tolerance=0.05)

    assert flags.empty
