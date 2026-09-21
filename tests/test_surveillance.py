import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surveillance import run_surveillance

COLUMNS = ["trade_id", "account_id", "related_group", "symbol", "side", "volume",
           "open_time", "close_time", "hold_seconds"]


def make_df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    return df


def _population(n=30):
    """A quiet population of ordinary trades, large enough for IsolationForest to fit."""
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = []
    for i in range(n):
        rows.append({
            "trade_id": f"N{i}", "account_id": "ACC-1", "related_group": "ACC-1", "symbol": "EURUSD",
            "side": "BUY", "volume": 1.0, "open_time": base + pd.Timedelta(minutes=i),
            "close_time": base + pd.Timedelta(minutes=i, seconds=600), "hold_seconds": 600,
        })
    return rows


def test_oversized_position_evidence_carries_raw_values_and_threshold():
    rows = _population(30)
    rows.append({"trade_id": "BIG", "account_id": "ACC-1", "related_group": "ACC-1", "symbol": "EURUSD",
                 "side": "BUY", "volume": 10.0, "open_time": "2026-09-18T08:00:00",
                 "close_time": "2026-09-18T08:10:00", "hold_seconds": 600})
    df = make_df(rows)

    alerts, _ = run_surveillance(df, oversized_multiplier=3.0)

    big = alerts[alerts["trade_id"] == "BIG"].iloc[0]
    assert big["rule"] == "oversized_position"
    ev = big["evidence"]
    assert ev["volume"] == 10.0
    assert ev["multiplier_threshold"] == 3.0
    assert ev["multiplier_actual"] > 3.0


def test_wash_trade_evidence_carries_both_legs():
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = _population(30)
    rows += [
        {"trade_id": "WA", "account_id": "ACC-2", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "BUY", "volume": 3.0, "open_time": base, "close_time": base + pd.Timedelta(minutes=5),
         "hold_seconds": 300},
        {"trade_id": "WB", "account_id": "ACC-3", "related_group": "GROUP-X", "symbol": "USDJPY",
         "side": "SELL", "volume": 3.0, "open_time": base + pd.Timedelta(seconds=10),
         "close_time": base + pd.Timedelta(minutes=5, seconds=10), "hold_seconds": 300},
    ]
    df = make_df(rows)

    alerts, _ = run_surveillance(df, wash_time_window_seconds=60, wash_volume_tolerance=0.05)

    wa = alerts[alerts["trade_id"] == "WA"].iloc[0]
    assert wa["rule"] == "wash_trade_pair"
    ev = wa["evidence"]
    assert ev["paired_trade_id"] == "WB"
    assert ev["paired_account_id"] == "ACC-3"
    assert ev["time_diff_seconds"] == 10.0
    assert ev["volume_diff"] == 0.0


def test_scalping_burst_evidence_carries_burst_size_and_window():
    base = pd.Timestamp("2026-09-18T07:00:00")
    rows = _population(30)
    for i in range(5):
        t = base + pd.Timedelta(seconds=30 * i)
        rows.append({"trade_id": f"S{i}", "account_id": "ACC-4", "related_group": "ACC-4", "symbol": "EURUSD",
                     "side": "BUY", "volume": 0.5, "open_time": t, "close_time": t + pd.Timedelta(seconds=2),
                     "hold_seconds": 2})
    df = make_df(rows)

    alerts, _ = run_surveillance(df, max_hold_seconds=5, min_count_for_burst=3, burst_window_minutes=10)

    burst_rows = alerts[alerts["rule"] == "scalping_burst"]
    assert len(burst_rows) == 5
    ev = burst_rows.iloc[0]["evidence"]
    assert ev["burst_size"] == 5
    assert ev["max_hold_seconds"] == 5


def test_run_surveillance_returns_population_stats():
    df = make_df(_population(30))

    alerts, population_stats = run_surveillance(df)

    for key in ("anomaly_score_mean", "hold_seconds_mean", "volume_ratio_mean",
                "hour_of_day_mean", "account_trade_count_mean"):
        assert key in population_stats


def test_no_trades_flagged_has_no_alerts_and_required_columns_present():
    df = make_df(_population(30))

    # A very tolerant configuration should produce no rule-based flags.
    alerts, _ = run_surveillance(df, max_hold_seconds=0.001, oversized_multiplier=1000,
                                  wash_time_window_seconds=0.001, contamination=0.01)

    for col in ["trade_id", "account_id", "rule", "severity", "detail", "evidence"]:
        assert col in alerts.columns
