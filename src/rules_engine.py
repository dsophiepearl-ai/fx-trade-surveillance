"""
Rule-based surveillance detectors. Each function takes a trades DataFrame
(columns: trade_id, account_id, related_group, symbol, side, volume,
open_time, close_time, hold_seconds) and returns a DataFrame of flagged
trades with a rule name, severity, a human-readable detail, and an
`evidence` dict carrying the raw numbers behind the flag - the actual
measured value, the threshold it was compared against, and (for a
wash-trade pair) the other leg of the trade - so a caller can show the
reasoning directly instead of just the detail sentence.
"""

from __future__ import annotations

import pandas as pd

ALERT_COLUMNS = ["trade_id", "account_id", "rule", "severity", "detail", "evidence"]


def detect_scalping(df: pd.DataFrame, max_hold_seconds: float = 5.0,
                     min_count_for_burst: int = 3, burst_window_minutes: float = 10.0) -> pd.DataFrame:
    """
    Flags trades held for less than max_hold_seconds. If an account has
    several of these close together in time (a "burst"), each trade in the
    burst is flagged at medium severity; an isolated fast trade is flagged
    at low severity (fast, but not necessarily a pattern).
    """
    candidates = df[df["hold_seconds"] <= max_hold_seconds].copy()
    flags = []
    for account, group in candidates.groupby("account_id"):
        group = group.sort_values("open_time")
        span_minutes = (group["open_time"].max() - group["open_time"].min()).total_seconds() / 60
        is_burst = len(group) >= min_count_for_burst and span_minutes <= burst_window_minutes
        for _, row in group.iterrows():
            if is_burst:
                flags.append({
                    "trade_id": row["trade_id"], "account_id": account,
                    "rule": "scalping_burst", "severity": "medium",
                    "detail": f"hold time {row['hold_seconds']:.1f}s, part of a {len(group)}-trade burst",
                    "evidence": {
                        "hold_seconds": round(float(row["hold_seconds"]), 2),
                        "max_hold_seconds": max_hold_seconds,
                        "burst_size": int(len(group)),
                        "min_count_for_burst": min_count_for_burst,
                        "burst_span_minutes": round(span_minutes, 2),
                        "burst_window_minutes": burst_window_minutes,
                    },
                })
            else:
                flags.append({
                    "trade_id": row["trade_id"], "account_id": account,
                    "rule": "fast_trade", "severity": "low",
                    "detail": f"hold time {row['hold_seconds']:.1f}s (isolated, not part of a burst)",
                    "evidence": {
                        "hold_seconds": round(float(row["hold_seconds"]), 2),
                        "max_hold_seconds": max_hold_seconds,
                    },
                })
    return pd.DataFrame(flags, columns=ALERT_COLUMNS)


def detect_oversized_positions(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Flags trades whose volume is more than `multiplier` times that account's own average volume."""
    avg_by_account = df.groupby("account_id")["volume"].mean()
    flags = []
    for _, row in df.iterrows():
        avg = avg_by_account[row["account_id"]]
        if avg > 0 and row["volume"] > multiplier * avg:
            flags.append({
                "trade_id": row["trade_id"], "account_id": row["account_id"],
                "rule": "oversized_position", "severity": "high",
                "detail": f"volume {row['volume']} lots vs. account average {avg:.2f} lots "
                          f"({row['volume'] / avg:.1f}x)",
                "evidence": {
                    "volume": float(row["volume"]),
                    "account_avg_volume": round(float(avg), 2),
                    "multiplier_actual": round(float(row["volume"]) / float(avg), 2),
                    "multiplier_threshold": multiplier,
                },
            })
    return pd.DataFrame(flags, columns=ALERT_COLUMNS)


def detect_wash_trades(df: pd.DataFrame, time_window_seconds: float = 60.0,
                        volume_tolerance: float = 0.05) -> pd.DataFrame:
    """
    Flags pairs of trades between two different accounts in the same
    related_group, on the same symbol, on opposite sides, with matching
    volume, opened within time_window_seconds of each other - the classic
    offsetting / wash-trade signature.
    """
    flags = []
    seen_pairs = set()

    for _, group in df.groupby("related_group"):
        if group["account_id"].nunique() < 2:
            continue
        group = group.sort_values("open_time").reset_index(drop=True)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group.iloc[i], group.iloc[j]
                if a["account_id"] == b["account_id"]:
                    continue
                if a["symbol"] != b["symbol"] or a["side"] == b["side"]:
                    continue
                dt = abs((b["open_time"] - a["open_time"]).total_seconds())
                if dt > time_window_seconds:
                    continue
                volume_diff = abs(a["volume"] - b["volume"])
                if volume_diff > volume_tolerance:
                    continue

                pair_key = tuple(sorted([a["trade_id"], b["trade_id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_id = "|".join(pair_key)

                for row, other in ((a, b), (b, a)):
                    flags.append({
                        "trade_id": row["trade_id"], "account_id": row["account_id"],
                        "rule": "wash_trade_pair", "severity": "high",
                        "detail": f"offsetting {row['symbol']} trade with a related account "
                                  f"within {dt:.0f}s, matching volume",
                        "evidence": {
                            "pair_id": pair_id,
                            "symbol": row["symbol"],
                            "related_group": row["related_group"],
                            "side": row["side"],
                            "volume": float(row["volume"]),
                            "open_time": row["open_time"],
                            "paired_trade_id": other["trade_id"],
                            "paired_account_id": other["account_id"],
                            "paired_side": other["side"],
                            "paired_volume": float(other["volume"]),
                            "paired_open_time": other["open_time"],
                            "volume_diff": round(float(volume_diff), 4),
                            "volume_tolerance": volume_tolerance,
                            "time_diff_seconds": round(dt, 1),
                            "time_window_seconds": time_window_seconds,
                        },
                    })
    return pd.DataFrame(flags, columns=ALERT_COLUMNS)


def run_rules(df: pd.DataFrame, max_hold_seconds: float = 5.0, min_count_for_burst: int = 3,
              burst_window_minutes: float = 10.0, oversized_multiplier: float = 3.0,
              wash_time_window_seconds: float = 60.0, wash_volume_tolerance: float = 0.05) -> pd.DataFrame:
    frames = [
        detect_scalping(df, max_hold_seconds=max_hold_seconds, min_count_for_burst=min_count_for_burst,
                         burst_window_minutes=burst_window_minutes),
        detect_oversized_positions(df, multiplier=oversized_multiplier),
        detect_wash_trades(df, time_window_seconds=wash_time_window_seconds,
                            volume_tolerance=wash_volume_tolerance),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
