"""
Combines the rule-based detectors and the statistical anomaly scorer into a
single ranked alert queue - the same two-layer approach described in the
project README: rules first, statistics as a second pass.
"""

from __future__ import annotations

import pandas as pd

from anomaly_model import score_anomalies
from rules_engine import ALERT_COLUMNS, run_rules

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def run_surveillance(df: pd.DataFrame, contamination: float = 0.03, max_hold_seconds: float = 5.0,
                      min_count_for_burst: int = 3, burst_window_minutes: float = 10.0,
                      oversized_multiplier: float = 3.0, wash_time_window_seconds: float = 60.0,
                      wash_volume_tolerance: float = 0.05):
    """
    Returns (alerts_df, population_stats): alerts_df is the ranked alert
    queue, population_stats is the mean of each engineered feature across
    every trade analyzed - the baseline a caller can show alongside any one
    trade's own feature values to explain why the anomaly model flagged it.
    """
    rule_flags = run_rules(
        df, max_hold_seconds=max_hold_seconds, min_count_for_burst=min_count_for_burst,
        burst_window_minutes=burst_window_minutes, oversized_multiplier=oversized_multiplier,
        wash_time_window_seconds=wash_time_window_seconds, wash_volume_tolerance=wash_volume_tolerance,
    )
    scored = score_anomalies(df, contamination=contamination)

    already_flagged = set(rule_flags["trade_id"]) if not rule_flags.empty else set()
    population_mean_score = float(scored["anomaly_score"].mean())
    population_stats = {
        "anomaly_score_mean": round(population_mean_score, 4),
        "hold_seconds_mean": round(float(scored["hold_seconds"].mean()), 2),
        "volume_ratio_mean": round(float(scored["volume_ratio"].mean()), 2),
        "hour_of_day_mean": round(float(scored["hour_of_day"].mean()), 1),
        "account_trade_count_mean": round(float(scored["account_trade_count"].mean()), 1),
    }

    anomaly_only = scored[scored["is_anomaly"] & ~scored["trade_id"].isin(already_flagged)].copy()
    anomaly_rows = []
    for _, row in anomaly_only.iterrows():
        anomaly_rows.append({
            "trade_id": row["trade_id"], "account_id": row["account_id"],
            "rule": "statistical_anomaly", "severity": "medium",
            "detail": "Flagged by the Isolation Forest model as a statistical outlier (no rule-based match).",
            "evidence": {
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "population_mean_score": round(population_mean_score, 4),
                "hold_seconds": round(float(row["hold_seconds"]), 2),
                "volume_ratio": round(float(row["volume_ratio"]), 2),
                "hour_of_day": int(row["hour_of_day"]),
                "account_trade_count": int(row["account_trade_count"]),
            },
        })
    anomaly_flags = pd.DataFrame(anomaly_rows, columns=ALERT_COLUMNS)

    combined = pd.concat([rule_flags, anomaly_flags], ignore_index=True)
    if combined.empty:
        combined["anomaly_score"] = []
        return combined, population_stats

    combined = combined.merge(scored[["trade_id", "anomaly_score"]].drop_duplicates("trade_id"),
                               on="trade_id", how="left")
    combined["severity_rank"] = combined["severity"].map(SEVERITY_RANK).fillna(0)
    combined = combined.sort_values(["severity_rank", "anomaly_score"], ascending=False)
    combined = combined.drop(columns="severity_rank").reset_index(drop=True)
    return combined, population_stats
