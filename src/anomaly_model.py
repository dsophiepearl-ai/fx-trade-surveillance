"""
Statistical anomaly-scoring layer, meant to run alongside (not instead of)
the rule-based detectors in rules_engine.py - it can catch patterns the
fixed rules miss, at the cost of needing a human to review borderline cases.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per trade: holding time, volume relative to that account's own
    average, hour of day, and how many trades that account made overall.
    """
    features = pd.DataFrame(index=df.index)
    avg_volume_by_account = df.groupby("account_id")["volume"].transform("mean")
    features["hold_seconds"] = df["hold_seconds"]
    features["volume_ratio"] = df["volume"] / avg_volume_by_account
    features["hour_of_day"] = pd.to_datetime(df["open_time"]).dt.hour
    features["account_trade_count"] = df.groupby("account_id")["trade_id"].transform("count")
    return features


def score_anomalies(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    """
    Fits an IsolationForest on the trade population and returns the
    original DataFrame with the engineered features plus two extra columns:
    anomaly_score (higher = more anomalous) and is_anomaly (bool). Keeping
    the features on the output lets a caller show *why* a trade scored high
    - which of hold time, relative volume, time of day, or trade frequency
    was unusual - not just the score itself.
    """
    features = build_features(df)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(features)

    enriched = df.copy()
    for col in features.columns:
        enriched[col] = features[col]
    enriched["anomaly_score"] = -model.score_samples(features)
    enriched["is_anomaly"] = model.predict(features) == -1
    return enriched
