"""
Streamlit dashboard for the FX Trade Surveillance / Anomaly Detector.

Two ways to get trade data in:
  - Generate sample data (default) - configurable seed, so the app works
    immediately with no upload required
  - Upload your own trades CSV

Run with:  streamlit run src/dashboard.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from generate_sample_trades import (
    ACCOUNTS,
    generate_normal_trades,
    generate_oversized_position,
    generate_scalping_burst,
    generate_wash_trade_pair,
)
from surveillance import run_surveillance

RULE_LABELS = {
    "scalping_burst": "Scalping burst",
    "fast_trade": "Fast trade (isolated)",
    "oversized_position": "Oversized position",
    "wash_trade_pair": "Wash-trade pair",
    "statistical_anomaly": "Statistical anomaly",
}
# Severity colors: red = high, amber = medium, gray = low - reserved status
# meanings, kept distinct from any categorical series color.
SEVERITY_COLORS = {"high": "#dc2626", "medium": "#d97706", "low": "#6b7280"}

st.set_page_config(page_title="FX Trade Surveillance", layout="wide")
st.title("FX Trade Surveillance / Anomaly Detector")
st.caption("Runs three explainable rules plus a statistical anomaly model over a book of trades, "
           "and shows the actual evidence behind every alert.")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Where should the trade data come from?",
                       ["Generate sample data", "Upload my own file"])

    trades_file = None
    if source == "Generate sample data":
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=5, step=1)
        n_normal = st.slider("Number of ordinary trades", 100, 800, 400, 50)
    else:
        trades_file = st.file_uploader(
            "Trades CSV (trade_id, account_id, related_group, symbol, side, volume, open_time, close_time)",
            type="csv",
        )

    st.header("Scalping")
    max_hold_seconds = st.slider("Max hold time to count as \"fast\" (s)", 1.0, 30.0, 5.0, 0.5)
    min_count_for_burst = st.slider("Min. trades to call it a burst", 2, 10, 3, 1)
    burst_window_minutes = st.slider("Burst window (minutes)", 1.0, 30.0, 10.0, 1.0)

    st.header("Oversized positions")
    multiplier = st.slider("Multiple of account's own average volume", 1.5, 10.0, 3.0, 0.5)

    st.header("Wash trades")
    wash_time_window = st.slider("Max time between offsetting legs (s)", 5.0, 300.0, 60.0, 5.0)
    wash_volume_tolerance = st.slider("Volume match tolerance (lots)", 0.0, 0.5, 0.05, 0.01)

    st.header("Statistical model")
    contamination = st.slider("Expected proportion of anomalous trades", 0.01, 0.10, 0.03, 0.01)

    run = st.button("Run surveillance", type="primary")

if not run:
    st.info("Configure your data source and detection thresholds in the sidebar, "
             "then click **Run surveillance**.")
    st.stop()

if source == "Generate sample data":
    rng = random.Random(int(seed))
    base_time = pd.Timestamp("2026-09-18T07:00:00").to_pydatetime()
    rows = generate_normal_trades(rng, int(n_normal), base_time)
    rows += generate_scalping_burst(rng, base_time, account=rng.choice(ACCOUNTS))
    rows += generate_oversized_position(base_time, account=rng.choice(ACCOUNTS))
    accounts_for_wash = rng.sample(ACCOUNTS, 2)
    rows += generate_wash_trade_pair(base_time, account_a=accounts_for_wash[0], account_b=accounts_for_wash[1])
    df = pd.DataFrame(rows)
    df["hold_seconds"] = (df["close_time"] - df["open_time"]).dt.total_seconds()
    df = df.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
else:
    if not trades_file:
        st.warning("Upload a trades CSV to run surveillance.")
        st.stop()
    df = pd.read_csv(trades_file, parse_dates=["open_time", "close_time"])
    df["hold_seconds"] = (df["close_time"] - df["open_time"]).dt.total_seconds()

alerts, population_stats = run_surveillance(
    df,
    contamination=contamination,
    max_hold_seconds=max_hold_seconds,
    min_count_for_burst=min_count_for_burst,
    burst_window_minutes=burst_window_minutes,
    oversized_multiplier=multiplier,
    wash_time_window_seconds=wash_time_window,
    wash_volume_tolerance=wash_volume_tolerance,
)

REQUIRED_ALERT_COLUMNS = ["trade_id", "account_id", "rule", "severity", "detail", "evidence"]
missing_cols = [c for c in REQUIRED_ALERT_COLUMNS if c not in alerts.columns]
if missing_cols:
    st.error(f"The surveillance engine's output is missing expected columns: {missing_cols}. "
             "This usually means rules_engine.py, surveillance.py, and dashboard.py are out of "
             "sync - make sure all three are the latest version, then reboot the app.")
    st.stop()

severity_counts = alerts["severity"].value_counts().to_dict() if not alerts.empty else {}

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trades analyzed", len(df))
col2.metric("Total alerts", len(alerts))
col3.metric("High severity", severity_counts.get("high", 0))
col4.metric("Medium severity", severity_counts.get("medium", 0))
col5.metric("Low severity", severity_counts.get("low", 0))

if not alerts.empty:
    chart_df = alerts.copy()
    chart_df["Rule"] = chart_df["rule"].map(RULE_LABELS).fillna(chart_df["rule"])
    chart_counts = chart_df.groupby(["Rule", "severity"]).size().reset_index(name="Alerts")
    fig = px.bar(chart_counts, x="Rule", y="Alerts", color="severity",
                 color_discrete_map=SEVERITY_COLORS, text="Alerts",
                 category_orders={"severity": ["high", "medium", "low"]})
    fig.update_layout(xaxis_title="", yaxis_title="Number of alerts", legend_title="Severity",
                       margin=dict(t=10, b=10))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")


def _fmt(value, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return f"{value}{unit}"


def render_comparison(label: str, actual_val, actual_unit: str, threshold_val, threshold_unit: str,
                       comparison: str, exceeded: bool = True):
    """One line of reasoning: the measured value, the threshold, and the verdict."""
    icon = "🔴" if exceeded else "🟢"
    c1, c2 = st.columns([1.3, 2])
    c1.markdown(f"**{label}**\n\n{_fmt(actual_val, actual_unit)}")
    c2.markdown(f"{icon} {comparison}: threshold is {_fmt(threshold_val, threshold_unit)}")


def render_scalping_card(row):
    ev = row.get("evidence") or {}
    rule = row.get("rule")
    if rule == "scalping_burst":
        headline = f"Held {_fmt(ev.get('hold_seconds'), 's')} — part of a {ev.get('burst_size')}-trade burst"
    else:
        headline = f"Held {_fmt(ev.get('hold_seconds'), 's')} — isolated, not a repeated pattern"
    with st.expander(f"🚩 {row.get('trade_id')} — {row.get('account_id')} — {headline}", expanded=False):
        render_comparison("Hold time", ev.get("hold_seconds"), "s", ev.get("max_hold_seconds"), "s",
                           "below the \"fast trade\" threshold")
        if rule == "scalping_burst":
            st.divider()
            render_comparison("Burst size", ev.get("burst_size"), " trades", ev.get("min_count_for_burst"),
                               f" trades in {ev.get('burst_window_minutes')} min", "at or above the burst threshold")
            st.caption(
                f"Reasoning: this account made {ev.get('burst_size')} trades, each held under "
                f"{ev.get('max_hold_seconds')}s, all within a {ev.get('burst_span_minutes')}-minute span "
                f"(the burst window allowed {ev.get('burst_window_minutes')} minutes). A single fast trade "
                "can be a legitimate scalping strategy; a repeated burst is what actually raises the "
                "priority, since it's the pattern - not any one trade - that looks deliberate."
            )
        else:
            st.caption(
                "Reasoning: this trade is fast enough to clear the same-hold-time threshold as a scalping "
                "burst, but it's the only fast trade from this account in the window, so it's kept at low "
                "severity rather than escalated — a single quick trade isn't unusual on its own."
            )


def render_oversized_card(row):
    ev = row.get("evidence") or {}
    headline = f"{_fmt(ev.get('volume'), ' lots')} vs. account average {_fmt(ev.get('account_avg_volume'), ' lots')}"
    with st.expander(f"🚩 {row.get('trade_id')} — {row.get('account_id')} — {headline}", expanded=False):
        render_comparison("This trade's volume", ev.get("volume"), " lots",
                           ev.get("account_avg_volume"), " lots", "vs. this account's own average")
        st.divider()
        render_comparison("Size multiple", ev.get("multiplier_actual"), "x", ev.get("multiplier_threshold"), "x",
                           "above the oversized-position threshold")
        st.caption(
            f"Reasoning: the threshold is relative to each account's own history, not a fixed lot size — "
            f"this account normally trades around {ev.get('account_avg_volume')} lots, so a "
            f"{ev.get('volume')}-lot trade ({ev.get('multiplier_actual')}x its own average) stands out "
            "as a real change in behavior, which is a more reliable signal than any single fixed cutoff."
        )


def render_wash_trade_card(row_a, row_b):
    ev = row_a.get("evidence") or {}
    with st.expander(
        f"🚩 {row_a.get('trade_id')} ↔ {ev.get('paired_trade_id')} — offsetting {ev.get('symbol')} pair "
        f"({ev.get('time_diff_seconds')}s apart)", expanded=False
    ):
        c1, c2 = st.columns(2)
        c1.markdown(
            f"**{row_a.get('account_id')}**\n\n"
            f"Trade: {row_a.get('trade_id')}\n\n"
            f"Side: {ev.get('side')}\n\n"
            f"Volume: {_fmt(ev.get('volume'), ' lots')}\n\n"
            f"Opened: {_fmt(ev.get('open_time'))}"
        )
        c2.markdown(
            f"**{ev.get('paired_account_id')}**\n\n"
            f"Trade: {ev.get('paired_trade_id')}\n\n"
            f"Side: {ev.get('paired_side')}\n\n"
            f"Volume: {_fmt(ev.get('paired_volume'), ' lots')}\n\n"
            f"Opened: {_fmt(ev.get('paired_open_time'))}"
        )
        st.divider()
        render_comparison("Time between the two legs", ev.get("time_diff_seconds"), "s",
                           ev.get("time_window_seconds"), "s", "within the wash-trade time window")
        render_comparison("Volume difference between legs", ev.get("volume_diff"), " lots",
                           ev.get("volume_tolerance"), " lots", "within the matching-volume tolerance")
        st.caption(
            f"Reasoning: both accounts belong to the same related group (\"{ev.get('related_group')}\"), "
            f"took opposite sides of the same {ev.get('symbol')} trade, in matching size, "
            f"{ev.get('time_diff_seconds')} seconds apart. No single one of those facts proves collusion — "
            "opposite sides on a liquid pair happen constantly — but the combination (related accounts, "
            "same size, opposite direction, seconds apart) is the specific signature of two people "
            "arranging trades to offset each other rather than trading independently."
        )


def render_anomaly_card(row, population_stats):
    ev = row.get("evidence") or {}
    headline = f"anomaly score {_fmt(ev.get('anomaly_score'))} vs. population average {_fmt(ev.get('population_mean_score'))}"
    with st.expander(f"🚩 {row.get('trade_id')} — {row.get('account_id')} — {headline}", expanded=False):
        st.markdown(
            "No single rule matched this trade. The Isolation Forest model instead compares its "
            "combination of features against the whole population and flags it as atypical:"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Hold time**\n\n{_fmt(ev.get('hold_seconds'), 's')}\n\n"
                    f"population avg: {_fmt(population_stats.get('hold_seconds_mean'), 's')}")
        c2.markdown(f"**Volume vs. own avg**\n\n{_fmt(ev.get('volume_ratio'), 'x')}\n\n"
                    f"population avg: {_fmt(population_stats.get('volume_ratio_mean'), 'x')}")
        c3.markdown(f"**Hour of day**\n\n{_fmt(ev.get('hour_of_day'))}:00\n\n"
                    f"population avg: {_fmt(population_stats.get('hour_of_day_mean'))}:00")
        c4.markdown(f"**Account activity**\n\n{_fmt(ev.get('account_trade_count'))} trades\n\n"
                    f"population avg: {_fmt(population_stats.get('account_trade_count_mean'))} trades")
        st.caption(
            "Reasoning: none of these four numbers has to be extreme on its own for the model to flag "
            "the trade — it's trained on all four together, so a trade that's only mildly unusual on "
            "several dimensions at once can score higher than one that's extreme on just one. That's "
            "exactly the kind of pattern a fixed rule would miss, and exactly why this runs as a second "
            "layer alongside the explainable rules rather than replacing them."
        )


tab_all, tab_scalp, tab_oversized, tab_wash, tab_anomaly = st.tabs(
    ["All alerts", "Scalping", "Oversized positions", "Wash trades", "Statistical anomalies"]
)

with tab_all:
    st.caption("Full ranked alert queue, one row per flagged trade, for reference or export.")
    st.dataframe(alerts.drop(columns="evidence"), width="stretch")
    st.download_button(
        "Download the alert queue as CSV",
        data=alerts.drop(columns="evidence").to_csv(index=False),
        file_name="alerts.csv",
        mime="text/csv",
    )

with tab_scalp:
    st.caption(
        "A burst means several fast trades from the same account clustered close together in time — "
        "the pattern most consistent with deliberate, repeated scalping rather than one quick trade. "
        "An isolated fast trade is shown too, at lower priority, since speed alone isn't suspicious."
    )
    scalp_alerts = alerts[alerts["rule"].isin(["scalping_burst", "fast_trade"])]
    if scalp_alerts.empty:
        st.success("No fast trades detected.")
    else:
        for _, row in scalp_alerts.iterrows():
            render_scalping_card(row)

with tab_oversized:
    st.caption(
        "The threshold here is relative, not absolute — a trade is flagged against that specific "
        "account's own average volume, so a large account trading large size isn't penalized for being large."
    )
    oversized_alerts = alerts[alerts["rule"] == "oversized_position"]
    if oversized_alerts.empty:
        st.success("No oversized positions detected.")
    else:
        for _, row in oversized_alerts.iterrows():
            render_oversized_card(row)

with tab_wash:
    st.caption(
        "Each card below is one pair of offsetting trades between related accounts — the two legs are "
        "shown side by side so you can see exactly why the pair was matched, the same way an analyst "
        "would review it: same instrument, opposite sides, matching size, seconds apart."
    )
    wash_alerts = alerts[alerts["rule"] == "wash_trade_pair"]
    if wash_alerts.empty:
        st.success("No wash-trade pairs detected.")
    else:
        seen_pairs = set()
        by_trade_id = {row["trade_id"]: row for _, row in wash_alerts.iterrows()}
        for _, row in wash_alerts.iterrows():
            pair_id = (row.get("evidence") or {}).get("pair_id")
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            paired_id = (row.get("evidence") or {}).get("paired_trade_id")
            paired_row = by_trade_id.get(paired_id, row)
            render_wash_trade_card(row, paired_row)

with tab_anomaly:
    st.caption(
        "These trades didn't trip any fixed rule, but the Isolation Forest model scored their overall "
        "feature profile as atypical against the rest of the population — worth a second look, with "
        "lower confidence than a rule-based flag, which is why it's kept at medium severity."
    )
    anomaly_alerts = alerts[alerts["rule"] == "statistical_anomaly"]
    if anomaly_alerts.empty:
        st.success("No statistical anomalies detected beyond what the rules already caught.")
    else:
        for _, row in anomaly_alerts.iterrows():
            render_anomaly_card(row, population_stats)
