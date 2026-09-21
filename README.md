# FX Trade Surveillance / Anomaly Detector

I built this to demonstrate two responsibilities that show up directly in forex Risk Analyst and Trading Operations postings: "investigate trading alerts and analyse flow," and "trade monitoring & market surveillance." It analyzes a book of trades across multiple accounts and produces a ranked queue of activity worth a closer look, the way a real surveillance system would.

**Live demo:** _add your link here once deployed — see "Seeing it run" below_

It runs as an interactive dashboard: generate a fresh, randomized book of trades (or upload your own), adjust every detection threshold, and see the results as live metrics and a chart. What matters most, though, is how each alert is shown: every flagged trade is its own expandable card with the actual evidence behind it — the measured value, the threshold it crossed, and (for a wash-trade pair) both legs of the trade side by side — the same reasoning an analyst would walk through before deciding whether an alert is worth escalating.

## The problem this solves

Not every unusual trade is a problem, and not every problem looks unusual on its own — which is why a surveillance system needs more than one lens. Some abusive patterns are obvious once you know what to look for (a position far bigger than an account normally takes, or two related accounts trading against each other to fake activity). Others don't fit a fixed rule and only show up as a statistical outlier once you look at the whole population of trades together. I built this with both approaches, because relying on just one leaves gaps a real desk can't afford.

## How it works

I split the detection into two layers that run together:

**Rule-based, first.** Three specific, explainable rules run against every trade:
- A **scalping burst** — several trades from the same account held for only a few seconds, clustered close together in time
- An **oversized position** — a trade far larger than that account's own historical average, so the threshold adapts to each account instead of using one fixed lot size for everyone
- A **wash-trade pair** — opposite-side trades on the same symbol, matching volume, between two accounts flagged as related, opened within seconds of each other

Each detector returns the raw numbers behind its decision, not just a flag — the actual hold time next to the threshold, the actual volume multiple next to the multiplier, both legs of a suspected wash trade next to each other — so the dashboard can show exactly why a trade was flagged, the same way I'd want to see it if I were the one reviewing the queue.

**Statistical, second.** I layered an Isolation Forest on top, scoring every trade on holding time, volume relative to that account's own average, time of day, and trading frequency. This catches outliers that don't match any fixed rule but still look unusual against the broader population. Its reasoning is less direct than a fixed rule, so the dashboard shows each flagged trade's own feature values next to the population average for context — worth a second look, but at lower confidence, which is why it only supplements the rule-based flags instead of replacing them.

Both layers feed into one ranked alert queue, sorted by severity and then by how anomalous the statistical model found the trade.

## What it detects

| Pattern | How it's identified | What the flag card shows |
|---|---|---|
| Scalping burst | Several trades held under a configurable threshold, clustered close together in time | Hold time vs. threshold, burst size vs. minimum, and the actual time span of the burst |
| Oversized position | A trade well above that account's own average volume | The trade's volume vs. that account's average, and the multiple vs. the configured threshold |
| Wash-trade pair | Opposite-side trades, same symbol, matching volume, between related accounts within a short window | Both legs side by side — account, side, volume, timing — plus the time gap and volume difference vs. tolerance |
| Statistical anomaly | Any trade the Isolation Forest scores as an outlier that the rules above didn't already catch | The trade's hold time, relative volume, hour of day, and account activity, each next to the population average |

## Skills this project demonstrates

| Skill | Where it shows up |
|---|---|
| Python (pandas, numpy for feature building and analysis) | `src/rules_engine.py`, `src/anomaly_model.py` |
| Rule-based pattern detection, with structured evidence per flag | `src/rules_engine.py` |
| Machine learning / anomaly detection (Isolation Forest, scikit-learn) | `src/anomaly_model.py` |
| Feature engineering | `build_features()` in `src/anomaly_model.py` |
| Combining multiple signals into one ranked, explainable output | `src/surveillance.py` |
| Interactive dashboarding & data visualization | `src/dashboard.py` (Streamlit, Plotly) |
| Designing reproducible test data with deliberate, known-answer patterns | `src/generate_sample_trades.py` |
| Unit testing | `tests/` — 13 tests, all passing |
| Command-line tooling (argparse) | `main.py` |

## Tech stack

Python 3, pandas, numpy, scikit-learn, Streamlit, Plotly

## Project structure

```
fx-trade-surveillance/
  main.py                       # command-line entry point
  src/
    generate_sample_trades.py   # builds sample data, with 3 patterns deliberately injected
    rules_engine.py              # the three rule-based detectors
    anomaly_model.py             # Isolation Forest feature-building and scoring
    surveillance.py               # combines both layers into one ranked, evidence-carrying alert queue
    dashboard.py                  # Streamlit UI
  data/
    sample_trades.csv            # generated
  tests/
    test_rules_engine.py
    test_surveillance.py
  requirements.txt
```

## Seeing it run

Real surveillance data isn't something I have access to for a portfolio project, so I wrote a generator that builds a realistic book of ordinary trades across ten accounts, then plants three deliberate patterns inside it — a scalping burst, an oversized position, and a wash-trade pair between two related accounts. That gives both the tool and the test suite a known answer to check against, which is how I confirmed the detectors actually work before trusting the output.

**Try it now — no installation needed:** once the app above is deployed, click the live demo link at the top. In the sidebar, pick "Generate sample data" (or upload your own trades CSV), adjust the detection thresholds, and click **Run surveillance** — the metrics and chart update immediately, and each tab (Scalping, Oversized positions, Wash trades, Statistical anomalies) gives you an expandable card per alert with the underlying evidence, not just a label.

If you'd like to run it yourself instead:

```bash
git clone https://github.com/dsophiepearl-ai/fx-trade-surveillance.git
cd fx-trade-surveillance
pip install -r requirements.txt
streamlit run src/dashboard.py
```

There's also a plain command-line version for automation or scripting, which runs the same two-layer detection without the UI:

```bash
python src/generate_sample_trades.py
python main.py --contamination 0.03
```

This writes `alerts.csv` — every flagged trade, its rule, severity, and the anomaly score behind it, ranked with the highest-priority alerts first.

## Deploying your own live version

Same process as any Streamlit app:

1. Push this repo to your own GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repository, branch `main`, and file path `src/dashboard.py`
4. Click **Deploy**, then add the resulting link to the "Live demo" line at the top of this README

## Testing it

```bash
pytest
```

All 13 tests pass. `test_rules_engine.py` checks each detector's classification logic — for example, that two offsetting trades between related accounts get flagged as a wash-trade pair, while the same two trades between unrelated accounts don't. `test_surveillance.py` checks that the combined output carries the structured evidence (raw values, thresholds, the paired trade for a wash-trade flag) that the dashboard depends on to show its reasoning, not just its own labels and text.
