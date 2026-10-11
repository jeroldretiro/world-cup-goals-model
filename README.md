# World Cup Goals Model

Leakage-free feature engineering (Elo ratings, rolling form) plus calibrated machine learning models — a Random Forest for Over/Under 2.5 goals and dual XGBoost Poisson models for full scoreline prediction — trained on ~49K international soccer matches and validated with true walk-forward testing across five World Cups.

Built as the "second opinion" layer for [prediction-market-analytics](https://github.com/larotalon/prediction-market-analytics): a model-only view to set against live prediction-market odds.

> **Note:** Educational/research project. Not prediction advice.

---

## What it does

| Script | What it does |
|--------|---------------|
| **`predict_wc.py`** | Second-opinion P(Over 2.5) for any international matchup — calibrated Random Forest (`CalibratedClassifierCV`, sigmoid) trained on ~37K matches |
| **`predict_score.py`** | Full scoreline board: two XGBoost Poisson goal models (home/away), derives an exact-score matrix, W/D/L, the O/U ladder, and BTTS |
| **`wc_sim.py`** | Exact bracket math (not Monte Carlo) for the remaining World Cup field, using the walk-forward-validated scoreline model |
| **`data/build.py`** | Feature pipeline — Elo ratings (tournament-weighted K-factor, +100 home advantage), rolling 10-game form, rest days. Every feature is computed strictly *before* the match it describes, so there's no leakage. |
| **`data/validate.py`** | Walk-forward validation harness — trains on everything before a given World Cup, tests only on that tournament, repeated across five Cups |

---

## Data

~49K international matches (1872–present) from the public [martj42/international_results](https://github.com/martj42/international_results) dataset.

---

## Quickstart

```bash
python3 predict_wc.py Spain Austria        # second-opinion Over 2.5 probability
python3 predict_score.py Argentina Egypt   # full scoreline board
python3 wc_sim.py                          # bracket math for the remaining WC field
python3 data/validate.py                   # walk-forward validation across 5 World Cups
```

Requires `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`. Refresh the source data before a live run:

```bash
curl -sSL -o data/raw/results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

---

## Validation results (walk-forward, out-of-sample)

Trained only on matches strictly before each World Cup, tested only on that tournament — 358 held-out matches across 2010, 2014, 2018, 2022, and 2026:

| World Cup | Train size | Test size | AUC | Log Loss | Brier | Predicted Over% | Actual Over% |
|---|---|---|---|---|---|---|---|
| 2010 | 21,437 | 64 | 0.548 | 0.681 | 0.244 | 45.5% | 42.2% |
| 2014 | 25,358 | 64 | 0.558 | 0.691 | 0.249 | 49.0% | 57.8% |
| 2018 | 29,066 | 64 | 0.485 | 0.700 | 0.253 | 46.2% | 48.4% |
| 2022 | 33,099 | 64 | 0.608 | 0.685 | 0.246 | 46.1% | 46.9% |
| 2026 | 36,792 | 102 | 0.523 | 0.695 | 0.251 | 49.3% | 53.9% |
| **Total** | | **358** | **0.550** | **0.691** | **0.249** | **47.4%** | **50.3%** |
| naive baseline | | | 0.500 | 0.693 | 0.250 | | |

**Calibration** (predicted-probability bucket → actual Over rate) tracks closely across every bucket, which matters more than AUC alone for a model meant to be *set against* a market price, not used blind:

| Predicted bucket | n | Predicted | Actual |
|---|---|---|---|
| 0.35–0.45 | 129 | 42.1% | 43.4% |
| 0.45–0.55 | 198 | 49.2% | 53.5% |
| 0.55–0.65 | 29 | 57.8% | 55.2% |
| 0.65–1.0 | 2 | 66.7% | 100.0% |

**Honest read:** the model clears a coin flip (0.550 vs. 0.500 AUC) but not by a lot, and it's noisier tournament-to-tournament (0.485–0.608) than the pooled number suggests. That's the actual point of a "second opinion" tool — it's not meant to replace the market, it's meant to flag the matches where the model and the market disagree enough to be worth a second look.

---

## What this demonstrates

- Leakage-free feature engineering — every feature is computed strictly pre-match, verified by construction, not just by intent
- True walk-forward validation methodology — chronological, out-of-sample by tournament, not k-fold or a random split
- Two model families solving the same underlying problem differently: a classifier (Random Forest) for a binary market, and Poisson regression (XGBoost) for full scoreline distributions
- Honest reporting of a modest, real result instead of overselling model performance

---

## Related project

Companion repo: [prediction-market-analytics](https://github.com/larotalon/prediction-market-analytics) — the live market-data tooling (Polymarket implied odds, smart-money tailing, value-finding, trader screening) that this model's output gets set against.

---

## License

MIT — see [LICENSE](LICENSE).
