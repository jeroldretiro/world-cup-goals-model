#!/usr/bin/env python3
"""OUR board for any international: score matrix + every derived market.

Usage:
  .../AlphaPy/.venv/bin/python predict_score.py Argentina Egypt
  ... predict_score.py Argentina Egypt --retrain   # after refreshing results.csv

Trains two XGBoost Poisson goal models on all internationals (cached in
model/scoreline.joblib), rebuilds both teams' current Elo/form from
data/raw/results.csv, and prints: expected goals, top exact scores, W/D/L,
the O/U ladder, and BTTS.

Model-only view: a forecast from the model's features alone, intended to be
compared with an independent external forecast.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))

from build import FEAT_COLS, HOME_ADV, MIN_PRIOR, build, k_factor  # noqa: E402
from predict_wc import pending_fixture, resolve  # noqa: E402
from scoreline import derive, fit, score_matrix  # noqa: E402

MODEL_PATH = os.path.join(HERE, "model", "scoreline.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    feats, elo, hist, last_date = build()
    if args.retrain or not os.path.exists(MODEL_PATH):
        mh, ma = fit(feats)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump((mh, ma), MODEL_PATH)
        print(f"trained on {len(feats)} internationals -> {MODEL_PATH}")
    else:
        mh, ma = joblib.load(MODEL_PATH)

    teams = [t for t in elo if len(hist[t]) >= MIN_PRIOR]
    home, away = resolve(args.home, teams), resolve(args.away, teams)

    fix = pending_fixture(home, away)
    if fix is not None:
        home, away = fix["home_team"], fix["away_team"]
        date = pd.Timestamp(fix["date"])
        neutral = fix["neutral"] in (True, "TRUE", "True")
        K = k_factor(fix["tournament"])
        where = f"{fix['city']} ({fix['tournament']})"
    else:
        date, neutral, K = pd.Timestamp.now(), True, 60
        where = "no listed fixture -- assumed World Cup, neutral"

    hh, ah = hist[home], hist[away]
    row = {
        "h_elo": elo[home], "a_elo": elo[away],
        "elo_diff": elo[home] - elo[away] + (0 if neutral else HOME_ADV),
        "elo_sum": elo[home] + elo[away],
        "h_gf": np.mean([g for g, _ in hh]),
        "h_ga": np.mean([g for _, g in hh]),
        "h_tot": np.mean([g + c for g, c in hh]),
        "a_gf": np.mean([g for g, _ in ah]),
        "a_ga": np.mean([g for _, g in ah]),
        "a_tot": np.mean([g + c for g, c in ah]),
        "h_rest": min((date - last_date[home]).days, 60),
        "a_rest": min((date - last_date[away]).days, 60),
        "neutral": int(neutral), "importance": K,
    }
    X = pd.DataFrame([row])[FEAT_COLS]
    lh, la = float(mh.predict(X)[0]), float(ma.predict(X)[0])
    m = score_matrix(lh, la)
    d = derive(m)

    print(f"{home} vs {away} -- {date.date()}, {where}")
    print(f"  expected goals: {home} {lh:.2f} - {la:.2f} {away}\n")
    flat = [(i, j, m[i, j]) for i in range(6) for j in range(6)]
    top = sorted(flat, key=lambda x: -x[2])[:6]
    print("  top scores: " + "  ".join(f"{i}-{j} {p:.1%}" for i, j, p in top))
    print(f"\n  {home} win {d['home']:.1%}   draw {d['draw']:.1%}   "
          f"{away} win {d['away']:.1%}")
    print("  O/U ladder: " + "  ".join(
        f"O{ln} {d[f'over{ln}']:.1%}" for ln in (1.5, 2.5, 3.5, 4.5)))
    print(f"  BTTS Yes {d['btts']:.1%}")
    print("\n  model-only view -- compare with an independent external forecast")


if __name__ == "__main__":
    main()
