#!/usr/bin/env python3
"""Second-opinion tool: model's P(Over 2.5) for a World Cup / international.

Usage:
  .../AlphaPy/.venv/bin/python predict_wc.py Spain Austria
  ... predict_wc.py Mexico England            # finds the pending fixture,
                                              # venue/neutral auto-detected
  ... predict_wc.py Brazil Norway --retrain   # after refreshing results.csv

Trains a calibrated RF on ~37k internationals (cached in
model/second_opinion.joblib). Elo + last-10 form come from the full history
in data/raw/results.csv -- refresh that file for latest results:
  curl -sSL -o data/raw/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Model-only forecast: no market data is used, so this is a data-only view to
compare with external forecasts and contextual factors.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))

from build import (FEAT_COLS, HOME_ADV, MIN_PRIOR, RESULTS,  # noqa: E402
                   build, k_factor)

MODEL_PATH = os.path.join(HERE, "model", "second_opinion.joblib")


def train_model(feats):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    gs = GridSearchCV(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        {"max_depth": [4, 6, 8], "min_samples_leaf": [5, 20, 50]},
        scoring="roc_auc", cv=3)
    gs.fit(feats[FEAT_COLS], feats["Over25"])
    clf = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=3)
    clf.fit(feats[FEAT_COLS], feats["Over25"])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"trained on {len(feats)} internationals -> {MODEL_PATH}")
    return clf


def resolve(name, teams):
    hits = [t for t in teams if name.lower() in t.lower()]
    exact = [t for t in hits if t.lower() == name.lower()]
    if exact:
        return exact[0]
    if len(hits) != 1:
        raise SystemExit(f"'{name}' -> {hits or 'no match'}")
    return hits[0]


def pending_fixture(home, away):
    df = pd.read_csv(RESULTS)
    df = df[df["home_score"].isna()]
    m = df[(df["home_team"] == home) & (df["away_team"] == away)]
    if m.empty:   # allow swapped order
        m = df[(df["home_team"] == away) & (df["away_team"] == home)]
    return m.iloc[0] if len(m) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--date", default=None)
    ap.add_argument("--neutral", choices=["yes", "no"], default=None)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    feats, elo, hist, last_date = build()
    clf = (train_model(feats)
           if args.retrain or not os.path.exists(MODEL_PATH)
           else joblib.load(MODEL_PATH))

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
        date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now()
        neutral = args.neutral != "no"      # internationals default neutral
        K = 60
        where = "no listed fixture -- assumed World Cup" + \
            (", neutral" if neutral else f", {home} at home")

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
    p = clf.predict_proba(pd.DataFrame([row])[FEAT_COLS])[0, 1]

    print(f"{home} vs {away} -- {date.date()}, {where}")
    print(f"  Elo: {home} {elo[home]:.0f}  {away} {elo[away]:.0f}  "
          f"(diff {row['elo_diff']:+.0f} incl. venue)")
    print(f"  last-10 goals f/a: {home} {row['h_gf']:.1f}/{row['h_ga']:.1f}"
          f"   {away} {row['a_gf']:.1f}/{row['a_ga']:.1f}")
    print(f"  rest: {home} {row['h_rest']}d, {away} {row['a_rest']}d")
    print(f"\n  MODEL P(Over 2.5) = {p:.1%}   P(Under 2.5) = {1 - p:.1%}")
    print(f"  implied decimal: Over {1 / p:.2f} / Under {1 / (1 - p):.2f}")
    print("\n  model-only forecast -- compare with external "
          "forecasts + contextual factors")


if __name__ == "__main__":
    main()
