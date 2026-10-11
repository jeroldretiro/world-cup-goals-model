#!/usr/bin/env python3
"""Walk-forward validation across World Cups for the Over 2.5 model.

For each WC (2010..2026-so-far): train on every international strictly
before that tournament, predict its matches. No odds benchmark exists free
for WC totals, so the verdict is AUC + calibration vs the naive base rate.
"""

import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

from build import FEAT_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
CUPS = ["2010", "2014", "2018", "2022", "2026"]
GRID = {"max_depth": [4, 6, 8], "min_samples_leaf": [5, 20, 50]}


def main():
    df = pd.read_csv(os.path.join(HERE, "features.csv"),
                     parse_dates=["date"])
    is_wc = df["tournament"] == "FIFA World Cup"

    print(f"{'WC':6}{'train':>7}{'test':>6}{'AUC':>8}{'LogLoss':>9}"
          f"{'Brier':>8}{'pred%':>7}{'act%':>7}")
    frames = []
    for year in CUPS:
        test = df[is_wc & (df["date"].dt.year == int(year))]
        train = df[df["date"] < test["date"].min()]

        gs = GridSearchCV(
            RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1),
            GRID, scoring="roc_auc", cv=3)
        gs.fit(train[FEAT_COLS], train["Over25"])
        clf = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid",
                                     cv=3)
        clf.fit(train[FEAT_COLS], train["Over25"])
        t = test.copy()
        t["p"] = clf.predict_proba(test[FEAT_COLS])[:, 1]
        frames.append(t)
        print(f"{year:6}{len(train):>7}{len(t):>6}"
              f"{roc_auc_score(t['Over25'], t['p']):>8.4f}"
              f"{log_loss(t['Over25'], t['p']):>9.4f}"
              f"{brier_score_loss(t['Over25'], t['p']):>8.4f}"
              f"{t['p'].mean():>7.1%}{t['Over25'].mean():>7.1%}")

    oos = pd.concat(frames)
    y, p = oos["Over25"], oos["p"]
    base = np.full(len(y), y.mean())
    print("-" * 58)
    print(f"{'TOTAL':6}{'':>7}{len(oos):>6}{roc_auc_score(y, p):>8.4f}"
          f"{log_loss(y, p):>9.4f}{brier_score_loss(y, p):>8.4f}"
          f"{p.mean():>7.1%}{y.mean():>7.1%}")
    print(f"{'naive':6}{'':>7}{'':>6}{0.5:>8.4f}{log_loss(y, base):>9.4f}"
          f"{brier_score_loss(y, base):>8.4f}")

    # calibration by prediction bucket
    oos["bucket"] = pd.cut(oos["p"], [0, .35, .45, .55, .65, 1])
    print("\ncalibration (pred bucket -> actual Over rate):")
    for b, g in oos.groupby("bucket", observed=True):
        print(f"  {str(b):12} n={len(g):3}  pred {g['p'].mean():.1%}  "
              f"actual {g['Over25'].mean():.1%}")
    oos.to_csv(os.path.join(HERE, "validate_oos.csv"), index=False)


if __name__ == "__main__":
    main()
