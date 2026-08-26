#!/usr/bin/env python3
"""Variant B: recency-weighted training (exponential half-life in years).
Same walk-forward as validate.py; adopt only if TOTAL AUC beats baseline."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from build import FEAT_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
CUPS = ["2010", "2014", "2018", "2022", "2026"]
HALF_LIFE = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
FEATURES = sys.argv[2] if len(sys.argv) > 2 else "features.csv"


def main():
    df = pd.read_csv(os.path.join(HERE, FEATURES), parse_dates=["date"])
    is_wc = df["tournament"] == "FIFA World Cup"
    frames = []
    for year in CUPS:
        test = df[is_wc & (df["date"].dt.year == int(year))]
        train = df[df["date"] < test["date"].min()]
        age = (test["date"].min() - train["date"]).dt.days / 365.25
        w = 0.5 ** (age / HALF_LIFE) if HALF_LIFE > 0 else None
        clf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, max_depth=6,
                                   min_samples_leaf=20, random_state=42,
                                   n_jobs=-1),
            method="sigmoid", cv=3)
        clf.fit(train[FEAT_COLS], train["Over25"],
                sample_weight=None if w is None else w.values)
        t = test.copy()
        t["p"] = clf.predict_proba(test[FEAT_COLS])[:, 1]
        frames.append(t)
    oos = pd.concat(frames)
    y, p = oos["Over25"], oos["p"]
    print(f"half_life={HALF_LIFE} features={FEATURES}  n={len(oos)}  "
          f"AUC {roc_auc_score(y, p):.4f}  LogLoss {log_loss(y, p):.4f}  "
          f"Brier {brier_score_loss(y, p):.4f}")


if __name__ == "__main__":
    main()
