#!/usr/bin/env python3
"""wc-scoreline: XGBoost Poisson goal models -> full score matrix.

Two gradient-boosted tree models (count:poisson objective) predict expected
home and away goals from the wc-goals feature set. A Poisson score matrix
then derives OUR probability for every market: W/D/L, the O/U ladder, BTTS,
and exact scores.

Run directly for walk-forward validation across World Cups 2010..2026:
compares the DERIVED Over 2.5 against the old direct classifier (AUC 0.561
benchmark) and scores W/D/L logloss vs an Elo-only baseline.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import poisson
from xgboost import XGBRegressor

from build import FEAT_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
MAXG = 9          # truncate Poisson at 9 goals a side
CUPS = ["2010", "2014", "2018", "2022", "2026"]


def make_models():
    """Home & away goal models. Monotonic: bigger elo_diff can only help
    the home side and hurt the away side."""
    mono_h = ",".join("1" if f == "elo_diff" else "0" for f in FEAT_COLS)
    mono_a = ",".join("-1" if f == "elo_diff" else "0" for f in FEAT_COLS)
    kw = dict(objective="count:poisson", n_estimators=400, learning_rate=0.05,
              max_depth=4, min_child_weight=20, subsample=0.9,
              colsample_bytree=0.8, random_state=42, n_jobs=-1)
    return (XGBRegressor(monotone_constraints=f"({mono_h})", **kw),
            XGBRegressor(monotone_constraints=f"({mono_a})", **kw))


def fit(train):
    mh, ma = make_models()
    mh.fit(train[FEAT_COLS], train["hg"])
    ma.fit(train[FEAT_COLS], train["ag"])
    return mh, ma


def score_matrix(lh, la):
    """P(home=i, away=j) for one match from Poisson means (independent v1)."""
    ph = poisson.pmf(np.arange(MAXG + 1), lh)
    pa = poisson.pmf(np.arange(MAXG + 1), la)
    m = np.outer(ph, pa)
    return m / m.sum()


def derive(m):
    """Every market from one score matrix."""
    i, j = np.indices(m.shape)
    tot = i + j
    out = {"home": m[i > j].sum(), "draw": np.trace(m),
           "away": m[i < j].sum(),
           "btts": m[(i > 0) & (j > 0)].sum()}
    for line in (1.5, 2.5, 3.5, 4.5, 5.5):
        out[f"over{line}"] = m[tot > line].sum()
    return out


def predict_markets(mh, ma, X):
    lh = mh.predict(X[FEAT_COLS])
    la = ma.predict(X[FEAT_COLS])
    rows = [derive(score_matrix(h, a)) for h, a in zip(lh, la)]
    df = pd.DataFrame(rows, index=X.index)
    df["lam_h"], df["lam_a"] = lh, la
    return df


def wdl_logloss(p_home, p_draw, p_away, hg, ag):
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    P = np.clip(np.column_stack([p_home, p_draw, p_away]), 1e-9, 1)
    return -np.log(P[np.arange(len(y)), y]).mean()


def main():
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(os.path.join(HERE, "features.csv"), parse_dates=["date"])
    is_wc = df["tournament"] == "FIFA World Cup"

    print(f"{'WC':6}{'test':>5}{'O2.5 AUC':>10}{'WDL logloss':>13}"
          f"{'elo-only':>10}{'goal MAE':>10}")
    frames = []
    for year in CUPS:
        test = df[is_wc & (df["date"].dt.year == int(year))]
        train = df[df["date"] < test["date"].min()]
        mh, ma = fit(train)
        pred = predict_markets(mh, ma, test)

        # elo-only W/D/L baseline: logistic on elo_diff, draw at base rate
        dr = (train["hg"] == train["ag"]).mean()
        ph_elo = 1 / (1 + 10 ** (-test["elo_diff"] / 400)) * (1 - dr)
        base = wdl_logloss(ph_elo, np.full(len(test), dr),
                           1 - dr - ph_elo, test["hg"], test["ag"])

        auc = roc_auc_score(test["Over25"], pred["over2.5"])
        ll = wdl_logloss(pred["home"], pred["draw"], pred["away"],
                         test["hg"], test["ag"])
        mae = (np.abs(pred["lam_h"] - test["hg"]).mean() +
               np.abs(pred["lam_a"] - test["ag"]).mean()) / 2
        print(f"{year:6}{len(test):>5}{auc:>10.4f}{ll:>13.4f}"
              f"{base:>10.4f}{mae:>10.3f}")
        t = test.copy()
        for c in pred.columns:
            t[c] = pred[c].values
        frames.append(t)

    oos = pd.concat(frames)
    auc = roc_auc_score(oos["Over25"], oos["over2.5"])
    ll = wdl_logloss(oos["home"], oos["draw"], oos["away"],
                     oos["hg"], oos["ag"])
    print("-" * 54)
    print(f"{'TOTAL':6}{len(oos):>5}{auc:>10.4f}{ll:>13.4f}")
    print(f"\nbenchmark: direct Over2.5 classifier walk-forward AUC = 0.561")
    print("calibration (derived P(home win) bucket -> actual):")
    oos["bucket"] = pd.cut(oos["home"], [0, .3, .45, .6, .75, 1])
    for b, g in oos.groupby("bucket", observed=True):
        act = (g["hg"] > g["ag"]).mean()
        print(f"  {str(b):12} n={len(g):3}  pred {g['home'].mean():.1%}"
              f"  actual {act:.1%}")
    oos.to_csv(os.path.join(HERE, "scoreline_oos.csv"), index=False)


if __name__ == "__main__":
    main()
