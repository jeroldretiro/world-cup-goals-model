#!/usr/bin/env python3
"""v1 (flat Poisson) vs v2 (state sim) on the stored OOS World Cup matches.

Leakage control: state multipliers here are fit ONLY on Euro/Copa games from
goal_minutes.csv; the 338 WC validation matches never touch the state fit
(their lam_h/lam_a are already out-of-sample from the original walk-forward).

Verdict metrics: Brier for draw, BTTS, Over 2.5 -- adopt v2 where it wins.
"""
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = 4000
FIT_LEAGUES = {"uefa.euro", "conmebol.america"}


def state_of(diff):
    return max(-2, min(2, diff))


def fit_multipliers():
    games = defaultdict(list)
    with open(os.path.join(HERE, "raw", "goal_minutes.csv")) as f:
        for r in csv.DictReader(f):
            if r["period"] == "ET" or r["league"] not in FIT_LEAGUES:
                continue
            games[r["event_id"]].append(
                (min(int(r["minute"]), 90), r["team"], r["home"]))
    goals = defaultdict(float)
    exp = defaultdict(float)

    def add(m0, m1, dh):
        for b in range(6):
            ov = max(0, min(m1, (b+1)*15) - max(m0, b*15))
            if ov:
                exp[(b, state_of(dh))] += ov
                exp[(b, state_of(-dh))] += ov

    for gs in games.values():
        gs.sort()
        h = a = 0
        prev = 0
        home = gs[0][2]
        for m, team, hm in gs:
            add(prev, m, h - a)
            sh = (team == home)
            goals[(min((m-1)//15, 5), state_of((h-a) if sh else (a-h)))] += 1
            h, a = h + sh, a + (not sh)
            prev = m
        add(prev, 90, h - a)
    base = sum(goals.values()) / sum(exp.values())
    grid = np.ones((6, 5))
    agg_s = {}
    for s in range(-2, 3):
        g = sum(goals[(b, s)] for b in range(6))
        e = sum(exp[(b, s)] for b in range(6))
        agg_s[s] = g / e / base if e > 500 else 1.0
    agg_b = {}
    for b in range(6):
        g = sum(goals[(b, s)] for s in range(-2, 3))
        e = sum(exp[(b, s)] for s in range(-2, 3))
        agg_b[b] = g / e / base if e > 500 else 1.0
    for b in range(6):
        for i, s in enumerate(range(-2, 3)):
            g, e = goals[(b, s)], exp[(b, s)]
            grid[b, i] = (g / e / base) if e > 300 else agg_b[b] * agg_s[s]
    return grid


def simulate(lh, la, grid, rng):
    """vectorized minute-by-minute sim; returns arrays of final (h, a)"""
    h = np.zeros(SIMS, dtype=np.int16)
    a = np.zeros(SIMS, dtype=np.int16)
    ph, pa = lh / 90.0, la / 90.0
    for minute in range(90):
        b = minute // 15
        dh = np.clip(h - a, -2, 2) + 2
        da = np.clip(a - h, -2, 2) + 2
        mh = grid[b, dh]
        ma = grid[b, da]
        r = np.random.random((2, SIMS))
        h += (r[0] < ph * mh).astype(np.int16)
        a += (r[1] < pa * ma).astype(np.int16)
    return h, a


def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def main():
    grid = fit_multipliers()
    print("state grid fit on Euro/Copa only (rows=15min buckets, "
          "cols=diff -2..+2):")
    for b in range(6):
        print("  " + "  ".join(f"{grid[b,i]:.2f}" for i in range(5)))

    rows = list(csv.DictReader(
        open(os.path.join(HERE, "scoreline_oos.csv"))))
    rng = np.random.default_rng(11)
    y_draw, y_btts, y_o25 = [], [], []
    v1 = {"draw": [], "btts": [], "o25": []}
    v2 = {"draw": [], "btts": [], "o25": []}
    for r in rows:
        hg, ag = float(r["hg"]), float(r["ag"])
        y_draw.append(hg == ag)
        y_btts.append(hg > 0 and ag > 0)
        y_o25.append(hg + ag >= 3)
        v1["draw"].append(float(r["draw"]))
        v1["btts"].append(float(r["btts"]))
        v1["o25"].append(float(r["over2.5"]))
        h, a = simulate(float(r["lam_h"]), float(r["lam_a"]), grid, rng)
        v2["draw"].append(np.mean(h == a))
        v2["btts"].append(np.mean((h > 0) & (a > 0)))
        v2["o25"].append(np.mean(h + a >= 3))

    print(f"\n{len(rows)} OOS WC matches, {SIMS} sims each")
    print(f"{'market':8}{'v1 Brier':>10}{'v2 Brier':>10}{'winner':>9}")
    for k, y in (("draw", y_draw), ("btts", y_btts), ("o25", y_o25)):
        b1, b2 = brier(y, v1[k]), brier(y, v2[k])
        print(f"{k:8}{b1:>10.4f}{b2:>10.4f}"
              f"{'v2' if b2 < b1 else 'v1':>9}")
    print(f"\navg predicted draw rate: v1 "
          f"{np.mean(v1['draw']):.1%}  v2 {np.mean(v2['draw']):.1%}  "
          f"actual {np.mean(y_draw):.1%}")


if __name__ == "__main__":
    main()
