#!/usr/bin/env python3
"""Build the international-match feature table for the WC Over 2.5 model.

Single chronological pass over results.csv (martj42/international_results):
  * Elo rating per team, eloratings.net style (K by tournament importance,
    goal-margin multiplier, +100 home advantage when not neutral)
  * rolling last-10 goals for/against + total-goals per team
  * rest days, neutral flag, importance

Every feature is computed strictly BEFORE the match it describes.
Writes data/features.csv (matches where both teams have >= 5 prior games).
"""

import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "raw", "results.csv")
OUT = os.path.join(HERE, "features.csv")

ROLL = 10
MIN_PRIOR = 5
MIN_YEAR = 1980          # feature rows before this are Elo warm-up only
HOME_ADV = 100.0
# Elo inactivity decay: fraction of (elo-1500) shed per idle year beyond the
# first. 0 = off (baseline). Adopted only if walk-forward AUC improves.
ELO_DECAY = float(os.environ.get("ELO_DECAY", "0"))

FEAT_COLS = ["h_elo", "a_elo", "elo_diff", "elo_sum",
             "h_gf", "h_ga", "h_tot", "a_gf", "a_ga", "a_tot",
             "h_rest", "a_rest", "neutral", "importance"]


def k_factor(tournament):
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return 60
    if any(x in t for x in ("euro", "copa américa", "copa america",
                            "africa cup", "asian cup", "gold cup")) \
            and "qualification" not in t:
        return 50
    if "qualification" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def margin_mult(diff):
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8


def build(results_path=RESULTS):
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    played = df.dropna(subset=["home_score", "away_score"]).sort_values("date")

    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=ROLL))   # (gf, ga) per team
    last_date = {}
    count = defaultdict(int)
    rows = []

    for r in played.itertuples():
        h, a = r.home_team, r.away_team
        neutral = r.neutral in (True, "TRUE", "True")
        K = k_factor(r.tournament)

        if ELO_DECAY:
            for t in (h, a):
                if t in last_date:
                    idle = (r.date - last_date[t]).days / 365.25 - 1.0
                    if idle > 0:
                        elo[t] += (1500.0 - elo[t]) * min(ELO_DECAY * idle, 1)

        if (count[h] >= MIN_PRIOR and count[a] >= MIN_PRIOR
                and r.date.year >= MIN_YEAR):
            hh, ah = hist[h], hist[a]
            rows.append({
                "date": r.date, "home": h, "away": a,
                "tournament": r.tournament,
                "h_elo": elo[h], "a_elo": elo[a],
                "elo_diff": elo[h] - elo[a] + (0 if neutral else HOME_ADV),
                "elo_sum": elo[h] + elo[a],
                "h_gf": np.mean([g for g, _ in hh]),
                "h_ga": np.mean([g for _, g in hh]),
                "h_tot": np.mean([g + c for g, c in hh]),
                "a_gf": np.mean([g for g, _ in ah]),
                "a_ga": np.mean([g for _, g in ah]),
                "a_tot": np.mean([g + c for g, c in ah]),
                "h_rest": min((r.date - last_date[h]).days, 60),
                "a_rest": min((r.date - last_date[a]).days, 60),
                "neutral": int(neutral), "importance": K,
                "hg": r.home_score, "ag": r.away_score,
                "Over25": int(r.home_score + r.away_score >= 3),
            })

        # ---- update state AFTER the snapshot ----
        exp_h = 1 / (1 + 10 ** (-(elo[h] - elo[a]
                                  + (0 if neutral else HOME_ADV)) / 400))
        res_h = 1.0 if r.home_score > r.away_score else \
            (0.5 if r.home_score == r.away_score else 0.0)
        delta = K * margin_mult(abs(int(r.home_score) - int(r.away_score))) \
            * (res_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
        hist[h].append((r.home_score, r.away_score))
        hist[a].append((r.away_score, r.home_score))
        last_date[h], last_date[a] = r.date, r.date
        count[h] += 1
        count[a] += 1

    return pd.DataFrame(rows), elo, hist, last_date


def main():
    feats, elo, _, _ = build()
    feats.to_csv(OUT, index=False)
    print(f"{len(feats)} feature rows -> {OUT}")
    print(f"Over 2.5 rate: {feats['Over25'].mean():.1%}")
    top = sorted(elo.items(), key=lambda x: -x[1])[:8]
    print("current Elo top 8:",
          ", ".join(f"{t} {e:.0f}" for t, e in top))


if __name__ == "__main__":
    main()
