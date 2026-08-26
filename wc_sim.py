#!/usr/bin/env python3
"""Tournament simulator: exact bracket math for the remaining World Cup teams.

Usage:
  .../AlphaPy/.venv/bin/python wc_sim.py

With four teams left this is exact arithmetic, not Monte Carlo:
  P(champion) = P(win your semi) * sum over other semi's outcomes of
                P(that opponent advances) * P(beat them in the final)
Advance probability per knockout game = P(win reg) + 0.5 * P(reg draw)
(extra time / pens treated as a coin flip -- same assumption the ledger uses).

Set against the Polymarket / Kalshi "World Cup winner" markets: where model
and market disagree by a lot, someone is wrong -- walk-forward says it is
usually (not always) us. Idea borrowed from ensemble bracket sims; inputs are
our own walk-forward-validated scoreline model, not an unvalidated ensemble.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))

from build import FEAT_COLS, MIN_PRIOR, build, k_factor  # noqa: E402
from predict_wc import pending_fixture, resolve  # noqa: E402
from scoreline import derive, fit, score_matrix  # noqa: E402

MODEL_PATH = os.path.join(HERE, "model", "scoreline.joblib")

# bracket: (semi 1, semi 2, final date)
SEMIS = [("France", "Spain"), ("England", "Argentina")]
FINAL_DATE = pd.Timestamp("2026-07-19")


def wdl(mh, ma, elo, hist, last_date, home, away, date, K):
    """W/D/L probabilities for a neutral-site knockout game."""
    hh, ah = hist[home], hist[away]
    row = {
        "h_elo": elo[home], "a_elo": elo[away],
        "elo_diff": elo[home] - elo[away],          # neutral: no home adv
        "elo_sum": elo[home] + elo[away],
        "h_gf": np.mean([g for g, _ in hh]),
        "h_ga": np.mean([g for _, g in hh]),
        "h_tot": np.mean([g + c for g, c in hh]),
        "a_gf": np.mean([g for g, _ in ah]),
        "a_ga": np.mean([g for _, g in ah]),
        "a_tot": np.mean([g + c for g, c in ah]),
        "h_rest": min((date - last_date[home]).days, 60),
        "a_rest": min((date - last_date[away]).days, 60),
        "neutral": 1, "importance": K,
    }
    X = pd.DataFrame([row])[FEAT_COLS]
    lh, la = float(mh.predict(X)[0]), float(ma.predict(X)[0])
    d = derive(score_matrix(lh, la))
    return d["home"], d["draw"], d["away"]


def main():
    feats, elo, hist, last_date = build()
    if not os.path.exists(MODEL_PATH):
        m_h, m_a = fit(feats)
        joblib.dump((m_h, m_a), MODEL_PATH)
    else:
        m_h, m_a = joblib.load(MODEL_PATH)

    teams_ok = [t for t in elo if len(hist[t]) >= MIN_PRIOR]
    K = k_factor("FIFA World Cup")

    # -- semis: advance probabilities (use listed fixture date if present)
    adv = {}          # team -> P(reach final)
    semi_of = {}      # team -> semi index
    for i, (a, b) in enumerate(SEMIS):
        a, b = resolve(a, teams_ok), resolve(b, teams_ok)
        fix = pending_fixture(a, b)
        date = pd.Timestamp(fix["date"]) if fix is not None else FINAL_DATE
        w, d, l = wdl(m_h, m_a, elo, hist, last_date, a, b, date, K)
        adv[a], adv[b] = w + 0.5 * d, l + 0.5 * d
        semi_of[a], semi_of[b] = i, i
        print(f"SEMI  {a} vs {b}:  {a} {w:.1%} / draw {d:.1%} / {b} {l:.1%}"
              f"   ->  advance: {a} {adv[a]:.1%}, {b} {adv[b]:.1%}")

    # -- finals: every cross-bracket pairing
    print()
    win_final = {}    # (team, opp) -> P(team beats opp in final)
    teams = list(adv)
    for t in teams:
        for o in teams:
            if semi_of[o] == semi_of[t]:
                continue
            w, d, l = wdl(m_h, m_a, elo, hist, last_date, t, o,
                          FINAL_DATE, K)
            win_final[(t, o)] = w + 0.5 * d
    for (t, o), p in sorted(win_final.items()):
        if t < o:
            print(f"FINAL {t} vs {o}:  {t} {p:.1%} / {o} {1-p:.1%}"
                  f"  (incl. pens)")

    # -- championship
    print("\nP(CHAMPION)  -- set against PM/Kalshi winner markets:")
    champ = {}
    for t in teams:
        p = adv[t] * sum(adv[o] * win_final[(t, o)]
                         for o in teams if semi_of[o] != semi_of[t])
        champ[t] = p
    for t, p in sorted(champ.items(), key=lambda x: -x[1]):
        print(f"  {t:12s} {p:6.1%}   (reach final {adv[t]:.1%})")
    print(f"  {'(sum)':12s} {sum(champ.values()):6.1%}   sanity: should be ~100%")


if __name__ == "__main__":
    main()
