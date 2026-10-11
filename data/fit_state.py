#!/usr/bin/env python3
"""Fit state multipliers for the v2 match simulator from goal_minutes.csv.

Reconstructs each game's running score from its goal sequence, accumulates
exposure minutes per (time bucket x state), and prints scoring-rate
multipliers relative to the all-game average. 0-0 tournament games absent
from the harvest are added from results.csv as pure level-state exposure.

States (from the perspective of the scoring/attacking team):
  level / trailing_1 / leading_1 / trailing_2+ / leading_2+
Time buckets: 0-15, 16-30, 31-45, 46-60, 61-75, 76-90 (reg time only).
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GM = os.path.join(HERE, "raw", "goal_minutes.csv")
RESULTS = os.path.join(HERE, "raw", "results.csv")

TOURNEYS = {"FIFA World Cup", "UEFA Euro", "Copa América"}
YEARS = {"2014", "2018", "2021", "2022", "2024", "2026"}


def state_of(diff):
    if diff == 0:
        return "level"
    if diff == -1:
        return "trailing_1"
    if diff == 1:
        return "leading_1"
    return "trailing_2+" if diff < -1 else "leading_2+"


def bucket(minute):
    return min((minute - 1) // 15, 5)


def main():
    # goals per game, ordered
    games = defaultdict(list)
    with open(GM) as f:
        for r in csv.DictReader(f):
            if r["period"] == "ET":
                continue
            m = min(int(r["minute"]), 90)
            games[r["event_id"]].append(
                (m, r["team"], r["home"], r["away"]))

    goals_st = defaultdict(float)      # (bucket, state) -> goals
    exp_st = defaultdict(float)        # (bucket, state) -> team-minutes
    n_games = 0

    def add_exposure(m0, m1, diff_home):
        """exposure for both teams between minute m0 and m1 at score diff"""
        for b in range(6):
            lo, hi = b * 15, (b + 1) * 15
            ov = max(0, min(m1, hi) - max(m0, lo))
            if ov:
                exp_st[(b, state_of(diff_home))] += ov      # home persp.
                exp_st[(b, state_of(-diff_home))] += ov     # away persp.

    for eid, gs in games.items():
        gs.sort()
        n_games += 1
        h = a = 0
        prev = 0
        home = gs[0][2]
        for m, team, hm, aw in gs:
            add_exposure(prev, m, h - a)
            scorer_home = (team == home)
            goals_st[(bucket(m),
                      state_of((h - a) if scorer_home else (a - h)))] += 1
            if scorer_home:
                h += 1
            else:
                a += 1
            prev = m
        add_exposure(prev, 90, h - a)

    # add 0-0 games from results.csv for the same tournament windows
    zz = 0
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            if (r["tournament"] in TOURNEYS and r["date"][:4] in YEARS
                    and r["home_score"] == "0" and r["away_score"] == "0"):
                zz += 1
                add_exposure(0, 90, 0)
    n_games += zz

    total_goals = sum(goals_st.values())
    total_exp = sum(exp_st.values())
    base = total_goals / total_exp          # goals per team-minute overall
    print(f"{n_games} games ({zz} scoreless added), {total_goals:.0f} reg "
          f"goals, base rate {base*90:.2f} goals/team/90\n")

    print(f"{'':12}" + "".join(f"{b*15}-{(b+1)*15:<7}" for b in range(6)))
    states = ["level", "trailing_1", "leading_1", "trailing_2+", "leading_2+"]
    mults = {}
    for s in states:
        row = ""
        for b in range(6):
            g, e = goals_st[(b, s)], exp_st[(b, s)]
            m = (g / e / base) if e > 200 else float("nan")
            mults[(b, s)] = m
            row += f"{m:>7.2f}    " if m == m else "      -    "
        print(f"{s:12}" + row)

    # compact aggregates for the simulator
    print("\nAGGREGATES (exposure-weighted, for v2):")
    for s in states:
        g = sum(goals_st[(b, s)] for b in range(6))
        e = sum(exp_st[(b, s)] for b in range(6))
        if e > 500:
            print(f"  {s:12} x{g/e/base:.3f}   ({g:.0f} goals / {e/90:.0f} team-games)")
    for b in range(6):
        g = sum(goals_st[(b, s)] for s in states)
        e = sum(exp_st[(b, s)] for s in states)
        print(f"  minutes {b*15:>2}-{(b+1)*15:<3} x{g/e/base:.3f}")


if __name__ == "__main__":
    main()
