"""Unit tests for the Elo and rolling-form feature logic in data/build.py."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import build  # noqa: E402


def make_results(tmp_path, matches):
    """matches: (date, home, away, hg, ag, tournament, neutral)"""
    df = pd.DataFrame(matches, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral"])
    df["city"] = df["country"] = "x"
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return str(path)


def schedule(n, hg=1, ag=0, tournament="Friendly"):
    """n matches between A and B, 10 days apart, starting 2000-01-01."""
    d = pd.Timestamp("2000-01-01")
    return [((d + pd.Timedelta(days=10 * i)).strftime("%Y-%m-%d"),
             "A", "B", hg, ag, tournament, "FALSE") for i in range(n)]


def test_k_factor_by_tournament_importance():
    assert build.k_factor("FIFA World Cup") == 60
    assert build.k_factor("FIFA World Cup qualification") == 40
    assert build.k_factor("UEFA Euro") == 50
    assert build.k_factor("Friendly") == 20
    assert build.k_factor("Some Cup") == 30


def test_margin_multiplier_grows_with_goal_difference():
    assert build.margin_mult(1) == 1.0
    assert build.margin_mult(2) == 1.5
    assert build.margin_mult(3) > build.margin_mult(2)


def test_win_raises_winner_and_lowers_loser(tmp_path):
    _, elo, _, _ = build.build(make_results(tmp_path, schedule(1)))
    assert elo["A"] > 1500 > elo["B"]


def test_elo_is_zero_sum(tmp_path):
    _, elo, _, _ = build.build(make_results(tmp_path, schedule(8)))
    assert elo["A"] + elo["B"] == pytest.approx(3000.0)


def test_draw_between_equal_neutral_teams_leaves_elo_unchanged(tmp_path):
    m = [("2000-01-01", "A", "B", 1, 1, "Friendly", "TRUE")]
    _, elo, _, _ = build.build(make_results(tmp_path, m))
    assert elo["A"] == pytest.approx(1500.0)
    assert elo["B"] == pytest.approx(1500.0)


def test_no_feature_rows_until_min_prior_games(tmp_path):
    feats, _, _, _ = build.build(make_results(tmp_path, schedule(5)))
    assert feats.empty  # 6th match is the first with 5 prior games each
    feats, _, _, _ = build.build(make_results(tmp_path, schedule(6)))
    assert len(feats) == 1


def test_features_use_only_past_matches(tmp_path):
    """Changing a match's own score must not change that match's features."""
    base = schedule(6)
    alt = base[:5] + [base[5][:3] + (9, 9) + base[5][5:]]
    f1, _, _, _ = build.build(make_results(tmp_path, base))
    f2, _, _, _ = build.build(make_results(tmp_path, alt))
    cols = build.FEAT_COLS
    pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_rolling_form_averages_prior_goals(tmp_path):
    feats, _, _, _ = build.build(make_results(tmp_path, schedule(6)))
    row = feats.iloc[0]
    assert row["h_gf"] == pytest.approx(1.0)   # A scored 1 in each of 5
    assert row["h_ga"] == pytest.approx(0.0)
    assert row["a_gf"] == pytest.approx(0.0)
    assert row["h_tot"] == pytest.approx(1.0)


def test_home_advantage_only_when_not_neutral(tmp_path):
    feats, _, _, _ = build.build(make_results(tmp_path, schedule(6)))
    row = feats.iloc[0]
    assert row["elo_diff"] == pytest.approx(
        row["h_elo"] - row["a_elo"] + build.HOME_ADV)
