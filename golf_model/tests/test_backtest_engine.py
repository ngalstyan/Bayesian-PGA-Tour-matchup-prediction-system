# ==============================================================================
# tests/test_backtest_engine.py
# ==============================================================================
#
# Regression tests for the event_id year-collision lookahead bug (AUDIT 2.1).
#
# DataGolf reuses event_id across annual editions of the same tournament.
# Before the fix, the backtest cache was keyed by event_id alone (later years
# overwrote earlier editions) and evaluate_matchups() joined matchup odds on
# event_id without a year filter — so earlier years' odds were priced with
# later-trained models (lookahead bias).
#
# Run from golf_model/:  python -m pytest tests/ -q
# ==============================================================================

import pickle

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from simulation.monte_carlo import SimulationResult
from validation.backtest import BacktestCache, BacktestEngine


EVENT_ID = 100  # same DataGolf event_id, two annual editions


def _make_settings(tmp_path):
    cfg = Settings(
        DATA_DIR=tmp_path / "data",
        PROCESSED_DIR=tmp_path / "processed",
        MODELS_DIR=tmp_path / "models",
        OUTPUTS_DIR=tmp_path / "outputs",
        LOGS_DIR=tmp_path / "logs",
    )
    cfg.MATCHUP_MIN_EDGE = 0.08
    cfg.KELLY_FRACTION = 0.25
    cfg.MATCHUP_MAX_BET_PCT = 0.015
    cfg.INITIAL_BANKROLL = 5000.0
    return cfg


def _make_events():
    """One tournament, two annual editions sharing the same event_id."""
    return pd.DataFrame({
        "event_id": [EVENT_ID, EVENT_ID],
        "event_name": ["Test Open", "Test Open"],
        "start_date": [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01")],
        "season": [2023, 2024],
    })


def _make_rounds():
    """Training rounds plus event rounds for both editions."""
    rows = []
    # Pre-2023 training history for players 1 and 2
    for pid in (1, 2):
        for d in pd.date_range("2022-01-09", periods=10, freq="7D"):
            rows.append({"player_id": pid, "event_id": 900 + pid,
                         "date": d, "sg_total": 0.5 if pid == 1 else -0.5})
    # 2023 edition (rounds stamped with completion date, after start).
    # Player 3 plays ONLY the 2023 edition; player 1 wins on SG.
    for pid, sg in ((1, 2.0), (2, -2.0), (3, -3.0)):
        rows.append({"player_id": pid, "event_id": EVENT_ID,
                     "date": pd.Timestamp("2023-06-04"), "sg_total": sg})
    # 2024 edition: only players 1 and 2; player 2 wins on SG.
    for pid, sg in ((1, -2.0), (2, 2.0)):
        rows.append({"player_id": pid, "event_id": EVENT_ID,
                     "date": pd.Timestamp("2024-06-04"), "sg_total": sg})
    return pd.DataFrame(rows)


def _fit_and_predict_stub(train_rounds, event_info, event_rounds):
    """Return distinguishable predictions per edition year.

    2023 model: P(player 1 beats player 2) = 0.9
    2024 model: P(player 1 beats player 2) = 0.1
    """
    year = pd.Timestamp(event_info["date"]).year
    p1_beats_p2 = 0.9 if year == 2023 else 0.1
    win_probs = {1: p1_beats_p2, 2: 1.0 - p1_beats_p2}
    return SimulationResult(
        win_probs=win_probs,
        top5_probs=win_probs,
        top10_probs=win_probs,
        top20_probs=win_probs,
        make_cut_probs={1: 1.0, 2: 1.0},
        n_simulations=1000,
        convergence_diagnostics={},
        h2h_probs={1: {2: p1_beats_p2}, 2: {1: 1.0 - p1_beats_p2}},
    )


def _make_matchup_odds():
    """One 72-hole matchup per edition, even odds, player 1 won both years."""
    rows = []
    for year in (2023, 2024):
        rows.append({
            "event_id": EVENT_ID,
            "event_name": "Test Open",
            "calendar_year": year,
            "bet_type": "72-hole Match",
            "p1_dg_id": 1, "p2_dg_id": 2,
            "p1_player_name": "Player One", "p2_player_name": "Player Two",
            "p1_close": 1.90, "p2_close": 1.90,
            "p1_outcome": 1.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def backtest(tmp_path):
    cfg = _make_settings(tmp_path)
    engine = BacktestEngine(cfg)
    cache_path = tmp_path / "cache.pkl"
    result = engine.run(
        events_df=_make_events(),
        rounds_df=_make_rounds(),
        odds_df=pd.DataFrame(),
        fit_and_predict_fn=_fit_and_predict_stub,
        holdout_seasons=[2023, 2024],
        cache_path=cache_path,
    )
    cache = BacktestCache.load(cache_path)
    return cfg, engine, result, cache


def test_cache_keeps_both_editions(backtest):
    """Both annual editions must survive in the cache — no overwriting."""
    _, _, _, cache = backtest
    assert (EVENT_ID, 2023) in cache.predictions
    assert (EVENT_ID, 2024) in cache.predictions
    assert cache.n_events == 2
    # The editions carry different predictions — they must not be conflated
    assert cache.h2h_predictions[(EVENT_ID, 2023)][1][2] == pytest.approx(0.9)
    assert cache.h2h_predictions[(EVENT_ID, 2024)][1][2] == pytest.approx(0.1)
    assert cache.event_metadata[(EVENT_ID, 2023)]["calendar_year"] == 2023
    assert cache.event_metadata[(EVENT_ID, 2024)]["calendar_year"] == 2024


def test_matchups_join_year_correctly(backtest):
    """Each year's odds must be priced with that year's model.

    The 2023 model says P(p1)=0.9 → bets p1 at even odds.
    The 2024 model says P(p1)=0.1 → bets p2 at even odds.
    Pre-fix, the 2024 model priced both years and both bets landed on p2.
    """
    cfg, engine, _, cache = backtest
    res = engine.evaluate_matchups(cache, _make_matchup_odds(),
                                   bet_type="72-hole Match")
    bets = res["bets_df"]
    assert len(bets) == 2
    side_by_year = dict(zip(pd.to_datetime(bets["date"]).dt.year, bets["bet_side"]))
    assert side_by_year[2023] == "p1"
    assert side_by_year[2024] == "p2"


def test_skip_accounting(backtest):
    """Nothing should be skipped on clean synthetic data, and the skip
    counters must exist so silent sample shrinkage is visible."""
    _, _, result, _ = backtest
    assert result.n_events_skipped == 0
    assert result.skip_reasons == {}
    assert result.total_events == 2


def test_event_rounds_are_per_edition(backtest):
    """Field size and winner must come from one edition, not all years merged.

    Player 3 plays only the 2023 edition; the SG-fallback winner is player 1
    in 2023 and player 2 in 2024. Pre-fix, both editions merged into one
    pool (field of 3 both years, single cross-year pseudo-winner).
    """
    _, _, _, cache = backtest
    meta_2023 = cache.event_metadata[(EVENT_ID, 2023)]
    meta_2024 = cache.event_metadata[(EVENT_ID, 2024)]
    assert meta_2023["n_players"] == 3
    assert meta_2024["n_players"] == 2
    assert meta_2023["winner_id"] == 1
    assert meta_2024["winner_id"] == 2


def test_get_winner_respects_playoff_fin_text(tmp_path):
    """Two players tied on strokes; fin_text '1' (playoff winner) must win."""
    engine = BacktestEngine(_make_settings(tmp_path))
    event_rounds = pd.DataFrame({
        "player_id": [1, 2, 3],
        "round_score": [270, 270, 275],
        "sg_total": [2.0, 2.1, 0.5],     # SG would (wrongly) pick player 2
        "fin_text": ["1", "T2", "3"],    # player 1 won the playoff
    })
    assert engine._get_winner(event_rounds) == 1


def test_get_winner_round_score_excludes_cut_missers(tmp_path):
    """Without fin_text, a 2-round cut-misser must not win on raw stroke sum."""
    engine = BacktestEngine(_make_settings(tmp_path))
    event_rounds = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 2, 2],
        "round_score": [68, 67, 69, 66, 72, 74],  # p2 missed cut: sum 146 < 270
    })
    assert engine._get_winner(event_rounds) == 1


def test_old_cache_versions_are_refused(tmp_path):
    """V1/V2 caches predate the collision fix and are corrupted — refuse them."""
    path = tmp_path / "old_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump({"version": 2, "cache": object()}, f)
    with pytest.raises(ValueError, match="event-year collision"):
        BacktestCache.load(path)
