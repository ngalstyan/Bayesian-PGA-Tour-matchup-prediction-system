# ==============================================================================
# tests/test_sharpe_and_exposure.py
# ==============================================================================
#
# Regression tests for AUDIT 3.1 (Sharpe over-annualization, zero-bet
# events excluded from the return series) and AUDIT 3.4 (no event-level
# exposure cap in the matchup backtest path).
#
# Run from golf_model/:  python -m pytest tests/ -q
# ==============================================================================

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from validation.backtest import BacktestCache, BacktestEngine, events_per_year


def test_events_per_year_from_dates():
    # 41 events spanning exactly two years → ~20.5 events/year
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=41)
    assert events_per_year(dates) == pytest.approx(41 / 2.0, rel=0.01)


def test_events_per_year_degenerate_cases():
    assert events_per_year(["2023-06-01"]) == 1.0
    assert events_per_year(["2023-06-01", "2023-06-01"]) == 1.0


def _make_cfg(tmp_path):
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
    cfg.MAX_TOURNAMENT_EXPOSURE_PCT = 0.30
    cfg.INITIAL_BANKROLL = 5000.0
    return cfg


def _make_cache(h2h_by_event, dates_by_event):
    return BacktestCache(
        created_at="test",
        model_description="synthetic",
        holdout_seasons=[2023],
        n_events=len(h2h_by_event),
        model_params={},
        predictions={k: {1: 0.5} for k in h2h_by_event},
        event_metadata={
            k: {"event_name": f"Event {k[0]}", "date": dates_by_event[k],
                "calendar_year": k[1], "n_players": 100, "winner_id": 1}
            for k in h2h_by_event
        },
        h2h_predictions=h2h_by_event,
    )


def _matchup_rows(event_id, year, pairs):
    rows = []
    for p1, p2 in pairs:
        rows.append({
            "event_id": event_id, "calendar_year": year,
            "event_name": f"Event {event_id}", "bet_type": "72-hole Match",
            "p1_dg_id": p1, "p2_dg_id": p2,
            "p1_player_name": f"P{p1}", "p2_player_name": f"P{p2}",
            "p1_close": 1.90, "p2_close": 1.90, "p1_outcome": 1.0,
        })
    return pd.DataFrame(rows)


def test_event_exposure_cap_binds(tmp_path):
    """50 max-edge matchups would stake 50 × 1.5% = 75% of bankroll
    uncapped; the event cap must scale total exposure down to 30%."""
    cfg = _make_cfg(tmp_path)
    pairs = [(2 * i + 1, 2 * i + 2) for i in range(50)]
    h2h = {}
    for p1, p2 in pairs:
        h2h.setdefault(p1, {})[p2] = 0.9   # huge edge vs even odds
        h2h.setdefault(p2, {})[p1] = 0.1
    cache = _make_cache({(100, 2023): h2h}, {(100, 2023): "2023-06-01"})
    odds = _matchup_rows(100, 2023, pairs)

    res = BacktestEngine(cfg).evaluate_matchups(cache, odds,
                                                bet_type="72-hole Match")
    assert res["n_bets"] == 50
    total_staked = res["bets_df"]["stake"].sum()
    cap = cfg.MAX_TOURNAMENT_EXPOSURE_PCT * cfg.INITIAL_BANKROLL
    assert total_staked == pytest.approx(cap, abs=1.0)
    # Proportional scaling: equal candidates stay equal
    assert res["bets_df"]["stake"].std() == pytest.approx(0.0, abs=0.01)


def test_zero_bet_events_count_in_sharpe(tmp_path):
    """An evaluated event with odds but no qualifying edge contributes a
    0.0 return to the Sharpe series instead of being dropped."""
    cfg = _make_cfg(tmp_path)
    h2h_edge = {1: {2: 0.9}, 2: {1: 0.1}}      # bets
    h2h_no_edge = {3: {4: 0.5}, 4: {3: 0.5}}   # no edge → no bets
    cache = _make_cache(
        {(100, 2023): h2h_edge, (200, 2023): h2h_no_edge},
        {(100, 2023): "2023-06-01", (200, 2023): "2023-07-01"},
    )
    odds = pd.concat([
        _matchup_rows(100, 2023, [(1, 2)]),
        _matchup_rows(200, 2023, [(3, 4)]),
    ], ignore_index=True)

    res = BacktestEngine(cfg).evaluate_matchups(cache, odds,
                                                bet_type="72-hole Match")
    assert res["n_bets"] == 1
    assert res["n_events_with_bets"] == 1
    assert res["n_events_evaluated"] == 2  # zero-bet event included
