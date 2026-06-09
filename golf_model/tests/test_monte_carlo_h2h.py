# ==============================================================================
# tests/test_monte_carlo_h2h.py
# ==============================================================================
#
# Regression tests for cut-aware H2H matchup settlement (AUDIT 3.2).
#
# Before the fix, simulated H2H probabilities were conditional on BOTH
# players making the cut. Real 72-hole match rules (and the recorded
# p1_outcome data) settle differently: if exactly one player misses the
# cut the other wins; if both miss, the lower 36-hole score wins; ties
# void. Conditioning on both-make-cut threw away exactly the scenarios
# where a steady cut-maker has his largest advantage.
#
# Run from golf_model/:  python -m pytest tests/ -q
# ==============================================================================

import numpy as np
import pytest

from config.settings import Settings
from simulation.monte_carlo import MonteCarloSimulator, PlayerAbility
from simulation.tournament import TournamentConfig, simulate_tournament_outcome


def _simulate(n_sims=2000):
    """Field of 60 average players plus 4 probes with deterministic fates.

    With cut_rule top_50_ties on a 64-man field:
      - Player 1001 (mu=+3):  always makes the cut.
      - Player 1002 (mu=-8):  always misses the cut.
      - Player 1003 (mu=-9):  always misses the cut.
      - Player 1004 (mu=-11): always misses the cut, worse than 1003.
    """
    cfg = Settings()
    cfg.RANDOM_SEED = 7
    sim = MonteCarloSimulator(cfg)

    field = [
        PlayerAbility(player_id=pid, mu_mean=0.0, mu_std=0.0, sigma=2.0)
        for pid in range(1, 61)
    ]
    field += [
        PlayerAbility(player_id=1001, mu_mean=3.0, mu_std=0.0, sigma=0.3),
        PlayerAbility(player_id=1002, mu_mean=-8.0, mu_std=0.0, sigma=0.3),
        PlayerAbility(player_id=1003, mu_mean=-9.0, mu_std=0.0, sigma=0.3),
        PlayerAbility(player_id=1004, mu_mean=-11.0, mu_std=0.0, sigma=0.3),
    ]

    config = TournamentConfig(
        event_id=1, event_name="Cut Test", field_size=len(field),
        cut_rule="top_50_ties",
    )
    return sim.simulate_tournament(field, config, n_simulations=n_sims,
                                   compute_h2h=True)


@pytest.fixture(scope="module")
def result():
    return _simulate()


def test_cut_maker_beats_cut_misser(result):
    """1001 always makes the cut, 1002 never does → P(1001 beats 1002) ≈ 1.

    Under the old both-make-cut conditioning this pair had (almost) no
    valid sims, so the probability was missing or based on noise.
    """
    p = result.h2h_probs[1001][1002]
    assert p > 0.99


def test_both_miss_cut_settles_on_36_holes(result):
    """1003 and 1004 both always miss; 1003's 36-hole score is better.

    Under the old conditioning this pair never produced a valid sim.
    """
    p = result.h2h_probs[1003][1004]
    assert p > 0.95


def test_h2h_probabilities_are_complementary(result):
    """P(A>B) + P(B>A) must equal 1 for every recorded pair (ties void)."""
    for pid_a, opponents in result.h2h_probs.items():
        for pid_b, p_ab in opponents.items():
            p_ba = result.h2h_probs[pid_b][pid_a]
            assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)


def test_outcome_dict_exposes_r2_scores():
    """simulate_tournament_outcome must return 36-hole totals for all players."""
    rng = np.random.default_rng(0)
    round_scores = rng.normal(0, 2, size=(80, 4))
    config = TournamentConfig(event_id=1, cut_rule="top_50_ties")
    outcome = simulate_tournament_outcome(round_scores, config, rng)
    assert "r2_scores" in outcome
    np.testing.assert_allclose(
        outcome["r2_scores"], round_scores[:, 0] + round_scores[:, 1]
    )
    # 36-hole totals are defined for cut-missers too (no NaN)
    assert not np.isnan(outcome["r2_scores"]).any()
