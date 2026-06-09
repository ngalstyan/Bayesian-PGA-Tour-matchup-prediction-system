# ==============================================================================
# tests/test_monte_carlo_noise.py
# ==============================================================================
#
# Regression test for the Student-t noise scale (AUDIT 4.3).
#
# PlayerAbility.sigma is the player's EMPIRICAL round-to-round standard
# deviation. A unit-scale t(ν) has variance ν/(ν-2) > 1, so the simulator
# must scale draws by √((ν-2)/ν) internally. Before the fix this
# correction lived in the notebook callbacks, and any caller that passed
# raw SDs (notebooks 06/07) overdispersed round noise by ~22% at ν=6.
#
# Run from golf_model/:  python -m pytest tests/ -q
# ==============================================================================

import numpy as np

from config.settings import Settings
from simulation.monte_carlo import MonteCarloSimulator


def test_round_noise_sd_matches_requested_sigma():
    cfg = Settings()
    cfg.RANDOM_SEED = 11
    cfg.ROUND_CORRELATION_RHO = 0.0  # isolate the noise term
    sim = MonteCarloSimulator(cfg)

    sigma = 2.75
    n_sims, n_players = 200_000, 2
    abilities = np.zeros((n_sims, n_players))
    sigmas = np.full(n_players, sigma)

    scores = sim._simulate_rounds(abilities, sigmas, n_sims, n_players)

    # Round-1 scores are pure noise around ability 0 (negated, SD unchanged)
    realized_sd = scores[:, :, 0].std()
    # t(6) tails make the sample SD noisy; 2% tolerance on 400K draws
    assert abs(realized_sd - sigma) / sigma < 0.02, (
        f"realized SD {realized_sd:.3f} vs requested {sigma}"
    )
