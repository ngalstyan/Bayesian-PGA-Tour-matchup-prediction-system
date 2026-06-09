# ==============================================================================
# tests/test_production_ewma.py
# ==============================================================================
#
# Smoke test for the production model module (AUDIT 5.3): the model that
# generates backtest and live predictions was previously duplicated
# notebook code; it now lives in models/production_ewma.py.
#
# Run from golf_model/:  python -m pytest tests/ -q
# ==============================================================================

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from models.production_ewma import ProductionEWMAModel
from simulation.monte_carlo import SimulationResult


def _make_train_rounds(n_players=12, n_events=15, seed=3):
    """Synthetic pre-event history: one row per player per event-round."""
    rng = np.random.default_rng(seed)
    skills = rng.normal(0, 1.0, n_players)
    rows = []
    for eid in range(1, n_events + 1):
        date = pd.Timestamp("2022-01-02") + pd.Timedelta(days=14 * eid)
        for pid in range(1, n_players + 1):
            for _ in range(4):
                sg = skills[pid - 1] + rng.normal(0, 2.5)
                parts = rng.dirichlet(np.ones(4)) * sg
                rows.append({
                    "player_id": pid, "event_id": eid, "date": date,
                    "sg_total": sg, "sg_ott": parts[0], "sg_app": parts[1],
                    "sg_arg": parts[2], "sg_putt": parts[3],
                })
    return pd.DataFrame(rows)


def test_predict_returns_simulation_result(tmp_path):
    cfg = Settings(
        DATA_DIR=tmp_path / "data",
        PROCESSED_DIR=tmp_path / "processed",
        MODELS_DIR=tmp_path / "models",
        OUTPUTS_DIR=tmp_path / "outputs",
        LOGS_DIR=tmp_path / "logs",
    )
    cfg.RANDOM_SEED = 5
    model = ProductionEWMAModel(cfg, n_simulations=2000)

    train = _make_train_rounds()
    player_ids = list(range(1, 13))
    result = model.predict(
        train_rounds=train,
        event_id=1,                      # has past editions in train
        event_name="Smoke Test Open",
        event_date=pd.Timestamp("2023-01-05"),
        player_ids=player_ids,
    )

    assert isinstance(result, SimulationResult)
    assert set(result.win_probs.keys()) == set(player_ids)
    assert sum(result.win_probs.values()) == pytest.approx(1.0, abs=0.01)
    # H2H pairs exist and are complementary
    assert result.h2h_probs
    p_ab = result.h2h_probs[1][2]
    p_ba = result.h2h_probs[2][1]
    assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)


def test_fit_and_predict_adapter_matches_engine_signature(tmp_path):
    cfg = Settings(
        DATA_DIR=tmp_path / "data",
        PROCESSED_DIR=tmp_path / "processed",
        MODELS_DIR=tmp_path / "models",
        OUTPUTS_DIR=tmp_path / "outputs",
        LOGS_DIR=tmp_path / "logs",
    )
    model = ProductionEWMAModel(cfg, n_simulations=500)
    train = _make_train_rounds()
    event_rounds = pd.DataFrame({
        "player_id": [1, 2, 3], "event_id": [99] * 3,
        "date": [pd.Timestamp("2023-02-01")] * 3, "sg_total": [0.0] * 3,
    })
    out = model.fit_and_predict(
        train,
        {"event_id": 99, "event_name": "Adapter Test", "date": "2023-02-01"},
        event_rounds,
    )
    assert isinstance(out, SimulationResult)
    assert set(out.win_probs.keys()) == {1, 2, 3}


def test_empty_inputs_return_empty():
    model = ProductionEWMAModel(n_simulations=100)
    assert model.predict(pd.DataFrame(), 1, "X", pd.Timestamp("2023-01-01"), [1]) == {}
    assert model.predict(_make_train_rounds(n_events=2), 1, "X",
                         pd.Timestamp("2023-01-01"), []) == {}
