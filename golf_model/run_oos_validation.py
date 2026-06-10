# ==============================================================================
# golf_model/run_oos_validation.py
# ==============================================================================
#
# FROZEN-PARAMETER OUT-OF-SAMPLE VALIDATION — 2025 to mid-2026
# -------------------------------------------------------------
# The 2023-2024 "holdout" was burned for selection: MATCHUP_MIN_EDGE was
# grid-searched on it, PROB_TEMPERATURE / sizing / devig method were
# chosen by re-running it (AUDIT.md finding 2.3). This script provides
# the only defensible out-of-sample estimate: an expanding-window walk
# over 2025–2026 events with every parameter frozen at the values in
# config/settings.py, evaluated against Pinnacle closing matchup odds.
#
#   conda activate golf_model
#   cd golf_model && python run_oos_validation.py            # full run
#   cd golf_model && python run_oos_validation.py --use-cache # re-evaluate only
#
# DISCLOSURES (recorded in the output JSON):
#   1. Parameters are frozen as of the remediation (June 2026), but live
#      betting ran during 2025-2026 and may have informally influenced
#      earlier parameter choices. This is the best available
#      approximation of out-of-sample, not a guarantee.
#   2. 2026 is a partial season (through early June).
#   3. The benchmark is the Pinnacle CLOSING line — the sharpest price,
#      generally not executable at size. Positive ROI here is a high
#      bar; live execution happens earlier at softer books.
#
# ==============================================================================

import json
import sys
from datetime import datetime

import pandas as pd

from config.settings import Settings
from data.loader import DataLoader
from models.production_ewma import ProductionEWMAModel
from validation.backtest import BacktestCache, BacktestEngine
from run_holdout_backtest import (
    check_cache_integrity,
    flat_roi_with_cis,
    git_sha,
    per_year_bet_split,
)
from utils.logger import get_logger

logger = get_logger(__name__)

OOS_SEASONS = [2025, 2026]
CACHE_NAME = "ewma25_50k_oos_2025_2026.pkl"
RESULTS_NAME = "oos_validation_2025_2026_results.json"


def load_oos_data(cfg: Settings):
    """Training rounds = everything (engine date-filters per event);
    walk events / odds / matchups = 2025-2026 PGA."""
    loader = DataLoader(cfg)

    # All root-dir rounds: training 2017-2022 (all tours) + PGA 2025/2026
    rounds_df = loader.load_rounds()
    if "dg_id" in rounds_df.columns and "player_id" not in rounds_df.columns:
        rounds_df = rounds_df.rename(columns={"dg_id": "player_id"})
    if "date" not in rounds_df.columns and "event_completed" in rounds_df.columns:
        rounds_df["date"] = pd.to_datetime(rounds_df["event_completed"], errors="coerce")

    # Add the 2023-2024 holdout rounds (training history for 2025+ events)
    holdout_dir = cfg.DATA_DIR / "holdout"
    rounds_holdout = pd.read_csv(
        holdout_dir / "sg_rounds_pga_2023_2024.csv", low_memory=False
    ).rename(columns={"dg_id": "player_id"})
    rounds_holdout["date"] = pd.to_datetime(
        rounds_holdout["event_completed"], errors="coerce"
    )

    rounds_combined = pd.concat([rounds_df, rounds_holdout], ignore_index=True)
    rounds_combined["date"] = pd.to_datetime(rounds_combined["date"])
    rounds_combined = rounds_combined.sort_values("date").reset_index(drop=True)

    # Walk events: 2025-2026 PGA schedules
    schedules = []
    for year in OOS_SEASONS:
        s = pd.read_csv(cfg.DATA_DIR / f"schedule_{year}.csv", low_memory=False)
        s = s[(s["tour"] == "pga") & (s["calendar_year"].isin(OOS_SEASONS))].copy()
        schedules.append(s)
    events = pd.concat(schedules, ignore_index=True)
    events["start_date"] = pd.to_datetime(events["date"], errors="coerce")
    events["season"] = events["calendar_year"].astype(int)

    # Outright odds
    odds = pd.concat(
        [pd.read_csv(cfg.DATA_DIR / f"odds_{y}.csv", low_memory=False)
         for y in OOS_SEASONS],
        ignore_index=True,
    ).rename(columns={
        "dg_id": "player_id", "bookmaker": "book", "close_odds": "decimal_odds",
    })
    if "market" in odds.columns:
        odds = odds[odds["market"].str.lower() == "win"]

    # Matchup odds
    matchup_odds = pd.concat(
        [pd.read_csv(cfg.DATA_DIR / f"matchup_odds_{y}.csv", low_memory=False)
         for y in OOS_SEASONS],
        ignore_index=True,
    )

    # OOS rounds only — for the winner integrity check
    oos_rounds = rounds_combined[
        pd.to_datetime(rounds_combined["date"]).dt.year.isin(OOS_SEASONS)
    ].copy()
    if "event_completed" not in oos_rounds.columns:
        oos_rounds["event_completed"] = oos_rounds["date"]

    logger.info(
        "OOS data | rounds=%d (train pool) | walk events=%d | odds=%d | matchups=%d",
        len(rounds_combined), len(events), len(odds), len(matchup_odds),
    )
    return rounds_combined, events, odds, matchup_odds, oos_rounds


def main(skip_run_if_cache_exists: bool = False):
    cfg = Settings()
    cfg.OVERROUND_METHOD = "proportional"

    cache_path = cfg.PROCESSED_DIR / "backtest_cache" / CACHE_NAME
    results_path = cfg.OUTPUTS_DIR / RESULTS_NAME

    rounds, events, odds, matchup_odds, oos_rounds = load_oos_data(cfg)

    engine = BacktestEngine(cfg)
    model = ProductionEWMAModel(cfg)

    if skip_run_if_cache_exists and cache_path.exists():
        logger.info("Using existing cache at %s", cache_path)
        run_result = None
    else:
        logger.info("Starting OOS expanding-window walk → %s", cache_path)
        run_result = engine.run(
            events_df=events,
            rounds_df=rounds,
            odds_df=odds,
            fit_and_predict_fn=model.fit_and_predict,
            holdout_seasons=OOS_SEASONS,
            cache_path=cache_path,
            model_description=(
                "OOS 2025-2026 | ProductionEWMAModel | frozen params | "
                "cut-aware H2H | v3 composite keys"
            ),
        )

    cache = BacktestCache.load(cache_path)

    integrity = check_cache_integrity(cache, oos_rounds)
    logger.info("Integrity: %s", {k: v for k, v in integrity.items()
                                  if k != "winner_mismatches"})

    outright = engine.evaluate(cache, events, odds)

    h2h = engine.evaluate_matchups(cache, matchup_odds, bet_type="72-hole Match")
    bets_df = h2h.pop("bets_df")

    h2h_gates = {
        "sample_size_ge_100": h2h["n_bets"] >= 100,
        "win_rate_gt_50": h2h["win_rate"] > 50.0,
        "roi_gt_0": h2h["roi_pct"] > 0,
        "sharpe_gt_0.3": h2h["sharpe"] > 0.3,
    }
    h2h_gates["all_passed"] = all(h2h_gates.values())

    report = {
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha(),
        "oos_seasons": OOS_SEASONS,
        "disclosures": [
            "Parameters frozen at config/settings.py values as of the "
            "June 2026 remediation; live betting ran during 2025-2026 and "
            "may have informally influenced earlier parameter choices.",
            "2026 is a partial season (through early June 2026).",
            "Benchmark is the Pinnacle CLOSING line (sharpest, generally "
            "not executable at size).",
        ],
        "model_params": {
            "EWMA_HALF_LIFE_ROUNDS": cfg.EWMA_HALF_LIFE_ROUNDS,
            "EWMA_HALF_LIFE_DAYS": cfg.EWMA_HALF_LIFE_DAYS,
            "OBSERVATION_DF": cfg.OBSERVATION_DF,
            "ROUND_CORRELATION_RHO": cfg.ROUND_CORRELATION_RHO,
            "COURSE_SG_BLEND": cfg.COURSE_SG_BLEND,
            "RECENT_FORM_BLEND": cfg.RECENT_FORM_BLEND,
            "COURSE_FIT_SHRINKAGE": cfg.COURSE_FIT_SHRINKAGE,
            "MU_STD_CAP": cfg.MU_STD_CAP,
            "PRODUCTION_N_SIMULATIONS": cfg.PRODUCTION_N_SIMULATIONS,
            "PROB_TEMPERATURE": cfg.PROB_TEMPERATURE,
            "MATCHUP_MIN_EDGE": cfg.MATCHUP_MIN_EDGE,
            "KELLY_FRACTION": cfg.KELLY_FRACTION,
            "MATCHUP_MAX_BET_PCT": cfg.MATCHUP_MAX_BET_PCT,
            "MAX_TOURNAMENT_EXPOSURE_PCT": cfg.MAX_TOURNAMENT_EXPOSURE_PCT,
            "OVERROUND_METHOD": cfg.OVERROUND_METHOD,
        },
        "integrity": integrity,
        "walk": {
            "n_events_scored": run_result.total_events if run_result else None,
            "n_events_skipped": run_result.n_events_skipped if run_result else None,
            "skip_reasons": run_result.skip_reasons if run_result else None,
        },
        "outright": {
            "n_events": outright.total_events,
            "model_avg_brier": round(outright.model_avg_brier, 5),
            "market_avg_brier": round(outright.market_avg_brier, 5),
            "n_bets": outright.total_bets,
            "total_staked": round(outright.total_staked, 2),
            "total_pnl": round(outright.total_pnl, 2),
            "roi_pct": outright.roi_pct,
            "sharpe": outright.sharpe,
            "max_dd_pct": outright.max_dd_pct,
            "gate_1_calibration": outright.gate_1_passed,
            "gate_2_significance": outright.gate_2_passed,
            "gate_3_betting": outright.gate_3_passed,
        },
        "h2h_matchups": {
            **h2h,
            "per_year": per_year_bet_split(bets_df),
            "flat_stake_roi": flat_roi_with_cis(bets_df),
            "gates": h2h_gates,
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nResults written to {results_path}")
    return report


if __name__ == "__main__":
    main(skip_run_if_cache_exists="--use-cache" in sys.argv)
