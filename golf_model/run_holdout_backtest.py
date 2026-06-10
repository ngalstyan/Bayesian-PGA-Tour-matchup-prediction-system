# ==============================================================================
# golf_model/run_holdout_backtest.py
# ==============================================================================
#
# HEADLESS 2023-2024 HOLDOUT BACKTEST (canonical reproduction)
# -------------------------------------------------------------
# Regenerates the expanding-window backtest end-to-end with the
# ProductionEWMAModel, evaluates outright and H2H matchup betting, runs
# integrity checks (no event-year collisions, winners match fin_text,
# per-year bet symmetry), and writes a results JSON to
# artifacts/outputs/backtest_2023_2024_results.json.
#
# This is the same flow as notebook 08, runnable without Jupyter:
#
#   conda activate golf_model
#   cd golf_model && python run_holdout_backtest.py
#
# Takes ~30-60 minutes (50K MC sims + H2H pairwise per event).
#
# IMPORTANT CAVEAT (recorded in the output JSON): the betting
# hyperparameters were historically selected by iterating on this same
# 2023-2024 holdout (AUDIT.md finding 2.3), so these numbers are a
# diagnostic, not an out-of-sample claim. See run_oos_validation.py for
# the frozen-parameter 2025-2026 validation.
#
# ==============================================================================

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings
from data.loader import DataLoader
from models.production_ewma import ProductionEWMAModel
from validation.backtest import BacktestCache, BacktestEngine
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_NAME = "ewma25_50k_v3.pkl"
RESULTS_NAME = "backtest_2023_2024_results.json"


# ==============================================================================
# Data assembly (mirrors notebook 08 cells 1 & 4)
# ==============================================================================

def load_combined_data(cfg: Settings):
    loader = DataLoader(cfg)

    rounds_df = loader.load_rounds()
    if "dg_id" in rounds_df.columns and "player_id" not in rounds_df.columns:
        rounds_df = rounds_df.rename(columns={"dg_id": "player_id"})
    if "date" not in rounds_df.columns and "event_completed" in rounds_df.columns:
        rounds_df["date"] = pd.to_datetime(rounds_df["event_completed"], errors="coerce")

    try:
        events_df = loader.load_events()
    except FileNotFoundError:
        events_df = rounds_df.groupby("event_id").agg(
            start_date=("date", "min"),
            event_name=("event_name", "first") if "event_name" in rounds_df.columns else ("event_id", "first"),
        ).reset_index()
        events_df["calendar_year"] = pd.to_datetime(events_df["start_date"]).dt.year
        if "season" in rounds_df.columns:
            events_df["season"] = rounds_df.groupby("event_id")["season"].first().values
    if "start_date" not in events_df.columns and "calendar_year" in events_df.columns:
        events_df["start_date"] = pd.to_datetime(
            events_df["calendar_year"].astype(int).astype(str) + "-01-01"
        )

    try:
        odds_df = loader.load_odds()
    except FileNotFoundError:
        odds_df = pd.DataFrame()

    holdout_dir = cfg.DATA_DIR / "holdout"
    rounds_holdout = pd.read_csv(holdout_dir / "sg_rounds_pga_2023_2024.csv", low_memory=False)
    rounds_holdout = rounds_holdout.rename(columns={"dg_id": "player_id"})
    rounds_holdout["date"] = pd.to_datetime(rounds_holdout["event_completed"], errors="coerce")

    odds_holdout = pd.read_csv(holdout_dir / "odds_2023_2024.csv", low_memory=False)

    schedule_holdout = pd.read_csv(holdout_dir / "schedule_2023_2024.csv", low_memory=False)
    schedule_holdout = schedule_holdout[
        schedule_holdout["calendar_year"].isin([2023, 2024])
        & (schedule_holdout["tour"] == "pga")
    ].copy()
    schedule_holdout["start_date"] = pd.to_datetime(schedule_holdout["date"], errors="coerce")

    events_combined = pd.concat([events_df, schedule_holdout], ignore_index=True)
    rounds_combined = pd.concat([rounds_df, rounds_holdout], ignore_index=True)
    odds_combined = pd.concat([odds_df, odds_holdout], ignore_index=True)

    odds_combined = odds_combined.rename(columns={
        "dg_id": "player_id",
        "bookmaker": "book",
        "close_odds": "decimal_odds",
    })
    if "market" in odds_combined.columns:
        odds_combined = odds_combined[odds_combined["market"].str.lower() == "win"]

    rounds_combined["date"] = pd.to_datetime(rounds_combined["date"])
    rounds_combined = rounds_combined.sort_values("date").reset_index(drop=True)

    matchup_odds = pd.read_csv(holdout_dir / "matchup_odds_2023_2024.csv")

    logger.info(
        "Data loaded | rounds=%d | events=%d | odds=%d | matchups=%d",
        len(rounds_combined), len(events_combined),
        len(odds_combined), len(matchup_odds),
    )
    return rounds_combined, events_combined, odds_combined, matchup_odds, rounds_holdout


# ==============================================================================
# Integrity checks
# ==============================================================================

def check_cache_integrity(cache: BacktestCache, rounds_holdout: pd.DataFrame) -> dict:
    """No year collisions; winners match fin_text per edition."""
    report = {}

    keys = list(cache.event_metadata.keys())
    report["n_cache_entries"] = len(keys)
    report["n_unique_event_ids"] = len({k[0] for k in keys})
    report["keys_match_meta_year"] = all(
        cache.event_metadata[k].get("calendar_year") == k[1] for k in keys
    )

    r = rounds_holdout.copy()
    r["date"] = pd.to_datetime(r["event_completed"], errors="coerce")
    r["yr"] = r["date"].dt.year

    winner_checked = winner_ok = 0
    mismatches = []
    for (event_id, year), meta in cache.event_metadata.items():
        edition = r[(r["event_id"] == event_id) & (r["yr"] == year)]
        if edition.empty or "fin_text" not in edition.columns:
            continue
        actual = edition[
            edition["fin_text"].astype(str).str.strip().isin(["1", "T1"])
        ]["player_id"].unique()
        if len(actual) == 0:
            continue
        winner_checked += 1
        if meta["winner_id"] in actual:
            winner_ok += 1
        else:
            mismatches.append({
                "event": meta["event_name"], "year": year,
                "cached": int(meta["winner_id"]),
                "actual": [int(a) for a in actual],
            })

    report["winners_checked"] = winner_checked
    report["winners_correct"] = winner_ok
    report["winner_mismatches"] = mismatches[:10]
    return report


def per_year_bet_split(bets_df: pd.DataFrame) -> dict:
    """Per-year bet stats. The pre-fix leak showed up as a 63.8% vs 50.3%
    win-rate asymmetry between years — this should now be unremarkable."""
    if len(bets_df) == 0:
        return {}
    df = bets_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    out = {}
    for year, g in df.groupby("year"):
        out[int(year)] = {
            "n_bets": int(len(g)),
            "win_rate_pct": round(float(g["won"].mean()) * 100, 1),
            "roi_pct": round(float(g["pnl"].sum() / g["stake"].sum()) * 100, 1)
            if g["stake"].sum() > 0 else 0.0,
        }
    return out


def flat_roi_with_cis(bets_df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """Flat-stake (unit) ROI with iid and by-event cluster-bootstrap CIs."""
    if len(bets_df) == 0:
        return {}
    unit = np.where(bets_df["won"], bets_df["decimal_odds"] - 1.0, -1.0)
    n = len(unit)
    mean = float(unit.mean())
    se = float(unit.std(ddof=1) / np.sqrt(n))

    rng = np.random.default_rng(seed)
    ev_ids = bets_df["event_id"].values
    uniq = np.unique(ev_ids)
    groups = {e: unit[ev_ids == e] for e in uniq}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        boots[i] = np.concatenate([groups[e] for e in sample]).mean()

    return {
        "flat_roi_pct": round(mean * 100, 2),
        "iid_ci95_pct": [round((mean - 1.96 * se) * 100, 2),
                         round((mean + 1.96 * se) * 100, 2)],
        "cluster_bootstrap_ci95_pct": [round(float(np.percentile(boots, 2.5)) * 100, 2),
                                       round(float(np.percentile(boots, 97.5)) * 100, 2)],
        "p_roi_leq_0": round(float((boots <= 0).mean()), 4),
        "n_bootstrap": n_boot,
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parent,
        ).strip()
    except Exception:
        return "unknown"


# ==============================================================================
# Main
# ==============================================================================

def main(skip_run_if_cache_exists: bool = False):
    cfg = Settings()
    cfg.OVERROUND_METHOD = "proportional"

    cache_path = cfg.PROCESSED_DIR / "backtest_cache" / CACHE_NAME
    results_path = cfg.OUTPUTS_DIR / RESULTS_NAME

    rounds, events, odds, matchup_odds, rounds_holdout = load_combined_data(cfg)

    engine = BacktestEngine(cfg)
    model = ProductionEWMAModel(cfg)

    # --- Expanding-window walk (slow) ---
    if skip_run_if_cache_exists and cache_path.exists():
        logger.info("Using existing cache at %s", cache_path)
        run_result = None
    else:
        logger.info("Starting expanding-window backtest → %s", cache_path)
        run_result = engine.run(
            events_df=events,
            rounds_df=rounds,
            odds_df=odds,
            fit_and_predict_fn=model.fit_and_predict,
            holdout_seasons=cfg.HOLDOUT_SEASONS,
            cache_path=cache_path,
            model_description=(
                "ProductionEWMAModel | 25r/120d EWMA + course-weights + "
                "recent-form + course-fit | cut-aware H2H | v3 composite keys"
            ),
        )

    cache = BacktestCache.load(cache_path)

    # --- Integrity checks ---
    integrity = check_cache_integrity(cache, rounds_holdout)
    logger.info("Integrity: %s", {k: v for k, v in integrity.items()
                                  if k != "winner_mismatches"})

    # --- Outright evaluation ---
    outright = engine.evaluate(cache, events, odds)

    # --- H2H matchup evaluation ---
    h2h = engine.evaluate_matchups(cache, matchup_odds, bet_type="72-hole Match")
    bets_df = h2h.pop("bets_df")

    year_split = per_year_bet_split(bets_df)
    roi_cis = flat_roi_with_cis(bets_df)

    # --- Gates ---
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
        "holdout_seasons": cfg.HOLDOUT_SEASONS,
        "caveat": (
            "Betting hyperparameters (MATCHUP_MIN_EDGE, PROB_TEMPERATURE, "
            "sizing) were historically selected on this same holdout "
            "(AUDIT.md 2.3). Treat these numbers as a diagnostic, not an "
            "out-of-sample estimate. See the 2025-2026 OOS validation."
        ),
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
        "h2h_matchups": {**h2h, "per_year": year_split,
                         "flat_stake_roi": roi_cis, "gates": h2h_gates},
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nResults written to {results_path}")
    return report


if __name__ == "__main__":
    skip = "--use-cache" in sys.argv
    main(skip_run_if_cache_exists=skip)
