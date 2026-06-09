# ==============================================================================
# golf_model/validation/backtest.py
# ==============================================================================
#
# EXPANDING-WINDOW BACKTESTING ENGINE
# --------------------------------------
# Simulates the model's real-time performance by walking through 
# historical tournaments chronologically, using ONLY data available 
# at the time of each prediction.
#
# CRITICAL: This is the primary defense against data leakage.
# The expanding window ensures:
#   - The model NEVER trains on future data.
#   - Each prediction uses only past information.
#   - Results represent what you would actually experience live.
#
# Protocol:
#   For each tournament t in the holdout period:
#     1. Training data = all rounds from events before t.
#     2. Fit/update model on training data.
#     3. Generate predictions for tournament t.
#     4. Record predicted probabilities.
#     5. After tournament: record actual outcome.
#     6. Compute per-event metrics (Brier, log-loss).
#     7. If betting: simulate bet placement and P&L.
#     8. Expand training window to include tournament t.
#     9. Move to tournament t+1.
#
# ==============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import Settings
from validation.metrics import brier_score, log_loss, roi, sharpe_ratio, max_drawdown
from validation.statistical_tests import diebold_mariano_test
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestEvent:
    """Results for a single backtested tournament."""
    event_id: int
    event_name: str = ""
    date: str = ""
    n_players: int = 0
    
    # Predictions
    model_probs: Optional[Dict[int, float]] = None    # {player_id: P(win)}
    market_probs: Optional[Dict[int, float]] = None   # {player_id: P(win)}
    
    # Outcome
    winner_id: int = 0
    
    # Metrics
    model_brier: float = 0.0
    market_brier: float = 0.0
    model_logloss: float = 0.0
    market_logloss: float = 0.0
    
    # Betting
    n_bets: int = 0
    stake_total: float = 0.0
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results across all events."""
    events: List[BacktestEvent] = field(default_factory=list)
    
    # Aggregate metrics
    total_events: int = 0
    model_avg_brier: float = 0.0
    market_avg_brier: float = 0.0
    model_avg_logloss: float = 0.0
    market_avg_logloss: float = 0.0
    
    # Betting aggregate
    total_bets: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    
    # Gate verdicts
    gate_1_passed: bool = False
    gate_2_passed: bool = False
    gate_3_passed: bool = False


@dataclass
class BacktestCache:
    """
    Cached simulation results for fast betting parameter iteration.

    V3: All prediction dicts are keyed by the composite event key
    (event_id, calendar_year) — NOT event_id alone. DataGolf reuses
    event_id across annual editions of the same tournament, so keying
    by event_id alone silently overwrites earlier editions and lets
    later-trained models be evaluated against earlier years' odds
    (lookahead bias).
    """
    VERSION: ClassVar[int] = 3

    created_at: str
    model_description: str
    holdout_seasons: List[int]
    n_events: int
    model_params: Dict[str, Any]
    predictions: Dict[Tuple[int, int], Dict[int, float]]       # {(event_id, year): {player_id: P(win)}}
    event_metadata: Dict[Tuple[int, int], Dict]                 # {(event_id, year): {event_name, date, calendar_year, n_players, winner_id}}

    # Full simulation results
    h2h_predictions: Optional[Dict[Tuple[int, int], Dict[int, Dict[int, float]]]] = None   # {(event_id, year): {pid_a: {pid_b: P(A>B)}}}
    top5_predictions: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None             # {(event_id, year): {player_id: P(top5)}}
    top10_predictions: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None            # {(event_id, year): {player_id: P(top10)}}
    top20_predictions: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None            # {(event_id, year): {player_id: P(top20)}}
    make_cut_predictions: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None         # {(event_id, year): {player_id: P(make_cut)}}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"version": self.VERSION, "cache": self}, f)
        logger.info("Cache saved to %s (%d events)", path, self.n_events)

    @classmethod
    def load(cls, path: Path) -> "BacktestCache":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "cache" in data:
            version = data.get("version", 1)
            cache = data["cache"]
        else:
            version = 1
            cache = data
        if version < 3:
            raise ValueError(
                f"Cache at {path} is version {version}, which predates the "
                "event-year collision fix: predictions were keyed by event_id "
                "alone and later years overwrote earlier editions, and simulated "
                "fields/winners merged multiple years. These caches produce "
                "lookahead-biased results and cannot be migrated. "
                "Regenerate with BacktestEngine.run()."
            )
        logger.info("Cache loaded from %s (%d events)", path, cache.n_events)
        return cache


class BacktestEngine:
    """
    Expanding-window backtesting engine.
    
    This engine orchestrates the backtest by:
    1. Walking through events chronologically.
    2. Calling user-provided functions for model fitting and prediction.
    3. Recording all predictions and outcomes.
    4. Computing validation metrics.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        logger.info("BacktestEngine initialized")

    def _sharpen_probs(self, probs: Dict[int, float]) -> Dict[int, float]:
        """Apply temperature scaling: p^(1/T) / Z.  T<1 sharpens, T=1 no-op."""
        T = getattr(self.settings, "PROB_TEMPERATURE", 1.0)
        if T == 1.0 or not probs:
            return probs
        inv_T = 1.0 / T
        raw = {pid: max(p, 1e-12) ** inv_T for pid, p in probs.items()}
        Z = sum(raw.values())
        return {pid: v / Z for pid, v in raw.items()}

    def run(
        self,
        events_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        fit_and_predict_fn: Callable,
        holdout_seasons: Optional[List[int]] = None,
        cache_path: Optional[Path] = None,
        model_description: str = "",
    ) -> BacktestResult:
        """
        Execute the expanding-window backtest.
        
        Parameters
        ----------
        events_df : pd.DataFrame
            Tournament metadata with columns: event_id, start_date, season.
            
        rounds_df : pd.DataFrame
            Round-level SG data with standard schema columns.
            
        odds_df : pd.DataFrame
            Historical odds with standard schema columns.
            
        fit_and_predict_fn : Callable
            User-provided function with signature:
                (train_rounds: pd.DataFrame,
                 event_info: dict,
                 event_rounds: pd.DataFrame)
                -> SimulationResult or Dict[int, float]

            Can return either a full SimulationResult (with h2h, top-N probs)
            or a simple {player_id: P(win)} dict for backward compatibility.

            The engine guarantees train_rounds contains ONLY data
            from BEFORE the target event.
            
        holdout_seasons : list of int, optional
            Seasons to backtest. Defaults to settings.HOLDOUT_SEASONS.
            
        Returns
        -------
        BacktestResult
            Complete backtest results.
        """
        holdout_seasons = holdout_seasons or self.settings.HOLDOUT_SEASONS

        # Sort events chronologically
        events = events_df.copy()
        events["start_date"] = pd.to_datetime(events["start_date"])
        events = events.sort_values("start_date").reset_index(drop=True)

        # Filter to holdout seasons
        if "season" in events.columns:
            holdout_events = events[events["season"].isin(holdout_seasons)]
        else:
            holdout_events = events[
                events["start_date"].dt.year.isin(holdout_seasons)
            ]

        logger.info(
            "Starting backtest | %d holdout events | seasons %s",
            len(holdout_events),
            holdout_seasons,
        )

        results = []
        bankroll = self.settings.INITIAL_BANKROLL
        bankroll_series = [bankroll]
        cache_predictions = {}
        cache_event_meta = {}
        cache_h2h = {}
        cache_top5 = {}
        cache_top10 = {}
        cache_top20 = {}
        cache_make_cut = {}

        for idx, event_row in holdout_events.iterrows():
            event_id = int(event_row["event_id"])
            event_date = event_row["start_date"]
            event_name = event_row.get("event_name", f"Event {event_id}")
            event_year = int(event_date.year)
            # DataGolf reuses event_id across annual editions — every cache
            # entry must be keyed by (event_id, year) to keep editions apart.
            event_key = (event_id, event_year)

            # --- CRITICAL: Training data = only events BEFORE this one ---
            round_dates = pd.to_datetime(rounds_df["date"])
            train_mask = round_dates < event_date
            train_rounds = rounds_df[train_mask]

            # Event-specific data: same event_id AND this edition's dates.
            # Rounds are stamped with the event completion date, which is
            # always within ~2 weeks after the start date. Without the date
            # window, recurring event_ids merge multiple years' editions
            # (wrong field, wrong winner).
            event_mask = (
                (rounds_df["event_id"] == event_id)
                & (round_dates >= event_date)
                & (round_dates <= event_date + pd.Timedelta(days=14))
            )
            event_rounds = rounds_df[event_mask]
            if len(odds_df) > 0:
                odds_mask = odds_df["event_id"] == event_id
                if "calendar_year" in odds_df.columns:
                    odds_mask &= odds_df["calendar_year"] == event_year
                event_odds = odds_df[odds_mask]
            else:
                event_odds = pd.DataFrame()

            if len(event_rounds) == 0:
                logger.debug("Skipping event %d (no round data)", event_id)
                continue

            # Determine actual winner
            winner_id = self._get_winner(event_rounds)
            if winner_id is None:
                logger.debug("Skipping event %d (no winner found)", event_id)
                continue

            n_players = event_rounds["player_id"].nunique()

            # --- Call user's model ---
            event_info = {
                "event_id": event_id,
                "event_name": event_name,
                "date": str(event_date.date()),
                "n_players": n_players,
            }

            try:
                sim_output = fit_and_predict_fn(
                    train_rounds, event_info, event_rounds
                )
            except Exception as e:
                logger.error("Model failed for event %d: %s", event_id, e)
                continue

            # Handle both SimulationResult and plain dict returns
            from simulation.monte_carlo import SimulationResult as _SimResult
            if isinstance(sim_output, _SimResult):
                model_probs = sim_output.win_probs
                sim_result = sim_output
            else:
                model_probs = sim_output
                sim_result = None

            if not model_probs:
                continue

            # Accumulate for cache (keyed by (event_id, year) — see BacktestCache)
            cache_predictions[event_key] = model_probs
            cache_event_meta[event_key] = {
                "event_name": event_name,
                "date": str(event_date.date()),
                "calendar_year": event_year,
                "n_players": n_players,
                "winner_id": winner_id,
            }

            # Cache extended sim results if available
            if sim_result is not None:
                if sim_result.h2h_probs:
                    cache_h2h[event_key] = sim_result.h2h_probs
                cache_top5[event_key] = sim_result.top5_probs
                cache_top10[event_key] = sim_result.top10_probs
                cache_top20[event_key] = sim_result.top20_probs
                cache_make_cut[event_key] = sim_result.make_cut_probs

            # Sharpen model probs (cache stores raw; scoring uses sharpened)
            model_probs = self._sharpen_probs(model_probs)

            # Get market probabilities from odds
            market_probs = self._get_market_probs(event_odds)

            # Compute per-event metrics
            bt_event = self._score_event(
                event_id=event_id,
                event_name=event_name,
                date=str(event_date.date()),
                n_players=n_players,
                model_probs=model_probs,
                market_probs=market_probs,
                winner_id=winner_id,
            )

            # Compute betting P&L using EdgeDetector + KellyCalculator
            if model_probs and not event_odds.empty:
                betting = self._compute_betting_pnl(
                    model_probs=model_probs,
                    event_odds=event_odds,
                    winner_id=winner_id,
                    bankroll=bankroll,
                )
                bt_event.n_bets = betting["n_bets"]
                bt_event.stake_total = betting["stake_total"]
                bt_event.pnl = betting["pnl"]

            results.append(bt_event)

            # Bankroll tracking
            bankroll += bt_event.pnl
            bankroll_series.append(bankroll)

            logger.debug(
                "Event %s: model_brier=%.4f, market_brier=%.4f, pnl=$%.2f",
                event_name[:30],
                bt_event.model_brier,
                bt_event.market_brier,
                bt_event.pnl,
            )

        # --- Save cache if requested ---
        if cache_path and cache_predictions:
            model_params = {
                "EWMA_HALF_LIFE_ROUNDS": self.settings.EWMA_HALF_LIFE_ROUNDS,
                "EWMA_HALF_LIFE_DAYS": self.settings.EWMA_HALF_LIFE_DAYS,
                "OBSERVATION_DF": self.settings.OBSERVATION_DF,
                "ROUND_CORRELATION_RHO": self.settings.ROUND_CORRELATION_RHO,
            }
            cache = BacktestCache(
                created_at=datetime.now().isoformat(),
                model_description=model_description,
                holdout_seasons=holdout_seasons,
                n_events=len(cache_predictions),
                model_params=model_params,
                predictions=cache_predictions,
                event_metadata=cache_event_meta,
                h2h_predictions=cache_h2h if cache_h2h else None,
                top5_predictions=cache_top5 if cache_top5 else None,
                top10_predictions=cache_top10 if cache_top10 else None,
                top20_predictions=cache_top20 if cache_top20 else None,
                make_cut_predictions=cache_make_cut if cache_make_cut else None,
            )
            cache.save(Path(cache_path))

        # --- Aggregate results ---
        result = self._aggregate_results(results, bankroll_series)

        logger.info(
            "Backtest complete | %d events | "
            "Model Brier=%.4f vs Market=%.4f | "
            "ROI=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            result.total_events,
            result.model_avg_brier,
            result.market_avg_brier,
            result.roi_pct,
            result.sharpe,
            result.max_dd_pct,
        )

        return result

    def evaluate(
        self,
        cache: BacktestCache,
        events_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Re-evaluate betting using cached model predictions.

        Skips the expensive fit_and_predict step entirely. Only re-runs
        devigging, scoring, and betting with the current settings. Use this
        to iterate on OVERROUND_METHOD, MIN_EDGE_THRESHOLD, MIN_BET_PROBABILITY,
        KELLY_FRACTION, etc. in seconds instead of minutes.
        """
        # Warn if model params differ from cache
        current_params = {
            "EWMA_HALF_LIFE_ROUNDS": self.settings.EWMA_HALF_LIFE_ROUNDS,
            "EWMA_HALF_LIFE_DAYS": self.settings.EWMA_HALF_LIFE_DAYS,
            "OBSERVATION_DF": self.settings.OBSERVATION_DF,
            "ROUND_CORRELATION_RHO": self.settings.ROUND_CORRELATION_RHO,
        }
        for k, v in current_params.items():
            cached_v = cache.model_params.get(k)
            if cached_v is not None and cached_v != v:
                logger.warning(
                    "Model param %s changed: cached=%s, current=%s — "
                    "cache may be stale, consider re-running run()",
                    k, cached_v, v,
                )

        # Sort events by date
        sorted_events = sorted(
            cache.event_metadata.items(),
            key=lambda x: x[1]["date"],
        )

        logger.info(
            "Evaluating %d cached events with current betting settings",
            len(sorted_events),
        )

        results = []
        bankroll = self.settings.INITIAL_BANKROLL
        bankroll_series = [bankroll]

        for event_key, meta in sorted_events:
            event_id, event_year = event_key
            model_probs = self._sharpen_probs(cache.predictions.get(event_key, {}))
            if not model_probs:
                continue

            event_name = meta["event_name"]
            date = meta["date"]
            n_players = meta["n_players"]
            winner_id = meta["winner_id"]

            # Get event odds — must match BOTH event_id and edition year
            if len(odds_df) > 0:
                odds_mask = odds_df["event_id"] == event_id
                if "calendar_year" in odds_df.columns:
                    odds_mask &= odds_df["calendar_year"] == event_year
                event_odds = odds_df[odds_mask]
            else:
                event_odds = pd.DataFrame()

            market_probs = self._get_market_probs(event_odds)

            bt_event = self._score_event(
                event_id=event_id,
                event_name=event_name,
                date=date,
                n_players=n_players,
                model_probs=model_probs,
                market_probs=market_probs,
                winner_id=winner_id,
            )

            if model_probs and not event_odds.empty:
                betting = self._compute_betting_pnl(
                    model_probs=model_probs,
                    event_odds=event_odds,
                    winner_id=winner_id,
                    bankroll=bankroll,
                )
                bt_event.n_bets = betting["n_bets"]
                bt_event.stake_total = betting["stake_total"]
                bt_event.pnl = betting["pnl"]

            results.append(bt_event)
            bankroll += bt_event.pnl
            bankroll_series.append(bankroll)

        result = self._aggregate_results(results, bankroll_series)

        logger.info(
            "Evaluate complete | %d events | "
            "Model Brier=%.4f vs Market=%.4f | "
            "ROI=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            result.total_events,
            result.model_avg_brier,
            result.market_avg_brier,
            result.roi_pct,
            result.sharpe,
            result.max_dd_pct,
        )

        return result

    def evaluate_matchups(
        self,
        cache: "BacktestCache",
        matchup_odds_df: pd.DataFrame,
        bet_type: str = "72-hole Match",
    ) -> Dict:
        """
        Backtest H2H matchup betting using cached simulation results.

        For each matchup offered by the market:
          1. Look up model P(A beats B) from cached h2h_probs.
          2. Compare to market implied probability from closing odds.
          3. If edge > threshold, bet on the side with value.
          4. Settle against actual outcome.

        Parameters
        ----------
        cache : BacktestCache
            Must have h2h_predictions (V2 cache).
        matchup_odds_df : pd.DataFrame
            Matchup odds with columns: event_id, p1_dg_id, p2_dg_id,
            p1_close, p2_close, p1_outcome, bet_type, etc.
        bet_type : str
            Filter to this bet_type (default "72-hole Match").

        Returns
        -------
        dict with summary stats and per-bet details.
        """
        if not cache.h2h_predictions:
            raise ValueError("Cache has no h2h_predictions. Re-run backtest with compute_h2h=True.")

        min_edge = getattr(self.settings, "MATCHUP_MIN_EDGE", 0.05)
        kelly_frac = getattr(self.settings, "KELLY_FRACTION", 0.25)
        bankroll = self.settings.INITIAL_BANKROLL

        # Filter to bet_type
        odds = matchup_odds_df[matchup_odds_df["bet_type"] == bet_type].copy()
        logger.info("Evaluating %d %s matchups", len(odds), bet_type)

        bets = []
        bankroll_series = [bankroll]

        sorted_events = sorted(
            cache.event_metadata.items(),
            key=lambda x: x[1]["date"],
        )

        for event_key, meta in sorted_events:
            event_id, event_year = event_key
            h2h = cache.h2h_predictions.get(event_key, {})
            if not h2h:
                continue

            # Get matchup odds for this event — must match BOTH event_id and
            # edition year. Matching on event_id alone evaluates one year's
            # model (trained on later data) against another year's odds,
            # which is lookahead bias.
            odds_mask = odds["event_id"] == event_id
            if "calendar_year" in odds.columns:
                odds_mask &= odds["calendar_year"] == event_year
            event_odds = odds[odds_mask]
            if event_odds.empty:
                continue

            event_pnl = 0.0
            event_bets = 0

            for _, row in event_odds.iterrows():
                p1_id = int(row["p1_dg_id"])
                p2_id = int(row["p2_dg_id"])

                # Get model probability
                model_p1 = None
                if p1_id in h2h and p2_id in h2h[p1_id]:
                    model_p1 = h2h[p1_id][p2_id]
                elif p2_id in h2h and p1_id in h2h[p2_id]:
                    model_p1 = 1.0 - h2h[p2_id][p1_id]

                if model_p1 is None:
                    continue

                model_p2 = 1.0 - model_p1

                # Market implied probabilities (from decimal odds, devigged)
                try:
                    p1_odds = float(row["p1_close"])
                    p2_odds = float(row["p2_close"])
                except (ValueError, TypeError):
                    continue
                if p1_odds <= 1.0 or p2_odds <= 1.0:
                    continue

                market_raw_p1 = 1.0 / p1_odds
                market_raw_p2 = 1.0 / p2_odds
                overround = market_raw_p1 + market_raw_p2
                market_p1 = market_raw_p1 / overround
                market_p2 = market_raw_p2 / overround

                # Check for edge on either side
                edge_p1 = model_p1 - market_p1
                edge_p2 = model_p2 - market_p2

                # Actual outcome
                outcome = row.get("p1_outcome")
                if pd.isna(outcome) or outcome == 0.5:  # tie = void
                    continue

                bet_side = None
                if edge_p1 > min_edge:
                    bet_side = "p1"
                    edge = edge_p1
                    model_p = model_p1
                    decimal_odds = p1_odds
                    won = (outcome == 1.0)
                elif edge_p2 > min_edge:
                    bet_side = "p2"
                    edge = edge_p2
                    model_p = model_p2
                    decimal_odds = p2_odds
                    won = (outcome == 0.0)

                if bet_side is None:
                    continue

                # Kelly sizing
                q = 1.0 - model_p
                b = decimal_odds - 1.0
                kelly_full = (model_p * b - q) / b if b > 0 else 0
                kelly_full = max(kelly_full, 0)
                stake_frac = kelly_full * kelly_frac
                stake_frac = min(stake_frac, self.settings.MATCHUP_MAX_BET_PCT)
                stake = stake_frac * bankroll

                if stake < 1.0:  # Min $1 bet
                    continue

                pnl = stake * (decimal_odds - 1.0) if won else -stake
                event_pnl += pnl
                event_bets += 1

                bets.append({
                    "event_id": event_id,
                    "event_name": meta["event_name"],
                    "date": meta["date"],
                    "p1_id": p1_id,
                    "p2_id": p2_id,
                    "p1_name": row.get("p1_player_name", ""),
                    "p2_name": row.get("p2_player_name", ""),
                    "bet_side": bet_side,
                    "model_p": round(model_p, 4),
                    "market_p": round(market_p1 if bet_side == "p1" else market_p2, 4),
                    "edge": round(edge, 4),
                    "decimal_odds": decimal_odds,
                    "stake": round(stake, 2),
                    "won": won,
                    "pnl": round(pnl, 2),
                })

            bankroll += event_pnl
            bankroll_series.append(bankroll)

        # Aggregate results
        bets_df = pd.DataFrame(bets) if bets else pd.DataFrame()
        total_staked = bets_df["stake"].sum() if len(bets_df) > 0 else 0
        total_pnl = bets_df["pnl"].sum() if len(bets_df) > 0 else 0
        n_won = bets_df["won"].sum() if len(bets_df) > 0 else 0
        n_bets = len(bets_df)

        br = np.array(bankroll_series)
        _, dd_pct = max_drawdown(br) if len(br) > 1 else (0, 0)

        # Sharpe from per-event returns
        event_returns = []
        if len(bets_df) > 0:
            for _, g in bets_df.groupby("event_id"):
                ev_staked = g["stake"].sum()
                if ev_staked > 0:
                    event_returns.append(g["pnl"].sum() / ev_staked)
        event_returns = np.array(event_returns) if event_returns else np.array([0.0])
        sharpe = round(sharpe_ratio(event_returns, annualization_factor=40), 2)

        result = {
            "n_bets": n_bets,
            "n_won": int(n_won),
            "win_rate": round(n_won / n_bets * 100, 1) if n_bets > 0 else 0,
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / total_staked * 100, 1) if total_staked > 0 else 0,
            "sharpe": sharpe,
            "max_dd_pct": round(dd_pct * 100, 1),
            "final_bankroll": round(bankroll, 2),
            "n_events_with_bets": bets_df["event_id"].nunique() if len(bets_df) > 0 else 0,
            "bets_df": bets_df,
        }

        logger.info(
            "Matchup backtest | %d bets | Win rate=%.1f%% | "
            "ROI=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            n_bets, result["win_rate"], result["roi_pct"],
            sharpe, result["max_dd_pct"],
        )

        return result

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _get_winner(self, event_rounds: pd.DataFrame) -> Optional[int]:
        """Determine the winner of a tournament from round data.

        Preference order:
        1. fin_text == "1" (DataGolf's official finish; playoff winners
           get "1", playoff losers get "T2"/"2" — so this respects playoffs).
        2. finish_position == 1 (alternative schema).
        3. Lowest summed round_score among players with the most rounds
           played (ignores playoffs, picks the 72-hole leader).
        4. Highest summed sg_total (last resort, same caveat).
        """
        if "fin_text" in event_rounds.columns:
            winners = event_rounds[
                event_rounds["fin_text"].astype(str).str.strip().isin(["1", "T1"])
            ]
            if len(winners) > 0:
                return int(winners["player_id"].iloc[0])

        if "finish_position" in event_rounds.columns:
            winners = event_rounds[
                event_rounds["finish_position"].astype(str).isin(["1", "1.0", "T1"])
            ]
            if len(winners) > 0:
                return int(winners["player_id"].iloc[0])

        # Fallback: lowest total strokes among players who completed all
        # rounds (cut-missers play fewer rounds and would otherwise "win"
        # on raw stroke sums)
        if "round_score" in event_rounds.columns:
            counts = event_rounds.groupby("player_id")["round_score"].count()
            finishers = counts[counts == counts.max()].index
            totals = (
                event_rounds[event_rounds["player_id"].isin(finishers)]
                .groupby("player_id")["round_score"].sum()
            )
            if len(totals) > 0:
                return int(totals.idxmin())

        if "score" in event_rounds.columns:
            totals = event_rounds.groupby("player_id")["score"].sum()
            if len(totals) > 0:
                return int(totals.idxmin())

        # Fallback: highest total SG
        if "sg_total" in event_rounds.columns:
            totals = event_rounds.groupby("player_id")["sg_total"].sum()
            if len(totals) > 0:
                return int(totals.idxmax())

        return None

    def _get_market_probs(self, odds_df: pd.DataFrame) -> Dict[int, float]:
        """Extract market probabilities from odds data."""
        if odds_df.empty:
            return {}

        # Prefer Pinnacle as benchmark
        from betting.odds_processing import process_tournament_odds

        try:
            devigged = process_tournament_odds(
                odds_df,
                book=self.settings.PRIMARY_BOOK,
                method=self.settings.OVERROUND_METHOD,
            )
            if len(devigged) > 0:
                return dict(zip(
                    devigged["player_id"].astype(int),
                    devigged["true_prob"].astype(float),
                ))
        except Exception as e:
            logger.debug("Could not process market odds: %s", e)

        return {}

    def _score_event(
        self,
        event_id: int,
        event_name: str,
        date: str,
        n_players: int,
        model_probs: Dict[int, float],
        market_probs: Dict[int, float],
        winner_id: int,
    ) -> BacktestEvent:
        """Compute metrics for a single backtested event."""
        # Build aligned arrays for scoring — use intersection so neither side
        # gets penalized with 0.0 for players it simply doesn't cover.
        all_players = set(model_probs.keys())
        if market_probs:
            all_players &= set(market_probs.keys())

        model_p = []
        market_p = []
        outcomes = []

        for pid in all_players:
            mp = model_probs.get(pid, 0.0)
            mkp = market_probs.get(pid, 0.0)
            won = 1.0 if pid == winner_id else 0.0

            model_p.append(mp)
            market_p.append(mkp)
            outcomes.append(won)

        model_p = np.array(model_p)
        market_p = np.array(market_p)
        outcomes = np.array(outcomes)

        # Normalize probabilities
        if model_p.sum() > 0:
            model_p = model_p / model_p.sum()
        if market_p.sum() > 0:
            market_p = market_p / market_p.sum()

        # Compute metrics
        m_brier = brier_score(model_p, outcomes)
        mk_brier = brier_score(market_p, outcomes) if market_p.sum() > 0 else 0
        m_ll = log_loss(model_p, outcomes)
        mk_ll = log_loss(market_p, outcomes) if market_p.sum() > 0 else 0

        return BacktestEvent(
            event_id=event_id,
            event_name=event_name,
            date=date,
            n_players=n_players,
            model_probs=model_probs,
            market_probs=market_probs,
            winner_id=winner_id,
            model_brier=m_brier,
            market_brier=mk_brier,
            model_logloss=m_ll,
            market_logloss=mk_ll,
        )

    def _compute_betting_pnl(
        self,
        model_probs: Dict[int, float],
        event_odds: pd.DataFrame,
        winner_id: int,
        bankroll: float,
    ) -> Dict:
        """
        Run the full betting pipeline for one event and return P&L.

        Uses Pinnacle closing odds (PRIMARY_BOOK) for edge detection and
        Kelly sizing. P&L is settled against the actual winner.
        """
        from betting.edge_detection import EdgeDetector
        from betting.kelly import KellyCalculator
        from betting.odds_processing import process_tournament_odds

        try:
            market_df = process_tournament_odds(
                event_odds,
                book=self.settings.PRIMARY_BOOK,
                method=self.settings.OVERROUND_METHOD,
            )
            if market_df.empty:
                return {"n_bets": 0, "stake_total": 0.0, "pnl": 0.0}

            opportunities = EdgeDetector(self.settings).find_edges(
                model_probs, market_df
            )
            if not opportunities:
                return {"n_bets": 0, "stake_total": 0.0, "pnl": 0.0}

            bets = KellyCalculator(self.settings, bankroll=bankroll).size_bets(
                opportunities
            )

            stake_total = sum(b.stake_dollars for b in bets)
            pnl = sum(
                b.stake_dollars * (b.decimal_odds - 1.0) if b.player_id == winner_id
                else -b.stake_dollars
                for b in bets
            )

            return {"n_bets": len(bets), "stake_total": stake_total, "pnl": pnl}

        except Exception as e:
            logger.debug("Betting P&L computation failed for event: %s", e)
            return {"n_bets": 0, "stake_total": 0.0, "pnl": 0.0}

    def _aggregate_results(
        self,
        events: List[BacktestEvent],
        bankroll_series: List[float],
    ) -> BacktestResult:
        """Aggregate per-event results into overall backtest result."""
        if not events:
            return BacktestResult()

        model_briers = np.array([e.model_brier for e in events])
        market_briers = np.array([e.market_brier for e in events if e.market_brier > 0])
        model_lls = np.array([e.model_logloss for e in events])
        market_lls = np.array([e.market_logloss for e in events if e.market_logloss > 0])

        total_pnl = sum(e.pnl for e in events)
        total_staked = sum(e.stake_total for e in events)

        br = np.array(bankroll_series)
        dd_dollar, dd_pct = max_drawdown(br) if len(br) > 1 else (0, 0)

        bet_returns = []
        for e in events:
            if e.stake_total > 0:
                bet_returns.append(e.pnl / e.stake_total)
        bet_returns = np.array(bet_returns) if bet_returns else np.array([0.0])

        result = BacktestResult(
            events=events,
            total_events=len(events),
            model_avg_brier=float(np.mean(model_briers)),
            market_avg_brier=float(np.mean(market_briers)) if len(market_briers) > 0 else 0,
            model_avg_logloss=float(np.mean(model_lls)),
            market_avg_logloss=float(np.mean(market_lls)) if len(market_lls) > 0 else 0,
            total_bets=sum(e.n_bets for e in events),
            total_staked=total_staked,
            total_pnl=total_pnl,
            roi_pct=round(roi(total_pnl, total_staked) * 100, 2) if total_staked > 0 else 0,
            sharpe=round(sharpe_ratio(bet_returns, annualization_factor=40), 2),  # ~40 PGA events/year
            max_dd_pct=round(dd_pct * 100, 1),
        )

        # Gate verdicts
        result.gate_1_passed = result.model_avg_brier < result.market_avg_brier

        # Gate 2: Diebold-Mariano test — model significantly better than market
        paired_events = [e for e in events if e.market_brier > 0]
        if len(paired_events) >= 10:
            model_losses = np.array([e.model_brier for e in paired_events])
            market_losses = np.array([e.market_brier for e in paired_events])
            dm_result = diebold_mariano_test(model_losses, market_losses)
            result.gate_2_passed = dm_result["p_value"] < self.settings.GATE_DM_PVALUE
        else:
            result.gate_2_passed = False

        result.gate_3_passed = (
            result.roi_pct > 0 and
            result.sharpe > self.settings.GATE_SHARPE_MIN and
            dd_pct < self.settings.GATE_MAX_DRAWDOWN_PCT
        )

        return result

    def results_to_dataframe(self, result: BacktestResult) -> pd.DataFrame:
        """Convert backtest results to a per-event DataFrame."""
        records = []
        for e in result.events:
            records.append({
                "event_id": e.event_id,
                "event_name": e.event_name,
                "date": e.date,
                "n_players": e.n_players,
                "model_brier": round(e.model_brier, 5),
                "market_brier": round(e.market_brier, 5),
                "brier_diff": round(e.model_brier - e.market_brier, 5),
                "model_logloss": round(e.model_logloss, 4),
                "market_logloss": round(e.market_logloss, 4),
                "n_bets": e.n_bets,
                "pnl": round(e.pnl, 2),
            })
        return pd.DataFrame(records)
