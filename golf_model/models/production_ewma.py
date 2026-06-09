# ==============================================================================
# golf_model/models/production_ewma.py
# ==============================================================================
#
# PRODUCTION EWMA MODEL
# ----------------------
# The model that actually generates backtest (notebook 08) and live
# (notebook 09) predictions. Extracted from the notebooks so both run
# provably identical code and all hyperparameters live in settings.py.
#
# Honest description (for decks and docs): this is NOT the PyMC
# hierarchical Bayesian model from notebook 04. It is an empirical-Bayes
# flavored heuristic stack:
#
#   1. Dual-decay EWMA skill estimates (rounds + calendar days)
#   2. Course-specific SG component re-weighting (50% blend, from the
#      historical variance of SG components at this event)
#   3. Recent-form blend (40% of last 8 rounds)
#   4. Event-history course-fit with normal-conjugate shrinkage (τ=0.50)
#   5. Student-t Monte Carlo tournament simulation (cut, playoff, H2H)
#
# Temporal integrity: every estimate is computed from `train_rounds`
# only, with the EWMA additionally filtered to dates <= event_date.
# The BacktestEngine guarantees train_rounds predates the target event.
#
# Sigma convention: sigmas passed to the simulator are EMPIRICAL
# round-to-round SDs; MonteCarloSimulator applies the Student-t scale
# correction internally.
#
# Usage (backtest):
#   model = ProductionEWMAModel(settings)
#   result = engine.run(..., fit_and_predict_fn=model.fit_and_predict, ...)
#
# Usage (live):
#   result = model.predict(rounds_df, event_id, event_name,
#                          pd.Timestamp.now(), player_ids)
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import Settings
from features.time_weighting import TimeWeighter
from simulation.monte_carlo import MonteCarloSimulator, PlayerAbility, SimulationResult
from simulation.tournament import TournamentConfig
from utils.logger import get_logger

logger = get_logger(__name__)

_SG_SUB_COLS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
_SG_EWMA_COLS = ["ewma_sg_ott", "ewma_sg_app", "ewma_sg_arg", "ewma_sg_putt"]
_TRAIN_COLS = ["player_id", "date", "event_id", "sg_total"] + _SG_SUB_COLS

# Fallback population round-to-round SD when no within-tournament
# variance can be estimated from the data.
_DEFAULT_POP_SIGMA = 2.75


class ProductionEWMAModel:
    """
    EWMA skill + course weighting + recent form + course-fit → MC simulation.

    Parameters
    ----------
    settings : Settings
        Provides all hyperparameters (COURSE_SG_BLEND, RECENT_FORM_BLEND,
        COURSE_FIT_SHRINKAGE, MU_STD_CAP, PRODUCTION_N_SIMULATIONS, ...).
    n_simulations : int, optional
        Override settings.PRODUCTION_N_SIMULATIONS.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        n_simulations: Optional[int] = None,
    ):
        self.cfg = settings or Settings()
        self.n_simulations = n_simulations or self.cfg.PRODUCTION_N_SIMULATIONS

        self.weighter = TimeWeighter(self.cfg)
        self.sim = MonteCarloSimulator(self.cfg)

        self.course_blend = self.cfg.COURSE_SG_BLEND
        self.min_course_rounds = self.cfg.MIN_COURSE_ROUNDS_FOR_WEIGHTS
        self.recent_blend = self.cfg.RECENT_FORM_BLEND
        self.recent_rounds = self.cfg.RECENT_FORM_ROUNDS
        self.recent_min_rounds = self.cfg.RECENT_FORM_MIN_ROUNDS
        self.tau_course_fit = self.cfg.COURSE_FIT_SHRINKAGE
        self.mu_std_cap = self.cfg.MU_STD_CAP

        logger.info(
            "ProductionEWMAModel | ewma=%sr/%sd | course_blend=%.2f | "
            "recent=%.2f×last%d | τ_cf=%.2f | n_sims=%s",
            self.cfg.EWMA_HALF_LIFE_ROUNDS, self.cfg.EWMA_HALF_LIFE_DAYS,
            self.course_blend, self.recent_blend, self.recent_rounds,
            self.tau_course_fit, f"{self.n_simulations:,}",
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def fit_and_predict(
        self,
        train_rounds: pd.DataFrame,
        event_info: Dict,
        event_rounds: pd.DataFrame,
    ) -> Union[SimulationResult, Dict]:
        """Adapter matching BacktestEngine's fit_and_predict callback."""
        player_ids = (
            event_rounds["player_id"].dropna().astype(int).unique().tolist()
        )
        return self.predict(
            train_rounds=train_rounds,
            event_id=int(event_info["event_id"]),
            event_name=str(event_info["event_name"]),
            event_date=pd.Timestamp(event_info["date"]),
            player_ids=player_ids,
        )

    def predict(
        self,
        train_rounds: pd.DataFrame,
        event_id: int,
        event_name: str,
        event_date: pd.Timestamp,
        player_ids: List[int],
        compute_h2h: bool = True,
    ) -> Union[SimulationResult, Dict]:
        """
        Build the field's ability estimates from train_rounds and simulate.

        Returns a SimulationResult, or {} when there is no usable data.
        """
        if len(train_rounds) == 0 or not player_ids:
            return {}

        available = [c for c in _TRAIN_COLS if c in train_rounds.columns]
        train = train_rounds[available].copy()
        train["date"] = pd.to_datetime(train["date"])
        train = train.dropna(subset=["date"])

        player_features = self.weighter.compute_weighted_sg(
            train, as_of_date=event_date
        )
        if player_features.empty:
            return {}

        feat_index = player_features.set_index("player_id")
        pop_mu = float(player_features["ewma_sg_total"].mean())

        # Population sigma (empirical SD — simulator applies the t-scale)
        if "ewma_sg_total_within_var" in player_features.columns:
            pop_within_var = player_features["ewma_sg_total_within_var"].dropna()
            pop_sigma = (
                float(np.sqrt(pop_within_var.median()))
                if len(pop_within_var) > 0 else _DEFAULT_POP_SIGMA
            )
        else:
            pop_sigma = _DEFAULT_POP_SIGMA

        # --- Course-specific SG weights ---
        has_sub = all(c in train.columns for c in _SG_SUB_COLS)
        course_weights = (
            self._compute_course_sg_weights(train, event_id) if has_sub else None
        )
        has_ewma_sub = all(c in feat_index.columns for c in _SG_EWMA_COLS)

        # --- Recent form ---
        recent_form = (
            train[train["player_id"].isin(player_ids)]
            .sort_values("date")
            .groupby("player_id")
            .tail(self.recent_rounds)
            .groupby("player_id")["sg_total"]
            .agg(["mean", "count"])
        )

        # --- Course-fit (event-history shrinkage) ---
        course_fits = self._compute_course_fit(
            train, event_id, player_ids, feat_index, pop_sigma
        )

        field = []
        mu_means_debug = []
        cf_means_debug = []
        n_course_adj = 0

        for pid in player_ids:
            cf_mean, cf_std = course_fits.get(pid, (0.0, 0.0))

            if pid in feat_index.index:
                row = feat_index.loc[pid]
                mu_mean = float(row["ewma_sg_total"])

                # Step 1: Course-specific SG weighting (with NaN guard)
                if course_weights is not None and has_ewma_sub:
                    try:
                        sub_skills = np.array(
                            [float(row[c]) for c in _SG_EWMA_COLS]
                        )
                        if not np.any(np.isnan(sub_skills)):
                            weights = np.array(
                                [float(course_weights[c]) for c in _SG_SUB_COLS]
                            )
                            mu_course = 4.0 * float(np.dot(weights, sub_skills))
                            if np.isfinite(mu_course):
                                mu_mean = (
                                    (1 - self.course_blend) * mu_mean
                                    + self.course_blend * mu_course
                                )
                                n_course_adj += 1
                    except (KeyError, TypeError):
                        pass

                # Step 2: Recent form adjustment
                if pid in recent_form.index:
                    rf = recent_form.loc[pid]
                    if int(rf["count"]) >= self.recent_min_rounds:
                        recent_mu = float(rf["mean"])
                        if np.isfinite(recent_mu):
                            mu_mean = (
                                (1 - self.recent_blend) * mu_mean
                                + self.recent_blend * recent_mu
                            )

                sg_var = max(float(row.get("ewma_sg_total_var", 1.0)), 0.25)
                ess = max(float(row.get("effective_sample_size", 5.0)), 1.0)

                within_var = row.get("ewma_sg_total_within_var", np.nan)
                if pd.notna(within_var) and within_var > 0.1:
                    sigma = float(np.sqrt(within_var))
                else:
                    sigma = pop_sigma

                mu_std = min(float(np.sqrt(sg_var / ess)), self.mu_std_cap)

                # Final NaN guard: fall back to pop_mu if mu_mean is corrupted
                if not np.isfinite(mu_mean):
                    mu_mean = pop_mu
                    mu_std = self.mu_std_cap

                mu_means_debug.append(mu_mean)
                cf_means_debug.append(cf_mean)
            else:
                mu_mean = pop_mu
                mu_std = self.mu_std_cap
                sigma = pop_sigma

            field.append(PlayerAbility(
                player_id=pid,
                mu_mean=mu_mean,
                mu_std=mu_std,
                sigma=sigma,
                course_fit_mean=cf_mean,
                course_fit_std=cf_std,
            ))

        if not field:
            return {}

        if mu_means_debug:
            mu_arr = np.array(mu_means_debug)
            cf_arr = np.array(cf_means_debug)
            n_with_cf = int(np.sum(np.abs(cf_arr) > 0.01))
            logger.info(
                "[%s] mu: [%.2f, %.2f] std=%.3f | course_weights=%s adj=%d | "
                "cf=%d/%d",
                event_name[:30], np.nanmin(mu_arr), np.nanmax(mu_arr),
                np.nanstd(mu_arr),
                "yes" if course_weights is not None else "no",
                n_course_adj, n_with_cf, len(cf_arr),
            )

        tourney = TournamentConfig(
            event_id=event_id,
            event_name=event_name,
            field_size=len(field),
            cut_rule=self.cfg.CUT_RULE,
        )

        return self.sim.simulate_tournament(
            field, tourney,
            n_simulations=self.n_simulations,
            compute_h2h=compute_h2h,
        )

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _compute_course_sg_weights(
        self,
        train_rounds: pd.DataFrame,
        event_id: int,
    ) -> Optional[pd.Series]:
        """Course-specific SG component weights from historical variance.

        Uses only this event's past editions present in train_rounds
        (temporally filtered upstream).
        """
        if "event_id" not in train_rounds.columns:
            return None
        event_hist = train_rounds[train_rounds["event_id"] == event_id]
        sub_cols_available = [c for c in _SG_SUB_COLS if c in event_hist.columns]
        if len(sub_cols_available) < 4:
            return None
        sub_data = event_hist[sub_cols_available].dropna()
        if len(sub_data) < self.min_course_rounds:
            return None
        variances = sub_data.var()
        total = variances.sum()
        if total < 0.01:
            return None
        return variances / total

    def _compute_course_fit(
        self,
        train_rounds: pd.DataFrame,
        event_id: int,
        player_ids: List[int],
        feat_index: pd.DataFrame,
        pop_sigma: float,
    ) -> Dict[int, Tuple[float, float]]:
        """Bayesian-shrunk course-fit estimates. Returns {pid: (cf_mean, cf_std)}.

        Normal-conjugate posterior on the player's residual (event-history
        mean SG minus current EWMA skill) with prior N(0, τ²).
        """
        tau2 = self.tau_course_fit ** 2
        sigma2 = pop_sigma ** 2

        if "event_id" not in train_rounds.columns:
            return {pid: (0.0, 0.0) for pid in player_ids}
        event_history = train_rounds[train_rounds["event_id"] == event_id]
        if len(event_history) == 0:
            return {pid: (0.0, 0.0) for pid in player_ids}

        player_event_stats = (
            event_history.groupby("player_id")["sg_total"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "event_sg_mean", "count": "n_rounds"})
        )

        result = {}
        for pid in player_ids:
            if pid not in player_event_stats.index:
                result[pid] = (0.0, 0.0)
                continue
            stats = player_event_stats.loc[pid]
            n = int(stats["n_rounds"])
            event_sg_mean = float(stats["event_sg_mean"])
            ewma_mu = (
                float(feat_index.loc[pid]["ewma_sg_total"])
                if pid in feat_index.index else 0.0
            )
            residual = event_sg_mean - ewma_mu
            precision_post = 1.0 / tau2 + n / sigma2
            posterior_mean = (n / sigma2 * residual) / precision_post
            posterior_std = float(np.sqrt(1.0 / precision_post))
            result[pid] = (float(posterior_mean), posterior_std)
        return result
