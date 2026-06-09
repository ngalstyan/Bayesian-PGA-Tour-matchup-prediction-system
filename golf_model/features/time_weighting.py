# ==============================================================================
# golf_model/features/time_weighting.py
# ==============================================================================
#
# EXPONENTIAL WEIGHTED MOVING AVERAGE (EWMA) TIME WEIGHTING
# -----------------------------------------------------------
# Applies recency weighting to player performance data.
#
# Mathematical Foundation:
#   Recent rounds reveal more about current skill than older rounds.
#   We weight observations using exponential decay:
#
#     w_j = exp(-λ · j)
#
#   where:
#     j    = "age" of the observation (rounds ago or days ago)
#     λ    = decay rate = ln(2) / half_life
#     w_j  = weight assigned to observation j
#
#   The half_life parameter controls how quickly old data is forgotten:
#     - half_life = 30 rounds → aggressive recency (volatile estimates)
#     - half_life = 60 rounds → moderate recency (balanced)
#     - half_life = 120 rounds → conservative (stable but slow to adapt)
#
# Dual EWMA (Baker & McHale, 2017):
#   We use TWO weighting schemes simultaneously:
#     1. Round-based: w_j = exp(-λ_r · j_rounds)  
#     2. Calendar-based: w_j = exp(-λ_d · j_days)
#
#   Final weight = geometric mean: w = sqrt(w_rounds × w_days)
#
#   Why dual? Round-based misses gaps (injuries, off-season).
#   Calendar-based misses rhythm (a player with 20 rounds in 30 days
#   has more "recent" data than one with 5 rounds in 30 days).
#
# Usage:
#   from features.time_weighting import TimeWeighter
#   weighter = TimeWeighter(Settings())
#   weighted_sg = weighter.compute_weighted_average(player_rounds_df)
#
# ==============================================================================

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeWeighter:
    """
    Apply EWMA time-weighting to player performance data.
    
    Parameters
    ----------
    settings : Settings
        Provides half-life parameters and SG column names.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.half_life_rounds = self.cfg.EWMA_HALF_LIFE_ROUNDS
        self.half_life_days = self.cfg.EWMA_HALF_LIFE_DAYS
        self.lambda_rounds = self.cfg.ewma_lambda_rounds
        self.lambda_days = self.cfg.ewma_lambda_days

    def compute_weighted_sg(
        self,
        rounds_df: pd.DataFrame,
        as_of_date: Optional[pd.Timestamp] = None,
        method: str = "dual",
    ) -> pd.DataFrame:
        """
        Compute time-weighted SG averages for all players.
        
        For each player, weights their historical rounds by recency
        and computes the weighted mean for each SG component.
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level data with columns:
            [player_id, date, sg_total, sg_ott, sg_app, sg_arg, sg_putt].
            Must be complete historical data (not pre-filtered).
            
        as_of_date : pd.Timestamp, optional
            Reference date for computing recency. If None, uses
            the maximum date in the dataset.
            All rounds AFTER this date are excluded (temporal integrity).
            
        method : str, default "dual"
            Weighting method:
            - "rounds": Weight by round sequence only
            - "days": Weight by calendar days only
            - "dual": Geometric mean of both (recommended)
            
        Returns
        -------
        pd.DataFrame
            One row per player with columns:
            [player_id, n_rounds, ewma_sg_total, ewma_sg_ott, ...,
             ewma_sg_putt, ewma_sg_total_var, ..., effective_sample_size]
             
        Notes
        -----
        The effective sample size (ESS) is computed as:
            ESS = (Σ w_j)² / Σ w_j²
            
        This quantifies how many "equivalent unweighted observations"
        the weighted average represents. Low ESS → high uncertainty.
        """
        df = rounds_df.copy()
        sg_cols = [self.cfg.SG_TOTAL_COL] + self.cfg.SG_COMPONENTS

        # Ensure date column
        if "date" not in df.columns:
            raise ValueError("rounds_df must contain 'date' column")

        df["date"] = pd.to_datetime(df["date"])

        # Temporal filter: only use rounds before as_of_date
        if as_of_date is not None:
            df = df[df["date"] <= as_of_date]

        if df.empty:
            logger.warning("No rounds available for time-weighted computation")
            return pd.DataFrame()

        reference_date = as_of_date or df["date"].max()
        logger.info(
            "Computing %s EWMA (λ_r=%.4f, λ_d=%.4f) as of %s | %d rounds",
            method, self.lambda_rounds, self.lambda_days,
            reference_date.strftime("%Y-%m-%d"), len(df),
        )

        # Sort by date descending (most recent first) within each player
        df = df.sort_values(["player_id", "date"], ascending=[True, False])

        results = []
        for player_id, player_df in df.groupby("player_id"):
            result = self._compute_player_ewma(
                player_df, reference_date, sg_cols, method
            )
            result["player_id"] = player_id
            results.append(result)

        result_df = pd.DataFrame(results)
        logger.info(
            "EWMA computed for %d players | mean ESS: %.1f",
            len(result_df),
            result_df["effective_sample_size"].mean(),
        )

        return result_df

    def _compute_player_ewma(
        self,
        player_df: pd.DataFrame,
        reference_date: pd.Timestamp,
        sg_cols: list,
        method: str,
    ) -> dict:
        """
        Compute EWMA for a single player.
        
        Parameters
        ----------
        player_df : pd.DataFrame
            Rounds for one player, sorted by date descending.
        reference_date : pd.Timestamp
            "Today" for computing recency.
        sg_cols : list
            SG column names to compute weighted averages for.
        method : str
            "rounds", "days", or "dual".
            
        Returns
        -------
        dict
            Weighted averages, variances, and metadata.
        """
        n = len(player_df)
        result = {"n_rounds": n}

        if n == 0:
            for col in sg_cols:
                result[f"ewma_{col}"] = np.nan
                result[f"ewma_{col}_var"] = np.nan
            result["effective_sample_size"] = 0.0
            return result

        # --- Compute weights ---
        if method == "rounds":
            weights = self._round_weights(n)

        elif method == "days":
            days_ago = (reference_date - player_df["date"]).dt.days.values
            weights = self._day_weights(days_ago)

        elif method == "dual":
            # Geometric mean of round-based and day-based weights
            w_rounds = self._round_weights(n)
            days_ago = (reference_date - player_df["date"]).dt.days.values
            w_days = self._day_weights(days_ago)
            weights = np.sqrt(w_rounds * w_days)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'rounds', 'days', or 'dual'.")

        # Normalize weights to sum to 1
        w_sum = weights.sum()
        if w_sum <= 0:
            for col in sg_cols:
                result[f"ewma_{col}"] = np.nan
                result[f"ewma_{col}_var"] = np.nan
            result["effective_sample_size"] = 0.0
            return result

        w_norm = weights / w_sum

        # --- Weighted mean and variance for each SG component ---
        for col in sg_cols:
            values = player_df[col].values

            # Handle NaN values: set their weight to 0
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() == 0:
                result[f"ewma_{col}"] = np.nan
                result[f"ewma_{col}_var"] = np.nan
                continue

            v = values[valid_mask]
            w = w_norm[valid_mask]
            w = w / w.sum()  # Re-normalize after dropping NaNs

            # Weighted mean
            wmean = np.average(v, weights=w)
            result[f"ewma_{col}"] = wmean

            # Weighted variance (Bessel-corrected for weights)
            # Using reliability weights formula:
            #   Var = Σ w_i (x_i - x̄)² / (1 - Σ w_i²)
            if len(v) > 1:
                w_sq_sum = np.sum(w ** 2)
                correction = 1.0 - w_sq_sum
                if correction > 0:
                    wvar = np.sum(w * (v - wmean) ** 2) / correction
                else:
                    wvar = np.var(v)
                result[f"ewma_{col}_var"] = wvar
            else:
                result[f"ewma_{col}_var"] = np.nan

        # --- Within-tournament variance for sg_total ---
        # Measures round-to-round noise WITHIN a single tournament,
        # excluding between-tournament effects (course fit, weather, conditions).
        # This is the correct noise parameter for tournament simulation.
        sg_total_col = sg_cols[0]  # "sg_total"
        if "event_id" in player_df.columns and sg_total_col in player_df.columns:
            within_vars = []
            within_wts = []

            for _, grp in player_df.groupby("event_id"):
                vals = grp[sg_total_col].values
                valid = ~np.isnan(vals)
                if valid.sum() < 2:
                    continue
                v = vals[valid]
                wt_var = np.var(v, ddof=1)

                # Weight = sum of EWMA weights for this tournament's rounds
                grp_positions = [player_df.index.get_loc(idx) for idx in grp.index]
                tournament_weight = weights[grp_positions].sum()

                within_vars.append(wt_var)
                within_wts.append(tournament_weight)

            if within_vars:
                within_vars = np.array(within_vars)
                within_wts = np.array(within_wts)
                ww = within_wts / within_wts.sum()
                result["ewma_sg_total_within_var"] = float(
                    np.average(within_vars, weights=ww)
                )
            else:
                result["ewma_sg_total_within_var"] = np.nan
        else:
            result["ewma_sg_total_within_var"] = np.nan

        # --- Effective sample size ---
        # ESS = (Σ w)² / Σ w²  (Kish, 1965)
        # This tells us: the weighted average has the precision of
        # approximately ESS unweighted observations.
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        result["effective_sample_size"] = ess

        return result

    def _round_weights(self, n: int) -> np.ndarray:
        """
        Compute round-based exponential weights.
        
        Weight for the j-th most recent round:
            w_j = exp(-λ_rounds · j)
        
        NOT normalized (normalization happens after combining with day weights).
        """
        j = np.arange(n, dtype=np.float64)
        return np.exp(-self.lambda_rounds * j)

    def _day_weights(self, days_ago: np.ndarray) -> np.ndarray:
        """
        Compute calendar-day-based exponential weights.
        
        Weight for a round played `d` days ago:
            w = exp(-λ_days · d)
        
        Parameters
        ----------
        days_ago : np.ndarray
            Number of calendar days since each round.
        """
        days = np.asarray(days_ago, dtype=np.float64)
        days = np.maximum(days, 0)  # Ensure non-negative
        return np.exp(-self.lambda_days * days)

    def optimize_half_life(
        self,
        rounds_df: pd.DataFrame,
        half_life_grid: Optional[list] = None,
        metric: str = "rmse",
    ) -> Tuple[float, pd.DataFrame]:
        """
        Cross-validate to find optimal half-life parameter.
        
        Uses expanding-window cross-validation:
            - For each tournament t in order:
                1. Compute EWMA using all data before t
                2. Predict SG for each player in t
                3. Measure prediction error (RMSE or MAE)
            - Average error across all tournaments for each half-life
            - Select half-life with lowest average error
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Full round-level data.
        half_life_grid : list, optional
            Half-life values to test. Default: [20, 30, 40, 50, 60, 80, 100, 120].
        metric : str, default "rmse"
            Error metric: "rmse" or "mae".
            
        Returns
        -------
        Tuple[float, pd.DataFrame]
            (best_half_life, results_df with all tested values and their errors)
            
        Notes
        -----
        This is a proper temporal cross-validation — no future data leakage.
        It directly optimizes the half-life for predictive accuracy.
        """
        if half_life_grid is None:
            half_life_grid = [20, 30, 40, 50, 60, 80, 100, 120]

        df = rounds_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Get unique events in order
        events = (
            df.groupby("event_id")["date"].min()
            .sort_values()
            .reset_index()
        )

        # Skip first N events (need enough history)
        min_history_events = 20
        if len(events) <= min_history_events:
            logger.warning("Not enough events for CV optimization")
            return self.half_life_rounds, pd.DataFrame()

        eval_events = events.iloc[min_history_events:]
        logger.info(
            "Optimizing half-life over %d events with grid %s",
            len(eval_events), half_life_grid,
        )

        results = []
        for hl in half_life_grid:
            # Temporarily override half-life
            original_hl = self.half_life_rounds
            self.half_life_rounds = hl
            self.lambda_rounds = np.log(2) / hl

            errors = []
            for _, event_row in eval_events.iterrows():
                eid = event_row["event_id"]
                event_date = event_row["date"]

                # Get EWMA predictions as of event start
                train = df[df["date"] < event_date]
                if train.empty:
                    continue

                ewma = self.compute_weighted_sg(
                    train, as_of_date=event_date, method="dual"
                )

                # Get actual performance in this event
                actual = (
                    df[df["event_id"] == eid]
                    .groupby("player_id")["sg_total"]
                    .mean()
                    .reset_index()
                    .rename(columns={"sg_total": "actual_sg"})
                )

                # Merge predictions with actuals
                merged = actual.merge(
                    ewma[["player_id", "ewma_sg_total"]],
                    on="player_id",
                    how="inner",
                )

                if merged.empty:
                    continue

                # Compute error
                err = merged["actual_sg"] - merged["ewma_sg_total"]
                if metric == "rmse":
                    errors.append(np.sqrt((err ** 2).mean()))
                elif metric == "mae":
                    errors.append(err.abs().mean())

            avg_error = np.mean(errors) if errors else np.nan
            results.append({
                "half_life_rounds": hl,
                f"avg_{metric}": avg_error,
                "n_eval_events": len(errors),
            })

            logger.info("  half_life=%d → avg_%s=%.4f", hl, metric, avg_error)

            # Restore original
            self.half_life_rounds = original_hl
            self.lambda_rounds = np.log(2) / original_hl

        results_df = pd.DataFrame(results)
        best_row = results_df.loc[results_df[f"avg_{metric}"].idxmin()]
        best_hl = best_row["half_life_rounds"]

        logger.info(
            "Optimal half-life: %.0f rounds (avg_%s=%.4f)",
            best_hl, metric, best_row[f"avg_{metric}"],
        )

        return float(best_hl), results_df
