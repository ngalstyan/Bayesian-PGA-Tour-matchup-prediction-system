# ==============================================================================
# golf_model/simulation/monte_carlo.py
# ==============================================================================
#
# MONTE CARLO TOURNAMENT SIMULATION ENGINE
# ------------------------------------------
# The core engine that converts posterior ability estimates into 
# win probabilities via simulation.
#
# For each simulation iteration:
#   1. Draw player abilities μ_i from posterior distributions.
#   2. Add course-fit adjustment: γ_c · δ_i.
#   3. Simulate 4 rounds with noise: ε ~ t(ν, 0, σ²).
#   4. Apply tournament structure (cuts, playoffs).
#   5. Record the winner.
#
# After N simulations:
#   P(win | player_i) = (# times player_i wins) / N
#
# These probabilities are then compared against market odds to detect edges.
#
# Performance:
#   - Vectorized across players (numpy).
#   - Parallelizable across simulations (numba JIT for inner loops).
#   - 100K simulations with 156 players ≈ 30–60 seconds on modern hardware.
#
# Mathematical framework:
#   Round score for player i in round r:
#     S_{i,r} = μ_i + γ_c · δ_i + ρ · (S_{i,r-1} - μ_i) + ε_{i,r}
#     ε_{i,r} ~ t(ν, 0, σ²_i)
#
#   where:
#     μ_i        = true skill (drawn from posterior)
#     γ_c · δ_i  = course-fit adjustment
#     ρ          = within-tournament round correlation (momentum)
#     ε_{i,r}    = round noise (Student-t for heavy tails)
#     σ²_i       = player-specific variance (some players more volatile)
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config.settings import Settings
from simulation.tournament import (
    TournamentConfig,
    simulate_tournament_outcome,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Added to cut-missers' 36-hole totals when settling H2H matchups, so any
# cut-maker beats any cut-misser while cut-missers keep their 36-hole order.
# Simulated scores are SG-scale (|score| < ~100), so 1e6 can never overlap.
_CUT_MISS_PENALTY = 1e6


@dataclass
class PlayerAbility:
    """
    A player's estimated ability parameters for simulation.
    
    These come from the Bayesian model's posterior distributions.
    
    Attributes
    ----------
    player_id : int
        DataGolf player ID.
    mu_mean : float
        Posterior mean of true skill (SG units/round).
    mu_std : float
        Posterior standard deviation (uncertainty in skill estimate).
    sigma : float
        Round-to-round noise standard deviation.
    course_fit_mean : float
        Expected course-fit adjustment at this specific course.
    course_fit_std : float
        Uncertainty in course-fit estimate.
    """
    player_id: int
    mu_mean: float
    mu_std: float
    sigma: float
    course_fit_mean: float = 0.0
    course_fit_std: float = 0.0


@dataclass
class SimulationResult:
    """
    Container for Monte Carlo simulation outputs.
    
    Attributes
    ----------
    win_probs : dict
        {player_id: P(win)} for all players.
    top5_probs : dict
        {player_id: P(finish top 5)} for all players.
    top10_probs : dict
        {player_id: P(finish top 10)} for all players.
    top20_probs : dict
        {player_id: P(finish top 20)} for all players.
    make_cut_probs : dict
        {player_id: P(make cut)} for all players.
    n_simulations : int
        Number of simulations run.
    convergence_diagnostics : dict
        Information about simulation convergence.
    """
    win_probs: Dict[int, float]
    top5_probs: Dict[int, float]
    top10_probs: Dict[int, float]
    top20_probs: Dict[int, float]
    make_cut_probs: Dict[int, float]
    n_simulations: int
    convergence_diagnostics: Dict
    h2h_probs: Optional[Dict[int, Dict[int, float]]] = None  # {pid_a: {pid_b: P(A beats B)}}


class MonteCarloSimulator:
    """
    Monte Carlo tournament simulation engine.
    
    Simulates complete tournaments by drawing from posterior ability
    distributions, adding noise, applying tournament rules, and 
    counting outcomes across many iterations.
    
    Parameters
    ----------
    settings : Settings
        Project configuration (provides N_SIMULATIONS, noise params, etc.).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.n_simulations = self.settings.N_SIMULATIONS
        self.observation_df = self.settings.OBSERVATION_DF
        self.round_correlation = self.settings.ROUND_CORRELATION_RHO

        # Set up RNG for reproducibility
        self.seed = self.settings.RANDOM_SEED
        self.rng = np.random.default_rng(self.seed)

        logger.info(
            "MonteCarloSimulator initialized | n_sims=%s, ν=%.1f, ρ=%.3f, seed=%s",
            f"{self.n_simulations:,}", self.observation_df,
            self.round_correlation, self.seed,
        )

    def simulate_tournament(
        self,
        player_abilities: List[PlayerAbility],
        tournament_config: TournamentConfig,
        n_simulations: Optional[int] = None,
        compute_h2h: bool = False,
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation of a tournament.
        
        Parameters
        ----------
        player_abilities : list of PlayerAbility
            Posterior ability estimates for each player in the field.
            
        tournament_config : TournamentConfig
            Tournament structure (cuts, field size, etc.).
            
        n_simulations : int, optional
            Override default simulation count.
            
        Returns
        -------
        SimulationResult
            Win probabilities and other market probabilities for all players.
        """
        n_sims = n_simulations or self.n_simulations
        n_players = len(player_abilities)

        logger.info(
            "Simulating %s: %d players × %s iterations",
            tournament_config.event_name,
            n_players,
            f"{n_sims:,}",
        )

        # Extract ability parameters into arrays for vectorized computation
        player_ids = np.array([p.player_id for p in player_abilities])
        mu_means = np.array([p.mu_mean for p in player_abilities])
        mu_stds = np.array([p.mu_std for p in player_abilities])
        sigmas = np.array([p.sigma for p in player_abilities])
        cf_means = np.array([p.course_fit_mean for p in player_abilities])
        cf_stds = np.array([p.course_fit_std for p in player_abilities])

        # Outcome counters
        win_counts = np.zeros(n_players, dtype=np.int64)
        top5_counts = np.zeros(n_players, dtype=np.int64)
        top10_counts = np.zeros(n_players, dtype=np.int64)
        top20_counts = np.zeros(n_players, dtype=np.int64)
        cut_counts = np.zeros(n_players, dtype=np.int64)

        # H2H pairwise counts: h2h_counts[i,j] = # sims where player i beat player j
        h2h_counts = np.zeros((n_players, n_players), dtype=np.int64) if compute_h2h else None

        # --- Main simulation loop ---
        # Process in batches for memory efficiency
        batch_size = min(1000, n_sims)
        n_batches = (n_sims + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            actual_batch = min(batch_size, n_sims - batch_idx * batch_size)

            # Step 1: Draw abilities from posterior
            # μ_i ~ N(μ̂_i, σ̂²_μ_i) for each simulation
            drawn_abilities = self._draw_abilities(
                mu_means, mu_stds, cf_means, cf_stds, actual_batch
            )
            # Shape: (batch_size, n_players)

            # Step 2: Simulate round scores
            round_scores = self._simulate_rounds(
                drawn_abilities, sigmas, actual_batch, n_players
            )
            # Shape: (batch_size, n_players, 4)

            # Step 3: Process each simulated tournament
            for sim_idx in range(actual_batch):
                outcome = simulate_tournament_outcome(
                    round_scores[sim_idx],  # (n_players, 4)
                    tournament_config,
                    self.rng,
                )

                # Record outcomes
                winner = outcome["winner_idx"]
                if winner >= 0:
                    win_counts[winner] += 1

                cut_counts += outcome["made_cut"].astype(np.int64)
                positions = outcome["positions"]

                # Top-N finishes
                valid_pos = ~np.isnan(positions)
                top5_counts[valid_pos & (positions <= 5)] += 1
                top10_counts[valid_pos & (positions <= 10)] += 1
                top20_counts[valid_pos & (positions <= 20)] += 1

                # H2H pairwise settlement per standard 72-hole match rules:
                #   - exactly one player makes the cut → he wins the matchup
                #   - both make the cut → lower 72-hole total wins
                #   - both miss the cut → lower 36-hole total wins
                #   - equal scores → void (counted for neither side)
                # Implemented via a single "settlement score" per player:
                # 72-hole total for cut-makers, 36-hole total plus a large
                # penalty for cut-missers (so any cut-maker beats any
                # cut-misser, while cut-missers keep their 36-hole order).
                if compute_h2h:
                    made_cut = outcome["made_cut"]
                    settlement = np.where(
                        made_cut,
                        outcome["total_scores"],
                        outcome["r2_scores"] + _CUT_MISS_PENALTY,
                    )
                    # wins_mat[i,j] = True if player i beat player j (lower score)
                    h2h_counts += settlement[:, None] < settlement[None, :]

            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                pct = (batch_idx + 1) / n_batches * 100
                logger.debug("Simulation progress: %.0f%%", pct)

        # Convert counts to probabilities
        win_probs = dict(zip(player_ids.tolist(), (win_counts / n_sims).tolist()))
        top5_probs = dict(zip(player_ids.tolist(), (top5_counts / n_sims).tolist()))
        top10_probs = dict(zip(player_ids.tolist(), (top10_counts / n_sims).tolist()))
        top20_probs = dict(zip(player_ids.tolist(), (top20_counts / n_sims).tolist()))
        make_cut_probs = dict(zip(player_ids.tolist(), (cut_counts / n_sims).tolist()))

        # H2H probabilities: P(A beats B) = wins_A / (wins_A + wins_B), ties excluded
        h2h_dict = None
        if compute_h2h and h2h_counts is not None:
            h2h_dict = {}
            for i in range(n_players):
                pid_a = int(player_ids[i])
                for j in range(i + 1, n_players):
                    pid_b = int(player_ids[j])
                    wins_a = int(h2h_counts[i, j])
                    wins_b = int(h2h_counts[j, i])
                    total = wins_a + wins_b
                    if total > 0:
                        h2h_dict.setdefault(pid_a, {})[pid_b] = wins_a / total
                        h2h_dict.setdefault(pid_b, {})[pid_a] = wins_b / total

            logger.info("H2H computed: %d players, %d valid pairs",
                        n_players, sum(len(v) for v in h2h_dict.values()) // 2)

        # Convergence diagnostics
        diagnostics = self._check_convergence(win_counts, n_sims)

        result = SimulationResult(
            win_probs=win_probs,
            top5_probs=top5_probs,
            top10_probs=top10_probs,
            top20_probs=top20_probs,
            make_cut_probs=make_cut_probs,
            n_simulations=n_sims,
            convergence_diagnostics=diagnostics,
            h2h_probs=h2h_dict,
        )

        # Log summary
        top_player = max(win_probs, key=win_probs.get)
        logger.info(
            "Simulation complete | Favorite: player %d (P=%.3f) | "
            "Sum P(win)=%.4f | Max SE=%.4f",
            top_player, win_probs[top_player],
            sum(win_probs.values()),
            diagnostics.get("max_se", 0),
        )

        return result

    # ==========================================================================
    # PRIVATE: Simulation mechanics
    # ==========================================================================

    def _draw_abilities(
        self,
        mu_means: np.ndarray,
        mu_stds: np.ndarray,
        cf_means: np.ndarray,
        cf_stds: np.ndarray,
        n_sims: int,
    ) -> np.ndarray:
        """
        Draw true abilities for each player in each simulation.
        
        For each simulation s and player i:
            ability_{s,i} = μ_i^{(s)} + cf_i^{(s)}
        where:
            μ_i^{(s)} ~ N(μ̂_i, σ̂²_μ_i)
            cf_i^{(s)} ~ N(cf̂_i, σ̂²_cf_i)
            
        This propagates BOTH skill uncertainty and course-fit uncertainty
        through to the final win probabilities.
        
        Returns
        -------
        np.ndarray
            Shape (n_sims, n_players) — drawn abilities.
        """
        n_players = len(mu_means)

        # Draw skill levels
        mu_draws = self.rng.normal(
            loc=mu_means[np.newaxis, :],     # (1, N)
            scale=mu_stds[np.newaxis, :],    # (1, N)
            size=(n_sims, n_players),
        )

        # Draw course-fit adjustments
        cf_draws = self.rng.normal(
            loc=cf_means[np.newaxis, :],
            scale=cf_stds[np.newaxis, :],
            size=(n_sims, n_players),
        )

        return mu_draws + cf_draws

    def _simulate_rounds(
        self,
        abilities: np.ndarray,
        sigmas: np.ndarray,
        n_sims: int,
        n_players: int,
    ) -> np.ndarray:
        """
        Simulate 4 rounds of scores for all players in all simulations.
        
        Round model with autocorrelation:
            S_{i,1} = ability_i + ε_{i,1}
            S_{i,r} = ability_i + ρ·(S_{i,r-1} - ability_i) + ε_{i,r}
            ε_{i,r} ~ t(ν, 0, σ²_i)
            
        The ρ term captures within-tournament momentum/conditions.
        Student-t noise handles the heavy tails in round scores.
        
        Returns
        -------
        np.ndarray
            Shape (n_sims, n_players, 4) — SG-relative round scores.
            Negative values = better than field average.
        """
        rho = self.round_correlation
        nu = self.observation_df

        round_scores = np.zeros((n_sims, n_players, 4))

        for r in range(4):
            # Student-t noise: ε ~ t(ν) * σ_i
            # scipy's t.rvs generates t with unit scale; multiply by σ_i
            noise = sp_stats.t.rvs(
                df=nu,
                size=(n_sims, n_players),
                random_state=self.rng.integers(0, 2**31),
            ) * sigmas[np.newaxis, :]

            if r == 0:
                # First round: no autocorrelation
                round_scores[:, :, r] = abilities + noise
            else:
                # Subsequent rounds: carry forward ρ fraction of previous deviation
                prev_deviation = round_scores[:, :, r - 1] - abilities
                round_scores[:, :, r] = abilities + rho * prev_deviation + noise

        # Convention: negate so that lower = better (like actual golf scores)
        # In SG convention, positive = good. In score convention, lower = good.
        # We keep SG convention here (negative round_score = below field avg = good)
        # tournament.py handles the "lower is better" for cuts/positions.
        return -round_scores

    # ==========================================================================
    # PRIVATE: Convergence diagnostics
    # ==========================================================================

    def _check_convergence(
        self,
        win_counts: np.ndarray,
        n_sims: int,
    ) -> Dict:
        """
        Check whether simulation has converged (enough iterations).
        
        Uses the standard error of proportion estimate:
            SE(p̂) = sqrt(p̂ * (1 - p̂) / N)
            
        For a typical 150-player field with 100K simulations:
            - Favorite (~5% win prob): SE ≈ 0.07%
            - Longshot (~0.5% win prob): SE ≈ 0.02%
            
        Target: max SE < 0.5% (0.005) for betting decisions.
        """
        probs = win_counts / n_sims
        standard_errors = np.sqrt(probs * (1 - probs) / n_sims)

        max_se = float(np.max(standard_errors)) if len(standard_errors) > 0 else 0
        mean_se = float(np.mean(standard_errors)) if len(standard_errors) > 0 else 0

        converged = max_se < 0.005  # Half a percentage point

        return {
            "max_se": round(max_se, 6),
            "mean_se": round(mean_se, 6),
            "converged": converged,
            "n_simulations": n_sims,
            "recommendation": (
                "Sufficient" if converged
                else f"Increase to ~{int(n_sims * (max_se / 0.005) ** 2):,} simulations"
            ),
        }

    # ==========================================================================
    # PUBLIC: Utility methods
    # ==========================================================================

    def results_to_dataframe(
        self,
        result: SimulationResult,
        player_names: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """
        Convert SimulationResult to a sorted DataFrame for inspection.
        
        Parameters
        ----------
        result : SimulationResult
        player_names : dict, optional
            {player_id: "Player Name"} for display.
            
        Returns
        -------
        pd.DataFrame
            Sorted by win probability (descending).
        """
        records = []
        for pid in result.win_probs:
            row = {
                "player_id": pid,
                "p_win": result.win_probs.get(pid, 0),
                "p_top5": result.top5_probs.get(pid, 0),
                "p_top10": result.top10_probs.get(pid, 0),
                "p_top20": result.top20_probs.get(pid, 0),
                "p_make_cut": result.make_cut_probs.get(pid, 0),
                "implied_odds": (
                    round(1 / result.win_probs[pid])
                    if result.win_probs[pid] > 0 else np.inf
                ),
            }
            if player_names and pid in player_names:
                row["player_name"] = player_names[pid]
            records.append(row)

        df = pd.DataFrame(records)
        df = df.sort_values("p_win", ascending=False).reset_index(drop=True)

        return df
