# ==============================================================================
# golf_model/simulation/tournament.py
# ==============================================================================
#
# TOURNAMENT STRUCTURE
# ---------------------
# Encodes the rules of a PGA Tour tournament:
#   - 4 rounds of 18 holes (72-hole stroke play)
#   - Cut after round 2 (typically top 65 and ties)
#   - Playoff for ties at the top (sudden death)
#   - Various field sizes and special rules
#
# This module converts raw simulated round scores into tournament outcomes
# (finish positions, made/missed cut, winner).
#
# Used by: simulation/monte_carlo.py
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TournamentConfig:
    """
    Configuration for a specific tournament's structure.
    
    Different PGA Tour events have different rules. This dataclass
    captures the key structural parameters.
    """
    event_id: int
    event_name: str = ""
    n_rounds: int = 4
    cut_round: int = 2                    # Cut happens after this round
    cut_rule: str = "top_65_ties"         # "top_65_ties", "top_70_ties", "no_cut"
    field_size: int = 156                 # Typical PGA Tour field
    par: int = 72                         # Per-round par
    has_playoff: bool = True              # Sudden death playoff for ties at top


# Standard cut rules: position threshold for making the cut
CUT_THRESHOLDS = {
    "top_65_ties": 65,
    "top_70_ties": 70,
    "top_50_ties": 50,
    "no_cut": 999,                        # Everyone plays all rounds
}


def apply_cut(
    scores_r1_r2: np.ndarray,
    cut_rule: str = "top_65_ties",
) -> np.ndarray:
    """
    Determine which players make the cut after round 2.
    
    Parameters
    ----------
    scores_r1_r2 : np.ndarray
        Shape (N_players,) — cumulative score through round 2 
        (lower = better, relative to par).
        
    cut_rule : str
        Cut rule identifier. See CUT_THRESHOLDS.
        
    Returns
    -------
    np.ndarray
        Boolean array of shape (N_players,). True = made cut.
        
    Notes
    -----
    "Top 65 and ties" means: find the score at position 65. Everyone
    at or below that score makes the cut. This often results in MORE
    than 65 players making the cut due to ties.
    
    Players with NaN scores are treated as withdrawn (missed cut).
    """
    if cut_rule == "no_cut":
        return np.ones(len(scores_r1_r2), dtype=bool)

    threshold_pos = CUT_THRESHOLDS.get(cut_rule, 65)

    # Handle NaN (withdrawn players)
    valid = ~np.isnan(scores_r1_r2)
    n_valid = valid.sum()

    if n_valid == 0:
        return np.zeros(len(scores_r1_r2), dtype=bool)

    # Sort valid scores to find the cut line
    valid_scores = scores_r1_r2[valid]
    sorted_scores = np.sort(valid_scores)

    # The cut line is the score at the threshold position
    # (position is 1-indexed, array is 0-indexed)
    cut_pos = min(threshold_pos, n_valid) - 1
    cut_line = sorted_scores[cut_pos]

    # Everyone at or below the cut line makes it
    made_cut = valid & (scores_r1_r2 <= cut_line)

    return made_cut


def determine_finish_positions(
    total_scores: np.ndarray,
    made_cut: np.ndarray,
) -> np.ndarray:
    """
    Assign finish positions based on total scores.
    
    Handles ties by assigning the same position to tied players.
    (e.g., two players tied for 3rd both get position 3, next player gets 5).
    
    Parameters
    ----------
    total_scores : np.ndarray
        Shape (N_players,) — total 72-hole scores (lower = better).
        NaN for players who missed the cut.
        
    made_cut : np.ndarray
        Boolean array. True = made cut and has valid 72-hole score.
        
    Returns
    -------
    np.ndarray
        Shape (N_players,) — finish positions.
        NaN for players who missed the cut.
        Position 1 = winner (or tied for first).
    """
    positions = np.full(len(total_scores), np.nan)

    cut_indices = np.where(made_cut)[0]
    if len(cut_indices) == 0:
        return positions

    cut_scores = total_scores[cut_indices]

    # Rank: method='min' gives tied players the same position
    # (e.g., T3 = position 3 for all tied players)
    ranks = pd.Series(cut_scores).rank(method="min").values

    positions[cut_indices] = ranks

    return positions


def determine_winner(
    total_scores: np.ndarray,
    made_cut: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Determine the tournament winner, handling ties via playoff.
    
    Parameters
    ----------
    total_scores : np.ndarray
        Shape (N_players,) — total 72-hole scores.
        
    made_cut : np.ndarray
        Boolean array of who made the cut.
        
    rng : np.random.Generator, optional
        Random number generator for playoff resolution.
        
    Returns
    -------
    int
        Index of the winning player (0-indexed into the original array).
        
    Notes
    -----
    Playoff model: each tied player has equal probability of winning.
    This is a simplification — in reality, skill matters in playoffs,
    but modeling playoff-specific skill requires more data than is 
    typically available.
    """
    if rng is None:
        rng = np.random.default_rng()

    cut_indices = np.where(made_cut)[0]
    if len(cut_indices) == 0:
        return -1  # No valid players

    cut_scores = total_scores[cut_indices]
    min_score = np.nanmin(cut_scores)

    # Find all players tied at the minimum score
    tied_mask = cut_scores == min_score
    tied_indices = cut_indices[tied_mask]

    if len(tied_indices) == 1:
        return int(tied_indices[0])

    # Playoff: random selection among tied players
    # (equal probability — simplification)
    winner_idx = rng.choice(tied_indices)
    return int(winner_idx)


def simulate_tournament_outcome(
    round_scores: np.ndarray,
    config: TournamentConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Process a full set of simulated round scores into tournament outcomes.
    
    Parameters
    ----------
    round_scores : np.ndarray
        Shape (N_players, 4) — simulated scores for each round.
        Values are strokes relative to field average (SG convention).
        Negative = better than average.
        
    config : TournamentConfig
        Tournament structure configuration.
        
    rng : np.random.Generator, optional
        For playoff resolution.
        
    Returns
    -------
    dict with keys:
        "winner_idx" : int — index of winner
        "made_cut" : np.ndarray — boolean (N_players,)
        "total_scores" : np.ndarray — 72-hole totals (N_players,)
        "r2_scores" : np.ndarray — 36-hole totals, all players (N_players,)
        "positions" : np.ndarray — finish positions (N_players,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_players = round_scores.shape[0]

    # Cumulative score after round 2 (for cut determination)
    # Note: in SG convention, LOWER (more negative) is BETTER
    score_after_r2 = round_scores[:, 0] + round_scores[:, 1]

    # Apply cut
    made_cut = apply_cut(score_after_r2, config.cut_rule)
    n_made_cut = made_cut.sum()

    # Total score (all 4 rounds for those who made cut)
    total_scores = np.full(n_players, np.nan)
    total_scores[made_cut] = round_scores[made_cut, :].sum(axis=1)

    # Finish positions
    positions = determine_finish_positions(total_scores, made_cut)

    # Winner
    winner_idx = determine_winner(total_scores, made_cut, rng)

    return {
        "winner_idx": winner_idx,
        "made_cut": made_cut,
        "total_scores": total_scores,
        "r2_scores": score_after_r2,
        "positions": positions,
        "n_made_cut": int(n_made_cut),
    }
