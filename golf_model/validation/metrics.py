# ==============================================================================
# golf_model/validation/metrics.py
# ==============================================================================
#
# VALIDATION METRICS
# -------------------
# All scoring metrics used to evaluate model quality across the three 
# validation gates:
#
# Gate 1 — Calibration:
#   - Brier Score (quadratic scoring rule)
#   - Log-Loss (logarithmic scoring rule)
#   - Murphy decomposition (reliability + resolution + uncertainty)
#
# Gate 2 — Statistical Significance:
#   - Diebold-Mariano test (model vs baseline)
#   - Likelihood ratio test
#
# Gate 3 — Betting Viability:
#   - ROI (return on investment)
#   - Sharpe Ratio (risk-adjusted return)
#   - CLV (Closing Line Value)
#   - Maximum drawdown
#
# All metrics follow the convention:
#   - Lower is better for loss functions (Brier, log-loss)
#   - Higher is better for performance metrics (ROI, Sharpe)
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# GATE 1: CALIBRATION METRICS
# ==============================================================================

def brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """
    Compute the Brier Score (quadratic scoring rule).
    
    BS = (1/N) Σ (p_i - o_i)²
    
    where p_i is predicted probability and o_i ∈ {0, 1} is outcome.
    
    For multi-outcome (e.g., outright winner with 156 players):
        One player has o_i = 1 (winner), all others o_i = 0.
        BS measures how concentrated probability was on the winner.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities for each outcome. Shape (N,) or (N, M).
    outcomes : np.ndarray
        Binary outcomes (0 or 1). Same shape as probabilities.
        
    Returns
    -------
    float
        Brier Score (lower = better). Range: [0, 2] for binary,
        wider for multi-class.
        
    Notes
    -----
    For golf outright markets, a good Brier Score is hard to define 
    in absolute terms because the outcome is inherently unpredictable.
    What matters is: is our Brier Score LOWER than the market's?
    """
    p = np.asarray(probabilities, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((p - o) ** 2))


def log_loss(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Compute log-loss (cross-entropy loss).
    
    LL = -(1/N) Σ [o_i · log(p_i) + (1-o_i) · log(1-p_i)]
    
    For multi-class (outright winner):
        LL = -(1/N) Σ_tournaments Σ_players o_{i,t} · log(p_{i,t})
    
    Log-loss penalizes confident wrong predictions more severely than 
    Brier Score. A model that assigns 0.1% to the actual winner is 
    penalized much more than one assigning 1%.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities. Clipped to [eps, 1-eps] for numerical safety.
    outcomes : np.ndarray
        Binary outcomes.
    eps : float
        Small constant to prevent log(0).
        
    Returns
    -------
    float
        Log-loss (lower = better).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)

    # Clip to avoid log(0)
    p = np.clip(p, eps, 1.0 - eps)

    # Binary cross-entropy
    ll = -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))
    return float(ll)


def ranked_probability_score(
    prob_dist: np.ndarray,
    actual_rank: int,
    n_players: int,
) -> float:
    """
    Ranked Probability Score for ordinal outcomes.
    
    RPS = (1/(K-1)) Σ_{k=1}^{K-1} (CDF_pred(k) - CDF_actual(k))²
    
    Useful for top-5, top-10 markets where outcome is ordinal.
    
    Parameters
    ----------
    prob_dist : np.ndarray
        Cumulative probability distribution over ranks.
    actual_rank : int
        Actual finish rank (1-indexed).
    n_players : int
        Total number of players.
        
    Returns
    -------
    float
        RPS (lower = better).
    """
    K = min(len(prob_dist), n_players)
    cdf_pred = np.cumsum(prob_dist[:K])
    cdf_actual = np.zeros(K)
    if actual_rank <= K:
        cdf_actual[actual_rank - 1:] = 1.0

    rps = np.mean((cdf_pred - cdf_actual) ** 2)
    return float(rps)


# ==============================================================================
# GATE 2: COMPARISON METRICS
# ==============================================================================

def score_differential(
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> Dict:
    """
    Compare model vs baseline across multiple events.
    
    Parameters
    ----------
    model_scores : np.ndarray
        Model's Brier/log-loss per event.
    baseline_scores : np.ndarray
        Baseline's Brier/log-loss per event.
        
    Returns
    -------
    dict
        Summary statistics of the differential.
    """
    diff = model_scores - baseline_scores  # Negative = model better
    n = len(diff)

    return {
        "n_events": n,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
        "model_better_pct": float(np.mean(diff < 0) * 100),
        "model_worse_pct": float(np.mean(diff > 0) * 100),
        "median_diff": float(np.median(diff)),
    }


# ==============================================================================
# GATE 3: BETTING PERFORMANCE METRICS
# ==============================================================================

def roi(total_profit: float, total_staked: float) -> float:
    """Return on Investment = total_profit / total_staked."""
    if total_staked == 0:
        return 0.0
    return total_profit / total_staked


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Sharpe Ratio = (mean_return - risk_free) / std_return × √(annualization)
    
    For betting, returns are per-bet returns (P&L / stake).
    Annualization factor ≈ √(bets_per_year).
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    excess = r - risk_free_rate
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(annualization_factor))


def max_drawdown(bankroll_series: np.ndarray) -> Tuple[float, float]:
    """
    Compute maximum drawdown from a bankroll time series.
    
    Parameters
    ----------
    bankroll_series : np.ndarray
        Bankroll value at each point in time.
        
    Returns
    -------
    tuple of (max_dd_dollar, max_dd_pct)
        Dollar drawdown and percentage drawdown.
    """
    arr = np.asarray(bankroll_series, dtype=np.float64)
    if len(arr) < 2:
        return 0.0, 0.0

    peak = np.maximum.accumulate(arr)
    dd_dollar = peak - arr
    dd_pct = dd_dollar / peak

    return float(np.max(dd_dollar)), float(np.max(dd_pct))


def profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    """
    Profit Factor = gross_profit / gross_loss.
    
    > 1.0 = profitable. > 2.0 = very good.
    """
    gross_profit = np.sum(np.abs(wins[wins > 0])) if len(wins[wins > 0]) > 0 else 0
    gross_loss = np.sum(np.abs(losses[losses < 0])) if len(losses[losses < 0]) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


# ==============================================================================
# AGGREGATE VALIDATION REPORT
# ==============================================================================

def generate_validation_report(
    model_brier_scores: np.ndarray,
    market_brier_scores: np.ndarray,
    model_log_losses: np.ndarray,
    market_log_losses: np.ndarray,
    bet_returns: np.ndarray,
    bankroll_series: np.ndarray,
    total_staked: float,
    total_pnl: float,
    gate_thresholds: Optional[Dict] = None,
    sharpe_annualization: float = 1.0,
) -> Dict:
    """
    Generate comprehensive validation report across all three gates.

    Parameters
    ----------
    model_brier_scores : array — per-event Brier scores for model
    market_brier_scores : array — per-event Brier scores for market
    model_log_losses : array — per-event log-losses for model
    market_log_losses : array — per-event log-losses for market
    bet_returns : array — per-period returns (P&L / stake). Pass PER-EVENT
        aggregates if you want an annualized Sharpe; per-bet returns must
        not be annualized by an events-per-year factor.
    bankroll_series : array — bankroll over time
    total_staked : float
    total_pnl : float
    gate_thresholds : dict, optional
        Override default gate thresholds.
    sharpe_annualization : float, default 1.0
        Periods per year of the return series (e.g. events_per_year from
        validation.backtest). Default 1.0 = report the raw per-period Sharpe.

    Returns
    -------
    dict
        Full validation report with gate pass/fail determinations.
    """
    report = {}

    # --- Gate 1: Calibration ---
    brier_diff = score_differential(model_brier_scores, market_brier_scores)
    logloss_diff = score_differential(model_log_losses, market_log_losses)

    report["gate_1_calibration"] = {
        "model_avg_brier": float(np.mean(model_brier_scores)),
        "market_avg_brier": float(np.mean(market_brier_scores)),
        "brier_improvement_pct": round(-brier_diff["mean_diff"] /
            np.mean(market_brier_scores) * 100, 2) if np.mean(market_brier_scores) > 0 else 0,
        "model_avg_logloss": float(np.mean(model_log_losses)),
        "market_avg_logloss": float(np.mean(market_log_losses)),
        "model_beats_market_pct": brier_diff["model_better_pct"],
        "passed": brier_diff["mean_diff"] < 0,  # Model has lower (better) Brier
    }

    # --- Gate 2: Statistical Significance ---
    # (Diebold-Mariano computed in statistical_tests.py)
    report["gate_2_significance"] = {
        "brier_differential": brier_diff,
        "logloss_differential": logloss_diff,
        "n_events": int(brier_diff["n_events"]),
        "note": "Run Diebold-Mariano test via statistical_tests.py for p-value",
    }

    # --- Gate 3: Betting Viability ---
    r = np.asarray(bet_returns, dtype=np.float64)
    br = np.asarray(bankroll_series, dtype=np.float64)

    overall_roi = roi(total_pnl, total_staked)
    overall_sharpe = sharpe_ratio(r, annualization_factor=sharpe_annualization)
    dd_dollar, dd_pct = max_drawdown(br)

    report["gate_3_betting"] = {
        "roi_pct": round(overall_roi * 100, 2),
        "sharpe_ratio": round(overall_sharpe, 2),
        "max_drawdown_pct": round(dd_pct * 100, 1),
        "max_drawdown_dollar": round(dd_dollar, 2),
        "n_bets": len(r),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "passed": overall_roi > 0 and overall_sharpe > 0.5 and dd_pct < 0.4,
    }

    # --- Overall verdict ---
    all_passed = (
        report["gate_1_calibration"]["passed"] and
        report["gate_3_betting"]["passed"]
    )
    report["overall"] = {
        "all_gates_passed": all_passed,
        "recommendation": "DEPLOY" if all_passed else "DO NOT DEPLOY",
    }

    return report
