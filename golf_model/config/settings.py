# ==============================================================================
# golf_model/config/settings.py
# ==============================================================================
#
# CENTRAL CONFIGURATION HUB
# --------------------------
# Every magic number, file path, API key, and hyperparameter lives here.
# No other file in the project should contain hardcoded constants.
#
# Usage:
#   from config.settings import Settings
#   cfg = Settings()
#   print(cfg.DATA_DIR)
#   print(cfg.EWMA_HALF_LIFE)
#
# Override for testing:
#   cfg = Settings(data_dir="/path/to/test/data")
#
# ==============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

@dataclass
class Settings:
    """
    Immutable project-wide configuration.
    
    All defaults are set here. Override by passing keyword arguments
    to the constructor, or by setting environment variables prefixed 
    with GOLF_ (e.g., GOLF_DATAGOLF_API_KEY).
    
    Sections:
        1. Paths           — file system locations
        2. API Keys        — external service credentials
        3. Data Parameters — train/holdout splits, tours, seasons
        4. Model Hypers    — mathematical model hyperparameters
        5. Betting Params  — Kelly, thresholds, bankroll limits
        6. Simulation      — Monte Carlo settings
        7. Validation      — gate thresholds for deployment
        8. Logging         — log level, format, file output
        9. News Research   — pre-bet news intelligence layer
    """

    # ==========================================================================
    # 1. FILE SYSTEM PATHS
    # ==========================================================================
    # 
    # PROJECT_ROOT: Absolute path to the golf_model/ directory.
    #               Auto-detected from this file's location.
    #
    # DATA_DIR: Where your raw CSV files live. 
    #           UPDATE THIS to match your local setup.
    #
    # PROCESSED_DIR: Where feature-engineered DataFrames are cached.
    #
    # MODELS_DIR: Where trained model artifacts are saved.
    #
    # OUTPUTS_DIR: Where validation reports, plots, and bet logs go.
    # ==========================================================================

    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")

    PROCESSED_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "models")
    OUTPUTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "outputs")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "logs")

    # ==========================================================================
    # 2. API KEYS
    # ==========================================================================
    #
    # DATAGOLF_API_KEY: Your DataGolf Scratch PLUS API key.
    #                   Set via environment variable GOLF_DATAGOLF_API_KEY
    #                   or pass directly to constructor.
    #
    # DATAGOLF_BASE_URL: Base URL for all DataGolf API endpoints.
    # ==========================================================================

    DATAGOLF_API_KEY: str = field(
        default_factory=lambda: os.environ.get("GOLF_DATAGOLF_API_KEY", "")
    )
    DATAGOLF_BASE_URL: str = "https://feeds.datagolf.com"

    # Open-Meteo (free tier — no key needed)
    OPENMETEO_BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # ==========================================================================
    # 3. DATA PARAMETERS
    # ==========================================================================
    #
    # TRAINING / HOLDOUT SPLIT:
    #   Training:  2019–2022 (~200 events). All parameters estimated here.
    #   Holdout:   2023–2024 (~100 events). Strict validation only.
    #   
    #   CRITICAL: Holdout data is NEVER used for parameter optimization.
    #             The expanding-window backtester enforces this by construction.
    #
    # TOURS: Which tours to include. PGA primary; KFT for rookie priors.
    #
    # SG_COMPONENTS: The four strokes-gained sub-skills from Broadie's 
    #                decomposition framework.
    #
    # MIN_ROUNDS_FOR_ESTIMATE: Minimum rounds a player must have before
    #                          we produce an individual (non-prior) estimate.
    #                          Below this, the player is fully shrunk to 
    #                          population mean.
    #
    # MIN_VISITS_FOR_COURSE_FIT: Minimum times a player must have visited
    #                            a specific course to contribute to course-fit
    #                            estimation. (With feature-based approach, this
    #                            is less binding but still useful for validation.)
    # ==========================================================================

    # Train / holdout split years (inclusive)
    TRAIN_SEASONS: List[int] = field(default_factory=lambda: [2019, 2020, 2021, 2022])
    HOLDOUT_SEASONS: List[int] = field(default_factory=lambda: [2023, 2024])

    # Tours to include
    TOURS: List[str] = field(default_factory=lambda: ["pga", "kft", "euro"])
    PRIMARY_TOUR: str = "pga"

    # Strokes-gained sub-components (Broadie decomposition)
    SG_COMPONENTS: List[str] = field(
        default_factory=lambda: ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    )
    SG_TOTAL_COL: str = "sg_total"

    # Minimum data requirements
    MIN_ROUNDS_FOR_ESTIMATE: int = 50        # Below this → full shrinkage to prior
    MIN_ROUNDS_FOR_VARIANCE: int = 100       # Below this → use population variance
    MIN_VISITS_FOR_COURSE_FIT: int = 3       # Per player-course pair (validation)

    # ==========================================================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================================================
    #
    # These are the mathematical tuning knobs of Method 1.
    # All should be treated as INITIAL VALUES subject to cross-validation.
    #
    # EWMA_HALF_LIFE_ROUNDS: Number of rounds for 50% weight decay.
    #   - Higher = slower decay = more weight on older rounds.
    #   - Literature suggests 40–80 rounds (Baker & McHale).
    #   - Will be optimized via CV in Pipe 4.
    #
    # EWMA_HALF_LIFE_DAYS: Alternative time-based decay (calendar days).
    #   - Used for the "dual EWMA" (both round-count and calendar).
    #   - Typical: 180–365 days.
    #
    # OBSERVATION_DF: Degrees of freedom ν for Student-t observation noise.
    #   - ν → ∞ recovers Gaussian. ν ≈ 5–8 typical for golf.
    #   - Estimated from data in Pipe 7.
    #
    # ROUND_CORRELATION_RHO: Within-tournament consecutive round correlation.
    #   - ρ ≈ 0.10–0.20 for PGA Tour (weather + momentum).
    #   - Estimated from data in Pipe 7.
    #
    # NUM_COURSE_FEATURES: Dimension of γ_c vector.
    #   - Features: length, rough, green_speed, wind_exposure, 
    #               elevation, fairway_width, green_size, water_hazard_pct
    # ==========================================================================

    # Time-weighting (Phase 2)
    EWMA_HALF_LIFE_ROUNDS: float = 25.0      # Rounds until 50% weight decay
    EWMA_HALF_LIFE_DAYS: float = 120.0       # Calendar days until 50% weight decay

    # Observation model (Phase 3)
    OBSERVATION_DF: float = 6.0              # Student-t ν (degrees of freedom)
    ROUND_CORRELATION_RHO: float = 0.15      # Within-tournament round correlation

    # Course features (Phase 4)
    NUM_COURSE_FEATURES: int = 8
    COURSE_FEATURE_NAMES: List[str] = field(default_factory=lambda: [
        "length_yards",       # Total course length
        "rough_height_in",    # Primary rough height (inches)
        "green_speed_stimp",  # Stimpmeter reading
        "wind_exposure",      # Wind exposure index (0–1 scale)
        "elevation_ft",       # Elevation above sea level
        "fairway_width_avg",  # Average fairway width (yards)
        "green_size_sqft",    # Average green size (sq ft)
        "water_hazard_pct",   # Pct of holes with water in play
    ])

    # Production EWMA model (the model behind notebooks 08/09).
    # These were previously hardcoded in the notebooks — kept here so the
    # backtest and live deployment provably run the same configuration.
    # FROZEN pending out-of-sample validation: do not re-tune against the
    # 2023–2024 holdout (it was already used for selection — see AUDIT.md 2.3).
    COURSE_SG_BLEND: float = 0.50            # Weight on course-variance-weighted SG components
    MIN_COURSE_ROUNDS_FOR_WEIGHTS: int = 40  # Min historical rounds at event for course weights
    RECENT_FORM_BLEND: float = 0.40          # Weight on recent-form mean
    RECENT_FORM_ROUNDS: int = 8              # Recent-form window (rounds)
    RECENT_FORM_MIN_ROUNDS: int = 6          # Min rounds in window to apply the blend
    COURSE_FIT_SHRINKAGE: float = 0.50       # τ for event-history course-fit shrinkage
    MU_STD_CAP: float = 0.10                 # Cap on skill-estimate uncertainty fed to MC
    PRODUCTION_N_SIMULATIONS: int = 50_000   # MC iterations for backtest/live predictions

    # Bayesian model (Phase 3)
    MCMC_DRAWS: int = 2000                   # Posterior samples (after warmup)
    MCMC_TUNE: int = 1000                    # Warmup/tuning iterations
    MCMC_CHAINS: int = 4                     # Parallel chains
    MCMC_TARGET_ACCEPT: float = 0.90         # NUTS acceptance rate
    ADVI_MAX_ITER: int = 50000               # ADVI iterations (fast approx.)
    CONVERGENCE_RHAT_THRESHOLD: float = 1.01 # R-hat must be below this
    CONVERGENCE_ESS_THRESHOLD: int = 400     # Effective sample size minimum

    # ==========================================================================
    # 5. BETTING PARAMETERS
    # ==========================================================================
    #
    # KELLY_FRACTION: Fraction of full Kelly to actually bet. 
    #   - Full Kelly (1.0) maximizes long-run growth but is extremely volatile.
    #   - 0.25 = 25% Kelly. Recommended for estimation uncertainty.
    #   - Range: 0.10 (very conservative) to 0.50 (aggressive).
    #
    # MIN_EDGE_THRESHOLD: Minimum P_model/P_market ratio to place a bet.
    #   - 1.05 = require at least 5% edge over market probability.
    #   - Below this, edge is likely noise, not signal.
    #
    # MIN_KELLY_FRACTION: Minimum Kelly fraction to justify a bet.
    #   - 0.002 = 0.2% of bankroll. Below this, transaction cost > edge.
    #
    # MAX_SINGLE_BET_PCT: Hard cap on any single bet as % of bankroll.
    #   - 0.02 = 2% max. Prevents ruin from single bad estimate.
    #
    # MAX_TOURNAMENT_EXPOSURE_PCT: Max total exposure across all bets
    #   in one tournament.
    #   - 0.08 = 8%. Prevents over-concentration in one event.
    #
    # OVERROUND_METHOD: Method for removing bookmaker vig from odds.
    #   - "shin": Shin (1991, 1993) — best for informed-trader markets.
    #   - "proportional": Simple proportional removal.
    #   - "power": Power method (Wisdom of Crowds).
    #
    # PRIMARY_BOOK: Sportsbook used as the "sharp" benchmark.
    #   - Pinnacle = sharpest book = best proxy for true probabilities.
    #
    # SOFT_BOOKS: Sportsbooks where you actually place bets.
    #   - Softer books = larger available edge vs model.
    # ==========================================================================

    # NOTE: these defaults are the values actually used by the notebook 08
    # backtest and notebook 09 live deployment (they previously drifted:
    # the notebooks overrode 1.05/0.01/2500 in-place).
    KELLY_FRACTION: float = 0.25
    MIN_EDGE_THRESHOLD: float = 1.50         # P_model / P_market ratio (outrights, high conviction)
    MIN_KELLY_FRACTION: float = 0.002        # Min fraction to justify bet
    MIN_BET_PROBABILITY: float = 0.05        # Skip outright bets where P_market < 5%
    PROB_TEMPERATURE: float = 0.35            # <1 sharpens model probs (chosen on the 2023-24 holdout — see AUDIT.md 2.3)
    MAX_SINGLE_BET_PCT: float = 0.02         # 2% of bankroll per bet (outrights)
    MAX_TOURNAMENT_EXPOSURE_PCT: float = 0.30 # 30% of bankroll per event (matchups)
    INITIAL_BANKROLL: float = 5000.0         # Starting bankroll ($)

    # Matchup betting parameters
    MATCHUP_MIN_EDGE: float = 0.08           # Min P_model - P_market edge for H2H bets
    MATCHUP_MAX_BET_PCT: float = 0.015       # 1.5% of bankroll per matchup bet ($75 max on $5K)

    OVERROUND_METHOD: str = "power"          # "shin", "proportional", "power"
    PRIMARY_BOOK: str = "pinnacle"           # Sharp benchmark
    SOFT_BOOKS: List[str] = field(default_factory=lambda: [
        "draftkings", "fanduel", "betmgm", "caesars", "bet365"
    ])

    # ==========================================================================
    # 6. SIMULATION PARAMETERS
    # ==========================================================================
    #
    # N_SIMULATIONS: Number of Monte Carlo tournament simulations.
    #   - 10,000 = fast development iteration (~seconds).
    #   - 100,000 = production predictions (~minutes).
    #   - 500,000 = high-precision validation (~tens of minutes).
    #
    # CUT_RULE: Standard PGA Tour cut rule.
    #   - "top_65_ties": Top 65 and ties make the cut after round 2.
    #   - Some events use "top_70_ties" or no cut.
    #
    # RANDOM_SEED: For reproducibility. Set to None for true randomness.
    # ==========================================================================

    N_SIMULATIONS: int = 100_000
    CUT_RULE: str = "top_65_ties"
    RANDOM_SEED: Optional[int] = 42

    # ==========================================================================
    # 7. VALIDATION GATE THRESHOLDS
    # ==========================================================================
    #
    # These are the minimum requirements from our analysis.
    # The model must pass ALL gates on the holdout before live deployment.
    #
    # Gate 1 — Calibration:
    #   Model's Brier Score < Market Brier Score
    #   Model's Log-Loss < Market Log-Loss
    #
    # Gate 2 — Statistical Significance:
    #   Diebold-Mariano p-value < 0.05
    #
    # Gate 3 — Betting Viability:
    #   Simulated ROI > 0% (after vig)
    #   Sharpe Ratio > 0.5 (annualized)
    #   Max Drawdown < 40% of bankroll
    # ==========================================================================

    GATE_DM_PVALUE: float = 0.05             # Diebold-Mariano significance
    GATE_SHARPE_MIN: float = 0.5             # Minimum annualized Sharpe
    GATE_MAX_DRAWDOWN_PCT: float = 0.40      # Max tolerable drawdown
    GATE_MIN_HOLDOUT_EVENTS: int = 80        # ~2 PGA seasons

    # ==========================================================================
    # 8. LOGGING
    # ==========================================================================

    LOG_LEVEL: str = "INFO"                  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE: bool = True                 # Also log to file (not just console)
    LOG_FORMAT: str = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"

    # ==========================================================================
    # 9. NEWS RESEARCH AGENT
    # ==========================================================================
    #
    # Read-only intelligence layer that researches recent news about players
    # in identified matchup bets. Produces briefings for human review —
    # never modifies bets, probabilities, or sizing.
    #
    # ANTHROPIC_API_KEY: Your Anthropic API key for Claude.
    #                    Set via environment variable ANTHROPIC_API_KEY.
    #
    # NEWS_AGENT_ENABLED: Toggle the research step on/off in notebook 09.
    #
    # NEWS_CACHE_TTL_DAYS: Cache research per player for this many days.
    #
    # NEWS_MAX_ARTICLES_PER_PLAYER: Max articles to fetch and analyze per player.
    #
    # NEWS_CLAUDE_MODEL: Claude model for analysis (Sonnet = cost-effective).
    #
    # NEWS_SEARCH_DAYS_BACK: How many days of news history to search.
    # ==========================================================================

    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    NEWS_AGENT_ENABLED: bool = True
    NEWS_CACHE_TTL_DAYS: int = 7
    NEWS_MAX_ARTICLES_PER_PLAYER: int = 5
    NEWS_CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    NEWS_SEARCH_DAYS_BACK: int = 14

    # ==========================================================================
    # POST-INIT: Create directories if they don't exist
    # ==========================================================================

    def __post_init__(self):
        """Create output directories on initialization."""
        for dir_path in [
            self.DATA_DIR,
            self.PROCESSED_DIR,
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    @property
    def ewma_lambda_rounds(self) -> float:
        """
        Convert half-life in rounds to exponential decay parameter λ.
        
        Formula: λ = ln(2) / half_life
        
        This λ is used in the weight formula: w_j = exp(-λ * j)
        where j is the number of rounds ago.
        """
        import numpy as np
        return np.log(2) / self.EWMA_HALF_LIFE_ROUNDS

    @property
    def ewma_lambda_days(self) -> float:
        """
        Convert half-life in days to exponential decay parameter λ.
        
        Formula: λ = ln(2) / half_life
        
        This λ is used in the weight formula: w_j = exp(-λ * Δt_j)
        where Δt_j is the number of days since round j.
        """
        import numpy as np
        return np.log(2) / self.EWMA_HALF_LIFE_DAYS

    @property
    def train_season_range(self) -> str:
        """Human-readable training period string."""
        return f"{min(self.TRAIN_SEASONS)}–{max(self.TRAIN_SEASONS)}"

    @property
    def holdout_season_range(self) -> str:
        """Human-readable holdout period string."""
        return f"{min(self.HOLDOUT_SEASONS)}–{max(self.HOLDOUT_SEASONS)}"

    def validate(self) -> List[str]:
        """
        Run basic validation checks on settings.
        Returns list of warning messages (empty = all good).
        """
        warnings = []

        if not self.DATAGOLF_API_KEY:
            warnings.append(
                "DATAGOLF_API_KEY is not set. "
                "Set env var GOLF_DATAGOLF_API_KEY or pass to Settings()."
            )

        if not self.DATA_DIR.exists():
            warnings.append(f"DATA_DIR does not exist: {self.DATA_DIR}")

        if self.KELLY_FRACTION > 0.5:
            warnings.append(
                f"KELLY_FRACTION={self.KELLY_FRACTION} is aggressive. "
                "Recommended range: 0.10–0.25."
            )

        if self.N_SIMULATIONS < 10_000:
            warnings.append(
                f"N_SIMULATIONS={self.N_SIMULATIONS} is low. "
                "Results may be noisy. Use ≥100,000 for production."
            )

        overlap = set(self.TRAIN_SEASONS) & set(self.HOLDOUT_SEASONS)
        if overlap:
            warnings.append(
                f"TRAIN and HOLDOUT seasons overlap: {overlap}. "
                "This causes data leakage — fix immediately."
            )

        if self.NEWS_AGENT_ENABLED and not self.ANTHROPIC_API_KEY:
            warnings.append(
                "ANTHROPIC_API_KEY is not set but NEWS_AGENT_ENABLED=True. "
                "Set env var ANTHROPIC_API_KEY or disable the news agent."
            )

        return warnings
