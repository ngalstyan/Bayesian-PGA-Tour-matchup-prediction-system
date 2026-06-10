# AUDIT.md — Independent Code & Methodology Audit

**Scope:** Full read-only audit of the golf betting model (Method 1) prior to presentation to quant funds.
**Date:** 2026-06-10.
**Method:** Every module in `golf_model/` was read; notebooks 01–09 were extracted and reviewed; the headline backtest numbers were **re-executed from the committed 103 MB prediction cache** (`data/processed/backtest_cache/ewma25_50k_v2.pkl`) in the `golf_model` conda environment, and a forensic decomposition of the bet population was performed. Audit scripts lived in `/tmp` (no repo files modified other than this one).

---

## Executive summary (read this first)

**The headline H2H result (1,854 bets, 56.6% win rate, +6.0% ROI, Sharpe 1.48) replicates exactly from the cache — and is invalid.** It is produced almost entirely by a lookahead-bias bug, not by model edge.

The mechanism: DataGolf reuses the same `event_id` for each annual edition of a tournament. The backtest cache is keyed by `event_id` alone, so 2024 predictions **overwrite** 2023 predictions for every recurring event. `evaluate_matchups()` then joins matchup odds on `event_id` **without a year filter**, so 2023 matchups are settled against predictions from a model trained on data through late 2024 — including the 2023 event's own outcome.

Empirical decomposition of the 1,854 bets (replicated bet-for-bet from the cache):

| Bet population | n | Win rate | ROI (staked) | Flat-stake ROI |
|---|---|---|---|---|
| **As reported (all)** | 1,854 | 56.6% | **+6.0%** | +9.1% |
| Leaked (2023 odds, 2024-trained model) | 865 | **63.8%** | **+18.9%** | +23.0% |
| **Clean (year-matched)** | 989 | **50.3%** | **−5.1%** | −3.0% |

Clean-subset flat ROI 95% CI: **[−9.1%, +3.0%]** (iid), **[−9.9%, +3.9%]** (cluster bootstrap by event); P(ROI ≤ 0) = 0.80. A 50.3% win rate at average odds 1.94 is exactly what a no-edge bettor paying ~3% round-trip vig against Pinnacle closing lines looks like.

A second, independent bug corrupts the outright backtest: the same missing year filter makes the engine merge two years' fields (37 of 58 cached events) and compute the "winner" by summing strokes-gained **across both years' editions** — so **29 of 58 cached events have the wrong winner recorded**. Outright Brier scores, the Diebold–Mariano test, and the +84.4% outright ROI are all settled against partly fictional outcomes.

**Bottom line:** the model has *not* been shown to beat Pinnacle closing lines on H2H matchups, and the live H2H deployment decision rests on a corrupted backtest. The 2023–2024 holdout is additionally burned by in-sample hyperparameter selection (edge-threshold grid search, probability temperature, sizing scheme — all chosen by re-running the holdout). After fixing the bugs, the model needs a fresh, untouched validation period before any live money conclusion.

The good news: the betting math (devig, Kelly, EV), the backtest engine's *training-window* hygiene (`date < event_start`), and the feature layer's temporal filters are correct; the bugs are concentrated in the cache/evaluation layer and are fixable. Replication infrastructure is excellent — the cache made this audit's forensics possible.

---

## PHASE 1 — Repo map and pipeline understanding

### Modules

| Layer | Files | Role |
|---|---|---|
| Config | `golf_model/config/settings.py` | All hyperparameters, paths, API keys (dataclass) |
| Data | `data/loader.py`, `data/schemas.py`, `data/api_client.py`, `data/weather.py` | CSV loading + schema validation; DataGolf/Open-Meteo clients |
| Features | `features/strokes_gained.py` (incl. `FieldStrengthCalculator` — there is **no** separate `field_strength.py` despite CLAUDE.md), `features/time_weighting.py`, `features/course_features.py`, `features/pipeline.py` | SG validation, field strength, dual-decay EWMA, orchestration |
| Models | `models/baseline.py` (incl. `ModelRegistry`), `models/bayesian_core.py`, `models/course_fit.py`, `models/priors.py` | EWMA baseline; PyMC hierarchical model; feature-based course fit (unused — no course CSV) |
| Simulation | `simulation/monte_carlo.py`, `simulation/tournament.py` | MC tournament engine (numpy, *not* numba despite docs), cut/playoff logic |
| Betting | `betting/odds_processing.py`, `betting/edge_detection.py`, `betting/kelly.py`, `betting/bankroll.py` | Devig (Shin/proportional/power), edge filters, fractional Kelly |
| Validation | `validation/backtest.py`, `validation/metrics.py`, `validation/calibration.py`, `validation/statistical_tests.py` | Expanding-window engine + cache, Brier/log-loss/ROI/Sharpe/DD, DM & bootstrap tests |
| Research | `research/news_agent.py` | Pre-bet news briefings (read-only; no effect on metrics) |
| Entry points | `run_pipeline.py` (train/backtest/predict — **uses BaselineModel**, diverges from notebooks), notebooks `01–09` (the real pipeline), `scripts/00–14` (data pulls) |

### Actual production data flow (what generated the headline numbers)

```
raw CSVs (sg_rounds_*, odds_*, matchup_odds_*; rounds dated by event_completed only)
  → NB08 cell 2 fit_and_predict callback, per holdout event:
      train_rounds = rounds[date < event_start]            (engine-enforced, correct)
      EWMA dual-decay skill (25r/120d) as_of event date
      + 50% blend toward course-variance-weighted SG components
      + 40% blend toward last-8-rounds mean
      + event-history Bayesian-shrunk "course fit" (τ=0.50)
      + t-scale correction (×0.8165), mu_std capped at 0.10
  → MonteCarloSimulator 50K sims, Student-t(6), ρ=0.15 → win/topN/H2H probs
  → BacktestCache (keyed by event_id ← THE BUG)
  → evaluate(): outrights vs Pinnacle close, devig, temperature 0.35, edge ≥ 1.50×
  → evaluate_matchups(): H2H vs Pinnacle close, edge ≥ 0.08, 25% Kelly capped 1.5%
  → ROI / Sharpe / MaxDD / Brier / DM → gates → README/CLAUDE.md claims
```

**Important framing fact:** the PyMC hierarchical Bayesian model (NB04) and the γ_c·δ_i course-fit model (`course_fit.py`) are **not part of this flow**. The deployed/backtested model is an EWMA heuristic stack. See Finding P4-1.

---

## PHASE 2 — Lookahead bias and data leakage

### Finding 2.1 — CRITICAL: H2H backtest leaks future models onto past odds via `event_id` year collision

**Location:**
- `golf_model/validation/backtest.py:308` (`cache_predictions[event_id] = model_probs` — later years overwrite earlier years; same for `cache_event_meta`, `cache_h2h` at lines 309–323)
- `golf_model/validation/backtest.py:571` (`event_odds = odds[odds["event_id"] == event_id]` — no `calendar_year` filter in `evaluate_matchups`, unlike `run()`/`evaluate()` which do filter odds by year at lines 260–262 and 467–469)

**Mechanism:** DataGolf reuses `event_id` across annual editions (35 of the 55 event IDs in `matchup_odds_2023_2024.csv` appear in both years). During the expanding-window walk, the 2023 and 2024 editions are each predicted correctly in-loop, but the cache dict keeps only the **2024** prediction. `evaluate_matchups()` then pulls **both years'** matchup rows for that ID and prices them all with the 2024-trained model. The 2024 model's training window includes the entire 2023 season — including the 2023 edition of the very event being "predicted" (its rounds enter the EWMA, the recent-form blend, and directly into `_compute_course_fit`'s event history).

**Verified impact (exact replication from the committed cache):** 865 of 1,854 bets (46.7%) are leaked. Leaked bets: 63.8% win rate, +18.9% ROI. Clean bets: 50.3% win rate, −5.1% ROI (flat-stake −3.0%, 95% CI [−9.9%, +3.9%], P(ROI≤0)=0.80). **The entire reported edge is this bug.** All four "H2H gates" (sample, win rate, ROI, Sharpe) fail on the clean subset except sample size.

**Severity: Critical.** Invalidates the headline result, the gate verdicts in CLAUDE.md/README, and the live-deployment decision.

**Proposed fix:** Key everything by a composite `(event_id, calendar_year)` — `cache_predictions`, `event_metadata`, `h2h_predictions`, and the odds join in `evaluate_matchups()` (matchup CSV already has `calendar_year`). Re-run the full backtest. Then treat the corrected 2023–24 numbers as *contaminated by prior tuning* (Finding 2.3) and confirm on untouched data.

### Finding 2.2 — CRITICAL: Outright backtest settles against wrong winners and simulates merged two-year fields

**Location:**
- `golf_model/validation/backtest.py:257` (`event_rounds = rounds_df[rounds_df["event_id"] == event_id]` — no year filter)
- `golf_model/validation/backtest.py:719–740` (`_get_winner`: rounds data has neither `finish_position` nor `score` columns — it has `fin_text` and `round_score` — so it always falls through to "highest summed `sg_total`")

**Mechanism:** For recurring events, `event_rounds` contains both years' editions. Consequences, all verified against the cache:
1. The simulated **field** merges both years' entrants — 37 of 58 cached events (e.g., AT&T Pebble Beach: 202 players simulated vs 80 actual), diluting every win probability.
2. The recorded **winner** is the player with the highest strokes-gained sum **across both editions** — **29 of 58 cached events store a winner_id that is not the actual winner** of the edition being scored. Even for single-year events, the SG-sum fallback picks the 72-hole leader, ignoring playoffs, and never sees `fin_text`.
3. Outright Brier, log-loss, the DM test (Gate 1/2), and betting P&L (Gate 3, the reported +84.4% ROI / Sharpe 1.97 / "PASS") are all computed against these pseudo-winners.

**Severity: Critical** for the outright numbers (which the README already de-emphasizes, but cites Gate 3 as PASS). Also the root enabler of Finding 2.1 (the merged field is why 2023 matchup pairs find H2H entries in the 2024-keyed cache).

**Proposed fix:** Filter `event_rounds` to the event's year (`rounds["date"].dt.year == event_date.year` or join on season); fix `_get_winner` to use `fin_text == "1"` first, then `round_score` sums; re-run.

### Finding 2.3 — HIGH: The 2023–2024 holdout was used for hyperparameter selection

**Location:** `notebooks/08_full_backtest.ipynb` cells 6, 9, 10; `config/settings.py:239` ("validated via backtest"), `settings.py:245`; CLAUDE.md ("MATCHUP_MIN_EDGE=0.08 is optimal (grid-search optimal)").

**Mechanism:** The cache exists precisely to "iterate on OVERROUND_METHOD, MIN_EDGE_THRESHOLD, … KELLY_FRACTION … in seconds" (`backtest.py:420`). Cell 9 grid-searches `MATCHUP_MIN_EDGE` over 10 values on the holdout and picks the Sharpe/ROI sweet spot; cell 10 picks the sizing scheme; cell 6 sets `PROB_TEMPERATURE=0.35`, `MIN_EDGE_THRESHOLD=1.50`, `MIN_BET_PROBABILITY=0.05` the same way. Cell 0's comment ("Power method over-compresses longshot probs, creating phantom edges") shows the devig method was also chosen from holdout results, and cell 2's "EWMA half-life reduced 60→25" plus the 0.40/0.50 blend weights have no documented training-period CV (a proper `optimize_half_life` CV routine exists in `time_weighting.py:332` but there is no evidence it produced these values on training data only).

**Why it matters:** Even after fixing 2.1/2.2, any re-run on 2023–24 with these parameters reports an in-sample-selected maximum, not an out-of-sample expectation. The "holdout" no longer functions as one.

**Severity: High.**

**Proposed fix:** Freeze all parameters, validate once on a period never used for any decision (2025 → mid-2026 data already exists in the repo, but note live betting since 2025 may have informed further tweaks — disclose accordingly). Alternatively nested/walk-forward parameter selection inside the training window only.

### Finding 2.4 — HIGH: Backtest assumes execution at Pinnacle **closing** prices

**Location:** `backtest.py:596–597` (`p1_close`/`p2_close`); outrights use `close_odds → decimal_odds` (NB08 cell 4).

**Mechanism:** Both open and close are available in the data (`p1_open`, `open_odds`), but only the close is used. This is a deliberate-looking choice (closing line = sharpest benchmark) and as a *calibration* test it is the right discipline — but as a *P&L simulation* it assumes you can get the closing price at Pinnacle limits, which no bettor reliably can, and live deployment (NB09) bets at soft books days before the close. The backtest therefore measures a quantity (edge vs Pinnacle close) different from the live strategy (edge vs soft-book early lines), in both directions: closing prices are unobtainable, but soft-book early lines are softer.

**Severity: High** (transparency/validity of the claim, not a code bug). Flag explicitly in any deck: the test is "model vs Pinnacle close." Note the clean-subset result means the model currently *loses* to the Pinnacle close.

**Proposed fix:** Report both: ROI vs close (calibration benchmark) and ROI vs open (rough executable proxy); add CLV tracking for live bets (the `compute_clv` helper exists in `edge_detection.py:224` but is unused).

### Verified clean (no leakage found)

- **Training-window hygiene:** `backtest.py:253` uses strictly `date < event_start`; the rounds' `date` is the event **completion** date (no per-round dates exist in the data), which makes the filter *conservative* — a concurrent week's results complete after the target event starts and are excluded. EWMA `as_of_date=event_date` with `<=` (`time_weighting.py:128`) is safe because train_rounds was already cut at `< start`.
- **EWMA/feature layer:** no forward peeking; `FieldStrengthCalculator` uses only `date < event_start` rounds; `FeaturePipeline.run_for_tournament` uses start−1 day; `optimize_half_life` trains strictly before each event.
- **No full-sample normalization:** no scalers are fit anywhere; course-feature Z-scoring is unused (no course CSV); probability temperature is a fixed scalar (tuned on holdout — see 2.3 — but not a leakage mechanism per se).
- **No model refits on holdout:** the expanding window only ever fits on pre-event data (the leak enters at the *cache/evaluation* layer, not the fitting layer).

---

## PHASE 3 — Betting and statistics

### Replication

- **H2H:** `engine.evaluate_matchups()` on the committed cache reproduces the headline **exactly**: 1,854 bets, 56.6% WR, +6.0% ROI, Sharpe 1.48, MaxDD 27.9%, P&L $37,176. A line-by-line reimplementation matched bet-for-bet (basis of the leak decomposition above).
- **Outright:** `engine.evaluate()` reproduces 28 bets, +84.4% ROI, Sharpe 1.97, MaxDD 11.4%, model Brier 0.01298 vs market 0.01103; top-3 events = $1,859 of $1,321 total P&L (>100%, confirming the concentration warning).
- **ROI confidence interval (as asked):** flat-stake per-bet ROI on all 1,854 bets is +9.1%, iid 95% CI [+4.7%, +13.5%]; cluster bootstrap by event [+3.0%, +14.9%] — *statistically* distinguishable from zero, **but only because of the leaked bets**. On the clean 989 bets: −3.0%, CI [−9.9%, +3.9%] — indistinguishable from zero and centered below it. Clean win rate 50.3% ± 3.1pp.

### Correct (verified)

- **Odds conversion & devig** (`odds_processing.py`): decimal→implied is right; proportional 2-way devig in `evaluate_matchups` is standard for H2H; Shin's root-find and the power method are correctly implemented with sane fallbacks. Everything is decimal odds end-to-end; no American-odds conversion errors exist because no American odds are used.
- **Kelly** (`kelly.py:89`, `backtest.py:636`): f\* = (bp−q)/b is correct; fractional multiplier, per-bet cap, and tournament exposure cap applied in a sensible order; EV formula `stake×(odds×p−1)` correct.
- **Edge math:** outrights use ratio `p_model/p_market ≥ 1.50`; H2H uses additive `p_model − p_market ≥ 0.08`. Internally consistent (only one side of a devigged 2-way market can clear an additive threshold).
- **ROI** = P&L / total staked — correct; ties voided consistent with the recorded `tie_rule` ("void in event of tie", outcome 0.5 skipped).
- **Bankroll compounding** updates per event before sizing the next — internally consistent.

### Finding 3.1 — MEDIUM: Sharpe is over-annualized and selection-biased

**Location:** `backtest.py:690` and `:912` (`annualization_factor=40`), `metrics.py:337`.

**Issues:** (a) The H2H sample is 55 events over **two** seasons (~27.5/yr); √40 vs √27.5 inflates Sharpe ~21% (reported 1.48 → ~1.23 at the correct frequency, on the contaminated sample). (b) Per-event returns are computed only over events **with bets** — quiet weeks (return ≈ 0) are excluded, overstating both mean and the comparability to "annualized." (c) `metrics.generate_validation_report` applies factor 40 to *per-bet* returns — wrong units entirely (~1,854 bets/2yr, not 40). The gate threshold "Sharpe > 0.5 annualized" is therefore not being tested as defined.

**Fix:** annualize by actual events-per-year of the strategy calendar (include zero-bet weeks), and never apply event-level annualization to per-bet return vectors.

### Finding 3.2 — MEDIUM: Model H2H probabilities are conditional on both players making the cut

**Location:** `monte_carlo.py:249–259` (pairs counted only when both `positions` are non-NaN, i.e., both made the cut in that sim; `tournament.py:255` sets missed-cut totals to NaN).

**Why it matters:** Pinnacle 72-hole match settlement (and DataGolf's `p1_outcome`) treats a missed cut as a loss unless both miss. The model's P(A>B) throws away all sims where exactly one player misses the cut — precisely the scenarios where a steady cut-maker has his largest advantage over a volatile player. The probabilities being compared to the market are therefore structurally mis-specified, which manufactures spurious "edges" against correctly-specified market prices. (The `h2h_valid` matrix is computed and never used — dead code from a half-finished fix?)

**Fix:** settle simulated H2H per the actual rule: one makes cut → he wins; both miss → lower 36-hole total (or void, matching book rules); both make → 72-hole total; ties void.

### Finding 3.3 — MEDIUM: Brier/DM comparison is run on temperature-sharpened probabilities with holdout-tuned T

**Location:** `backtest.py:160–168, 326, 455`; T=0.35 from NB08 cell 6.

The calibration gate (model Brier vs market Brier) is evaluated after applying a sharpening exponent chosen by looking at the same holdout. Combined with Finding 2.2 (wrong winners in 29/58 events), neither Gate 1 nor the DM p-value (Gate 2) is currently meaningful. DM implementation itself (`statistical_tests.py:30`) is fine for h=1 (sample variance of the loss differential, normal approx, one-sided "less"), with the standard small-n caveat at n=58; the bootstrap alternative is also correctly implemented.

**Fix:** re-run gates after 2.1/2.2 fixes, with T fixed ex-ante (or T=1), on uncontaminated data.

### Finding 3.4 — LOW: Within-event Kelly stakes are sized independently against the same bankroll

`evaluate_matchups` sizes every matchup in an event at up to 1.5% of the same pre-event bankroll with no per-event exposure cap (the outright path has `MAX_TOURNAMENT_EXPOSURE_PCT`, the matchup path doesn't; with ~34 bets/event, theoretical event exposure can exceed 50% of bankroll). MaxDD 27.9% in the contaminated backtest hints at the realized risk. **Fix:** apply an event-level exposure cap in the matchup path (NB09 live flow should match).

---

## PHASE 4 — MCMC / model soundness

### Finding 4.1 — HIGH (presentation integrity): The "hierarchical Bayesian model" is not the model that produced any reported result

**Location:** `models/bayesian_core.py` (fit only in NB04); NB08 cell 2 / NB09 (production callback).

The backtested and live model is: dual-decay EWMA + 50% course-variance reweighting + 40% recent-form blend + normal-conjugate event-history shrinkage + Student-t Monte Carlo. The PyMC model is fitted once in NB04, its posteriors are never loaded by NB08/NB09, and `course_fit.py` (the γ_c·δ_i model in the math writeup) has never run for lack of course data. A fund deck describing the live strategy as "hierarchical Bayesian" with this repo as evidence would be materially inaccurate. **Fix:** either wire the posterior into the simulator (and re-validate) or describe the production model as what it is (empirical-Bayes-flavored EWMA heuristic).

### Finding 4.2 — MEDIUM: NUTS diagnostics cannot be verified from the repo

All notebook outputs are stripped (0 output cells in NB01–08) and no trace/posterior artifact is saved (`artifacts/models/` contains only two BaselineModel pickles). The claimed diagnostics (R-hat max 1.0048, ESS min 1,071, 0 divergences for 5,445 parameters) are plausible for this model/data size but unverifiable. The diagnostic code itself (`check_convergence`, `backtest.py` analog in `bayesian_core.py:274`) is correctly written (R-hat<1.01, ESS>400, divergence count). **Fix:** persist the InferenceData (`az.to_netcdf`) and a diagnostics JSON alongside model artifacts.

### Finding 4.3 — MEDIUM: Model specification notes (would matter if the Bayesian model were promoted to production)

- **Static skill:** `mu_player` is constant over 2019–2022 — no time dynamics — directly contradicting the project's core premise (form decay) and the μ_{i,t} notation in the math writeup. Predictions from it would be stale by construction.
- **Centered parameterization** (`mu_player ~ N(mu_pop, sigma_pop)`): with hundreds of rounds per regular player this usually samples fine (consistent with claimed diagnostics), but players with 1–5 rounds create funnel geometry; a non-centered version is the safe default.
- **No field-strength term in the likelihood:** KFT/Euro/PGA rounds are pooled with raw `sg_total`, so a KFT regular's μ is inflated relative to PGA SG (see 5.4).
- **Priors are reasonable:** μ_pop~N(0,0.5), σ_pop~HN(2) (loose vs true ~0.6–1.5 spread but harmless), τ_i~HN(4) weakly informative, ν=2+Gamma(3,1) sensible. Identifiability of μ_pop vs μ_i is fine given the hierarchy. `priors.py` documents InvGamma but code uses HalfNormal — doc/code mismatch only.
- **Simulator t-noise scaling:** `MonteCarloSimulator._simulate_rounds` multiplies a unit-scale t(ν=6) by σ_i, giving variance 1.5σ²; the √((ν−2)/ν)=0.8165 correction is applied *by the NB08/09 callback*, not by the simulator. NB06/NB07 and any other caller that passes raw σ overdisperses by 22%. The correction belongs inside the simulator.
- **Playoff = uniform coin flip** among tied players (`tournament.py:208`) — documented simplification, second-order.

---

## PHASE 5 — Reproducibility and code quality

### Finding 5.1 — MEDIUM: Silent failure modes in the backtest engine

`backtest.py:287–293` catches **all** exceptions from the model callback and skips the event (events silently vanish from the sample); `_compute_betting_pnl` (`:871`) and `_get_market_probs` (`:761`) catch `Exception` and return "no bets"/"no probs" at *debug* log level — a broken betting pipeline is indistinguishable from "no edges found." **Fix:** count and report skipped events in `BacktestResult`; log at WARNING+; consider failing fast in audit runs.

### Finding 5.2 — MEDIUM: Environment is not pinned

`environment.yml` uses only lower bounds (`pandas>=2.0`, `pymc>=5.10`, …). A clean clone a year from now resolves different versions; MCMC and even pandas groupby behavior can shift results. No lock file, no `pip freeze` snapshot. **Fix:** commit a `conda-lock` / explicit-spec file alongside the loose spec.

### Finding 5.3 — MEDIUM: Two pipelines, one truth problem

`run_pipeline.py --mode backtest` runs `BaselineModel` via `FeaturePipeline` — a different model from NB08's callback (which bypasses `FeaturePipeline` entirely). The documented entry point does not produce the reported numbers; only the notebook does. Notebook outputs are stripped, so the repo contains **no executable artifact of the claimed results** other than the cache. **Fix:** move the NB08 callback into a module (e.g., `models/production_ewma.py`), call it from both NB08 and `run_pipeline.py`, and commit a small results JSON per backtest run.

### Finding 5.4 — MEDIUM: Field-strength adjustment is computed but never used

`FeaturePipeline` creates `sg_*_adj` columns, but `TimeWeighter.compute_weighted_sg` reads the raw `sg_total`/components (`time_weighting.py:118`), and the NB08/NB09 production callback never touches field strength at all while training on PGA+Euro+KFT rounds pooled together. Cross-tour SG is treated as exchangeable, biasing μ for players with weaker-tour histories. **Fix:** either point the EWMA at the `_adj` columns or restrict production training to PGA rounds; measure the delta.

### Finding 5.5 — LOW: Assorted

- **Stale/incorrect docs:** CLAUDE.md lists `features/field_strength.py` (doesn't exist), says `ModelRegistry` is "not created" (it is, in `baseline.py:224`), says the project is "not a git repo" (it is), claims "numba >=0.58 (JIT for Monte Carlo)" — **no numba import exists anywhere**; the MC loop is pure numpy + Python.
- **Settings drift:** `INITIAL_BANKROLL=2500` in settings vs 5000 in NB08/NB09; the notebook callback hardcodes `_RECENT_BLEND`, `_COURSE_BLEND`, `_TAU_COURSE_FIT`, `_MIN_COURSE_ROUNDS`, 50K sims, mu_std cap 0.10 — violating the project's own "all config in settings.py" convention; CLAUDE.md's hyperparameter table (e.g., `MIN_BET_PROBABILITY 0.05`, `MIN_EDGE_THRESHOLD 1.50`) reflects notebook overrides, not `settings.py` defaults (0.01 / 1.05).
- **Cache staleness check** (`evaluate`, `backtest.py:424`) omits `PROB_TEMPERATURE`, the blend weights, and the bet-side parameters — exactly the knobs the cache exists to iterate on.
- **Dead/unused code:** `h2h_valid` matrix (monte_carlo.py:203, computed, never read), `exponential_weights` import in `time_weighting.py:50` (unused), `compute_clv` never called, `bankroll.py`/`calibration.py` unused in the reported flow, `kelly_fraction_with_uncertainty` unused.
- **`tests/` is empty** — every numerical claim is validated only by notebooks whose outputs are stripped. Given Findings 2.1/2.2, a unit test on a synthetic two-year duplicated `event_id` would have caught both bugs.
- **Seeds:** `RANDOM_SEED=42` is wired through MC and PyMC; MC results are deterministic per-process given fixed call order (scipy t draws are seeded from the generator). Good. The cache makes the betting layer fully reproducible (this audit's replication was exact).
- **Pickle for caches/models** — fine locally; don't ship to third parties.

---

## What I would do, in order

1. **Fix Finding 2.1 and 2.2** (composite `(event_id, year)` keys; year-filtered `event_rounds`; `fin_text`-based winner). Re-run NB08 end-to-end. Expect the H2H result to land near the clean subset: ~50% WR, ~0 to −5% ROI vs Pinnacle close.
2. **Pause live H2H betting** until step 1's result is known — the deployment gates were passed on corrupted numbers, and the clean-subset evidence is consistent with paying vig at no edge. (Your call, but the audit's evidence points firmly that way.)
3. **Fix Finding 3.2** (cut-aware H2H settlement in the simulator) — the most plausible genuine model improvement available.
4. Re-validate with frozen parameters on 2025–2026 data (never used for tuning), reporting ROI vs both open and close, correct Sharpe annualization, and cluster-robust CIs.
5. Fix the medium/low items (exposure cap in matchup path, t-scale inside simulator, field-strength usage, pinned env, tests for the backtest engine, honest model description in the deck).

## What genuinely checks out

Exact replication of every headline number from the committed cache; correct devig/Kelly/EV/ROI arithmetic; a properly leak-free *training window* in the expanding walk; temporally careful feature code (`as_of_date` discipline throughout); tie handling consistent with book rules; deterministic seeding; and an honest pre-existing self-diagnosis in NB08 cell 7 (concentration risk, 1.88× overconfidence) — the skepticism was pointed at the right place, one layer above where the bugs actually were.
