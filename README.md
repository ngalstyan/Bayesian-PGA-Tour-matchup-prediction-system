# Golf Betting Model — Hierarchical Bayesian Strokes-Gained

A quantitative system for betting PGA Tour golf. It models player skill with a hierarchical Bayesian
strokes-gained framework, simulates tournaments via Monte Carlo, finds edges against sportsbook odds,
and sizes bets with fractional Kelly.

> **Status (June 2026):** A code audit found a lookahead bug in the original backtest (DataGolf reuses
> `event_id` across annual editions; ~47% of H2H bets were priced with models trained on future data).
> After the fix, the corrected 2023–24 backtest shows **no demonstrated H2H edge vs the Pinnacle closing
> line** (49.8% win rate, −5.3% ROI). **Live betting is paused** pending the frozen-parameter 2025–26
> out-of-sample validation. Outright-winner betting still fails calibration. See [Results](#results).

---

## Motivation

I love a lot of things, but also: golf and prediction markets. This project started as an attempt to put them together —
can a model actually beat the book on PGA Tour events?

My first instinct was the obvious one: **predict the outright winner.** It didn't work. A golf tournament
is ~150 players over 4 rounds with enormous variance — the winner is often a 30/1 to 80/1 longshot, and
there simply isn't enough signal to consistently identify them ahead of the market. Too little data, too
much noise. My model was systematically *overconfident* on longshots (its probability spread was ~1.9× the
market's), and the backtest profits, while positive, were concentrated in a handful of lucky events — not a
real, repeatable edge.

So I pivoted to **head-to-head matchup betting** ("will Player A beat Player B this week?"). This turned out
to be the right call:

- A matchup is a **2-outcome** market instead of a 150-outcome one — far less noise to fight.
- Sportsbook **vig is ~2–3%** on matchups vs. ~25% overround baked into outright markets.
- The model only has to rank two players relative to each other, which is exactly what strokes-gained data
  is good at.

The matchup thesis (less noise, less vig, pairwise ranking is what strokes-gained data is good at) still
stands — but the original backtest that appeared to validate it contained a lookahead bug, and the corrected
numbers do not show an edge against the Pinnacle closing line. The honest state of the project is below.

---

## Results

Evaluated on a **2023–2024 holdout** (expanding window) after training on 2017–2022. Canonical record:
[`golf_model/artifacts/outputs/backtest_2023_2024_results.json`](golf_model/artifacts/outputs/backtest_2023_2024_results.json),
reproducible headlessly with `cd golf_model && python run_holdout_backtest.py`.

### The bug that invalidated the original results

The originally published H2H numbers (1,854 bets, 56.6% win rate, +6.0% ROI, Sharpe 1.48) were an artifact
of a **lookahead bug**: DataGolf reuses `event_id` across annual editions of the same tournament, the
backtest cache was keyed by `event_id` alone (2024 predictions silently overwrote 2023), and the matchup
evaluation joined odds without a year filter — so ~47% of 2023 bets were priced with models that had been
trained on data through 2024. The same collision also merged two years' fields and recorded wrong winners
for the outright evaluation. Both are fixed (composite `(event_id, year)` keys end-to-end, regression
tests in [`golf_model/tests/`](golf_model/tests/)).

### Head-to-head matchups — corrected (live betting paused)

| Metric | Original (leaked) | Corrected |
|---|---|---|
| Bets | 1,854 | 2,075 across 55 events |
| Win rate | 56.6% | **49.8%** |
| ROI (Kelly-sized) | +6.0% | **−5.3%** |
| Flat-stake ROI | — | −4.7% (cluster-bootstrap 95% CI: −9.8% to +0.3%) |
| Sharpe (annualized) | 1.48 | −1.33 |
| P(true ROI ≤ 0) | — | 0.97 |

Per-year results are now symmetric (2023: 49.0% WR, 2024: 50.6% WR) — the original run's suspicious
63.8%/50.3% split between years was the leak's signature. **All H2H gates fail except sample size.**

Two further caveats on even these corrected numbers:

- The betting hyperparameters (edge threshold, probability temperature, sizing) were historically tuned on
  this same 2023–24 holdout, so this is a diagnostic, not an out-of-sample estimate. A frozen-parameter
  2025–26 OOS validation is the next gating step before any live betting resumes.
- The benchmark is the **Pinnacle closing line** — the sharpest available price. Live execution at softer
  books / earlier lines is a lower bar, but no edge vs the close means no demonstrated skill.

### Outright winners — still shelved

The corrected backtest confirms the original verdict:

- **Calibration:** model Brier 0.0149 > market Brier 0.0127 (worse than the book). **FAIL**
- **Significance:** model does not beat the market at any reasonable confidence. **FAIL**
- The betting-metrics gate technically passes (+21% ROI on 84 bets), but the sample is small and
  concentration-prone — the same pattern the audit flagged as luck, not edge.

The honest takeaway: **the model currently ranks players roughly as well as the closing line, not better.**
The original "diversified matchup edge" conclusion was an artifact of the lookahead bug.

---

## How it works

Conceptual generative model for golfer *i* in round *r* of tournament *t*:

```
Y_{i,r,t} = μ_{i,t} + γ_{c(t)} · δ_i + ε_{i,r,t}
```

**What actually runs in production** ([`golf_model/models/production_ewma.py`](golf_model/models/production_ewma.py))
is an empirical-Bayes-flavored heuristic stack, *not* the full PyMC hierarchical model (that exists in
[`golf_model/models/bayesian_core.py`](golf_model/models/bayesian_core.py) as research code, validated in
notebook 04 but never promoted):

1. **μ_{i,t}** — dual-decay EWMA skill estimates over strokes-gained (25 rounds / 120 days half-life)
2. Course-specific SG component re-weighting (from the historical variance of SG components at the event)
3. Recent-form blend (40% weight on the last 8 rounds)
4. Event-history course-fit with normal-conjugate shrinkage — a simplified stand-in for **γ_{c(t)} · δ_i**
5. **ε ~ Student-t(ν=6)** — heavy-tailed round noise with within-tournament correlation ρ=0.15

The pipeline then runs **50K Monte Carlo tournament simulations** (cut rules, playoffs, cut-aware H2H
settlement per real 72-hole match rules) to produce P(win), P(top-5/10), P(make-cut), and P(A beats B) for
every player. Sportsbook odds are de-vigged, compared to model probabilities to detect edges, and bets are
sized with **fractional (¼) Kelly** under per-bet and per-event exposure caps.

Deeper detail: [`ARCHITECTURE.md`](ARCHITECTURE.md) (15-pipe design) and
[`mathbehind/`](mathbehind/) (full mathematical writeup).

---

## What you need to run it (and why the data isn't here)

This repo is **code only**. To run it for real you need two paid/external pieces that I can't redistribute:

1. **DataGolf — Scratch PLUS subscription** ([datagolf.com](https://datagolf.com)). This is the source of
   round-level strokes-gained data, historical closing odds, and matchup odds. It's a **paid tier and the
   data is proprietary**, so none of the CSVs are committed here. You bring your own subscription and pull
   the data with the scripts in [`scripts/`](scripts/) (e.g. `scripts/14_pull_season_data.py`).
2. **Anthropic Claude API key** — only needed for the optional news-research briefing (see below). ~$0.30/week.

Free dependency: **Open-Meteo** for weather (no auth required). (if you develop thsi part of the project tho)

**Excluded from the repo** (see [`.gitignore`](.gitignore)): the `data/` directory (proprietary DataGolf
data), trained model `artifacts/`, research `cache/`, run logs, and all secrets. You regenerate data and
models locally.

---

## Quick start

```bash
# 1. Environment
conda env create -f golf_model/environment.yml          # loose spec (latest compatible versions)
# conda env create -f golf_model/environment.lock.yml   # exact pinned versions — use this to
#                                                       # reproduce the published backtest numbers
conda activate golf_model

# 2. Credentials
cp golf_model/.env.example golf_model/.env
#    Fill in GOLF_DATAGOLF_API_KEY and (optionally) ANTHROPIC_API_KEY

# 3. Pull data (requires a DataGolf Scratch PLUS subscription)
python scripts/14_pull_season_data.py --key YOUR_DATAGOLF_API_KEY

# 4. Train / backtest / predict
python golf_model/run_pipeline.py --mode train
python golf_model/run_pipeline.py --mode backtest
python golf_model/run_pipeline.py --mode predict --event_id 12345

# 5. Reproduce the canonical validation runs (results JSONs in golf_model/artifacts/outputs/)
cd golf_model
python run_holdout_backtest.py    # 2023-24 holdout backtest + integrity checks (~30-60 min)
python run_oos_validation.py      # frozen-parameter 2025-26 out-of-sample validation

# 6. Tests (regression tests for the lookahead fix, H2H settlement, Sharpe, sizing caps)
cd golf_model && python -m pytest tests/ -q
```

The weekly workflow lives in the notebooks — see
[`golf_model/notebooks/09_live_deployment.ipynb`](golf_model/notebooks/09_live_deployment.ipynb) and
[`golf_model/README.md`](golf_model/README.md) for the full operational routine.

---

## Repository layout

```
golf_data-model/
├── ARCHITECTURE.md          # 15-pipe architecture spec
├── mathbehind/              # Mathematical framework writeup (LaTeX + PDF)
├── golf_model/              # Main Python package
│   ├── config/              # All config & hyperparameters (settings.py)
│   ├── data/                # Loaders, schemas, DataGolf + weather API clients
│   ├── features/            # SG decomposition, EWMA time-weighting, course features
│   ├── models/              # Baseline, hierarchical Bayesian, course-fit
│   ├── simulation/          # Monte Carlo tournament engine
│   ├── betting/             # De-vig, edge detection, Kelly sizing, bankroll
│   ├── validation/          # Brier, calibration, backtest, statistical tests
│   ├── research/            # News-intelligence agent (Claude API)
│   └── notebooks/           # Sequential pipeline 01–09
└── scripts/                 # DataGolf pull scripts (00–14)
```

---

## Roadmap / ways to improve

- **Weather integration.** Wind, rain, and temperature meaningfully shift scoring and favor certain player
  profiles. An Open-Meteo client is already scaffolded in
  [`golf_model/data/weather.py`](golf_model/data/weather.py); the next step is folding round-level weather
  into the noise/skill model so a bomber's edge in high wind (or a short-hitter's edge on a soft, calm track)
  is priced in.
- **Per-player course/track optimization.** This is the `γ_c · δ_i` course-fit term (Phase 4), currently
  optional because the course-feature dataset isn't populated. The goal: an 8-dimensional course vector
  (length, rough height, green speed, wind exposure, elevation, fairway width, green size, water) crossed
  with each player's strengths, so the model knows *this* player overperforms on *this kind* of track.
  Plan is Bayesian-ridge shrinkage with leave-one-course-out cross-validation.
- **News-intelligence agent.** A read-only pre-bet briefing
  ([`golf_model/research/news_agent.py`](golf_model/research/news_agent.py)) that scrapes recent Google News
  RSS for each player in a flagged matchup and uses the Claude API to summarize injury, form, equipment, and
  personal-life signals for human review before placing bets. It never modifies bets and is cached per
  player per week. Future extensions: Twitter/X integration (blocked for now by free-tier limits) and
  golf-podcast transcript analysis.
- **Outright model improvements.** Fix the overconfidence on longshots (probability tempering / better tail
  calibration) so the outright market can eventually clear the calibration gate.
- **Live odds + automated logging.** Tighter integration with sportsbook feeds and an automated P&L/CLV
  tracker.

---

***Thank you for your time!!! I greatly enjoyed working on this! Thank you for my friends for being a great motivation for me, and Sonnet for being my personal assistant on the way.***


---

## Disclaimer

This is a personal research project for studying prediction markets and Bayesian modeling. It is **not
financial advice**, and nothing here is a guarantee of profit, backtest results are historical and sports
betting carries real risk of loss. Bet responsibly and only what you can afford to lose, and follow the laws
and sportsbook terms in your jurisdiction.
