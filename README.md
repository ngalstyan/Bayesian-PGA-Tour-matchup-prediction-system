# Golf Betting Model — Hierarchical Bayesian Strokes-Gained

A quantitative system for betting PGA Tour golf. It models player skill with a hierarchical Bayesian
strokes-gained framework, simulates tournaments via Monte Carlo, finds edges against sportsbook odds,
and sizes bets with fractional Kelly.

> **Status:** Head-to-head (H2H) matchup betting passes all validation gates and is the live strategy.
> Outright-winner betting does **not** pass calibration — see [Results](#results).

---

## Motivation

I love two things: golf and prediction markets. This project started as an attempt to put them together —
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

On the matchup side the results held up across a proper expanding-window backtest, and that's what the
system bets today.

---

## Results

Validated on a strict **2023–2024 holdout** (expanding window, no lookahead), after training on 2017–2022.

### Head-to-head matchups — the live strategy

| Metric | Value |
|---|---|
| Bets | 1,854 across 55 events |
| Win rate | 56.6% |
| ROI | +6.0% |
| Sharpe | 1.48 |
| Max drawdown | 27.9% |
| Backtest P&L | +$37,176 |
| Avg edge / odds / stake | 13.4% / 1.94 / $336 |

All H2H gates pass: sample ≥ 100, win rate > 50%, ROI > 0%, Sharpe > 0.3.

### Outright winners — shelved

Backtest ROI looked high (84%), but it **fails the gates that matter**:

- **Calibration:** model Brier 0.013 > market Brier 0.011 (worse than the book).
- **Significance:** Diebold-Mariano p ≈ 0.99; the model beat the market in only ~18% of events.
- **Concentration:** ~100% of the profit came from 3 events — luck, not edge.

The takeaway that drove the whole project: **a diversified stream of small matchup edges beats chasing
longshot winners.**

---

## How it works

Core generative model for golfer *i* in round *r* of tournament *t*:

```
Y_{i,r,t} = μ_{i,t} + γ_{c(t)} · δ_i + ε_{i,r,t}
```

- **μ_{i,t}** — player skill, estimated with Bayesian shrinkage over strokes-gained components
  (off-the-tee, approach, around-the-green, putting), time-weighted with dual EWMA decay (25 rounds / 120 days).
- **γ_{c(t)} · δ_i** — player × course-fit interaction (optional Phase 4).
- **ε ~ Student-t(ν=6)** — heavy-tailed round noise, because golf scoring has fat tails.

The pipeline then runs **50K–100K Monte Carlo tournament simulations** (cut rules, round correlation,
playoffs) to produce P(win), P(top-5/10), P(make-cut), and P(A beats B) for every player. Sportsbook odds are
de-vigged (Shin's method), compared to model probabilities to detect edges, and bets are sized with
**fractional (¼) Kelly** under exposure caps.

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

Free dependency: **Open-Meteo** for weather (no auth required).

**Excluded from the repo** (see [`.gitignore`](.gitignore)): the `data/` directory (proprietary DataGolf
data), trained model `artifacts/`, research `cache/`, run logs, and all secrets. You regenerate data and
models locally.

---

## Quick start

```bash
# 1. Environment
conda env create -f golf_model/environment.yml
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

## Disclaimer

This is a personal research project for studying prediction markets and Bayesian modeling. It is **not
financial advice**, and nothing here is a guarantee of profit — backtest results are historical and sports
betting carries real risk of loss. Bet responsibly and only what you can afford to lose, and follow the laws
and sportsbook terms in your jurisdiction.
