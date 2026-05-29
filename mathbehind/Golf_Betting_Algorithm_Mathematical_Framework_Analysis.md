# Golf Betting Algorithm: Mathematical Framework Analysis

## Executive Summary

This analysis presents three mathematically rigorous frameworks for developing a PGA Tour outright winner betting algorithm, each drawn from distinct theoretical traditions and offering complementary strengths. The frameworks are selected to span the spectrum from Bayesian probabilistic modeling, through ranking-theoretic approaches, to latent state-space methods—ensuring the decision-maker can choose (or combine) approaches based on their data infrastructure, risk tolerance, and desired edge profile.

**Framework 1**, the Hierarchical Bayesian Strokes-Gained Decomposition Model, is the most golf-native approach. It directly leverages Broadie's strokes-gained revolution, decomposes player skill into sub-components (driving, approach, short game, putting), and uses Bayesian shrinkage to handle the massive heterogeneity in sample sizes across players. Its primary theoretical edge comes from properly quantifying uncertainty in player ability—something the betting market systematically underestimates for mid-tier players.

**Framework 2**, the Dynamic Plackett-Luce Model with Score-Driven Dynamics, treats each tournament as a ranking-generating process and directly models the probability of any finishing order. Rooted in Luce's choice axiom and generalized via Plackett's permutation model, it naturally handles the multi-competitor structure of golf tournaments without reducing them to pairwise comparisons. Its time-varying worth parameters, updated via the GAS (Generalized Autoregressive Score) mechanism, capture form cycles with mathematical elegance.

**Framework 3**, the Latent Factor State-Space Model with Stochastic Volatility, borrows from quantitative finance to model each golfer as a time-varying latent process with heteroskedastic performance variance. By allowing both skill level and consistency to fluctuate independently (analogous to asset price and volatility dynamics), this framework captures the empirically observed phenomenon that golfer variance is itself predictive—streaky players and those entering volatile form periods represent distinct betting opportunities.

The three frameworks are not mutually exclusive. The optimal long-run strategy likely involves an ensemble approach, weighting each model's win probabilities using a meta-learner calibrated on out-of-sample log-loss.

---

## Framework 1: Hierarchical Bayesian Strokes-Gained Decomposition Model

### 1. Framework Name & Classification

**Formal Classification:** Hierarchical Bayesian Random-Effects Model with Empirical Bayes Shrinkage

**Primary Theoretical Foundation:** Bayesian probability theory, empirical Bayes estimation (Efron, 2010), and the strokes-gained decomposition framework (Broadie, 2008, 2012, 2014)

### 2. Mathematical Foundation

**Core Generative Model.** For golfer $i$ in round $r$ of tournament $t$, the adjusted strokes-gained score is modeled as:

$$Y_{i,r,t} = \mu_{i,t} + \boldsymbol{\gamma}_{c(t)}^\top \boldsymbol{\delta}_i + \varepsilon_{i,r,t}$$

where:
- $Y_{i,r,t}$ is the adjusted strokes-gained relative to a baseline (field average or scratch benchmark)
- $\mu_{i,t}$ is golfer $i$'s latent true ability at tournament $t$
- $\boldsymbol{\gamma}_{c(t)}$ is a course-specific feature vector for course $c$ associated with tournament $t$
- $\boldsymbol{\delta}_i$ is golfer $i$'s course-fit interaction coefficient vector
- $\varepsilon_{i,r,t} \sim \mathcal{N}(0,\, \sigma_i^2)$ is the round-level noise with player-specific variance

**Skill Decomposition.** Following Broadie (2012, 2014) and the empirical Bayes analysis of Brill and Wyner (2025), latent ability is decomposed into sub-components:

$$\mu_{i,t} = \mu_{i,t}^{\text{OTT}} + \mu_{i,t}^{\text{APP}} + \mu_{i,t}^{\text{ARG}} + \mu_{i,t}^{\text{PUTT}}$$

where OTT = off-the-tee, APP = approach, ARG = around-the-green, PUTT = putting.

For each sub-component $s \in \{\text{OTT},\, \text{APP},\, \text{ARG},\, \text{PUTT}\}$, we observe hole-level strokes gained:

$$X_{i,j,s} \mid \mu_{i,s} \;\sim\; \mathcal{N}\!\left(\mu_{i,s},\; \sigma_{s,\text{obs}}^2\right)$$

$$\mu_{i,s} \;\sim\; \mathcal{N}\!\left(\mu_s,\; \tau_s^2\right)$$

This yields the standard hierarchical normal-normal model. The empirical Bayes posterior for each golfer-component is:

$$\mathbb{E}\!\left[\mu_{i,s} \mid \mathbf{X}\right] = B_s \cdot \hat{\mu}_s + (1 - B_s) \cdot \bar{X}_{i,s}$$

where the shrinkage factor is:

$$B_s = \frac{\sigma_{s,\text{obs}}^2 \,/\, N_{i,s}}{\tau_s^2 + \sigma_{s,\text{obs}}^2 \,/\, N_{i,s}}$$

This is the James-Stein shrinkage estimator generalized to the hierarchical setting, with $\tau_s^2$ and $\sigma_{s,\text{obs}}^2$ estimated via marginal maximum likelihood (Efron, 2010; Brill & Wyner, 2025).

**Key Insight from Recent Research.** Brill and Wyner (2025, arXiv:2506.21822) demonstrate via the Benjamini-Hochberg multiple testing procedure that at $\alpha = 0.10$, approximately 259 golfers show statistically significant driving skill, 72 show significant approach skill, but putting skill is "nearly indistinguishable from noise." This has profound implications: the model should invest far more degrees of freedom in tee-to-green components and treat putting estimates with heavy shrinkage toward the grand mean.

**Temporal Dynamics.** Following the DataGolf methodology (Courchene & Courchene), latent ability evolves via an exponentially weighted moving average (EWMA) scheme. For golfer $i$, the predicted ability before round $r$ at time $t$ is:

$$\hat{\mu}_{i,t} = \sum_{k < t} w(k;\, \lambda) \cdot Y_{i,k}$$

with weights:

$$w(k;\, \lambda) = \frac{\lambda^{\,(t-k)}}{\displaystyle\sum_{j < t} \lambda^{\,(t-j)}}$$

The decay parameter $\lambda$ is estimated from historical data, and separate decay rates are maintained for (a) sequence-weighted averaging (round order, ignoring calendar gaps) and (b) time-weighted averaging (calendar distance between rounds). The final prediction blends these two averages, with the blend ratio depending on the regularity of the golfer's schedule.

**Regression to the Mean.** The predicted ability incorporates Bayesian shrinkage toward the population mean, governed by the number of rounds in the estimation window:

$$\hat{\mu}_{i,t}^{\,\text{shrunk}} = \bigl(1 - \alpha(N_i)\bigr) \cdot \hat{\mu}_{i,t} + \alpha(N_i) \cdot \bar{\mu}$$

where $\alpha(N_i)$ is a decreasing function of rounds played, estimated empirically.

**Tournament Simulation.** Given estimated means $\{\hat{\mu}_{i,t}\}$ and variances $\{\hat{\sigma}_i^2\}$ for all $n$ golfers in a field, tournament outcomes are simulated $M$ times (typically $M = 10{,}000$–$100{,}000$):

For simulation $m$:
1. For each golfer $i$, draw 4 round scores: $Y_{i,r}^{(m)} \sim \mathcal{N}\!\left(\hat{\mu}_{i,t},\; \hat{\sigma}_i^2\right)$ for $r = 1,\ldots,4$
2. Apply course-fit adjustments: $Y_{i,r}^{(m),\text{adj}} = Y_{i,r}^{(m)} + \boldsymbol{\gamma}_{c(t)}^\top \hat{\boldsymbol{\delta}}_i$
3. Apply weather/wave adjustments: $Y_{i,r}^{(m),\text{final}} = Y_{i,r}^{(m),\text{adj}} + w_{r,\text{wave}(i)}$
4. Compute 72-hole total: $S_i^{(m)} = \displaystyle\sum_{r=1}^{4} Y_{i,r}^{(m),\text{final}}$
5. Determine winner: $\text{Winner}^{(m)} = \arg\min_i \; S_i^{(m)}$

Win probability estimate:

$$\hat{P}(i \text{ wins}) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\!\left[\text{Winner}^{(m)} = i\right]$$

**Probability Distributions Employed:**
- Normal distribution for round-level scores (justified by CLT over 18 holes, though heavier tails are observed empirically)
- Hierarchical normal-normal for the Bayesian shrinkage structure
- Optional: skew-normal or Student-$t$ for more realistic tail behavior (DataGolf notes scores are "not quite normal")

**Key Theorems & Principles:**
- James-Stein estimator dominance (Efron & Morris, 1973): the shrinkage estimator dominates the MLE in terms of total squared error when $\geq 3$ parameters are estimated simultaneously
- Benjamini-Hochberg FDR control (1995): provides principled determination of which sub-skills are statistically distinguishable from noise
- Central Limit Theorem: justifies the approximate normality of 18-hole round scores as sums of hole-level outcomes

### 3. Golf-Specific Adaptations

**Course Fit.** The course-fit interaction $\boldsymbol{\gamma}_{c(t)}^\top \boldsymbol{\delta}_i$ captures the empirically significant phenomenon that certain golfers systematically outperform at specific course types. The feature vector $\boldsymbol{\gamma}_{c(t)}$ encodes:
- Course length and par configuration
- Green complexity (stimpmeter, slopes, size)
- Fairway width and rough penalty
- Wind exposure and altitude
- Historical strokes-gained-by-category averages for the field

The golfer's response vector $\boldsymbol{\delta}_i$ captures their differential sensitivity to these features, estimated from historical course-level data.

**Form Cycles.** The dual EWMA structure (sequence-weighted and time-weighted) naturally captures both gradual skill evolution and the impact of extended layoffs. The DataGolf methodology specifically handles the case of golfers returning from injury (where the time-weighted average diverges dramatically from the sequence-weighted average).

**Variance Heterogeneity.** Player-specific variance $\sigma_i^2$ is critical in outright winner markets. Two golfers with the same expected ability but different variances have very different win probabilities—the higher-variance player is more likely to win in large fields. This is estimated from residuals of the ability model.

**Field Strength.** Win probabilities automatically account for field strength through the simulation mechanism: when stronger players are present, the distribution of minimum scores shifts, reducing all other players' win probabilities proportionally.

### 4. Analogous Applications

**Horse Racing (Ziemba & Hausch, 1986, Management Science).** The place-and-show betting system at racetracks uses a fundamentally similar approach: estimate true win probabilities from fundamentals, compare to market-implied probabilities, and exploit systematic discrepancies. The hierarchical structure mirrors the treatment of horse ability as a function of trainer, bloodline, and surface—directly analogous to golfer ability as a function of skill components and course fit.

**Baseball (PECOTA, Nate Silver / Baseball Prospectus).** PECOTA uses comparable aging curves and similarity-based shrinkage to project player performance. The mathematical parallel is the use of hierarchical models to borrow strength across similar players, with temporal decay functions capturing career trajectory dynamics.

**Macroeconomic Forecasting (Federal Reserve DSGE Models).** Dynamic stochastic general equilibrium models use a similar hierarchical structure: observable economic indicators are noisy realizations of latent state variables (potential output, natural rate), estimated via Bayesian filtering. The shrinkage principle is identical—extreme observations are discounted toward prior beliefs about parameter stability.

**Mathematical Parallel:** In all cases, the fundamental structure is $\text{observation} = \text{latent signal} + \text{noise}$, with hierarchical priors enabling partial pooling across related units (golfers/horses/economies).

### 5. Data Requirements & Feature Engineering

**Required Variables:**
- $\text{SG}_{i,r,t}^{\text{Total}}$: Total strokes gained per round (continuous, $\in \mathbb{R}$)
- $\text{SG}_{i,r,t}^{\text{OTT}}$, $\text{SG}_{i,r,t}^{\text{APP}}$, $\text{SG}_{i,r,t}^{\text{ARG}}$, $\text{SG}_{i,r,t}^{\text{PUTT}}$: Sub-component strokes gained
- $\text{CourseID}_t$: Course identifier linking to course features
- $\text{Date}_t$: Calendar date for time-weighted calculations
- $\text{FieldStrength}_t$: Derived from participating golfers' current ratings
- $\text{Weather}_{r,t}$: Wind speed, precipitation, temperature (for wave adjustments)
- $\text{WaveAssignment}_{i,r,t}$: AM/PM tee time for weather wave effects

**Transformations:**
- Scores are adjusted to strokes-gained-relative-to-field (removes course/round difficulty)
- Course features are standardized to $z$-scores across all courses
- Temporal weights are computed via exponential decay with estimated half-life
- Interaction terms: $\text{SG}_{\text{component}} \times \text{Course}_{\text{feature}}$ for course-fit modeling

**Data Structure:** Unbalanced panel data (golfer $\times$ tournament $\times$ round), with time-series structure within each golfer and cross-sectional structure within each tournament-round.

**Minimum Sample Sizes:**
- For individual golfer ability: $\geq 50$ rounds for reasonable shrinkage (Brill & Wyner suggest $150+$ holes per stroke category for sub-component analysis)
- For course-fit interactions: $\geq 20$ tournaments per course, $\geq 3$ visits per golfer-course pair
- For variance estimation: $\geq 100$ rounds per golfer
- For the overall model: $\geq 3$ years of PGA Tour data (${\sim}150$ events) for stable parameter estimation

### 6. Advantage Analysis

**Theoretical Edge.** The primary edge comes from the hierarchical Bayesian structure's superior handling of uncertainty. Market makers and casual bettors typically anchor on recent performance without proper shrinkage. A golfer who gains $+3$ strokes in one tournament may be priced as if their true ability is $+3$, when the Bayesian posterior (accounting for noise) might suggest $+1.5$. This creates systematic mispricing.

**Specific Market Inefficiencies Exploited:**
1. **Recency bias in longshot pricing:** Mid-field players with one strong recent result get temporarily over-backed; the model's shrinkage provides more accurate probabilities
2. **Course-fit neglect:** Market odds often under-weight the systematic ${\sim}0.5$–$1.0$ strokes-gained advantage that certain players possess at specific courses
3. **Form versus noise confusion:** The dual EWMA structure distinguishes genuine form changes from statistical noise more effectively than simple recent-performance averages
4. **Variance blindness:** Outright winner markets may systematically underprice high-variance players in large fields (and overprice them in small fields)

**Robustness Properties:**
- Bayesian shrinkage inherently guards against overfitting by regularizing extreme estimates
- The EWMA structure is non-parametric in nature, requiring only the decay rate to be estimated
- Monte Carlo simulation propagates all sources of uncertainty into win probability estimates

### 7. Limitation Analysis

**Violated Assumptions:**
- **Normality of round scores:** Golf scores exhibit slight right-skew and excess kurtosis (catastrophic holes). Mitigation: use skew-$t$ distributions or non-parametric simulation.
- **Independence of rounds within a tournament:** Psychological momentum and weather persistence create within-tournament correlation. Mitigation: include auto-correlated error terms or tournament random effects.
- **Stationarity of skill within a tournament week:** Injuries, equipment changes, or "hot hand" effects may cause intra-tournament skill shifts.

**Edge Cases:**
- Players returning from extended absence (limited recent data for EWMA)
- Course redesigns or novel courses (no historical course-fit data)
- Extreme weather events outside the range of training data
- Tournament format changes (no-cut events, modified scoring)

**Computational Complexity:** $\mathcal{O}(M \times n \times 4)$ for simulation, where $M$ is simulation count, $n$ is field size. With $M = 100{,}000$ and $n = 156$, this is $\approx 62.4$ million draws—trivially fast on modern hardware.

### 8. Betting Strategy Integration

**Probability → Edge → Stake Pipeline:**

1. **Compute model probability:** $\hat{P}_{\text{model}}(i \text{ wins})$ from simulation
2. **Extract market probability:** $P_{\text{market}}(i \text{ wins}) = 1 \,/\, \bigl(d_i \cdot (1 - \text{overround\_share})\bigr)$
3. **Calculate edge:** $\text{Edge}_i = \hat{P}_{\text{model}}(i) - P_{\text{market}}(i)$
4. **Kelly staking:** For outright bets with odds $b$ (net decimal odds $- 1$):

$$f_i^{*} = \frac{b \cdot \hat{P}_{\text{model}}(i) - \bigl(1 - \hat{P}_{\text{model}}(i)\bigr)}{b}$$

5. **Fractional Kelly:** Apply fraction $\kappa \in [0.10,\, 0.25]$ to account for estimation uncertainty:

$$f_i = \kappa \cdot \max\!\left(0,\; f_i^{*}\right)$$

**Correlated Bets.** In a single tournament, all outright bets are mutually exclusive (only one winner). However, bets across top-5/top-10/outright markets on the same golfer are positively correlated. The simultaneous Kelly criterion for correlated bets in an outright market (where exactly one of $n$ mutually exclusive outcomes occurs) is:

$$f_i^{*} = \hat{P}_{\text{model}}(i) - \frac{1 - \hat{P}_{\text{model}}(i)}{b_i}$$

subject to $\sum_i f_i \leq 1$ (can't bet more than the bankroll), which is naturally satisfied for outright winner markets where all $f_i^{*}$ are small.

**Threshold Criteria:**
- Minimum edge: $\hat{P}_{\text{model}}(i) \,/\, P_{\text{market}}(i) > 1.05$ (5% edge threshold)
- Minimum Kelly fraction: $f_i > 0.002$ (0.2% of bankroll) to justify transaction costs
- Maximum single bet: 2% of bankroll (hard cap regardless of Kelly)
- Maximum tournament exposure: 8% of bankroll across all bets in one event

### 9. Comparison Metrics

**Primary Metrics:**
- **Log-loss (cross-entropy):**

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{n_t} \Bigl[y_{i,t} \cdot \ln\!\bigl(\hat{P}(i)\bigr) + (1 - y_{i,t}) \cdot \ln\!\bigl(1 - \hat{P}(i)\bigr)\Bigr]$$

  where $y_{i,t} = 1$ if golfer $i$ won tournament $t$. The most principled metric for probabilistic forecasts.

- **Brier Score:**

$$\text{BS} = \frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{n_t} \bigl(\hat{P}(i) - y_{i,t}\bigr)^2$$

  Less sensitive to extreme probabilities than log-loss.
- **Calibration:** Compare predicted vs. observed win rates across deciles of predicted probability.
- **ROI:** Net profit / total amount wagered. The ultimate practical metric, but highly variance-dependent over short horizons.
- **Sharpe Ratio:** Annualized $(\bar{r} - r_f) \,/\, \sigma(r)$, calculated on a per-tournament basis.

**Backtesting Methodology:**
- Expanding window: train on all data up to tournament $t$, predict tournament $t+1$
- Never use future data for any component (course-fit, ability estimates, variance estimates)
- Account for survivorship bias: include all golfers who were in the field, not just those who made the cut
- Simulate realistic bet placement: use closing odds, not opening odds

**Validation Tests:**
- Probability integral transform (PIT) histogram: check that predicted CDFs are uniformly distributed
- Diebold-Mariano test: compare forecast accuracy against the market benchmark
- Murphy decomposition: separate Brier score into reliability, resolution, and uncertainty components

### 10. Development Roadmap

**Phase 1 (Foundation):** Build the basic strokes-gained regression model
- Collect $\geq 3$ years of round-level strokes-gained data
- Implement course-round difficulty adjustments
- Validate: check that model RMSE < naive (field average) RMSE

**Phase 2 (Temporal Dynamics):** Add EWMA weighting
- Estimate optimal decay parameters via cross-validation on historical data
- Implement dual (time-weighted / sequence-weighted) averaging
- Validate: check improvement in out-of-sample $R^2$ over Phase 1

**Phase 3 (Hierarchical Structure):** Add Bayesian shrinkage and sub-component decomposition
- Estimate sub-component shrinkage factors using empirical Bayes
- Add player-specific variance estimation
- Validate: check log-loss improvement over Phase 2 via simulation

**Phase 4 (Course Fit):** Add golfer $\times$ course interactions
- Build course feature database
- Estimate course-fit interaction coefficients
- Validate: check for statistically significant course-fit effects via leave-one-course-out testing

**Phase 5 (Betting Integration):** Connect to market odds and implement staking
- Build odds scraping and normalization pipeline
- Implement fractional Kelly staking
- Backtest on $\geq 2$ years of out-of-sample data
- Milestone: demonstrate positive expected ROI at $\geq 2\%$ significance level

---

## Framework 2: Dynamic Plackett-Luce Model with Score-Driven Dynamics

### 1. Framework Name & Classification

**Formal Classification:** Time-Varying Plackett-Luce Ranking Model with Generalized Autoregressive Score (GAS) Dynamics

**Primary Theoretical Foundation:** Luce's choice axiom (1959), Plackett's permutation distribution (1975), Generalized Autoregressive Score models (Creal, Koopman & Lucas, 2013)

### 2. Mathematical Foundation

**The Plackett-Luce Distribution.** For a tournament with $N$ golfers producing a complete ranking $\mathbf{Y} = (y_1, y_2, \ldots, y_N)$ where $y_1$ finished first, $y_2$ second, etc., the probability of observing this specific ranking is:

$$P\!\left(\mathbf{Y} = (y_1,\ldots,y_N) \mid \mathbf{f}\right) = \prod_{k=1}^{N} \frac{\exp(f_{y_k})}{\displaystyle\sum_{j=k}^{N} \exp(f_{y_j})}$$

where $\mathbf{f} = (f_1, \ldots, f_N)^\top$ is the vector of log-worth parameters. The $\exp(\cdot)$ transformation ensures positive worth values while allowing $f_i \in \mathbb{R}$.

**Win Probability.** The probability that golfer $i$ wins the tournament is simply the first factor:

$$P(i \text{ wins} \mid \mathbf{f}) = \frac{\exp(f_i)}{\displaystyle\sum_{j=1}^{N} \exp(f_j)}$$

This is the multinomial logit (softmax) function. It satisfies the Independence of Irrelevant Alternatives (IIA) property, also known as Luce's choice axiom: the ratio of win probabilities for any two golfers is independent of the other competitors in the field.

**Time-Varying Worth Parameters via GAS Dynamics.** Following the framework of Creal, Koopman, and Lucas (2013), and as applied to ranking data by Hol and Koopman (2021, arXiv:2101.04040), the log-worth parameters evolve over time:

$$f_{i,t+1} = \omega_i + A \cdot s_{i,t} + B \cdot f_{i,t} + \mathbf{X}_{i,t}^\top \boldsymbol{\beta}$$

where:
- $\omega_i$ is a golfer-specific intercept (long-run ability level)
- $s_{i,t}$ is the scaled score of the Plackett-Luce log-likelihood evaluated at the observed ranking in tournament $t$
- $A$ and $B$ are autoregressive parameters (shared across golfers or group-specific)
- $\mathbf{X}_{i,t}$ is a vector of exogenous covariates (course fit, recent rest, etc.)
- $\boldsymbol{\beta}$ is the coefficient vector on exogenous covariates

**The Score Function.** The score $s_{i,t}$ is the derivative of the log-likelihood with respect to $f_i$, evaluated at the current parameter values:

$$s_{i,t} = \frac{\partial \ln P(\mathbf{Y}_t \mid \mathbf{f}_t)}{\partial f_i} = \mathbf{1}[i \text{ was ranked}] - \sum_{\substack{k:\, i \text{ still in} \\ \text{contention at stage } k}} \frac{\exp(f_i)}{\displaystyle\sum_{j=k}^{N} \exp(f_{y_j})}$$

Intuitively, the score reflects the discrepancy between where golfer $i$ finished and where the model expected them to finish. If a low-rated golfer wins, their score is strongly positive; if a top-rated golfer finishes last, their score is strongly negative. The GAS update mechanism uses this "surprise" information to update worth parameters.

**Fisher Information Scaling.** In the full GAS framework, the score is scaled by the inverse Fisher information matrix:

$$\tilde{s}_{i,t} = \mathcal{I}(\mathbf{f}_t)^{-1} \cdot \nabla_{f_i} \ln P(\mathbf{Y}_t \mid \mathbf{f}_t)$$

For large $N$, the Fisher information is computationally expensive (it involves sums over all permutations). In practice, one uses Monte Carlo approximation or the diagonal approximation:

$$\mathcal{I}_{ii}(\mathbf{f}_t) \approx \text{Var}_{\mathbf{Y} \mid \mathbf{f}_t}\!\left[\frac{\partial \ln P(\mathbf{Y} \mid \mathbf{f}_t)}{\partial f_i}\right]$$

**Partial Rankings and Truncation.** In golf, we observe the full finishing order for made-cut golfers and only a partial ranking (relative to the cut line) for those who miss the cut. The truncated Plackett-Luce model (Henderson & Kirrane, 2018, Bayesian Analysis) handles this by conditioning only on the observed top positions:

$$P(y_1,\ldots,y_K \mid \mathbf{f}) = \prod_{k=1}^{K} \frac{\exp(f_{y_k})}{\displaystyle\sum_{j=k}^{N} \exp(f_{y_j})}$$

where $K < N$ is the number of positions observed (those making the cut), and the denominator at each stage still includes all remaining golfers (including those who didn't reach that stage of selection).

**Identification.** The Plackett-Luce model is identified only up to an additive constant (adding $c$ to all $f_i$ leaves probabilities unchanged). The standard normalization is $\sum_i f_i = 0$ or fixing one golfer's parameter at zero.

### 3. Golf-Specific Adaptations

**Course-Conditioned Worth.** The exogenous covariate term $\mathbf{X}_{i,t}^\top \boldsymbol{\beta}$ allows course characteristics to shift golfer worth parameters tournament-by-tournament:

$$f_{i,t} = f_{i,t}^{\text{base}} + \sum_k \beta_k \cdot \bigl(\text{CourseFeature}_k \times \text{GolferAttribute}_{k,i}\bigr)$$

This captures the reality that a golfer's relative strength depends on the course. A bombers' course elevates long hitters' worth parameters; a precision course elevates accurate iron players.

**Field Composition Effects.** The Plackett-Luce model automatically handles field composition: win probabilities are computed over the specific set of competitors in each tournament. When Scottie Scheffler enters an event, all other golfers' win probabilities decrease proportionally to their relative worth.

**Handling Ties.** The extended Plackett-Luce model with Davidson (1970) tie parameters accommodates the tied finishing positions common in golf:

$$P(\text{tie between } i \text{ and } j) \propto \delta \cdot \sqrt{\exp(f_i) \cdot \exp(f_j)}$$

where $\delta$ is a tie prevalence parameter. Dead-heat rules in golf betting make proper tie handling critical for accurate probability estimation.

**Form vs. Class Decomposition.** Following Baker and McHale (2022), who apply an Ornstein-Uhlenbeck process to golf, the GAS dynamics naturally decompose into long-run class (captured by $\omega_i$ and the slow-moving component $B \cdot f_{i,t}$) and short-run form (captured by the recent score update $A \cdot s_{i,t}$). The ratio $A/B$ determines the relative responsiveness to recent results versus long-term track record.

### 4. Analogous Applications

**Formula 1 Racing (Henderson & Kirrane, 2018, Bayesian Analysis).** A time-weighted truncated Plackett-Luce model was applied to F1 results from 2010-2013, producing season-ahead championship probability forecasts. The mathematical structure is directly analogous to golf: $N$ competitors produce a ranking in each event, abilities evolve over time, and partial rankings (DNFs) require truncation. The authors demonstrate that the Plackett-Luce model outperforms simple Elo-style updates for motorsport rankings.

**Horse Racing (Stern, 1990).** Stern's foundational work on ranking models for horse racing used a related parametric model to estimate horse abilities from finishing orders, establishing the precedent for applying ranking distributions to betting markets.

**Search Engine Ranking / Information Retrieval (Liu, 2011).** The Plackett-Luce model is widely used in Learning-to-Rank frameworks (e.g., Google's search ranking), where documents are ranked by relevance. The mathematical optimization (maximizing the likelihood of observed click-through rankings) is identical in structure to maximizing the likelihood of observed tournament finishing orders.

**Large Language Model Alignment (RLHF).** The Bradley-Terry model (the pairwise special case of Plackett-Luce) is used in Reinforcement Learning from Human Feedback to rank model outputs. The mathematical framework of inferring latent quality from observed preference orderings is shared.

### 5. Data Requirements & Feature Engineering

**Required Variables:**
- $\text{Rank}_{i,t}$: Finishing position of golfer $i$ in tournament $t$ (ordinal)
- $\text{MadeCut}_{i,t}$: Binary indicator for making the 36-hole cut
- $\text{Score}_{i,r,t}$: Raw stroke total per round
- $\text{CourseFeatures}_{c(t)}$: Vector of course characteristics (length, fairway width, green speed, etc.)
- $\text{GolferAttributes}_{i,t}$: Current ability decomposition (SG:OTT, SG:APP, etc.)

**Transformations:**
- Rankings are converted to the ordering format required by the Plackett-Luce likelihood
- Tied positions are handled via the extended PL tie model
- Covariates are standardized and interaction terms constructed for course-fit effects
- Recent tournament results are implicitly encoded in the GAS state variable

**Data Structure:** Time-ordered sequence of tournament rankings (panel of tournaments $\times$ golfers), with each observation being a complete or partial ranking.

**Minimum Sample Sizes:**
- For GAS parameter estimation ($A$, $B$, $\omega$): $\geq 30$ tournaments per golfer for stable individual-level estimates; $\geq 100$ tournaments for shared GAS dynamics
- For covariate effects ($\boldsymbol{\beta}$): Standard regression requirements; $\geq 10$ observations per covariate
- For the full model: $\geq 2$ years of complete PGA Tour data ($\approx 90$ events with full-field rankings)

### 6. Advantage Analysis

**Theoretical Edge.** The Plackett-Luce framework directly models the object of interest (the probability of any specific finishing order) rather than requiring an intermediate step of modeling scores and simulating. This eliminates discretization error and distributional assumptions about score distributions.

**Specific Market Inefficiencies Exploited:**
1. **IIA violations as a feature, not a bug:** While the base PL model satisfies IIA, the GAS dynamics cause effective IIA violations through covariate interactions. When the model detects that a specific field composition particularly favors certain golfer profiles, it adjusts worth parameters in ways that simple rating-based models miss.
2. **Rank-order information utilization:** Bookmakers primarily use score-based models. The PL model extracts additional information from the rank order that is lost when reducing tournaments to raw scores. Two golfers who both shoot -12 in different tournaments convey different information depending on where they finished relative to the field.
3. **Dynamic form detection:** The GAS score mechanism provides a theoretically optimal update rule (in the information-theoretic sense) for adjusting golfer ratings based on observed performance. This may be more responsive than ad-hoc EWMA approaches.

**Robustness Properties:**
- The GAS framework has known optimality properties for score-driven time series models (Blasques, Koopman & Lucas, 2015)
- The model is semi-parametric: it makes distributional assumptions about rankings but not about the underlying score distribution
- Rank-based inference is inherently robust to outliers in raw scores

### 7. Limitation Analysis

**Violated Assumptions:**
- **Independence of Irrelevant Alternatives (IIA):** The base Plackett-Luce model implies that adding or removing golfer $k$ doesn't change the win probability ratio $P(i \text{ wins})/P(j \text{ wins})$. In reality, course-fit effects create substitution patterns (e.g., adding a long hitter disproportionately hurts other long hitters on a bombers' course). Mitigation: mixed logit extensions with random coefficients.
- **Constant variance implicit in log-worth:** Unlike Framework 1, the basic PL model doesn't explicitly model player-specific performance variance. A high-variance and low-variance golfer with the same expected ability are assigned the same worth parameter, despite having different win probabilities in large fields.
- **Luce's choice axiom:** Strict IIA may not hold empirically. Testing via the Hausman-McFadden test or Henderson-Plackett diagnostics is recommended.

**Edge Cases:**
- Very small fields (e.g., 30-player events): the PL model's multinomial logit structure may over-concentrate probability on top-rated players
- Tournaments with unusual formats (match play, team events): the ranking structure doesn't naturally apply
- Golfers with very few tournaments: GAS dynamics may not have converged to a stable estimate

**Computational Complexity:** Likelihood evaluation is $\mathcal{O}(N)$ per tournament (due to the sequential product structure). GAS updates are $\mathcal{O}(N)$ per golfer per tournament. For full parameter estimation via ML over $T$ tournaments, complexity is $\mathcal{O}(T \times N^2)$ if computing the Fisher information matrix. Monte Carlo approximation or diagonal Fisher methods reduce this.

### 8. Betting Strategy Integration

**Direct Win Probability Output.** The Plackett-Luce model produces win probabilities directly via the softmax function—no simulation required:

$$\hat{P}(i \text{ wins}) = \frac{\exp(\hat{f}_{i,t})}{\displaystyle\sum_{j} \exp(\hat{f}_{j,t})}$$

**Kelly Criterion for Mutually Exclusive Outcomes.** Since tournament outcomes are mutually exclusive, the simultaneous Kelly criterion applies. For $n$ outright bets with edges:

$$\text{Maximize:} \quad \sum_{i} \hat{P}_i \cdot \ln(1 + f_i \cdot b_i) + \left(1 - \sum_{i} \hat{P}_i\right) \cdot \ln\!\left(1 - \sum_{i} f_i\right)$$

This is a concave optimization problem solvable via standard methods.

**Fractional Kelly Implementation.** Apply $\kappa \in [0.15,\, 0.25]$ multiplier:

$$f_i = \kappa \cdot f_i^{*}$$

**Threshold Criteria:**
- Edge threshold: $\hat{P}_{\text{model}} / P_{\text{market}} > 1.08$ (8% edge to account for PL model's less rich uncertainty quantification compared to Framework 1)
- Minimum edge in expected units: $\mathbb{E}[\text{profit}] > 3\%$ of stake
- Maximum single position: 1.5% of bankroll
- Diversification: prefer spreading bets across 3-6 golfers with positive edge rather than concentrating

### 9. Comparison Metrics

**Primary Metrics:**
- **Ranked Probability Score (RPS):** Specifically designed for ordinal forecasts, it measures the cumulative squared error across all rank positions. More appropriate than Brier score for ranking models.
- **Log-loss on winner prediction:** Standard probabilistic accuracy metric, directly comparable across frameworks.
- **Kendall's $\tau$ and Spearman's $\rho$:** Rank correlation between model-predicted ranking and actual finishing order. Measures the model's overall ordering accuracy.
- **Top-$K$ accuracy:** Fraction of actual top-$K$ finishers that were in the model's predicted top-$K$. Relevant for top-5/top-10 betting markets.

**Backtesting Methodology:**
- Rolling estimation: re-estimate all GAS parameters using data up to tournament $t-1$, then forecast tournament $t$
- Use proper scoring rules (log score, ranked probability score) to avoid incentivizing hedging
- Compare against a static Plackett-Luce baseline to isolate the value of GAS dynamics

**Validation Tests:**
- Hausman-McFadden test for IIA violations
- Likelihood ratio test: dynamic PL vs. static PL
- Cross-validated log-likelihood comparison against the Bayesian simulation model (Framework 1)

### 10. Development Roadmap

**Phase 1 (Static Plackett-Luce):** Estimate time-invariant worth parameters from historical rankings
- Collect complete finishing orders for $\geq 2$ years of PGA Tour events
- Fit static PL model via maximum likelihood
- Validate: compare win probability predictions against market odds using log-loss

**Phase 2 (Time-Varying PL):** Add GAS dynamics
- Implement the GAS update mechanism with shared $A$, $B$ parameters
- Estimate via profile likelihood or two-stage estimation
- Validate: demonstrate out-of-sample log-loss improvement over static model

**Phase 3 (Covariates):** Add course-fit and exogenous variables
- Construct course feature database
- Add $\mathbf{X}_{i,t}^\top \boldsymbol{\beta}$ term to the state equation
- Validate: $F$-test on joint significance of course-fit covariates

**Phase 4 (Extensions):** Partial rankings, ties, and robustness
- Handle missed cuts via truncated PL likelihood
- Implement Davidson tie extension
- Test for IIA violations and consider mixed logit extensions if needed

**Phase 5 (Betting Integration):** Market comparison and staking
- Same as Framework 1, Phase 5
- Milestone: achieve positive out-of-sample ROI in simulated betting over $\geq 100$ tournaments

---

## Framework 3: Latent Factor State-Space Model with Stochastic Volatility

### 1. Framework Name & Classification

**Formal Classification:** Multi-Factor State-Space Model with Player-Specific Stochastic Volatility and Ornstein-Uhlenbeck Mean-Reversion

**Primary Theoretical Foundation:** State-space models (Durbin & Koopman, 2012), stochastic volatility (Taylor, 1986; Jacquier, Polson & Rossi, 1994), Ornstein-Uhlenbeck processes for mean-reverting dynamics (Baker & McHale, 2022)

### 2. Mathematical Foundation

**Latent State Representation.** Each golfer $i$ at time $t$ is characterized by a latent state vector:

$$\boldsymbol{\theta}_{i,t} = \bigl(\mu_{i,t},\; h_{i,t}\bigr)$$

where:
- $\mu_{i,t}$ is the latent true ability (expected strokes gained per round)
- $h_{i,t} = \ln(\sigma_{i,t}^2)$ is the log-volatility (capturing consistency/inconsistency)

**Observation Equation.** Round-level scores are generated by:

$$Y_{i,r,t} = \mu_{i,t} + \boldsymbol{\gamma}_{c(t)}^\top \boldsymbol{\delta}_i + \exp\!\left(\frac{h_{i,t}}{2}\right) \cdot \varepsilon_{i,r,t}$$

where $\varepsilon_{i,r,t} \sim \mathcal{N}(0,1)$ and the $\exp(h_{i,t}/2)$ term creates player- and time-specific heteroskedasticity.

**State Transition - Ability (Ornstein-Uhlenbeck Process).** The latent ability follows a continuous-time mean-reverting process, discretized to the inter-tournament interval $\Delta t_i$:

$$\mu_{i,t+1} = \bar{\mu}_i + \varphi_i \cdot \bigl(\mu_{i,t} - \bar{\mu}_i\bigr) + \sigma_{\mu,i} \cdot \sqrt{1 - \varphi_i^2} \cdot \eta_{i,t}$$

where:
- $\bar{\mu}_i$ is golfer $i$'s long-run equilibrium ability ("class")
- $\varphi_i = \exp(-\kappa_i \cdot \Delta t_i)$ is the autoregressive coefficient, determined by the mean-reversion speed $\kappa_i$ and the time gap $\Delta t_i$ between tournaments
- $\sigma_{\mu,i}$ is the innovation standard deviation for ability (how much true skill fluctuates)
- $\eta_{i,t} \sim \mathcal{N}(0,1)$ is the ability innovation

This is the discrete-time approximation to the Ornstein-Uhlenbeck SDE:

$$d\mu_{i,t} = \kappa_i \cdot \bigl(\bar{\mu}_i - \mu_{i,t}\bigr)\, dt + \sigma_{\mu,i}\, dW_t$$

The OU process is particularly well-suited to golf because it captures both the persistence of skill (high $\varphi$ means form carries over) and the inevitable regression to one's class level ($\bar{\mu}_i$ acts as an attractor).

Baker and McHale (2022) applied exactly this OU formulation to distinguish short-term "form" from long-term "class" in professional golf, finding that hot-hand effects can persist across multiple consecutive tournaments—a finding with direct betting implications.

**State Transition - Volatility (Stochastic Volatility).** The log-volatility follows its own autoregressive process:

$$h_{i,t+1} = \alpha_h + \rho_h \cdot h_{i,t} + \sigma_h \cdot \xi_{i,t}$$

where:
- $\alpha_h$ is the volatility intercept
- $\rho_h \in (0,1)$ is the volatility persistence parameter
- $\sigma_h$ is the volatility-of-volatility
- $\xi_{i,t} \sim \mathcal{N}(0,1)$ is the volatility innovation, potentially correlated with $\eta_{i,t}$ via:

$$\text{Corr}\!\left(\eta_{i,t},\; \xi_{i,t}\right) = \rho_{\eta\xi}$$

This correlation term $\rho_{\eta\xi}$ captures the "leverage effect" from finance—the analog in golf being that declining ability may coincide with increasing inconsistency (e.g., a player losing confidence becomes both worse on average and more variable).

**Multi-Factor Extension.** The ability state can be decomposed into sub-factors:

$$\mu_{i,t} = \sum_{s \,\in\, \{\text{OTT},\, \text{APP},\, \text{ARG},\, \text{PUTT}\}} \mu_{i,t}^{s}$$

where each sub-factor follows its own OU process with factor-specific parameters:

$$\mu_{i,t+1}^{s} = \bar{\mu}_i^{s} + \varphi^{s} \cdot \bigl(\mu_{i,t}^{s} - \bar{\mu}_i^{s}\bigr) + \sigma_\mu^{s} \cdot \sqrt{1 - (\varphi^{s})^2} \cdot \eta_{i,t}^{s}$$

This allows driving skill and putting skill to evolve at different rates and with different mean-reversion speeds, consistent with the empirical finding (Broadie, 2014) that tee-to-green skills are more persistent than putting.

**Inference via Sequential Monte Carlo (Particle Filter).** Since the model is non-linear (due to the stochastic volatility), exact Bayesian filtering (Kalman filter) is inapplicable. Inference proceeds via the particle filter:

1. **Initialize:** Draw $N_{\text{particles}}$ samples from the prior: $\boldsymbol{\theta}_{i,0}^{(p)} \sim \pi(\boldsymbol{\theta}_{i,0})$
2. **Predict:** Propagate each particle through the state transition: $\boldsymbol{\theta}_{i,t|t-1}^{(p)} \sim p\!\left(\boldsymbol{\theta}_{i,t} \mid \boldsymbol{\theta}_{i,t-1}^{(p)}\right)$
3. **Update:** Compute weights based on the observation likelihood: $w_t^{(p)} = p\!\left(Y_{i,t} \mid \boldsymbol{\theta}_{i,t|t-1}^{(p)}\right)$
4. **Resample:** Resample particles with probability proportional to weights
5. **Estimate:** The filtered state is:

$$\mathbb{E}\!\left[\boldsymbol{\theta}_{i,t} \mid Y_{1:t}\right] \approx \sum_{p=1}^{N_{\text{particles}}} w_t^{(p)} \cdot \boldsymbol{\theta}_{i,t}^{(p)}$$

Alternatively, for a fully Bayesian approach, MCMC methods (e.g., Jacquier, Polson & Rossi, 1994) can jointly estimate latent states and hyperparameters.

**Win Probability via Simulation.** Given filtered states $\{(\hat{\mu}_{i,t},\, \hat{h}_{i,t})\}$ for all golfers in a field:

$$\hat{P}(i \text{ wins}) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\!\left[i = \arg\min_j \sum_{r=1}^{4} Y_{j,r}^{(m)}\right]$$

where $Y_{j,r}^{(m)} \sim \mathcal{N}\!\left(\hat{\mu}_{j,t} + \boldsymbol{\gamma}_{c(t)}^\top \hat{\boldsymbol{\delta}}_j,\; \exp(\hat{h}_{j,t})\right)$ is drawn from each golfer's current filtered distribution.

**Probability Distributions Employed:**
- Normal for round scores (conditional on latent states)
- Log-normal for volatility ($h_{i,t}$ is log-variance, so $\sigma_{i,t}^2$ is log-normally distributed)
- Optionally: Student-$t$ for observation errors (to handle the excess kurtosis in golf scores)

**Key Theorems & Principles:**
- Bayesian filtering theorem: the posterior distribution of latent states given observations is updated recursively via Bayes' rule
- Ornstein-Uhlenbeck process properties: stationary distribution is Gaussian, autocorrelation decays exponentially with known rate
- Particle filter convergence (Del Moral, 2004): as $N_{\text{particles}} \to \infty$, the particle approximation converges to the true filtering distribution

### 3. Golf-Specific Adaptations

**Stochastic Volatility as a Betting Signal.** This is the key golf-specific insight of Framework 3. In outright winner markets, a golfer's performance variance is a first-order determinant of win probability—far more so than in sports where the mean outcome is more decisive. Consider two golfers both expected to shoot $-1$ per round, but one with $\sigma = 2.0$ strokes and another with $\sigma = 3.5$ strokes. In a 156-player field, the high-variance golfer has a meaningfully higher win probability (because winning requires tail performance). The stochastic volatility component allows the model to detect when a golfer is entering a high-variance or low-variance regime.

**Empirical Evidence for Heteroskedasticity.** Connolly and Rendleman (2008, JASA) document substantial player-specific variance heterogeneity on the PGA Tour, and their cubic spline model for time-varying ability implicitly captures volatility changes. The stochastic volatility formalization provides a principled framework for this.

**Irregular Timing.** The continuous-time OU formulation naturally handles the irregular spacing of golf tournaments—golfers don't play every week, and the gaps between events vary. The parameter $\Delta t_i$ (calendar time between golfer $i$'s consecutive events) enters directly through the AR coefficient $\varphi_i = \exp(-\kappa \cdot \Delta t_i)$, causing more shrinkage toward the long-run mean after longer absences. This is mathematically superior to the ad-hoc solutions required in discrete-time models.

**Pressure and Momentum.** The latent state framework can be extended to include a "momentum" state variable capturing the empirical hot-hand effect documented in golf by Baker and McHale (2022). This would add a third state dimension representing short-run psychological momentum, mean-reverting to zero:

$$m_{i,t+1} = \varphi_m \cdot m_{i,t} + \sigma_m \cdot \zeta_{i,t}$$

with the observation equation becoming $Y_{i,r,t} = \mu_{i,t} + m_{i,t} + \cdots$

### 4. Analogous Applications

**Quantitative Finance - Stochastic Volatility Models (Heston, 1993; Taylor, 1986).** The stochastic volatility framework is the workhorse of options pricing and risk management. The mathematical parallel is exact: asset returns are analogous to golf scores, latent volatility is analogous to golfer consistency, and the leverage effect ($\rho_{\eta\xi}$) is analogous to the skill-volatility correlation. The rich literature on SV estimation (MCMC, particle filters, importance sampling) transfers directly.

**Epidemiology - Disease Surveillance (Durbin & Koopman, 2012).** State-space models are used to estimate the latent true infection rate from noisy reported cases, with the observation noise and process noise playing roles identical to measurement error and true skill evolution in golf.

**NBA Player Tracking (Mews & Ötting, 2023).** Applied a continuous-time state-space model (OU process) to detect hot-hand effects in NBA free-throw data. The irregular timing between shots parallels the irregular timing between golf tournaments. They found "modest but persistent" hot-hand effects, consistent with Baker and McHale's golf findings.

**Macroeconomic Forecasting (Stock & Watson, 2007 - Unobserved Components Model).** The decomposition of GDP into trend (class) and cycle (form) components using the Kalman filter is structurally identical to the golf model's decomposition of ability into long-run equilibrium and short-run fluctuation.

### 5. Data Requirements & Feature Engineering

**Required Variables:**
- $\text{Score}_{i,r,t}$: Round-level score (raw or adjusted strokes-gained)
- $\text{SG}_{i,r,t}^{s}$: Sub-component strokes gained per round (if using multi-factor version)
- $\text{Date}_{i,t}$: Calendar date (for computing inter-tournament intervals $\Delta t_i$)
- $\text{CourseFeatures}_{c(t)}$: Course characteristics for course-fit adjustment
- $\text{Weather}_{r,t}$: Round-level weather data

**Transformations:**
- Compute inter-tournament intervals: $\Delta t_i = \text{Date}_{i,t+1} - \text{Date}_{i,t}$ (in days or weeks)
- Adjust raw scores for course/round difficulty (pre-processing step identical to Framework 1)
- Log-transform variance estimates for the stochastic volatility specification
- Compute residual variance from a preliminary ability model for initializing the SV component

**Data Structure:** Irregularly-spaced time series (one per golfer), with panel structure across golfers sharing hyperparameters.

**Minimum Sample Sizes:**
- For individual OU parameter estimation ($\kappa_i$, $\bar{\mu}_i$): $\geq 80$ tournaments per golfer
- For shared hyperparameters: $\geq 3$ years of data across $\geq 100$ golfers
- For stochastic volatility: $\geq 120$ rounds per golfer (need sufficient observations to distinguish variance changes from noise)
- For particle filter stability: $N_{\text{particles}} \geq 500$ per golfer

### 6. Advantage Analysis

**Theoretical Edge.** The primary edge is the explicit modeling of time-varying volatility—a dimension almost entirely ignored by the betting market. The market prices outright winner odds based on some assessment of each golfer's expected performance (mean ability), but rarely accounts for whether a golfer is entering a high- or low-variance phase.

**Specific Market Inefficiencies Exploited:**
1. **Variance regime detection:** When the model detects a golfer entering a high-volatility phase (e.g., returning from a swing change, early-season rust), their win probability in large fields increases even as their expected score may remain unchanged. The market may overprice them for top-10 but underprice them for outright win.
2. **Layoff effect mispricing:** The OU formulation provides optimal shrinkage after layoffs. A golfer returning from 3 months off has $\varphi = \exp(-\kappa \cdot 90/7) \approx 0.3$, meaning 70% shrinkage toward long-run class. If the market instead uses the golfer's form from before the layoff, there's a systematic discrepancy.
3. **Mean-reversion arbitrage:** The OU process implies that extreme recent form (either very hot or very cold) will revert toward class. The rate of this reversion ($\kappa$) is estimated from data, not assumed. If the market reverts too slowly (chasing hot form) or too quickly (dismissing cold spells), the model captures the edge.
4. **Momentum detection:** The hot-hand extension explicitly models the persistence of form streaks, which Baker and McHale (2022) document as lasting across multiple consecutive PGA Tour events.

**Robustness Properties:**
- The OU process has a stationary distribution, preventing parameter drift
- The particle filter is non-parametric in its approximation to the posterior, avoiding Gaussian approximation errors
- Multi-factor decomposition allows the model to remain stable even if one sub-skill has a structural break (e.g., injury affecting driving but not putting)

### 7. Limitation Analysis

**Violated Assumptions:**
- **Gaussian observation errors:** Golf scores have heavier tails than Gaussian. Mitigation: use Student-t observation errors (straightforward in the particle filter framework).
- **Constant OU parameters:** The mean-reversion speed $\kappa$ and innovation variance $\sigma_\mu$ may change over a golfer's career (young golfers improve systematically, aging golfers decline). Mitigation: allow $\kappa$ to depend on age or experience via a structural break test.
- **Independence across golfers:** The model treats each golfer's state as evolving independently. In reality, competitive effects may exist (one golfer's dominance demoralizing others). Mitigation: include field-strength covariates.

**Edge Cases:**
- Rookies and golfers with <30 tournaments: insufficient data for individual OU parameter estimation. Require hierarchical structure to borrow strength from similar golfers.
- Course history effects: the OU model has no explicit memory of course-specific performance. Course-fit must be handled entirely through the covariate term.
- Major championship premium: the heightened stakes of majors may alter both mean and variance in ways not captured by the base model.

**Computational Complexity:** The particle filter requires $\mathcal{O}(N_{\text{particles}} \times N_{\text{golfers}})$ operations per tournament. With 500 particles and 156 golfers, this is ${\sim}78{,}000$ state propagations per tournament. Over 100 tournaments, total computation is ${\sim}7.8$ million operations—manageable but substantially heavier than Framework 1 or 2. The multi-factor extension multiplies this by the number of sub-factors.

### 8. Betting Strategy Integration

**Variance-Aware Kelly Criterion.** Because the model explicitly estimates both mean and variance, the Kelly criterion can account for estimation uncertainty in a principled way. The "Modified Kelly" of Chu, Wu, and Swartz (2018, SFU) provides a decision-theoretic framework:

$$f_i^{*} = \arg\max_{f} \;\; \mathbb{E}_{\boldsymbol{\theta} \sim \text{posterior}}\!\left[\ln\!\Big(1 + f \cdot b_i \cdot \mathbf{1}[i \text{ wins}] - f \cdot \mathbf{1}[i \text{ doesn't win}]\Big)\right]$$

This integrates over the posterior uncertainty in θ (including both ability and volatility), automatically producing more conservative stakes when the model is uncertain. This directly addresses the key weakness of standard Kelly—its sensitivity to probability estimation errors.

**Variance-Conditioned Bet Selection.** The model naturally suggests a sophisticated bet selection strategy:
- When a golfer's filtered volatility $\exp(\hat{h}_{i,t})$ is elevated: prefer outright winner bets (the fat tails favor outright payoffs)
- When volatility is low: prefer top-10/top-20 bets (the golfer will likely finish near their mean)
- This "volatility trading" strategy is unique to Framework 3 and has no analog in Frameworks 1 or 2

**Threshold Criteria:**
- Edge threshold: $\hat{P}_{\text{model}} / P_{\text{market}} > 1.10$ (higher threshold due to greater model complexity and estimation noise)
- Volatility confidence: require that the 80% credible interval for current volatility excludes the prior mean (i.e., the model is "confident" in its volatility estimate)
- Bankroll limits: same as Framework 1 (fractional Kelly with $\kappa = 0.10$–$0.20$, max 1.5% per bet)

### 9. Comparison Metrics

**Primary Metrics:**
- **Log-loss on winner prediction:** Comparable across all three frameworks
- **Conditional coverage:** Does the model's predicted variance match the realized variance of residuals? Test via Mincer-Zarnowitz regression of squared residuals on predicted variance.
- **Volatility forecast accuracy:** Evaluate whether detected "high-volatility" periods actually coincide with more extreme outcomes, using the hit rate of variance-conditioned bet selection.
- **Calibration across form states:** Separate calibration analysis for golfers in "hot," "neutral," and "cold" form (as classified by the OU state relative to long-run mean).

**Backtesting Methodology:**
- Same expanding-window approach as Frameworks 1 and 2
- Additional test: out-of-sample variance prediction (does the model correctly predict which golfers will have more variable outcomes?)
- Track separately the performance of mean-driven bets vs. variance-driven bets

**Validation Tests:**
- Ljung-Box test on standardized residuals (no residual autocorrelation after filtering)
- Jarque-Bera test on standardized residuals (normality, or adequacy of $t$-distribution)
- KPSS test on filtered ability states (confirm stationarity of OU process)
- DIC (Deviance Information Criterion) for model selection across different numbers of sub-factors

### 10. Development Roadmap

**Phase 1 (Linear State-Space):** Implement a Kalman filter version with fixed (known) variance
- Estimate OU parameters for ability using the Kalman filter
- Validate: compare filtered ability estimates against DataGolf's published ratings

**Phase 2 (Stochastic Volatility):** Add time-varying variance via particle filter
- Implement basic SV model with shared hyperparameters
- Validate: demonstrate that predicted variance tracks realized score dispersion

**Phase 3 (Multi-Factor):** Decompose ability into sub-skill state variables
- Separate OTT, APP, ARG, PUTT components with independent OU dynamics
- Validate: check that sub-factor decay rates match known persistence patterns (tee-to-green > putting)

**Phase 4 (Course-Fit and Covariates):** Add exogenous adjustments
- Implement course-fit interaction terms
- Add weather adjustments
- Validate: $F$-test on covariate significance

**Phase 5 (Betting Integration):** Variance-aware staking and market comparison
- Implement modified Kelly with posterior integration
- Backtest variance-conditioned bet selection strategy
- Milestone: demonstrate that variance-driven bets contribute positive ROI independently of mean-driven bets

---

## Comparative Analysis

### Side-by-Side Comparison

| Attribute | Framework 1: Hierarchical Bayesian SG | Framework 2: Dynamic Plackett-Luce | Framework 3: State-Space with SV |
|---|---|---|---|
| **Theoretical Tradition** | Bayesian statistics / Empirical Bayes | Ranking theory / Score-driven TS | State-space / Stochastic processes |
| **Primary Output** | Win probabilities via simulation | Win probabilities via softmax | Win probabilities via simulation |
| **Data Requirement** | Moderate (round-level SG) | Moderate (finishing orders) | High (round-level SG + enough obs for SV) |
| **Computational Cost** | Low (MC simulation) | Low-Medium (MLE + GAS updates) | High (particle filter) |
| **Handles Player Variance** | Yes (static, player-specific) | No (implicit only) | Yes (time-varying, dynamic) |
| **Handles Course Fit** | Yes (via interaction terms) | Yes (via covariates in state eq.) | Yes (via covariates in obs. eq.) |
| **Handles Form Cycles** | Partially (via EWMA) | Yes (via GAS dynamics) | Yes (via OU mean-reversion) |
| **Handles Irregular Timing** | Partially (dual EWMA) | Implicitly (by tournament order) | Naturally (continuous-time OU) |
| **Overfitting Risk** | Low (shrinkage protects) | Medium (many parameters in full model) | Medium-High (particle filter can overfit) |
| **Interpretability** | High (strokes gained is intuitive) | Medium (worth parameters less intuitive) | Medium (latent states require inference) |
| **Theoretical Novelty** | Low (well-established approach) | Medium (GAS-PL is relatively new) | High (SV in sports is novel) |
| **Edge Source** | Shrinkage + course fit | Rank-order information + dynamics | Volatility regime detection |
| **Peer-Reviewed Foundation** | Strong (Broadie, Efron, Brill & Wyner) | Strong (Luce, Plackett, Henderson & Kirrane) | Strong (Taylor, Heston, Baker & McHale) |

### Recommendation Logic

**If your priority is reliability and interpretability:** Choose Framework 1. The hierarchical Bayesian SG model has the most direct connection to golf-specific data, the strongest regularization properties (shrinkage), and produces the most interpretable results. This is the closest analog to what DataGolf and other professional golf analytics operations use.

**If your priority is mathematical elegance and theoretical optimality:** Choose Framework 2. The Plackett-Luce model directly models the observable outcome (rankings) without intermediate distributional assumptions. The GAS update mechanism has known optimality properties for score-driven dynamics. This framework is the most theoretically "clean" of the three.

**If your priority is finding novel edges and you have strong quantitative finance background:** Choose Framework 3. The stochastic volatility component is the most differentiating feature, offering a dimension of analysis (variance regime detection) that is almost entirely unexploited in golf betting markets. The "volatility trading" strategy (betting outright on high-variance golfers, top-10 on low-variance golfers) is unique and potentially highly profitable.

**The ensemble approach:** The theoretically optimal strategy is to estimate all three models and combine their win probabilities via a calibrated meta-learner:

$$\hat{P}_{\text{ensemble}}(i \text{ wins}) = \alpha_1 \cdot \hat{P}_1(i) + \alpha_2 \cdot \hat{P}_2(i) + \alpha_3 \cdot \hat{P}_3(i)$$

where the weights $\alpha_k$ are estimated by minimizing log-loss on a held-out validation set. Theoretical results from the forecast combination literature (Bates & Granger, 1969) guarantee that the ensemble will weakly dominate any individual model in expected log-loss.

---

## Next Steps for Decision Making

### Critical Questions to Answer Before Committing

1. **Data availability audit:** Can you access round-level strokes-gained decomposition data (SG:OTT, SG:APP, etc.)? If yes, Frameworks 1 and 3 are feasible. If only finishing positions are available, Framework 2 is the natural choice.

2. **Computational resources:** Do you have the infrastructure for particle filtering (Framework 3) or is computational simplicity preferred (Framework 1)?

3. **Market access:** Are you betting pre-tournament outrights only, or also in-play and derivative markets (top-5, top-10, matchups)? Framework 3's variance insights are most valuable when you can express views across multiple market types.

4. **Time horizon:** For rapid deployment (weeks), Framework 1 is fastest to implement. For a multi-month research project aiming for maximum edge, the ensemble approach is optimal.

5. **Risk tolerance:** Framework 1 with fractional Kelly is the most conservative approach. Framework 3 with volatility-conditioned betting is the most aggressive and has the highest variance in returns.

### Validation Before Full Commitment

For any chosen framework, conduct the following validation before committing to live betting:

1. **Calibration test:** Bin predicted win probabilities into deciles and compare against observed win rates. A well-calibrated model should produce a 45-degree calibration plot.

2. **Beating the market test:** Compare model log-loss against the log-loss of market-implied probabilities (derived from closing odds) over $\geq 50$ tournaments. A paired $t$-test on per-tournament log-loss differences should reject the null of equal accuracy at $p < 0.05$.

3. **Economic significance test:** Backtest the complete betting strategy (including fractional Kelly staking and transaction costs) over $\geq 2$ years. The 95% confidence interval for annualized ROI should exclude zero.

4. **Stability test:** Confirm that model parameters are stable across rolling estimation windows. Large parameter swings suggest overfitting to specific periods.

### Key Academic References

- Broadie, M. (2012). "Assessing golfer performance on the PGA Tour." *Interfaces*, 42(2), 146-165.
- Brill, R. & Wyner, A. (2025). "Putting Skill as Nearly Indistinguishable from Noise: An Empirical Bayes Analysis." *arXiv:2506.21822*.
- Connolly, R.A. & Rendleman, R.J. (2008). "Skill, luck, and streaky play on the PGA Tour." *JASA*, 103(481), 74-88.
- Drappi, C. & Ting Keh, L.C. (2019). "Predicting golf scores at the shot level." *J. Sports Analytics*, 5(2), 1-9.
- Baker, R.D. & McHale, I.G. (2022). "Form vs. class in professional golf: An OU model approach." [Applied to detecting persistent hot-hand effects across PGA Tour tournaments]
- Henderson, D.A. & Kirrane, L.J. (2018). "A comparison of truncated and time-weighted Plackett-Luce models." *Bayesian Analysis*, 13(2), 335-358.
- Cattelan, M., Varin, C. & Firth, D. (2013). "Dynamic Bradley-Terry modelling of sports tournaments." *J. Royal Statistical Society: Series C*, 62(1), 135-150.
- Creal, D., Koopman, S.J. & Lucas, A. (2013). "Generalized autoregressive score models with applications." *J. Applied Econometrics*, 28(5), 777-795.
- Jacquier, E., Polson, N.G. & Rossi, P.E. (1994). "Bayesian analysis of stochastic volatility models." *J. Business & Economic Statistics*, 12(4), 371-389.
- Chu, D., Wu, Y. & Swartz, T.B. (2018). "Modified Kelly Criteria." *SFU Working Paper*.
- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods*. Cambridge University Press.
- Kelly, J.L. (1956). "A new interpretation of information rate." *Bell System Technical Journal*, 35, 917-926.
