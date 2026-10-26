# Design notes — rough-vol-nn

This document explains the mathematics behind every component and records each modelling
choice together with the alternatives that were considered and why they were rejected.
Numbers quoted in the README come from `results/*.json`; this document deliberately
contains no results, only reasoning.

---

## 1. Why rough volatility

**Empirical fact 1 — roughness of realised volatility.** Gatheral, Jaisson & Rosenbaum
(2018) regress the log of realised variance increments on the lag Δ:
E|log σ_{t+Δ} − log σ_t|^q ∝ Δ^{qH}. Across indices and frequencies the scaling is
linear in q (monofractal) with **H ≈ 0.1**, far below the H = 1/2 of any diffusive
(Markovian) stochastic-volatility model. Log-volatility behaves like a fractional Brownian
motion with Hurst index ≈ 0.1: its paths are Hölder continuous of order H − ε only, i.e.
much rougher than Brownian motion.

**Empirical fact 2 — the short-dated skew explodes.** The at-the-money skew
ψ(T) = |∂σ_BS(k, T)/∂k|_{k=0} of equity indices behaves like T^{−α} with α ≈ 0.4 from
about a week to a couple of years. Classical one-factor models (Heston, SABR with β = 1,
...) produce a skew that tends to a *constant* as T → 0.

The rough Bergomi model explains both facts with one parameter: the same H that sets the
path regularity of volatility sets the skew exponent, α = 1/2 − H. Section 4 derives this.

## 2. The rough Bergomi model

Bayer, Friz & Gatheral (2016), under the pricing measure, forward-normalised (S_0 = F = 1,
zero rates; rates and dividends only enter through the forward and the discount factor,
which the market loader strips out):

```
dS_t / S_t = sqrt(V_t) dB_t,          B = ρ W + sqrt(1 − ρ²) W⊥
V_t        = ξ_0(t) · exp( η Ŵ_t − ½ η² t^{2H} )
Ŵ_t        = sqrt(2H) ∫_0^t (t − s)^{H − 1/2} dW_s
```

* **Ŵ** is a Riemann–Liouville fractional Brownian motion: a Volterra (moving-average)
  integral of the *same* Brownian motion W that drives the spot through ρ. With the
  normalisation sqrt(2H), Var Ŵ_t = 2H ∫_0^t (t−s)^{2H−1} ds = t^{2H}. Hence
  exp(η Ŵ_t − ½η² t^{2H}) has mean one and E[V_t] = ξ_0(t): ξ_0 *is* the forward variance
  curve, which is what makes the model "Bergomi" (a forward-variance model).
* **Parameters**: H (roughness), η (vol-of-vol), ρ (spot–vol correlation, the leverage
  effect), ξ_0 (forward variance). Following Horvath, Muguruza & Tomas (2021) we use a
  **flat** forward variance ξ_0(t) ≡ ξ_0, so the model has four parameters
  (H, η, ρ, ξ_0). A piecewise-constant ξ_0(t) is the natural extension (Section 10).
* **Not Markovian.** V_t depends on the whole path of W through the kernel; there is no
  finite-dimensional state, no PDE, no closed-form characteristic function. Pricing is
  Monte Carlo — which is exactly why a learned pricer is valuable.
* **Why Riemann–Liouville rather than Mandelbrot–van Ness fBm.** RL fBm is driven by one
  Brownian motion on [0, t] and is adapted to the filtration of W, so the correlation with
  the spot is a single ρ and the Volterra structure gives the clean variance t^{2H}. MvN
  fBm has stationary increments but needs the infinite past and a non-explicit
  correlation structure with the spot. The price paid by RL: increments are not
  stationary near t = 0 — irrelevant for pricing, which only needs the law on [0, T].

### 2.1 The fractional kernel

K(t) = sqrt(2H) t^{H−1/2}. For H < 1/2 the exponent α = H − 1/2 is negative, so the
kernel is **singular at 0** but square-integrable (∫_0^t K² = t^{2H} < ∞), which is all a
Wiener integral needs. The singularity is the whole story:

* Most of the variance of Ŵ_t − Ŵ_{t−δ} comes from the most recent noise:
  Var(Ŵ_{t} − Ŵ_{t−δ}) ~ c δ^{2H}, and δ^{2H} ≫ δ for small δ when H < 1/2. The volatility
  moves *a lot* over short horizons — roughness.
* Numerically, a left-point Riemann sum of the kernel misses the mass of the singularity
  in the last cell and badly underestimates that variance. This motivates the hybrid
  scheme.

## 3. Simulation: the hybrid scheme

Bennedsen, Lunde & Pakkanen (2017). On a grid t_i = i/n write

```
Ŵ_{t_i} = sqrt(2H) Σ_{k=1}^{i} ∫_{t_{i−k}}^{t_{i−k+1}} (t_i − s)^α dW_s .
```

* **Near field (k ≤ κ):** keep the power kernel *exactly*. For κ = 1 the term is
  W̃_{i−1} := ∫_{t_{i−1}}^{t_i} (t_i − s)^α dW_s, which is Gaussian and jointly Gaussian
  with the Brownian increment ΔW_{i−1} of the same cell:

  ```
  Var ΔW = Δ,   Cov(ΔW, W̃) = Δ^{α+1}/(α+1),   Var W̃ = Δ^{2α+1}/(2α+1),   Δ = 1/n.
  ```

  So each cell needs two correlated normals, drawn via the 2×2 Cholesky factor
  (`rbergomi.hybrid_kernel`).
* **Far field (k ≥ κ+1):** the kernel is smooth away from the diagonal, so replace it by
  a constant on each cell, evaluated at the point b_k that minimises the mean-squared
  approximation error,

  ```
  b_k = ( (k^{α+1} − (k−1)^{α+1}) / (α+1) )^{1/α},    weight G_k = (b_k/n)^α .
  ```

  The far-field sum Σ_{k≥2} G_k ΔW_{i−k} is a **discrete convolution**, computed for all
  i at once by FFT: O(N n log n) for N paths and n steps instead of O(N n²).
* **Volterra process on the grid:** Ŵ_i = sqrt(2H) (W̃_{i−1} + Σ_{k=2}^{i} G_k ΔW_{i−k}).

**Exact compensator.** Instead of the continuous-time t^{2H}, the code uses the exact
variance of the *discretised* Ŵ_i, 2H (Δ^{2α+1}/(2α+1) + Σ_{k=2}^{i} G_k² Δ). Then
E[V_{t_i}] = ξ_0 holds exactly on the grid (tested), so the forward-variance input is
honoured exactly even at coarse resolution.

**Spot.** Log-Euler with the left-point variance (Itô): V_{t_i} uses noise only up to
t_i (W̃_{i−1} and ΔW_{i−k}, k ≥ 2), so it is independent of the increment ΔW_i that
multiplies it.

**Grid.** 500 steps/year, chosen so every maturity of the training grid
(0.02 … 1.5) is an integer number of steps (10 steps for the 1-week-ish T = 0.02).
The discretisation ablation (README) measures the bias against 2,000 steps/year.

| alternative | why rejected |
|---|---|
| Exact Cholesky of the joint (Ŵ, W) covariance | exact, but O(n³) setup *per H* (H changes with every training sample) and O(n²) per path; memory for 1,500×1,500 per sample. The hybrid scheme is O(n log n) and its error is small next to MC noise (measured). |
| Riemann sum (κ = 0) | misses the singular mass near the diagonal; biased variance for small H — the regime we care about. |
| κ ≥ 2 | marginal gain over κ = 1 (BLP 2017 Fig. 2) for a larger Gaussian block per cell. |
| Markovian (multi-factor) approximations of the kernel | introduce a second approximation layer and extra hyperparameters; useful for PDE methods, unnecessary for MC. |

### 3.1 Variance reduction

**Antithetic variates.** All normals Z are paired with −Z. Pair means are i.i.d., and
standard errors are computed from pair means (not from individual paths, which would
understate the error because partners are negatively correlated by construction).

**Conditional Monte Carlo ("mixing", Romano & Touzi 1997).** Condition on the path of W
(the volatility driver). Then

```
ln S_T | W  ~  N( ρ I_T − ½ ρ² Q_T − ½ (1−ρ²) Q_T ,  (1−ρ²) Q_T ),
I_T = ∫_0^T sqrt(V_t) dW_t,   Q_T = ∫_0^T V_t dt .
```

So E[(S_T − K)^+ | W] is a **Black price** with forward F̃ = exp(ρ I_T − ½ρ² Q_T) and
total variance (1−ρ²) Q_T. The estimator averages Black prices over paths of W:

* it removes the W⊥ noise entirely (variance reduction is largest when |ρ| is small and
  smallest as |ρ| → 1, see the ablation);
* every path contributes a strictly positive, smooth price, so deep out-of-the-money cells
  never produce zero prices and IV inversion never fails;
* prices are smooth in the parameters for fixed random numbers. This matters for
  Monte-Carlo-in-the-loop calibration with common random numbers: finite-difference
  Jacobians of indicator-type payoffs would be noise.

It is still unbiased: E[E[payoff | W]] = E[payoff] (tower property). The plain estimator
is kept and used in tests (martingale check, agreement test).

**Tests** (`tests/test_rbergomi.py`): η = 0 reproduces Black–Scholes within 4 SE; E[S_T] = 1
within 4 SE for rough and smooth parameter sets; the reported SE matches the spread of 40
independent replications; SE scales like N^{−1/2}; plain and mixing agree; H = 1/2
reproduces Brownian variance exactly.

## 4. Why roughness explains the short-dated skew

Write σ_t = sqrt(V_t). For a general stochastic-volatility model dS/S = σ_t dB_t with
correlation ρ between B and the volatility driver W, the Alòs–León–Vives (2007)
decomposition gives, to leading order in T,

```
ψ(T) ≈ ρ / (σ_0 T²) · ∫_0^T ∫_s^T E[ D_s σ_t ] dt ds ,
```

where D_s σ_t is the Malliavin derivative: the sensitivity of σ_t to a shock of the
Brownian motion W at time s ≤ t. Intuitively, a negative ρ makes down-moves of the spot
coincide with up-moves of vol, fattening the left tail; the skew measures how much the
spot-vol covariance has accumulated *by T, relative to the total variance σ_0² T*.

* **Markovian SV (Heston, H = 1/2).** σ_t responds to a shock at s with a sensitivity that
  is O(1) as t − s → 0 (for Heston, D_s σ_t → σ_v/2, the vol-of-variance over two). The
  double integral is ≈ (σ_v/2) T²/2, so ψ(T) → ρ σ_v / (4 σ_0): **a constant**. To create
  a large short-dated skew Heston needs a huge vol-of-vol, which then over-steepens long
  maturities (mean reversion κ fights back, but with a single exponential time scale).
* **Rough Bergomi.** σ_t = sqrt(ξ_0) exp(½ η Ŵ_t − …), so
  D_s σ_t = σ_t · ½ η sqrt(2H) (t − s)^{H−1/2}. The kernel singularity means a shock has a
  huge effect on volatility *immediately after* it happens. Then

  ```
  ∫_0^T ∫_s^T (t−s)^{H−1/2} dt ds = T^{H+3/2} / ((H+½)(H+3/2)),
  ψ(T) ≈ |ρ| η sqrt(2H) / (2 (H+½)(H+3/2)) · T^{H−1/2} .
  ```

  A **power law** with exponent H − 1/2 ≈ −0.4 for H = 0.1 — exactly the empirical
  term structure. Equivalent statement: the effective vol-of-vol at horizon τ scales like
  τ^{H−1/2} and blows up as τ → 0. (Rigorous versions: Fukasawa 2011; Bayer, Friz,
  Gulisashvili, Horvath & Stemper 2019, who also derive the constant above.)

The same derivation explains the sanity check used throughout the repo: the Heston
limit ρσ/(4√v_0) is tested in `tests/test_heston_skew.py`, and the rough power law is
measured from Monte Carlo in `results/atm_skew.json` (figure `assets/atm_skew_loglog.png`).

**Caveat (honest):** the formula is a small-T asymptotic. At T ~ 1y the MC skew bends
away from the line; the fitted slope is therefore reported on T ≤ 0.1 only. Very short
maturities are limited by the time step (a 1.5-day option has ~4 steps at 2,500/yr).

## 5. Implied-volatility inversion

`implied_vol.py`: given a forward-normalised undiscounted price, invert the Black formula
for the total standard deviation s = σ√T.

* Convert to the **out-of-the-money side** with put–call parity (C − P = 1 − e^k in
  forward units). OTM prices carry all the time value; ITM inversion loses digits to the
  intrinsic value.
* **Safeguarded Newton on ln(price)**, not on price. For deep OTM options the price is
  exponentially small in 1/s, so f(s) = price(s) − p is extremely flat then extremely
  steep — Newton overshoots. ln price(s) is much closer to linear in 1/s.
* Start at s* = sqrt(2|k|), the inflection point of s ↦ price(s) (maximal vega), Jäckel's
  classic starting point; maintain a bracket [lo, hi] updated from the sign of f; any
  Newton step that leaves the bracket is replaced by bisection (geometric when the bracket
  spans more than a factor 4).
* Prices outside the no-arbitrage bounds (≤ 0, ≥ forward/strike upper bound) return NaN,
  never a guess.
* Vectorised over arbitrary arrays; tested on 20,000 random (k, T, σ) to 1e−9 relative
  accuracy, plus extremes (T = 1e−4, σ√T = 3.4, k = −2).

| alternative | why rejected |
|---|---|
| py_vollib / *Let's Be Rational* | machine-precision in two iterations, but the package is unmaintained, pins old numpy/pandas behaviour and is scalar-first. Our vectorised Newton–bisection reaches 1e−9 in < 10 iterations on the grid, which is far below the MC noise (1e−3 relative). |
| Brent per option | robust but scalar; 7M inversions in data generation make vectorisation matter. |

Standard error of an implied vol: delta method, SE_σ = SE_price / vega.

## 6. The training set

* **Parameter box** (paper): H ∈ [0.025, 0.5], η ∈ [0.5, 4], ρ ∈ [−0.95, −0.1],
  ξ_0 ∈ [0.01, 0.16]; sampled **uniformly** (alternative: Sobol — marginally better
  coverage, but i.i.d. sampling keeps the train/val/test split statistically clean).
* **Grid:** 8 maturities {0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5} × 11 strikes on a
  **√T-scaled log-moneyness grid** k = z√T, z ∈ [−0.42, 0.28] (step 0.07, ATM included).

  | alternative | why rejected |
  |---|---|
  | paper grid: fixed K ∈ {0.5 … 1.5}, T ∈ {0.1 … 2.0} | at T = 0.1 and σ = 10%, K = 0.5 is 22 standard deviations out of the money: the price underflows and the IV is undefined/noise. A fixed-K grid also puts most cells deep ITM/OTM at the short end — exactly where roughness lives. Scaling by √T keeps every cell at a comparable number of standard deviations, so every cell is informative. |
  | starting at T = 0.1 | loses the region where the T^{H−1/2} skew is distinctive. 0.02 ≈ 1 week is the shortest maturity that still has ≥ 10 time steps. |
  | asymmetric z | the range [−0.42, 0.28] is skewed left because equity-index smiles are; liquid index strikes cover roughly ±0.35 in z. |

* **Labels:** IVs from 40,000 paths (20,000 antithetic pairs), mixing estimator. The
  per-cell standard error is stored with every label.
* **Chunked, resumable generation** (`data.py`): the parameter table is a pure function of
  the seed; sample i uses MC seed `seed + i`, so a chunk is bit-identical regardless of
  interruptions (tested); chunks are written atomically (temp + rename) and carry a
  config fingerprint, and a resume with a different config is refused.
* **Reference set:** 256 surfaces at 640,000 paths (16× more, so 4× smaller SE): the
  ground truth used to score the surrogates. Without it, errors measured against noisy
  labels conflate network error with label noise.

## 7. Surrogate (step 1 of the two-step approach)

**Grid-based** (Horvath, Muguruza & Tomas 2021): the network maps the 4 parameters to all
88 IVs at once. Compared with a *pointwise* network f(θ, T, k):

* one forward pass prices the whole surface (88× fewer evaluations);
* the Jacobian ∂σ/∂θ, needed by Levenberg–Marquardt, is a single 88×4 forward-mode
  autodiff product;
* the output has the structure of the data (smiles are smooth across neighbouring
  outputs), which the last linear layer captures.

Off-grid quotes are handled by interpolating the network's grid (Section 9).

**Scaling.** Inputs mapped affinely to [−1, 1] using the box; outputs standardised per
cell (mean/std over the training set). Loss: MSE on standardised IV — equivalent to a
per-cell relative weighting, so wings (small vega, larger noise) do not dominate.

**Models.**
* *Grid MLP*: 3 hidden layers × 30 ELU units (the paper's architecture).
* *Residual MLP*: input layer to width 128, 3 pre-activation residual blocks
  (h ← h + W₂ σ(W₁ σ(h))), SiLU, linear head. Smooth activations because calibration
  differentiates the network: ReLU would make ∂σ/∂θ piecewise constant.
  **No batch/layer norm:** normalisation layers couple samples or rescale features
  non-smoothly and are unnecessary at this depth.

**Training.** Adam (lr 1e−3), batch 256, ReduceLROnPlateau on validation loss, early
stopping on validation, best-validation weights restored. All on GPU, a few minutes.

**Noisy labels and the noise floor.** Each label is (truth + discretisation bias + MC
noise ε) with E[ε] = 0 and ε independent across samples. A regression network fitted by
MSE estimates the conditional mean, i.e. it *averages* the noise of neighbouring
parameter sets. A surrogate can therefore be **more accurate than any single training
label** — the comparison against the noise floor in the README tests exactly this. It
cannot remove the discretisation bias (common to all labels and to the reference set);
that is quantified separately by the time-step ablation.

| alternative | why rejected |
|---|---|
| learning prices instead of IVs | prices span orders of magnitude across the grid; IVs are O(1) and are what is calibrated. |
| relative-error loss | equivalent in spirit to per-cell standardisation, but unstable for tiny targets. |
| pointwise network | see above; would be needed only for quotes far outside the grid. |
| Gaussian process / Chebyshev tensor surrogate | GP scales as O(n³) in 64k training points; tensor interpolation in 4 dims needs structured (not random) MC designs. |

## 8. Calibration (step 2)

Objective: Σ_cells w (σ_model(θ) − σ_market)², over quoted cells only.

**Levenberg–Marquardt** (`calibration.lm_calibrate`), batched in torch:
* box constraints by reparametrisation θ = lo + (hi − lo) · sigmoid(u);
* exact Jacobian ∂r/∂u by forward-mode autodiff (`torch.func.jacfwd` + `vmap`), so
  thousands of surfaces are calibrated in one batched loop on the GPU;
* **Marquardt scaling** (λ · diag(JᵀJ) rather than λ · I): makes the step invariant to
  per-parameter rescaling, including the sigmoid's vanishing slope near the box edges;
* multi-start (box centre + random starts), best cost kept;
* stopping: an accepted step that lowers the cost by less than `tol` (relative), or a
  damping λ above `lambda_max` (steps have become negligible). The first version stopped
  only on accepted steps, so a converged start kept iterating through rejected steps
  until λ reached 1e11 — a 2–3× waste found by profiling.

**Jacobian on CPU, single surface.** Forward-mode autodiff in torch (`jacfwd`) costs
~6 ms per call for this tiny network on CPU — pure framework overhead (a plain forward
pass is 0.2 ms). For single-surface calibration we therefore use **float64 central
differences through the network** (8 extra rows in one batched forward pass,
h = 1e−6). The network is C^∞ (SiLU/ELU), so the truncation error is O(h²) ≈ 1e−12 and
rounding error in float64 ≈ 1e−10/h ≈ 1e−10; measured agreement with autodiff is ~3e−8
absolute (5e−7 relative) and calibrated parameters are identical to 4 digits, at 8× less
wall-clock. Batched GPU calibration and the market interpolation path keep exact
autodiff (`jacobian: autodiff`), where its cost is amortised. Rejected alternatives:
reverse-mode with batched cotangents (4 ms — still overhead-bound), float32 finite
differences (1e−3 relative error).

**L-BFGS-B** (scipy) on θ with the box as native bounds and autodiff gradients — the
general-purpose alternative; per surface.

**Monte Carlo in the loop** (`mc_calibrate`): scipy trust-region least squares with the MC
pricer, common random numbers (fixed seed) and finite-difference Jacobians; smooth thanks
to the mixing estimator. This is the "before" in the speed comparison; it is started from
the same point as the surrogate LM.

**Identifiability.** ξ_0 sets the ATM level, ρ the sign and size of the skew, η and H
trade off: the skew *level* scales like η·sqrt(2H)/((H+½)(H+3/2)) while its *term
structure* is T^{H−1/2}. Only the slope across maturities separates η from H, so
recovering H requires several short maturities — another reason for the 0.02/0.05/0.1
nodes. The recovery study reports noise-free targets (identifiability only), MC-noisy
targets (realistic) and high-precision targets separately so these effects can be told
apart.

## 9. Real data

`market.py` reads a user-supplied CSV (canonical layout, NSE legacy bhavcopy or NSE UDiFF
bhavcopy). **Nothing is invented: without a file the step prints a message and records
`"status": "skipped"`.**

1. **Liquidity filters:** minimum price (tick-size floor), minimum volume / open interest,
   maximum relative bid–ask spread when bid/ask exist, minimum days to expiry.
2. **Forward and discount factor by put–call parity.** C − P = D(F − K) is linear in K:
   regress C − P on K over the strikes nearest the money (where both sides are liquid);
   slope = −D, intercept = DF. This uses the options market's own forward (it embeds
   dividends, futures basis and funding) rather than an assumed rate; D outside
   (0.8, 1.05) flags bad data and the expiry is dropped.
3. **OTM only:** puts below F, calls at/above F, normalised as p = price/(D F).
4. **Static-arbitrage filter** on equivalent call prices c(K): bounds (1 − K)^+ ≤ c ≤ 1,
   monotonically decreasing (no vertical-spread arbitrage), convex (no butterfly
   arbitrage); offending quotes are removed greedily. Calendar violations (ATM total
   variance decreasing in T) are counted and reported, not removed.
5. IV inversion; quotes outside the surrogate's domain (z range, T range) are dropped and
   counted.

**Calibrating off-grid quotes:** the surrogate grid is interpolated to each quote —
linearly in z on each grid maturity, then **linearly in total variance σ²T across
maturities**, which preserves monotone total variance and is the standard
arbitrage-friendly choice. The interpolation is written in torch, so the LM Jacobian flows
through it.

ACT/365 calendar-day time to expiry. Weekly NIFTY expiries shorter than the first grid
maturity (0.02y ≈ 7 days) are outside the surrogate domain.

## 10. Baseline: Heston via the COS method

Heston (forward-normalised): dv = κ(θ − v)dt + σ√v dW, d⟨B, W⟩ = ρ dt; five parameters
(v0, κ, θ, σ, ρ).

* **Characteristic function** in the *little Heston trap* form (Albrecher et al. 2007):
  with d = sqrt((κ − iρσu)² + σ²(iu + u²)) and g = (κ − iρσu − d)/(κ − iρσu + d),
  using e^{−dT} keeps the complex logarithm on its principal branch for long maturities
  (the original Heston form jumps across the branch cut).
* **COS method** (Fang & Oosterlee 2008): expand the density of X = ln(S_T/K) in a cosine
  series on [a, b]; coefficients come from the characteristic function, payoff
  coefficients are closed-form. Error decays exponentially in the number of terms for
  smooth densities.
* **Puts + parity:** the put payoff is bounded, so COS is insensitive to the upper
  truncation; calls priced directly blow up with e^b.
* **Truncation:** [a, b] = c₁ ± L sqrt(c₂) with c₂ computed numerically from ln φ (the
  closed-form c₂ in the literature has typos — we hit one; a unit test now compares with
  the numerical derivative). L = 16 was needed for strongly negative ρ at T = 1.5
  (verified against Gil-Pelaez quadrature to 1e−8 in `tests/test_heston_skew.py`).
* Calibration: scipy trust-region least squares with bounds, multi-start. The Feller
  condition is **not** imposed (it is routinely violated by market calibrations and is
  irrelevant for the characteristic function).

## 11. Known limitations

* **Flat forward variance.** One ξ_0 cannot fit a market ATM term structure that is not
  flat; on real data the rBergomi residuals will partly be term-structure error, not
  roughness error. The standard fix is a piecewise-constant ξ_0(t) (8 more inputs);
  the grid design already supports it.
* **Discretisation bias is learned — and it is larger than the MC noise.** The surrogate
  learns the 500-step hybrid scheme, not the continuous model. The time-step ablation
  (README) shows the 500-step bias relative to 2,000 steps exceeds the Monte Carlo
  standard error of the comparison, and is largest at the shortest maturity and smallest
  H. In other words, at this budget more *time steps* would buy more accuracy than more
  *paths*; the next data run should use 1,000+ steps/year, or Richardson extrapolation
  across two resolutions.
* **Grid domain.** Quotes outside z ∈ [−0.42, 0.28] or T ∉ [0.02, 1.5] cannot be priced
  by the surrogate; extrapolation is deliberately refused.
* **Constant rates per expiry.** The implied discount factor is per expiry; no term
  structure model.
* **Single date.** Joint calibration across days (parameter stability) is not attempted.
