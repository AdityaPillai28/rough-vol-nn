# Interview notes — the 15 hardest questions

Honest answers, including the weak spots. Numbers are from `results/*.json` (as rendered
in the README); if the results are regenerated, re-read them there.

---

**1. Why should I believe volatility is rough and not just mean-reverting fast?**
Two separate pieces of evidence. (a) Time series: the scaling of log realised-variance
increments, E|Δ log σ|^q ∝ Δ^{qH}, is linear in q with H ≈ 0.1 across assets and lags
(Gatheral–Jaisson–Rosenbaum 2018). A fast-mean-reverting OU process looks rough only
below its mean-reversion time and has H = 1/2 at the smallest lags. (b) Options: the ATM
skew follows ≈ T^{−0.4} over two decades of maturities. A one-factor Markov model gives a
flat short-end skew; reproducing T^{−0.4} needs a continuum of time scales, which is what
a power-law kernel is. Caveat I would volunteer: realised-variance estimates have
measurement noise that biases H downwards, and there is a literature arguing that H is
somewhat higher than 0.1 once you correct for it. The options evidence does not depend on
that estimator.

**2. Derive the T^{H−1/2} skew.**
Alòs–León–Vives: ψ(T) ≈ ρ/(σ₀T²) ∫₀^T∫_s^T E[D_sσ_t] dt ds. For rBergomi,
D_sσ_t = σ_t · ½η√(2H)(t−s)^{H−1/2}. The double integral is
T^{H+3/2}/((H+½)(H+3/2)), so ψ(T) ≈ |ρ|η√(2H)/(2(H+½)(H+3/2)) · T^{H−1/2}. For Heston,
D_sσ_t → σ/2, which gives ρσ/(4√v₀), a constant. Empirically (README skew table) the
MC slopes for H ∈ {0.05, 0.1, 0.2, 0.3} are within 0.016 of H − ½, and a Heston model fitted
to the same H = 0.1 surface has slope −0.03 ± 0.014 (flat).

**3. Your MC skew lies below the theoretical line at short T. Is your simulator wrong?**
The slopes match, but the level at T = 0.004 is ~20–30% below the asymptotic constant.
Two reasons. The formula is a T → 0 limit with corrections of relative order T^{2H} (and
T^{2H} = 0.33 even at T = 0.004 for H = 0.1). And a 1.5-day option has only 10 time steps
at 2,500/yr. I would check this by refining the step at fixed T. What makes me believe the
simulator itself: E[V_t] = ξ₀ exactly, martingale and η = 0 → Black–Scholes tests pass
within 4 SE, and the Volterra variance matches t^{2H} to < 2% away from t = 0.

**4. Why the hybrid scheme and not exact simulation?**
Exact Cholesky on the joint (Ŵ, W) grid is O(n³) to set up for each H, and H changes with
every one of the 80k training samples. It is O(n²) per path and needs a 1,500 × 1,500
factor per sample. The κ = 1 hybrid scheme integrates the singular part of the kernel
exactly in the cell next to the diagonal, approximates the smooth far field at the
MSE-optimal points b_k, and does the far field as an FFT convolution: O(N n log n), about
75 ms for 40k paths × 750 steps on a laptop GPU. A plain Riemann sum is biased exactly
where it matters (small H), because it misses the singular mass.

**5. What is the time-discretisation error, and does it matter?**
It matters more than anything else in this pipeline, and I measured it. At 500 steps/yr
the mean absolute IV difference against 2,000 steps/yr is 0.10 vol points (0.16 at
T ≤ 0.1, 0.27 in the rough corner H = 0.05). The MC SE of that difference is 0.06. The
surrogate's own error is ~0.1% relative (≈ 0.02 vol points). So the surrogate faithfully
learns the 500-step model, and the gap to the continuous model is the step size. Next
time I would spend the GPU budget on 1,000+ steps or Richardson extrapolation, not on
more paths.

**6. Your surrogate beats the Monte Carlo noise floor. How is that possible?**
Each label is truth + ε with E[ε] = 0 and ε independent across parameter sets. An MSE
regressor estimates the conditional mean, so it averages the noise of nearby parameter
sets (implicit smoothing over 64k samples). Against a 640k-path reference, the residual
MLP's mean relative error is 0.099%, versus 0.257% for a single 40k-path label (ratio
0.39). The paper's 3×30 MLP is at 0.94 of the noise floor: it is capacity-limited, not
noise-limited. Two caveats: the reference has its own noise (0.078% relative SE), so the
measured 0.099% is an upper bound; and smoothing removes noise, not bias.

**7. Why grid-based rather than a network that takes (T, k) as inputs?**
One forward pass gives all 88 vols. The 88 × 4 Jacobian for LM is one cheap
differentiation. The output layer can exploit cross-strike correlation. The cost is a
fixed domain: off-grid quotes need interpolation (I do linear in z, linear in total
variance across T, which is differentiable) and nothing outside the grid can be priced.
For market data with weekly expiries below 7 days, that is a real limitation.

**8. Why does the paper's MLP do worse than yours? Isn't that just more parameters?**
Mostly, yes: 4.7k vs 111k parameters. The residual structure helps optimisation at that
width, and SiLU gives smooth Jacobians. I did not run a width-matched plain MLP, so I
cannot attribute the gain to the residual connections specifically; that is the
ablation I would add.

**9. How identifiable are the parameters?**
On noise-free targets (the surrogate's own output), LM recovers all four parameters to
~1e−7, so the map is locally invertible on the box. With MC-noisy targets the MAEs are
H 0.0028, η 0.018, ρ 0.0077, ξ₀ 0.0004. On the high-precision reference: H 0.0012,
η 0.006, ρ 0.003. ξ₀ is pinned by the ATM level. ρ and η both steepen the skew, and H
only enters through the term structure of the skew. So η and H trade off, worst near
H = 0.5 (flat skew term structure) and for strongly negative ρ. The short maturities
(0.02, 0.05, 0.1) are what make H identifiable.

**10. How do you handle the box constraints in LM, and why Marquardt scaling?**
θ = lo + (hi − lo)·sigmoid(u). The sigmoid's slope vanishes near the edges, which would
make plain LM (λI damping) crawl. Marquardt damping λ·diag(JᵀJ) is invariant to
per-parameter rescaling, so the reparametrisation does not distort the steps. Edge
parameters are still recovered less well: fewer training neighbours and a flat sigmoid.
I also fixed a stopping bug found by profiling: rejected steps never triggered
convergence, so converged starts kept iterating until λ hit 1e11.

**11. Your speed-up is only 68×. The paper reports much more.**
My baseline is strong: GPU Monte Carlo, the conditional (mixing) estimator and common
random numbers, which make the MC pricer smooth in θ. That gives ~3.9 s per calibration
(52 pricer calls). A CPU, plain-estimator baseline would be orders of magnitude slower,
and the ratio would look better for no good reason. The surrogate single-surface LM is
57 ms on one CPU thread (4 starts). Batched on the GPU it is ~1 ms per surface, ≈4,000×.
Also worth saying: MC-in-the-loop recovered parameters *worse* than the surrogate on the
same surfaces, because its finite-difference Jacobian with 40k paths is noisy and
max_nfev bounds it.

**12. Why finite differences for the Jacobian when you have autodiff?**
Only on the single-surface CPU path, and only after measuring. torch's forward-mode
`jacfwd` costs ~6 ms per call for this network on CPU. That is framework overhead: a
forward pass is 0.2 ms. Central differences in float64 on the smooth network cost
0.65 ms, agree with autodiff to 3e−8 absolute, and give identical parameters to four
digits. Truncation error is O(h²) ≈ 1e−12 for a C^∞ network, and float64 rounding is
≈ 1e−10. Batched GPU runs and the market path keep exact autodiff.

**13. What does the conditional (mixing) estimator buy, and when is it weakest?**
Conditional on the vol driver W, ln S_T is Gaussian, so the payoff expectation is a Black
price with forward exp(ρI − ½ρ²Q) and variance (1 − ρ²)Q. It removes the W⊥ noise exactly,
never produces zero prices (so IV inversion never fails: 0 failures in 7M cells), and is
smooth in θ under common random numbers. The gain shrinks as |ρ| → 1, because then
almost all the noise comes from W. Measured plain/mixing SE ratio: median 1.6, range
1.0–30 across parameter sets.

**14. How would this work on NIFTY, and what would go wrong first?**
Loader: one trade date. Forward and discount from a put–call-parity regression near the
money (this absorbs dividends and futures basis), OTM quotes only, liquidity filters,
then a greedy static-arbitrage cleanup (monotone, convex calls). Calendar violations are
reported. I have not run it: no file was supplied, and I do not fabricate market data.
What I expect to go wrong first: (a) flat ξ₀ cannot fit a sloped ATM term structure, so
residuals will be term-structure error. The fix is piecewise-constant ξ₀(t), i.e. 8 more
inputs. (b) Weekly expiries under 7 days fall outside the grid. (c) Bhavcopies carry
settlement prices, not bid/ask, so the spread filter cannot run. (d) A one-day
calibration says nothing about parameter stability.

**15. If you had another week, what would you change?**
In order: (1) regenerate with 1,000–2,000 steps/yr. Discretisation, not MC noise or the
network, is now the binding error. (2) Piecewise-constant forward variance, for real
data. (3) A width-matched plain-MLP ablation, to separate depth/residual effects from
capacity. (4) Calibrate on a real chain and study parameter stability across days.
(5) A control variate (e.g. the timer-option approach of McCrickerd–Pakkanen) for
strongly negative ρ, where mixing helps least.
