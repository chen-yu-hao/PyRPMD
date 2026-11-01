# H + CH4: proposed second, excluded reference pilot

**Design only: unimplemented, not adopted and not executed.** The only computation
in this directory replays the complete saved 768-draw pilot without loading a
PES, drawing a random number or changing any prior result. The proposed next
experiment would use at most **4,000 P1 PES geometries on one CPU**, including all
proposal calibration and controls. It needs independent method/implementation
review and a separate operational adoption before execution.

The purpose is to test whether a better proposal can make the existing finite-
domain classical flux integrals practical. This is not a thermal-rate benchmark,
a quantum calculation, a transmission calculation or a paper-agreement test.
The original [geometry derivation](../hch4-reference-feasibility/README.md),
[768-draw design](../hch4-reference-feasibility/pilot-design/README.md) and
[complete excluded result](../hch4-reference-pilot/README.md) remain immutable.

## 1. What the complete old data diagnose

[saved-pilot-diagnosis.json](saved-pilot-diagnosis.json) uses every original draw,
all 641 successful energies, all 127 geometric zeros and all eight diagnostic
seeds. Its [script](diagnose_saved_pilot.py) rechecks the frozen input hashes and
reconstructs the saved importance weights from energies and geometric prefactors.
It imports no potential or dynamics code and makes no new random draws.

| Classical target | Empirical ESS / 256 | Largest weight | Top four weights | Draws carrying 90% of observed weight |
| --- | ---: | ---: | ---: | ---: |
| Surface alpha = 0.999 | 4.33 | 27.65% | 93.48% | 4 |
| Surface alpha = 1.007 | 7.18 | 22.01% | 67.56% | 8 |
| Entry at 30 bohr | 6.14 | 33.00% | 65.47% | 9 |

At alpha = 0.999, the negative- and positive-mirror Gaussian components supplied
101 and 102 draws, yet contributed 95.95% and 4.05% of the observed weight.
The proposal and physical reflection symmetry call for equal mirror treatment;
this noisy imbalance must not become unequal fitted mixture weights. The
observed weighted standard deviation of the active C-H radius was only
0.0327 bohr, compared with the old proposal's 0.15 bohr. However, four effective
weights cannot estimate an 11-dimensional covariance reliably. Leave-one-seed-
out moments and correlation spectra are retained as diagnostics, not adopted
proposal parameters.

The uniform-box components contributed negligibly to the *observed* weight.
Only 11 uniform draws per target survived geometry checks, and their energies
were high. This does not bound narrow unseen regions of the finite domain or
mass outside it. Similarly, tiny observed weight near the old boundaries does
not establish molecular-domain convergence. The new proposal therefore retains
an explicit positive density across a wider audit domain and separately reports
tail-component and boundary-shell behavior.

The new diagnosis initially expected the saved completion-status literal
`complete`; the historical report actually says `complete_cost_pilot`. That
read-only guard failure and exact initial script are retained in
[initial-diagnosis-schema-failure.json](initial-diagnosis-schema-failure.json).
The correction accepts only the actual frozen status and still requires the
entire 768/641/127 membership. No original input was changed.

## 2. Keep the physical integral and channel accounting fixed

The equal-isotope masses, beta, CBE/PES-2008 identity, reference bond difference,
atom labels and finite separation remain those of the original design. Both
alpha = 0.999 and 1.007 still use **one bead**. The latter is only a classical
surface diagnostic; it does not reproduce the published P128 surface ensemble.

Use the same nine methane coordinates

\[
z=(r,s,v,a_x,a_y,a_z,b_x,b_y,b_z),\qquad J_z=r^2s^2,
\]

and the same four charts, active-maximum indicator, positive radial root and
connected-component condition. With the original notation,

\[
f_{h,\alpha}(z,u)=I_\mathcal D I_h J_z
\frac{\rho_h^2}{|\partial S_h/\partial\rho|}
\sqrt{\frac{G_\alpha}{2\pi\beta}}
\exp[-\beta(V-E_0)].
\]

The entry integrand remains

\[
f_0(z,u;L)=I_\mathcal D J_z
\frac{L^2}{\sqrt{2\pi\beta\mu}}
\exp[-\beta(V(z,d+Lu)-E_0)].
\]

Each integral is with respect to `dz dOmega`. Sum the **four separately sampled
channel means once**, and divide by the one entry-chart mean. There is no new
factor four, rotor symmetry factor, mass correction, accepted-draw normalization
or absolute partition-function approximation. The common energy offset and
global rotation volume cancel as before. Every geometric zero remains in its
original unconditional component and seed count.

## 3. A normalized joint shape/direction proposal

The old product of diagonal shape Gaussians and an independent direction law
cannot represent shape/approach correlations. For the TS proposal only, use a
stereographic direction coordinate `t=(tx,ty)`:

\[
u(t)=\frac{(2t_x,2t_y,1-|t|^2)}{1+|t|^2},\qquad
J_\Omega(t)=\frac{4}{(1+|t|^2)^2},\qquad d\Omega=J_\Omega\,d^2t.
\]

A normalized 11-dimensional density `g(z,t)` therefore has density
`g(z,t)/J_Omega(t)` with respect to `dz dOmega`. At the single excluded pole,
the Gaussian/Student contributions have their zero limiting value; the uniform
sphere component remains defined. The implementation must not multiply the
final importance weight by this Jacobian a second time. Equivalently, both
numerator and proposal can be represented in `dz dt` and the same factor cancels.

Two local quadratic models, one per alpha, supply joint means and covariances.
For each model retain both exact reflected proposal components: negate `ay`,
`by` and `ty` in the mean, and transform the covariance by the same diagonal
reflection matrix. All four alpha/mirror components have equal weights within
their family. The canonical numeric shape law is reused in each of the four
labeled charts; the four target integrals are still sampled and summed separately.

The fixed TS density is

\[
q_{TS}(z,u)=\frac{1}{8}\frac{I_{B_+}(z)}{|B_+|4\pi}
+\frac{3}{4}\frac14\sum_{j=1}^4\frac{\mathcal N_{11}(z,t;\mu_j,C_j)}{J_\Omega(t)}
+\frac{1}{8}\frac14\sum_{j=1}^4\frac{T_{5,11}(z,t;\mu_j,\Sigma_j)}{J_\Omega(t)},
\]

where the Student distribution has five degrees of freedom and
`Sigma_j = 4*(5-2)/5 * C_j`. Its covariance is four times `C_j`, supplying broader
tails without changing the target. All component densities, including components
not selected for that draw, enter the log-summed denominator.

The entry proposal keeps a uniform physical direction. Its nine-dimensional
shape density has uniform weight 1/8, total Gaussian weight 3/4 split equally
between the two mirrors, and total Student weight 1/8 split equally. Divide that
shape mixture by `4*pi` to obtain its density with respect to `dz dOmega`.
Do not learn an entry-direction preference from the old weighted direction mean:
that estimate also came from only about six effective weights.

The wider box is `r,s in [1.2,5.0]`, `v in [-1,1]`, and each Cartesian coordinate
of `a,b in [-5,5]`. It contains every shape in the audit domain defined below.
Thus `q >= 1/(8*|B_+|*4*pi)` there. All Gaussian and Student components remain
untruncated; invalid target points have zero weight. This is a support proof,
not a bound on unobserved importance weights or an assertion of physical accuracy.

### Fixed stratification and independent seeds

The proposed fresh seeds are **2026112201 through 2026112208**, disjoint from the
original seeds. Each seed has 32 latent TS draws per channel, evaluated at both
alpha values with the same `z,u`: four uniform draws, six from each of the four
Gaussians, and one from each of the four Student components. It also has 128
entry draws: 16 uniform, 48 per Gaussian mirror, and eight per Student mirror.

These fixed counts exactly match the mixture probabilities. The unconditional
mean of `f/q` remains unbiased for each integral because

\[
\sum_j\frac{n_j}{n}\int q_j(x)\frac{f(x)}{\sum_k(n_k/n)q_k(x)}dx=\int f(x)dx.
\]

This identity requires the *full stratified mixture density*. Do not average
selected-component ratios, renormalize surviving draws, or treat the paired
alphas as independent seeds. Each fresh seed keeps all its channel, entry,
mirror, broad-component and paired-control observations together.

## 4. Deterministic calibration, fully charged to the budget

The old weighted covariance is not used. Instead, construct a local quadratic
model around the exact distributed TS geometry, mapped to the alpha surface
using the already reviewed root equation. Use the regular tetrahedral entry
geometry from the old design, with incoming direction `+z` at 30 bohr. The entry
model only builds a proposal; actual entry integration still samples the full
sphere and full six-atom energy.

For TS calibration use the smooth negative log density in `x=(z,t)`:

\[
W_\alpha(x)=\beta(V(x)-E_0)-\log(J_z\rho^2f_S/|S_\rho|)-\log J_\Omega(t).
\]

For the entry shape model use its corresponding `W_0(z)` at the fixed training
direction, without an angular fitting parameter. Transform to dimensionless
coordinates using the old fixed shape standard deviations. The TS direction
scales are both `1/(2*sqrt(8))`, the local stereographic width of the old
concentration-eight direction component. These are scaling conventions, not
estimated physical widths.

Use **energy-only** central finite differences at dimensionless step `h=0.02`.
The pinned [PES adapter](../../../../../benchmarks/hch4/pes.py) documents a
2.06 ppm legacy difference between the reported force scale and the derivative
of reported energy. This design does not alter that adapter or silently treat
those forces as exact energy derivatives. A proposal Hessian built from the
reported energy avoids that issue entirely.

For dimension `d`, a full central Hessian/gradient stencil needs `1+2*d^2`
geometries. The three models have dimensions 11, 11 and 9. Diagonal second
differences are also checked at `h/2`. No optimizer, uncounted initialization,
force-difference stencil or hidden Hessian evaluations are permitted.

Require all stencil points to remain in the original domain and selected active
chart. Require a finite symmetric positive Hessian, minimum dimensionless
eigenvalue above `1e-6`, condition number at most `1e8`, and maximum diagonal
step-refinement discrepancy
`abs(H_h(ii)-H_h/2(ii))/max(1,abs(H_h/2(ii))) <= 0.05`. These are fixed proposal-
construction checks, not physical convergence tolerances.

Form `C_y=H_y^-1` and a single Newton displacement `delta=-H_y^-1*g_y`.
Require `sqrt(delta.T*H_y*delta) <= 3` and a valid shifted center. Do not clip
eigenvalues, clip the displacement, optimize further or change the proposal if
this construction fails. If `S` is the fixed diagonal scaling matrix and
`x=x0+S*y`, the actual proposal parameters in the original chart are
`mean_x=x0+S*delta` and `C_x=S*H_y^-1*S.T`. Use these transformed parameters in
every Gaussian/Student density and reflection operation. Order the eigenpairs of
`H_y` by ascending eigenvalue. Its unit-width probes in original coordinates are
`mean_x +/- S*eigenvector_i/sqrt(eigenvalue_i)`; the first axes therefore refer to
the softest directions in the declared scaled coordinates. Evaluate each shifted
center and these positive/negative probes. Require their maximum absolute
deviation from the quadratic prediction `W(center)+1/2` to be no larger than
five dimensionless energy units. This is an efficiency screen; passing does not
establish harmonic physics or control distant tails.

Finally evaluate eight reflected preexisting points per model: the shifted
center, both signs on its first three Hessian axes, and the positive fourth-axis
point. Require their reflected energy differences to be at most `1e-10 hartree`
and distance changes at most `1e-12 bohr`. Preserve every failed calibration
point and stop; do not silently create a different local proposal.

| Stage | Maximum PES geometries |
| --- | ---: |
| Full Hessian/gradient stencils: 243 + 243 + 163 | 649 |
| Half-step diagonal checks: 22 + 22 + 18 | 62 |
| Three shifted centers | 3 |
| Unit-width axis probes: 22 + 22 + 18 | 62 |
| Eight reflected existing points per model | 24 |
| **Total deterministic calibration** | **800** |
| Fresh fixed TS and entry diagnostic geometries | 3,072 |
| Paired entry-only 60-bohr control geometries | 128 |
| **Entire proposed experiment** | **4,000** |

Write and hash the initial 711-geometry stencil ledger before its first call.
The final 89 calibration geometries are deterministic functions of that complete
stencil, not its rate or integral estimate; save their ledger before evaluating
them. Freeze the resulting proposal matrices and all calibration outcomes before
generating or evaluating any fresh diagnostic draw. The full 3,072+128 diagnostic
ledger must then be materialized before any of its PES calls.

## 5. Domain, tail and separation diagnostics

The primary mathematical pilot target stays exactly the old finite domain:
all four CH radii in `[1.4,4.5]` bohr and every pair at least 0.7 bohr. Root,
active-max and connected-component conditions also stay unchanged.

The separate audit envelope extends the CH range to `[1.2,5.0]` and the minimum
pair to 0.6 bohr. Evaluate sampled geometries throughout this envelope and save
three indicators: an inner domain `[1.6,4.0]` with minimum pair 0.9, the original
domain, and the outer envelope. The original-domain weight is still zero outside
its original boundary. Report the inner/original and outer-shell masses, counts,
concentration and independent-seed participation separately. The wider domain
is an unvalidated PES/molecular diagnostic, not a new physical target or a way
to improve a paper comparison. A shell with no sampled weight remains unresolved;
no zero-observation upper bound is asserted.

For each seed, preselect the first two uniform entry draws, first six draws from
each Gaussian mirror and first draw from each Student mirror. Those 16 draws
have exactly the same component fractions as the full entry mixture. Re-evaluate
them with the incoming atom at 60 rather than 30 bohr, keeping methane shape and
direction fixed. This is 128 additional target geometries, including retained
geometric zeros. Compare **the same selected 16 draws per seed at both 30 and
60 bohr**, dividing out each side's `L^2`. Do not compare the full 128-draw
30-bohr mean with the 16-draw 60-bohr mean. The primary flux denominator still
uses all 128 original entry draws per seed; this separate paired subset only
serves the separation diagnostic.
This isolates an entrance-potential separation diagnostic; it does **not** prove
full finite-R-infinity convergence of the alpha surfaces or the complete rate.

The output must retain per-target and per-channel empirical ESS, maximum weight,
top-1% weight share, each mixture-component contribution, mirror balance,
leave-one-whole-seed-out sensitivity, all nested-domain statistics and all paired
separation results. Broad-tail and uniform draws are never removed or replaced.

## 6. Interpretation and stopping rules

Predeclare a proposal-efficiency screen for the original-domain targets: each
of the eight channel/alpha groups must have ESS at least 32 out of 256 and maximum
normalized weight at most 0.10; each pooled target must have ESS at least 128 out
of 1,024 and maximum weight at most 0.025. Also require its top 1% of draws to
carry at most 20% of the observed weight and removing any whole seed to change
the log integral by at most 0.25. These thresholds assess whether a later
independent precision study is plausible. They are not a guarantee against
unseen tails and never produce a validated-rate or paper-pass flag.

The central diagnostic ratio remains the ratio of pooled integral means,
`Abar/Bbar`, not the mean of seed ratios or exponentiated mean log ratios.
If approximate Monte Carlo intervals are reported, use the eight conditional
independent seed vectors. For `log(Abar/Bbar)`, the delta-method influence is
`d_i=A_i/Abar-B_i/Bbar`; use its sample standard deviation divided by `sqrt(8)`
and the nominal Student-t multiplier with seven degrees of freedom. Preserve
the paired-alpha and entry covariance. This small-sample approximation is not
a valid tail bound, a formal coverage guarantee or a physical-rate interval.
An exactly zero integral is unresolved; no epsilon substitution may be used to
create a finite log ratio or uncertainty estimate.

There is one fixed attempt and no automatic extension, adaptive sample size,
seed replacement, proposal refitting after diagnostic observations, physical
rate conversion, quantum correction or paper-agreement test. A failed calibration,
PES error, nonfinite result, exhausted budget, incomplete cohort or inadequate
ESS is retained honestly. A later experiment needs a new design.

Propose the same bounded resource contract as the earlier pilot: one CPU and
one thread, whole-process hard caps of two seconds for preparation, three for
supervision and 55 for the worker (60 seconds total), a combined soft admission
limit of 55 seconds, and an outer wall ceiling of 120 seconds. The supervisor's
overhead and finalization must remain charged. These are ceilings, not an
assurance that all 4,000 geometries will complete. New code must retain the
reviewed invocation-token/one-attempt protections and all failure receipts.

## 7. Required checks before any adoption

An independent reviewer must check the stereographic density/Jacobian, reflection
map, full-mixture normalization, deterministic stratification, unchanged four-
channel coarea measure, nested-domain membership and pooled-ratio covariance.
Implementation tests must include analytic Gaussian and Student normalization,
sphere-Jacobian identities, known finite-domain integrals, four-chart versus one-
entry counting, unconditional zeros, complete staged-ledger counts, invalid-
Hessian/failed-PES interruption, no fresh sampling before proposal freeze, and
all hard resource/one-attempt paths. No actual new evaluation is authorized by
this document or by a successful synthetic gate.
