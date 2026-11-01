# Enhanced sampling and ML: quantitative validation

This campaign separates three questions: whether the equations are implemented
correctly, whether finite trajectories sample their intended distributions,
and whether a published numerical result has actually been reproduced.
The live implementation comparison uncovered and fixed an OPES initialization
error. The much shorter Müller trajectories are convergence stress tests;
they must not be described as reproducing the paper's sampling/convergence
curves. The paper itself does not claim those plotted Müller runs are fully
converged.

The reusable driver is
[`sampling.py`](../../../../benchmarks/tier2_literature/sampling.py), with
[`plot_sampling.py`](../../../../benchmarks/tier2_literature/plot_sampling.py)
for figures. [The source map](source_map.json) records primary sources,
exact input-file revisions, and the scope of each comparison.
[Numerical results](RESULTS.md) report all completed independent seeds.

## What is compared

| Experiment | Reference | Evidential scope |
| --- | --- | --- |
| Müller basin free energies | Printed values in Invernizzi and Parrinello's 2022 SI | Direct comparison of published numbers with independent numerical quadrature |
| Fixed OPES and OPES-explore | Real PLUMED 2.9.4, same prescribed depositions | Bias energy and force equations, initialization, fixed physical widths; no trajectory or adaptive-width parity claim |
| Müller stochastic trajectories | Exact marginals and basin integrals of the published model | Finite-budget sampling performance on the same model, with stated algorithm and budget differences |
| Shared well-tempered MetaD | Exact harmonic frozen-bias density and ideal well-tempered marginal | Synchronous shared-history and thermal-sampling engineering checks |
| Temperature/Hamiltonian exchange | Exact finite-bead harmonic Gaussian moments | Equilibrium ensemble check using the public propagator and exchange implementation |
| Biased first-passage clock | Computed overdamped reflecting/absorbing mean first-passage time | Analysis API test using a static, explicitly transition-free bias; no native infrequent-MetaD or OPES-flooding driver validation |
| TorchScript committee and Δ-learning | Held-out analytic energy/force function | Trained-model lifecycle engineering test; no molecular ML-paper reproduction |

## Published model and independently calculated reference

The [2022 paper](https://doi.org/10.1021/acs.jctc.2c00152), SI S3,
defines the free-energy integrals on x∈[−1.3,1.0], y∈[−0.2,2.0].
The printed basin differences are **5.712(6)** for the x=0 separator and
**5.690(5)** for y=x+1.3. We preserve that printed notation without assigning
an unreported confidence-interval meaning to the parentheses.
Independent nested adaptive quadrature gives **5.712591741222511** and
**5.69051678439633** kBT, respectively. Tightening the quadrature tolerances
changes the results by about 4×10⁻¹². These are computed reference values,
not estimates from our trajectories. The separately checked analytic force
agrees with central finite differences to 1.8×10⁻⁸ in model force units.

The model coefficients, 0.15 energy scale, wall convention, and four selected
initial positions/seeds come from the
[authors' pinned inputs](https://github.com/invemichele/OPES-explore/tree/47b5d4c92f87920dd7750900c751a1e37011ed6a/mueller).
The code implements the potential and its analytic derivative independently.
An independent harmonic z coordinate embeds the 2D model in the public
`[P,N,3]` API and factors out of every x/y statistic.

| Parameter | Authors' input | Local primary campaign |
| --- | --- | --- |
| kBT, mass, friction | 1, 1, 10 | Same numerical model units |
| Timestep, deposition pace | 0.005, 500 | Same |
| Barrier, bias factor | 20, default γ=20 | 20, explicit γ=20 |
| Biased coordinate | x | x and a separate y control |
| MD steps per trajectory | 200,000,000 | 80,000 adaptive + 40,000 frozen |
| Width and compression | PLUMED adaptive widths, compressed kernels | Literal fixed σ=0.1, or this package's deposition-Welford rule; no compression |
| Independent runs | Paper Fig. 4 averages 25 runs; four input files are selected here | Seeds 57859, 28661, 38795, 53730 from those four selected inputs |

The paper's plotted iteration count is not the MD step count: each update
takes 500 MD steps. Our adaptation budget is 2,500 times smaller.
The local adaptive rule is documented in
[`tier2-sampling.md`](../../../tier2-sampling.md); it is not PLUMED's
exponentially updated diffusion-window width estimator.

## A real bug found by comparison with PLUMED

For both OPES variants, the empty-density limit is

\[
V_0(s)=c\log\epsilon=-\mathrm{barrier},\qquad \nabla V_0=0.
\]

The original PyRPMD empty history returned zero bias. For OPES-METAD,
whose next kernel weight is exp(βV), that made the first weight 1 instead
of exp(−β barrier). The offset is not harmless once it changes that first
weight relative to later weights.

The live comparison uses 80 deterministic deposition centers, 21 probes per
deposition, β=1, barrier=γ=20, and physical σ=0.1. Both codes use fixed widths,
compression is disabled, and PLUMED's kernel cutoff is 12 standard deviations.
There is no spline or trajectory interpolation in this one-dimensional,
nonperiodic Cartesian-CV comparison.
PLUMED's explore input multiplies SIGMA by √γ; its input is therefore
0.1/√20 to match PyRPMD's literal physical width.

| Method | Legacy maximum energy difference | Corrected maximum energy difference | Corrected maximum force difference |
| --- | ---: | ---: | ---: |
| OPES-METAD | 11.225256 | 7.11×10⁻¹⁵ | 9.24×10⁻¹⁴ |
| OPES-explore after the first deposition | 2.75×10⁻¹⁴ | 2.75×10⁻¹⁴ | 9.95×10⁻¹⁴ |

Explore kernels are unweighted, so the old empty constant did not affect
its post-first-update bias or forces. Two already completed explore histories
are retained with their original schema-1 provenance, as recorded in
[`legacy_explore_reuse.json`](legacy_explore_reuse.json). All corrected METAD
trajectories use schema 2. The original METAD data and direct comparison are
preserved separately under `opes_before_initialization_fix/` and
`plumed_comparator_before_initialization_fix/`.

New checkpoints persist `initial_bias=-barrier` in bias state schema 2.
Recognized old schema-1 checkpoints continue exactly with their old weights
and zero empty-history bias, including an empty checkpoint. They remain
schema 1 and analysis reports that provenance. Tests cover exact continuation,
unknown/inconsistent state rejection, workflow resume/reanalysis, and a live
PLUMED regression. `PYRPMD_REQUIRE_PLUMED=1` turns a missing runtime into
a test failure instead of a skip.
Partially started legacy jobs initialize missing replicas with the same
legacy convention. Missing replicas alongside mixed old/new checkpoints
are rejected as ambiguous; existing streams remain available for reanalysis.

## Reading the trajectory statistics

Each OPES history uses the public native `Dynamics.sample_step` path with
P=1 and the normal-mode BAB Langevin splitting. Between depositions, an
energy-consistent cubic spline accelerates the immutable bias. Its 4,097-point
grid is audited against direct energy and derivative calls after every
deposition; maxima are retained for each run. Out-of-grid evaluations call
the direct bias. This numerical acceleration is a stated difference from
a fully direct force trajectory; it is not used for the live PLUMED test.

Trajectories save every ten MD steps. Analysis drops the first 8,000 frozen
steps and uses the remaining 3,200 frames. It conditions x/y samples on the
published integration rectangle, records the retained fraction, and compares
80-bin CV histograms with the corresponding exact marginal. Reported errors
include:

- Direct KDE FES RMSE over bins with exact F−Fmin≤10 kBT, after fitting one
  additive constant. This checks the bias's density estimate.
- Frozen-bias reweighting with exp(βV), ΔF for both basin definitions,
  histogram FES RMSE, and Jensen–Shannon divergence using natural logarithms.
- Missing reference probability, because a low global divergence can hide an
  unvisited rare basin. A missing basin makes ΔF undefined; it is stored as
  null and is never replaced by zero or silently counted as converged.
- Crossings between oblique basin cores y−x−1.3>0.2 and <−0.2; passage through
  the intervening strip retains the previous core and is not counted twice.
- Distance from the exact stationary density for each frozen bias, separately
  from distance to the ideal well-tempered marginal. This distinguishes
  inadequate equilibration from an imperfect or regularized adaptive bias.
- Weight-only Kish ESS, explicitly excluding autocorrelation. Four-seed spread
  and 95% Student-t intervals describe independent-run variability; intervals
  over only the finite subset of ΔF estimates are not evidence of convergence.

The finite regularizer also means the ideal well-tempered law is an
approximate target: for explore with barrier=γ=20, ε≈0.349. Its deviation
must not automatically be attributed to a propagator error. In the poor-x-CV
stress test, a globally small reweighting divergence can coexist with zero
crossings and an undefined basin difference. That is a failure to demonstrate
convergence, regardless of a favorable scalar FES score.

## Other engineering checks

Replica exchange compares centroid, bead, and momentum variances with the
finite-P harmonic distribution, including the β-dependent spring energy.
The references are derived Gaussian moments; comparisons use independent
seeds and retained thermodynamic slots after exchange. No protein-folding
result is inferred from this test.

The shared MetaD experiment checks that all four walkers receive heights
computed from the same pre-update bias and that permuting a simultaneous
batch leaves its final bias unchanged. A frozen stage samples exp[−β(U+V)],
then reweights to the known harmonic distribution. Adaptive deposition does
not obey a stationary detailed-balance assertion; the stationary claim is
tested only after deposition stops.

The kinetics experiment integrates an independent overdamped SDE in
U(x)=6(x²−1)² with reflection at −1.8 and absorption at 0.8. Its smooth static
bias is supported strictly below x=−0.35; bias and gradient are audited as
zero throughout the transition region. Every integration interval contributes
to the rescaled clock. The exact reflecting/absorbing mean first-passage
integral supplies the reference. This exercises the analysis API on actual
biased paths and does not validate an adaptive flooding or infrequent-MetaD
deposition policy. Rate-unit conversions are not chemical predictions.

The ML experiment fits four bootstrap polynomial models with PyTorch,
exports real TorchScript files, evaluates held-out energies and autograd
forces, and runs an actual monitored PIMD trajectory. Monitoring must leave
the trajectory unchanged and collect only unlabeled candidate geometries.
Full Δ-learning is compared with the corresponding full model, and contracted
forces are checked by finite differences of the approximate RPC Hamiltonian.
Committee disagreement is a population spread, not a calibrated error bound.
The known target is `U=sum(0.3*q**2 + 0.04*q**4 + 0.02*(1-cos(3*q)))`.
Training uses 512 geometries in [−0.8,0.8]³ with energy-label noise standard
deviation 0.0002, fitting the three features sum(q²), sum(q⁴), and sum(q⁶).
Held-out interpolation uses 400 new geometries in the same box; extrapolation
uses 400 points from [−2,2]³ whose maximum absolute coordinate exceeds 1.2.
This deliberately limited model class exposes common model bias as well as
committee spread.

## Reproduction

Run from the repository root with project dependencies, SciPy, Matplotlib,
and optional CPU PyTorch installed. Build the native extension for the
trajectory campaigns. Set BLAS/OpenMP threads to one when using concurrent
campaign processes.

```bash
python benchmarks/tier2_literature/sampling.py reference
python benchmarks/tier2_literature/sampling.py opes --steps 80000 --freeze 40000 --grid-points 4097 --cvs x
python benchmarks/tier2_literature/sampling.py opes --steps 80000 --freeze 40000 --grid-points 4097 --cvs y
python benchmarks/tier2_literature/sampling.py summarize
python benchmarks/tier2_literature/sampling.py plumed
python benchmarks/tier2_literature/sampling.py exchange --steps 24000
python benchmarks/tier2_literature/sampling.py shared --steps 20000 --freeze 30000
python benchmarks/tier2_literature/sampling.py kinetics --replicas 128 --dt 0.001
python benchmarks/tier2_literature/sampling.py kinetics --replicas 128 --dt 0.0005
python benchmarks/tier2_literature/sampling.py ml
python benchmarks/tier2_literature/sampling_sensitivity.py
python benchmarks/tier2_literature/plot_sampling.py
python benchmarks/tier2_literature/report_sampling.py
```

The OPES driver refuses to reuse cached results with different run settings
or a legacy METAD schema. Use a separate output directory when changing a
budget. Source files and parameters are retained.

### Published raw data versus rebuildable caches

[`CURATED_FILES.txt`](CURATED_FILES.txt) is the exact repository-relative
publication list; [`raw_manifest.json`](raw_manifest.json) gives content hashes
and explains omissions. The **50 retained NPZ files contain scientific arrays**,
not restart q/p/RNG state. They are required to replay the saved-data figures
and distributions without rerunning millions of stochastic steps. The general
repository `*.npz` ignore rule must be overridden for these specific paths.

| Included raw paths | Count | Unique information |
| --- | ---: | --- |
| `opes_v2/*/trajectory.npz` | 36 | Actual x/y/bias time series, FES snapshots, retained histograms: 32 primary runs and 4 sensitivity runs |
| `opes_before_initialization_fix/*/trajectory.npz` | 8 | Original stochastic histories documenting the pre-fix behavior |
| `shared_metadynamics/seed_*.npz` | 4 | Simultaneous four-walker trajectories and frozen/reweighted distributions |
| `replica_exchange/raw_moments.npz` | 1 | Slot-resolved time series of moments and replica labels |
| `ml_lifecycle/training_and_heldout.npz` | 1 | Training geometries/labels and held-out predictions, exact forces, disagreement and errors |

Eight small `ml_lifecycle/member_*.pt` / `delta_*.pt` files preserve the actual
trained models. The 80 monitoring records are exported losslessly as
`ml_lifecycle/candidate_records.jsonl`, including array shapes/dtypes and source
hashes; the monitored and unmonitored sample arrays are in
`ml_lifecycle/trajectory_samples.json`. They make the selection cap and unchanged
trajectory auditable without publishing internal monitor checkpoint files.
The five `drivers/*.py.gz` archives retain launch-source variants and the
OPES module used by the trajectory jobs, with the reconstruction/identity scope
stated in [`campaign_code_provenance.json`](campaign_code_provenance.json).

The publication list omits duplicate `opes/` files, PLUMED `bck.*` files,
rebuildable ML run/monitor checkpoints, four illustrative kinetics trace arrays,
and `reference.npz`. All first-passage lifetimes are retained as CSV. The
reference grid is retained in `reference_x.csv` and `reference_y.csv`;
`sampling.py` automatically recomputes its optional NPZ cache when absent.
`plot_sampling.py` does not read that cache: the exact comparison arrays it
needs are already stored with each retained trajectory.
