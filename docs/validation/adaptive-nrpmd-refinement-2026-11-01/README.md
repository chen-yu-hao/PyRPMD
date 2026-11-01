# Longer-budget follow-up of the two initial P4 failures

This directory records the longer-budget calculations for **both** endpoint-
voltage P4 cases that failed the positive effective-sample gate in the
[initial nine-case study](../adaptive-nrpmd-study-2026-11-01/README.md):
`U=-0.6 V` and `U=+0.6 V`. The saved arrays contain **30,720 production
configurations per case, 61,440 in total**. This is a stored-configuration
count, not a count of independent observations.

Both follow-up production estimates pass the unchanged gates.
The original failed records remain unchanged in the sibling study directory.
The calculations retain `physical_converged=false` and `rate_constant=null`;
gate success does not establish equilibrium, path, bead, or real-time
convergence. The [implementation guide](../../manual/adaptive-nrpmd.md) and
[original report](../adaptive-nrpmd-study-2026-11-01/README.md) define the model,
signed SEO estimators and finite-bead reference.

## Selection, fixed parameters and changed budgets

The [runner](run_refinement.py) explicitly selects both original P4 endpoint
cases whose reports contain `target_effective_samples_below_threshold`.
The selection and the original-report hashes are recorded in [plan.json](plan.json).
There is no search over random seeds or diagnostic thresholds. The already
completed U=-0.6 follow-up was reused when the runner was expanded to include
U=+0.6; the case cache verifies its configuration, relevant source hashes and
raw-artifact hashes.

Only four configuration fields changed from the initial study:

| Budget | Initial calculation | Follow-up |
|---|---:|---:|
| Pilot samples per window/round | 96 | 512 |
| Pilot burn-in sweeps | 96 | 256 |
| Production samples per chain/window | 512 | 2,048 |
| Production burn-in sweeps | 256 | 512 |

The original voltage/bead ordering and seed **20261101** are preserved so the
random-stream assignment is unchanged. The mass is 1.5 electron masses,
beta is 3 hartree⁻¹, and the x/y umbrella stiffness remains 5 hartree/bohr².
There are five windows and three independent
source chains per window, and both save strides remain two sweeps. The maximum
pilot budget remains three rounds. Mean-sign threshold 0.1, positive-ESS
threshold 20, overlap threshold 0.01, 64 bootstrap draws, and the dynamics
configuration are unchanged.

Each follow-up saves `3 × 5 × 2048 = 30,720` configurations and performs
`3 × 5 × (512 + 2 × 2048) = 69,120` production Monte Carlo sweeps including
burn-in. The two selected cases therefore use 138,240 production sweeps.
The longer pilot can change the frozen path as well as the production
distribution under its umbrellas, so changes in diagnostics cannot be
attributed to saved sample count alone.

**The initial and follow-up stages are not independent replicates.** They
reuse the same seed and stream identities, with different pilot paths and
burn-in lengths. No source samples or uncertainty draws are pooled across
stages. Bootstrap resampling operates only on the three source chains per
window within each follow-up. Follow-up selection was prompted by the initial
failures; reported intervals are not adjusted for that selection.

## Before and after diagnostics

`ESS+` is the correlation-adjusted positive-weight diagnostic. The separately
reported signed diagnostic is `ESSsigned = ESS+ × mean_sign²`. The production
gate uses ESS+, not ESSsigned; neither diagnostic establishes independent
sample counts or finite electronic-projector variance.

| U / V and stage | Mean sign | ESS+ | ESSsigned | Production | Path stopping reason |
| --- | --- | --- | --- | --- | --- |
| [-0.6, Initial](../adaptive-nrpmd-study-2026-11-01/Um0_6_P4/report.json) | 0.435648 | 13.796 | 2.618 | Unavailable: ESS | unresolved_pilot_sign |
| [-0.6, Longer budget](Um0_6_P4/report.json) | 0.312258 | 323.786 | 31.571 | Available | adaptation_budget_exhausted |
| [+0.6, Initial](../adaptive-nrpmd-study-2026-11-01/Up0_6_P4/report.json) | 0.193379 | 15.882 | 0.594 | Unavailable: ESS | unresolved_pilot_sign |
| [+0.6, Longer budget](Up0_6_P4/report.json) | 0.296626 | 1089.606 | 95.871 | Available | adaptation_budget_exhausted |

The initial P4 pilots stopped before moving any centers. The new central
centers and path statuses are:

| U / V | Initial center | Follow-up center / bohr | Follow-up stopping reason |
| --- | --- | --- | --- |
| -0.6 | (0, 0) | (-0.000029, 0.233642) | adaptation_budget_exhausted |
| +0.6 | (0, 0) | (0.001898, 0.237386) | adaptation_budget_exhausted |

A path that stops at `adaptation_budget_exhausted` has used its available
rounds; the label is not a path-convergence claim. Endpoint centers remain
fixed at `(−1,0)` and `(1,0)`.

## Follow-up estimates against the matching finite-bead reference

The existing [reference.json](../adaptive-nrpmd-study-2026-11-01/reference.json)
is reused without modification. The analyzer verifies model-source hashes,
point identity, beta, mass and reservoir chemical potential. It uses the P4
`preferred_records`: a 49×61 grid, x in [−3,3] bohr and y in
[−3.125,3.625] bohr. This is a deterministic **finite-P** equilibrium reference
with separately recorded spacing/domain sensitivities. It supplies no exact
real-time or infinite-bead validation.

The following values are saved in [comparison.csv](comparison.csv) and
[comparison.json](comparison.json). Means x and y use bohr, second moments and
squared radius of gyration use bohr², and populations and model electron
numbers are dimensionless. Bead and centroid second moments retain their
different estimators. Initial-stage means remain unavailable because their
production gates failed; no initial estimates are reconstructed from raw data.

### U=-0.6 V, P=4

| Observable | Estimate | Bootstrap SE | Pointwise 95% interval | Finite-P reference | Reference in interval |
| --- | --- | --- | --- | --- | --- |
| Mean x | 0.201991 | 0.085177 | [0.039898, 0.358291] | 0.075084 | Yes |
| Mean y | 0.174474 | 0.067619 | [0.052073, 0.311948] | 0.216520 | Yes |
| Bead x² | 0.740025 | 0.086404 | [0.585534, 0.929256] | 0.643202 | Yes |
| Bead y² | 0.347218 | 0.051600 | [0.261844, 0.482157] | 0.327039 | Yes |
| Bead xy | -0.079924 | 0.060860 | [-0.184976, 0.036832] | -0.009034 | Yes |
| Centroid x² | 0.624281 | 0.091752 | [0.462330, 0.818944] | 0.524258 | Yes |
| Centroid y² | 0.245942 | 0.054723 | [0.162247, 0.386103] | 0.222696 | Yes |
| Centroid xy | -0.079593 | 0.058891 | [-0.181160, 0.036355] | -0.006152 | Yes |
| Population 0 | 0.431544 | 0.033984 | [0.379678, 0.500847] | 0.432965 | Yes |
| Population 1 | 0.568456 | 0.033984 | [0.499153, 0.620322] | 0.567035 | Yes |
| Electron number N | 0.221015 | 0.023348 | [0.171733, 0.263617] | 0.186306 | Yes |
| xy radius of gyration squared | 0.217019 | 0.010454 | [0.194614, 0.234401] | 0.223286 | Yes |

### U=+0.6 V, P=4

| Observable | Estimate | Bootstrap SE | Pointwise 95% interval | Finite-P reference | Reference in interval |
| --- | --- | --- | --- | --- | --- |
| Mean x | 0.083709 | 0.067216 | [-0.038177, 0.211113] | 0.054645 | Yes |
| Mean y | 0.236932 | 0.050997 | [0.132185, 0.323317] | 0.215428 | Yes |
| Bead x² | 0.605598 | 0.061446 | [0.490684, 0.711905] | 0.643552 | Yes |
| Bead y² | 0.291485 | 0.029212 | [0.240966, 0.331649] | 0.326645 | Yes |
| Bead xy | -0.036333 | 0.040416 | [-0.117675, 0.018058] | -0.006651 | Yes |
| Centroid x² | 0.486994 | 0.063801 | [0.361470, 0.594925] | 0.524561 | Yes |
| Centroid y² | 0.194862 | 0.029013 | [0.145155, 0.245360] | 0.222294 | Yes |
| Centroid xy | -0.031071 | 0.036821 | [-0.108728, 0.018907] | -0.004553 | Yes |
| Population 0 | 0.420174 | 0.025461 | [0.376036, 0.470769] | 0.435873 | Yes |
| Population 1 | 0.579826 | 0.025461 | [0.529231, 0.623964] | 0.564127 | Yes |
| Electron number N | 0.137608 | 0.018867 | [0.104613, 0.176375] | 0.127168 | Yes |
| xy radius of gyration squared | 0.215227 | 0.009135 | [0.198948, 0.230194] | 0.223341 | Yes |

All 24 available pointwise 95% intervals contain the matching
reference. This count includes correlated and algebraically related
observables, including complementary populations and the electron-number
observable linear in x and y. It is a descriptive check, not a coverage
calibration, a simultaneous confidence statement, or a declaration of
equilibrium convergence. The initial study's interval misses remain part of
the evidence and are not replaced by these follow-ups.

## Uncertainty and remaining limitations

The bootstrap resamples source chains within each window and circular blocks
within each selected source chain, refitting MBAR on every draw. Intervals
require all requested draws to resolve MBAR, overlap and signed normalization,
and at least eight complete blocks per source stream. The recorded diagnostics
are:

| U / V | Valid draws | Block-length range | Minimum complete blocks/stream | Intervals |
| --- | --- | --- | --- | --- |
| -0.6 | 64/64 | 10–21 | 97 | Available |
| +0.6 | 64/64 | 9–17 | 120 | Available |

The remaining concentration of positive weights is visible directly in the
saved source arrays:

| U / V | Largest weight | Top two weight mass | Their SEO signs |
| --- | --- | --- | --- |
| -0.6 | 1.314% | 2.368% | +1, +1 |
| +0.6 | 0.404% | 0.770% | -1, -1 |

Connected window overlap and finite weight diagnostics do not establish
coverage of all unbiased tails. Near nodal SEO factors, projector ratios can
become large; finite-run standard errors and bootstrap intervals do not prove
finite variance or Gaussian asymptotics. Three chains and 64 bootstrap draws
cannot diagnose every long correlation time or recover unsampled regions.
Intervals remain conditional on the new frozen path and chosen block lengths;
they do not include uncertainty over alternative learned paths.

The common BO charging model and analytic voltage reference are unchanged.
There is no SHE/RHE calibration or state-dependent Faradaic exchange. The
original P8/P16 reference checks continue to show appreciable bead dependence,
so the P4 follow-up does not establish nuclear-quantum convergence.

## Charge and short propagation checks

The largest chemical-potential stationarity residual over saved beads in these
two cases is 1.8735e-15 hartree. Each case uses its own 16 complete parents,
reused at dt=0.02, 0.01 and 0.005 to t=2 atomic time units. Maximum absolute
sum-bead propagation-Hamiltonian drifts, in hartree, are:

| U / V | dt=0.02 | dt=0.01 | dt=0.005 |
| --- | --- | --- | --- |
| -0.6 | 0.00798392 | 0.00199573 | 0.000498917 |
| +0.6 | 0.00655549 | 0.00163631 | 0.000408918 |

These are conditional, short-time integration checks. They establish neither
agreement with exact quantum dynamics nor a rate plateau. Saved ancestry and
initial signs remain attached to each complete parent; resampling creates no
new independent equilibrium observations.

## Raw data, hashes and reproduction

Each case directory contains its [negative-voltage report](Um0_6_P4/report.json)
or [positive-voltage report](Up0_6_P4/report.json), `source.npz`, and, when
available, `dynamics.npz`. The source archives retain full nuclear/mapping phase
space, SEO signs/projectors, positive weights, observables, frozen centers,
chain/window identities and sample indices. The dynamics archives retain
paired trajectories and source-parent ancestry. This directory contains
4 NPZ archives totaling 25,259,928 bytes.

The plan records hashes of both initial reports, the runner and all relevant
study/model source files. Case reports record their raw-artifact hashes. The
comparison records its summary/reference inputs and every case-report hash.
Important recorded-file hashes are:

| Artifact | SHA256 |
| --- | --- |
| `run_refinement.py` | `6690704e5d3656ee2294e22c0d72a2d32f8d4f4bdabc5922245455c80ba187e4` |
| `plan.json` | `676d4b19678ba1a34ad4e42eab4ca144f88abd9ce0c9cf303d8b716370705043` |
| `summary.json` | `84b2191767f596e550c156656162520df0e8768e97b87cc619a8795ac9fbcacb` |
| Shared original `reference.json` | `b7aa70b3346fcc7ab02a4f3cec6278c9e9357dee2df60ed5a39763127461bc0f` |

From the repository root, after installing the checkout and creating the
initial study evidence:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python docs/validation/adaptive-nrpmd-refinement-2026-11-01/run_refinement.py
python examples/electrochemistry/analyze_adaptive_nrpmd_study.py \
  docs/validation/adaptive-nrpmd-refinement-2026-11-01 \
  --reference docs/validation/adaptive-nrpmd-study-2026-11-01/reference.json
```

The runner writes into its own directory and reuses completed cases only after
the existing configuration/source/artifact checks pass. It does not resume an
interrupted case midway through sampling. The analysis command only reads the
initial reference; it writes comparison files into this follow-up directory
and reruns neither sampling nor dynamics. The original study's raw sources,
comparisons and reference remain unchanged.
