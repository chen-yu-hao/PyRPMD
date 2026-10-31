# Dynamics efficiency and validation record — 2026-10-31

This record distinguishes software equivalence, model performance, and chemical
rate validation. The software/model checks pass. **The short H + H₂ calculations
do not establish agreement with the published rate.** None of the requested
reaction set is marked physically converged by this record.

## Implemented scope

| Priority | Delivered | Remaining scope |
| --- | --- | --- |
| Potential reuse | One actual provider/process pool per compatible worker task batch; exception eviction, explicit stateful opt-out, orderly cleanup | Reuse does not extend across separate `run_tasks` calls |
| Trajectory batching | Vectorized deterministic integrator, masks, independent rings, gas/fixed-surface recrossing pair dispatch | Full dRPMD, thermostatted umbrella and periodic-site dispatch; real ML/GPU measurements |
| RPC/MTS | Explicit contraction, physical difference potential and symmetric MTS Python APIs | Production workflow/checkpoint integration; convergence for actual expensive PESs |
| TRPMD/spectroscopy | Internal-mode Langevin dynamics, centroid/dipole/flux correlations, power/relative IR spectra, rate sensitivity option | PA-CMD, absolute IR absorption, batched TRPMD, molecular spectroscopy validation |
| Published rates | Source registry, convention-aware comparison gate, real BKMP2 pilot and diagnostic records | Converged H3 and isotope series; matched PES runs for the rest of the requested chemistry |

See [the feature guide](../../performance-dynamics.md) for inputs, capability
restrictions and the public APIs. Existing scalar RPMD remains the default;
resource reuse is the only default execution change.

## Software verification

- Full native-enabled regression run: **1,821 passed, 13 skipped, 2 deselected,
  153 subtests passed**, in 146.25 s. The real BKMP2 library was present.
  [Raw pytest result](pytest-full.txt).
- The two socket protocol tests excluded from that run passed separately in
  an environment permitting local socket binding.
- The final targeted input, batched/TRPMD recrossing and reference-comparison
  run passed **107 tests**, covering the added method-identity and early input
  guards. [Output](pytest-final-targeted.txt). This overlaps the main suite;
  it is not an additional 107 unique tests.
- Strict Sphinx build (`-W --keep-going`) passed; [build output](sphinx-strict.txt).
- Source distribution creation passed. The new modules and benchmark registry
  are included. Whitespace validation (`git diff --check`) passed.
- Local MPI was unavailable. The existing GitHub Actions CPU matrix runs the
  extended MPI integration and workflow tests; consult the PR checks for those
  results. Local skips cover optional ASE/PyTorch and independent-reference
  packages, not successful tests of those unavailable dependencies.

The machine-readable [verification record](verification.json) includes commands
and a source-file digest inventory. Benchmark files retain their own original
run-time source hashes; later documentation/capability guards do not rewrite
historical benchmark provenance.

## Measured performance and numerical behavior

All timings below are local CPU measurements, with the workload and raw repeats
recorded in the linked JSON. They are not predictions for ML or ab initio models.

| Experiment | Baseline | Changed execution | Interpretation |
| --- | ---: | ---: | --- |
| Real BKMP2, scalar provider lifecycle | 0.1313 s, reload each pair | 0.1112 s, reuse | 1.18× measured speedup |
| Real BKMP2, two bead workers | 10.3552 s, reload pool | 6.1872 s, reuse pool | 1.67× measured speedup; cheap PES still favors serial native execution |
| Harmonic 16-trajectory model | 0.2614 s, scalar | 0.1314 s, batch | 1.99× measured speedup, identical final coordinates/momenta |
| RPC/MTS slow geometries | 25,632, full small step | 603, 32→3 beads and four inner steps | 42.5× fewer slow geometries, with an approximation error |

The [real recrossing benchmark](recrossing-performance.json) uses 8 child pairs,
16 beads, 128 steps, and three repeats per variant, including setup/checkpoint
costs. All 18 runs have identical flux numerators, denominators and endpoint
energy errors. Batching four/eight pairs takes 0.4183/0.3601 s, **slower** than the
0.1112 s scalar fused native loop on this cheap Fortran PES. That is why batching
is opt-in, and why no universal batching speedup is claimed.

The [harmonic batching record](../batched-dynamics-2026-10-31.json) reduces calls
from 4,096 to 129. This count includes reuse of endpoint forces, so it should
not be interpreted as a 31.75× wall-time result from batching alone.

The [RPC/MTS model](rpc-mts.json) reports both workload and approximation error:
MTS alone gives coordinate RMS error 4.62e-8 against full small-step dynamics;
RPC plus MTS gives 2.54e-4 in the model's atomic units. These are short model
trajectories, not accuracy evidence for chemical rates.

The [TRPMD harmonic spectrum](trpmd-spectrum.json) checks friction scales
0, 0.5 and 1. All resolve the analytic 0.7 angular-frequency mode within one
finite-record bin. Centroid velocities agree with RPMD to 4.4e-13; the finite
ensemble's zero-lag covariance and its uncertainty are reported explicitly.
This is not a molecular IR validation.

## Published H + H₂ comparison: validation withheld

The reference is Table 2 of [Suleimanov, Allen and Green, CPC 184 (2013)](https://doi.org/10.1016/j.cpc.2012.10.017):
BKMP2, 300 K, 128 beads, with both equivalent product channels included.
The published QTST factor is 2.51e-16 cm³ molecule⁻¹ s⁻¹, transmission
coefficient 0.776, and rate 1.95e-16 cm³ molecule⁻¹ s⁻¹.

Both calculations below use only 50 fs equilibration plus 100 fs production
per umbrella/replica, two replicas and short recrossing sampling. The original
publication uses much longer equilibrium sampling. The pilot integrator,
thermostat and initialization protocol also differ; physical parameters alone
do not establish numerical convergence.

| Quantity | Exploratory weak restraint | Source-matched restraint |
| --- | ---: | ---: |
| Input umbrella coefficient / hartree | 0.09500435 (100 kBT) | 30 |
| PMF barrier / hartree | 0.01072582455 | 0.02601403039 |
| Candidate rate / cm³ molecule⁻¹ s⁻¹ | 1.75330e-13 | 1.92758e-20 |
| Candidate / published rate | 899.13 | 9.88499e-5 |
| Transmission coefficient | 0.57619 | 0.61722 |
| Median effective samples per stream | 4.21 | 9.66 |
| Observed histogram support at dividing surface | No | Yes |
| Physical-validation `passed` | `null` | `null` |

The weak-restraint Gaussian UI curve extrapolates through an unsampled barrier
region. Formal MBAR replay rejects that dividing surface. The source-matched
run has support, but independent replica barriers differ by about 8 kBT and
most streams have fewer than 20 effective samples. On the same stiff-restraint
samples, MBAR changes the barrier by only 7.84e-5 hartree, far too little to
explain the discrepancy. PMF sampling/normalization instability dominates the
observed rate variation; these runs do not prove that no other issue remains.

An [audit of the original RPMDrate implementation](https://github.com/chen-yu-hao/PyRPMD/blob/feat/tier1-dynamics-efficiency-20261031/benchmarks/published_rates/umbrella-normalization.md)
confirms that `kappa="30 hartree"` is correct and must not be divided by 128.
Changing this coefficient to improve numerical agreement would not be a
valid correction. See [replica/time-half diagnostics](pmf-sampling-diagnostic.json)
and [same-sample estimator/support diagnostics](pmf-estimator-diagnostic.json).

The archived [weak-restraint](broad-pilot/manifest.json) and
[source-matched](stiff-pilot/manifest.json) summaries include settings, comparison
reports, PMF/recrossing curves and digests. Their `driver.py` files are exact
historical source snapshots for provenance, not runnable entry points from
their archive directories. The [offline replay script](pmf-replay.md) accepts
compact CV-sample archives and can reproduce the estimator/support analysis
without the PES or dynamics. Those binary sample archives and the full
coordinates/momenta/checkpoint files are not included in this commit.
The guide records the locally generated archive hashes and how to create
archives from fresh runs; diagnostic input hashes identify the original run.

## Next physical validation work

1. Extend source-matched umbrella equilibration/production and compare separate
   replica and time-half PMFs before increasing recrossing sampling. The current
   `h3.py --profile extended` proposes 1 ps/2 ps per window/replica; it has not
   been run here and is not a convergence guarantee.
2. Establish bead-count, timestep, reactant-separation and statistical convergence,
   including a half-timestep child check. The short pilots' worst endpoint energy
   deviations also exceed their configured 0.1 kBT-per-bead diagnostic threshold.
3. Compare the complete PMF-plus-recrossing uncertainty to the matched published
   value. Do not use a favorable time half, a bare exit code, or a single ratio
   as a physical pass.
4. Obtain/verify the matched PES and conventions for the remaining reactions.
   The registry contains 31 sourced numerical rows and 5 explicit unavailable
   entries. Unresolved nuclear-spin conventions block automatic equality checks
   against the currently listed H3 exact-quantum values.
