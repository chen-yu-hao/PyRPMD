# Shared RPMDrate reaction coordinate for an explicit quantum correction

**Research opt-in; native default rate workflow is unchanged.**

Target base: `chen-yu-hao/PyRPMD`, main
`ffa14f247e453e9626b2e3fb3c288da0ca753d2e`.
This patch is independent of unmerged PRs #137/#138. It adds files rather than
replacing those branches or the native PMF, RATTLE, parent/child, or rate code.

## Coordinate definition, corrected explicitly

RPMDrate's coordinate is **not** the normalized difference of two bond lengths.
Reuse `pyrpmd.cv.RPMDrateCoordinate`, constructed by the native rate factory:

```
s0 = rinf - reactant_COM_distance
s1 = min_bond(max_equivalent_TS(reference_forming - reference_breaking
                              - current_forming + current_breaking))
xi = s0/(s0-s1)
S(xi_star) = (1-xi_star)*s0 + xi_star*s1
```

Lengths are bohr, masses electron masses, and all atom indices are zero based.
The RPMD estimator is `xi(mean(bead_positions))`, **not** `mean(xi(beads))`.
For quantum physical geometries the same scalar and side operator `S>0` are
used; the quantum wavefunction does not acquire fictitious ring-polymer masses.
The adapter uses native values and derivatives. Its independent min/max pass
only detects nonsmooth branch seams. Quantum charts reject seams, singular
coordinate maps, and nonpositive `s0-s1`; no smoothing or clipping is introduced.

Native TS preparation may change the geometry and reaction definition. The
wrapper reads its `preparation.json` and uses native `prepared_system()` before
binding the quantum coordinate. Reusing the initial guessed TS would be wrong.

## What is implemented

1. `analysis.rpmdrate_shared`: mathematical-definition/surface hashes, centroid
   gradient and flux adapters, local full coordinate chart whose first axis is
   the exact native xi, and explicit 1--3D constrained-slice metrics.
2. `analysis.rpmdrate_quantum`: 1--3D mass-lumped finite-element active-slice
   Hamiltonians from a physical PES, exact finite-basis Kubo flux-side trace,
   shared side projector, executable finite-basis backend, matched active-flux
   containers, and a conditional correction with diagnostic plateau checks.
3. `workflows.rate_quantum`: call the ordinary native rate workflow, keep its
   PMF/QTST and output untouched, bind the same prepared reaction/surface, call
   an explicit active backend, and write a separately labelled candidate result.
4. Tests: coordinate/normalization consistency, nonlinear kinetic geometry,
   finite-basis propagation, algebraic limits, mismatches and refusal paths.

A supplied active Hamiltonian can include an independently justified bath
closure. The built-in grid builder is only a **fixed-environment constrained
manifold** model. It does not construct a general bath influence functional.
It includes the induced metric, active kinetic cross terms and induced measure;
it does not silently use the projected Cartesian inverse-mass block as the
inverse of a frozen-bath metric. No extrinsic confinement potential is assumed.

## The correction and its limits

The experimental candidate is

```
k_candidate = k_QTST(native) * [kappa_full_RPMD
                               + kappa_active_Q - kappa_active_RPMD].
```

Both active kappas must be formed from **matched** equilibrium-normalized
flux-side numerators divided by the SAME active positive reference flux.
They are ratios of sums, not means of individual ratios. The full QTST factor
is never multiplied by an additional tunneling or partition-function correction.
This avoids an automatic extra static correction; it is **not** a proof that
the selected dynamical defect transfers to the coupled full system.

`kubo_flux_side()` returns `Cfs/Z` and `Cfs/Qr`, not a dimensionless RPMDrate
kappa and not a bimolecular rate. `FiniteBasisFluxBackend` demands an explicit
`quantum_to_common_measure` conversion and independent matched RPMD flux blocks.
Sharing a scalar xi alone cannot determine that conversion or the environment
feedback. A scattering transmission probability is not an acceptable substitute.

Quantum finite-basis recurrences, discretization error, native PMF uncertainty,
native recrossing uncertainty and active/full covariance require separate audits.
The new reported SE is **active-block sampling conditional on the native
baseline**; it is not the total rate uncertainty. Native original uncertainty
remains in the unmodified native summary.

`rate_estimate_cm3_s` preserves the finite-time candidate, including a negative
candidate. `rate_cm3_s` remains `None`, `physical_converged=False`, and no general
exact-rate claim is made, even when numerical prerequisites pass. Evidence
records are explicit caller attestations, not automatic physical proofs.

## Public use

```python
from pyrpmd.input import JobSpec
from pyrpmd.analysis.rpmdrate_quantum import FiniteBasisFluxBackend
from pyrpmd.workflows.rate_quantum import run_shared_quantum_rate
from my_active_solver import build_matched_problem

config, system = JobSpec.from_file("rate.toml").resolve()
backend = FiniteBasisFluxBackend(build_matched_problem)
result = run_shared_quantum_rate(
    config, system, "runs/shared", backend,
    pes_identity="your-pinned-PES-version-and-parameters",
    reactant_normalization="explicit matching active/full reactant convention",
    kinetic_mapping="native-xi-plus-auxiliary-coordinates; declared metric/bath",
    plateau_start=100,  # MUST reproduce the native reported plateau tail.
)
```

`build_matched_problem(**kwargs)` receives `context`, `coordinate`, `time_au`,
`native_directory`, `native_summary`, prepared `config` and prepared `system`.
It must return exactly the fields documented by `FiniteBasisFluxBackend`.
The active quantum solver is executed, not a fake/scaled spectrum. The active
RPMD reference and a physical environment model must still be supplied.

For a deliberately fixed-coordinate model:

```python
from pyrpmd.analysis.rpmdrate_shared import LocalXiChart
from pyrpmd.analysis.rpmdrate_quantum import build_active_slice_hamiltonian

chart = LocalXiChart(coordinate, reference_geometry, radius=trust_radius)
model = build_active_slice_hamiltonian(
    chart, [xi_nodes, auxiliary_nodes],
    lambda geometries: physical_pes.evaluate(geometries).energy,
    environment_coordinates=omitted_chart_coordinates,
)
# model["hamiltonian"] and model["grid_geometries"] are suitable inputs to
# the finite-basis backend. This fixed-coordinate model alone is NOT a
# generally valid dynamically coupled quantum/RPMD correction.
```

The CLI helper is `run_shared_rate.py`; `--help` lists native input and trusted
Python provider options. No invented `[quantum_correction]` TOML section is
silently accepted by the native schema. Periodic, fixed-atom, surface-rate and
TRPMD inputs are rejected by this gas-phase RPMD bridge.

## Reproduce checks

```bash
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_rpmdrate_shared.py
OPENBLAS_NUM_THREADS=1 python benchmarks/rpmdrate_shared/barrier_validation.py
```

The numerical barrier uses a **physical** `sech(x)^2` PES, not a PMF. It changes
coordinates to RPMDrate xi without changing the 1D physical Hamiltonian. It
compares a finite-box Kubo flux kernel with an independent analytic Eckart
scattering integral and separately runs 1024 P=1 flux-launch trajectories.
It is a kernel check, **not** a molecular/environment correction benchmark.

In the limited local patch-only environment the full repository could not be
cloned or imported. The explicitly recorded fallback was:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_rpmdrate_shared.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python benchmarks/rpmdrate_shared/barrier_validation.py --formula-oracle
```

`coordinate_oracle.py` is an independent formula oracle for this fallback and
unit tests. Production modules never import it. The real native-coordinate
parity test skips when `pyrpmd.cv` is absent. Native `run_rate` integration was
mock-tested; a complete real PMF/recrossing job was **not** executed here.

## Measured local results

- 45 passed, 1 skipped (native package parity), not the full repository suite.
- Same-scalar evaluation deviation: <= 8.89e-16 in the barrier coordinate test.
- Primary time window [2,4]: finest-grid kernel differs from analytic scattering
  by -2.588%; increasing 321 to 481 nodes changes that estimate by ~0.036%.
- Separately reported, not used to redefine the primary window: [4,6] differs by
  -0.271%; its peak-to-peak variation is still ~3.05% of the analytic reference.
- No new chemical reaction rate, bath-closure accuracy, or physically converged
  hybrid kappa is claimed.

All numeric outputs and the finite-time sensitivity are saved in the small
validation JSON. The full downloadable archive also contains the raw curves.

## Verified upstream sources

- PyRPMD base above: `cv.py`, `workflows/rate.py`, `rate_constraints.py`,
  `rate_pmf.py`, `prepare.py`, `rate_analysis.py`, `rate_artifacts.py` and
  `analysis/recrossing.py`.
- GreenGroup/RPMDrate `rpmdrate/_surface.f90`, Git blob
  `cb8c5c60059f211eb0b4e2882c5c579dfee8ca5d` (MIT), checked for the
  paired-bond/equivalent-path min/max and sign conventions. No Fortran code or
  external runtime was copied or executed. Native PyRPMD already carries
  attribution in `cv.py` and its third-party notices.
