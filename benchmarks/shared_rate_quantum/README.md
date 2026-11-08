# Shared RPMDrate coordinate and quantum-flux bridge

**Research/opt-in workflow, stacked on PR #138. No default rate algorithm is changed.**
This connects the native PMF, constrained-parent and paired-child workflow to an
explicitly supplied quantum correction provider. It also implements an actual
finite quantum flux-side kernel on the same native dividing surface. It does
**not** supply a general quantum/environment propagator or establish that the
correction improves arbitrary molecular rates.

## One authoritative coordinate, not a normalized bond difference

The existing `pyrpmd.cv.RPMDrateCoordinate` is reused directly:

    xi = s0 / (s0 - s1)
    s0 = Rinf - distance(fragment centers of mass)
    s1 = min_bond(max_equivalent_TS(reference_difference - r_forming + r_breaking))
    S(x; xi_star) = (1 - xi_star) s0(x) + xi_star s1(x)

The product side is **S > 0**. It agrees with `xi > xi_star` only on a positive
`s0-s1` patch. The quantum coordinate charts reject poles and detected branch
seams rather than silently smoothing the native definition. Native equivalent
paths, isotope masses, Rinf and the selected surface are preserved.

The RPMD coordinate is `xi(mean(Cartesian beads))`, not `mean(xi(beads))`.
The quantum operator uses the same *physical-coordinate* surface. This shared
geometric definition is necessary for comparison but does not identify a
quantum position operator with a centroid estimator.

## Implemented components

- `analysis.shared_rate_coordinate`: immutable-by-identity native coordinate
  snapshot, native surface flux metric, auxiliary CV metric diagnostics, and
  one- to three-dimensional affine physical core charts reparameterized by xi.
- `analysis.shared_quantum_flux`: finite-element weak side operator with the
  correct volume measure; finite thermal Kubo flux-side kernel; explicit parent
  trace mapping; matched-flux correction with native chain/block statistics.
- `workflows.shared_rate_quantum`: opt-in orchestration around unchanged
  `run_rate`; export the **actual** PMF-selected xi_star and parent/child data;
  authenticate native child fingerprints and numerator/denominator sums.

The full-chart contravariant CV block is not silently treated as an effective
reduced kinetic operator. The implemented `XiSlice`/`XiAffineChart` Hamiltonians
explicitly hold omitted Cartesian directions fixed. Their induced metric and
all mixed kinetic terms describe that restricted model only. A caller must
justify how that model couples to the remaining RPMD environment.

## Python API

The native workflow still computes `k_QTST` and the full-system RPMD fluxes.
A provider receives constrained parent configurations and the original grouped
positive-flux normalization:

```python
from pyrpmd.workflows.shared_rate_quantum import run_shared_rate

report = run_shared_rate(
    config, system, output_directory,
    quantum_provider=matched_conditional_provider,
    bootstrap=256,
)
```

The provider must return `active_rp`, `active_quantum` (`MatchedFluxBlocks`) and
`evidence`. Both use the same shared coordinate identity, ordered parent IDs,
time grid and reference denominator. Missing/mismatched data are rejected.

An actual restricted quantum kernel can be built as follows:

```python
from pyrpmd.analysis.shared_rate_coordinate import XiSlice
from pyrpmd.analysis.shared_quantum_flux import solve_shared_core

# native_reference is supplied to the provider by the workflow.
shared = native_reference.coordinate
path = XiSlice(shared, anchor, explicit_cartesian_direction, bracket)
core = solve_shared_core(
    path, [xi_nodes], physical_pes,
    potential_identity=shared.pes_identity, states=32,
)
trace = core.flux_side(native_reference.full_flux.time_au)
```

`xi_nodes` must put `xi_star` at an interior finite-element node. For two or
three quantum coordinates, use `XiAffineChart` with explicit independent
Cartesian directions. Its first coordinate is the same native xi; the remaining
coordinates are declared affine coefficients. This is not an automatic global
reaction-coordinate atlas or a bath-elimination prescription.

`trace['flux_side_per_partition']` is `C_fs^K(t)/Z` of the finite core. It is
**not** the native recrossing numerator or kappa. The provider must derive and
supply its per-parent `quantum_traces_to_blocks(..., scales=...)` conversion,
including conditional initial weights and the common positive-flux measure.
No automatic t=0 division is used: the quantum Kubo C_fs(0) is zero.

The experimental assembly is

    kappa_mix(t) = sum_parent(N_full + N_Q,active - N_RP,active) / sum_parent(D_full)
    k_candidate = k_QTST,native * plateau(kappa_mix)

This is an explicit **transfer approximation**, not an exact general identity.
A quantum transmission probability cannot replace kappa. A PMF cannot replace
the physical PES inside the quantum propagator. No spectra, kappas or negative
rate estimates are clipped, fitted, or renormalized to conceal a failure.

## Evidence and uncertainty

Provider evidence has named `passed`/`record` entries for the physical core
Hamiltonian, kinetic metric, conditional initial measure, common positive flux,
environment coupling and quantum numerical convergence. These labels record
external evidence; they are not proofs supplied by this library.

The native plateau diagnostic is reused. Parent/chain correlations are retained;
multiple paired children from one parent are not counted as independent quantum
samples. Reported uncertainty is **conditional on fixed QTST**, and excludes
PMF uncertainty, transfer-model error and unresolved quantum sampling. A provider
must not hide extra quantum Monte Carlo noise in deterministic parent traces.

All output retains `physical_converged=False`; `rate_cm3_s=None`. The finite-time
candidate is stored separately as `rate_estimate_cm3_s`, with unmet evidence and
plateau gates. Even passing all declared gates produces a research estimate,
not a universal exact-rate certification.

Native export verifies completed child caches against the prepared config,
optimized system, PES, beta, surface, starting configuration and native stream
fingerprints; their sums must equal `recrossing.npz`. The digest includes parent
and child checkpoint bytes. Changed references require a new correction folder.
Rate extensions must be exported from a committed generation explicitly. This
wrapper is serial at the outer level; a native task executor can still parallelize
sampling. No new CLI task, MPI-collective quantum layer or Fortran RPMDrate engine
is claimed.

## Reproduce the actual checks

Run from a checkout containing this PR (Python >=3.11, NumPy/SciPy/pytest):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_shared_rate_quantum.py tests/test_rate_workflows.py \
  tests/test_rate_module_compatibility.py tests/test_general_coordinate_rates.py \
  tests/test_rate_pmf_composition.py tests/test_rpmdrate_import.py \
  tests/test_curvilinear_propagation.py

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python benchmarks/shared_rate_quantum/run.py \
  --input benchmarks/shared_rate_quantum/inputs/native_smoke.json \
  --output runs/shared-native-quantum
```

The native smoke input has artificial two-electron-mass particles and a free
PES. It is an execution/normalization regression, **not a chemical reaction**.
It executes PMF sampling, constrained parents and paired children, then a NOOP
provider; unchanged results are a transport check, not a quantum improvement.

The independent quantum control is a fixed-transverse Cartesian double well
`40*(x_H^2-0.49)^2`, transformed to the actual native xi. Its finite-box,
finite-eigenbasis quantum Cfs is compared with a separately discretized Cartesian
finite-difference reference, not with fitted or rescaled data. The shared weak
side operator is Hermitian in the same FE measure used for the Hamiltonian.

Measured records are in `docs/validation/shared-rate-quantum-20261108/summary.json`.
Raw native outputs and flux curves are in the accompanying downloadable archive,
not checked into the repository. No new molecular quantum rate is reported.

## Method references

The coordinate and rate implementation are the existing `cv.py`,
`workflows/rate.py`, `rate_constraints.py`, `rate_pmf.py`, `rate_recrossing.py`,
and `analysis/recrossing.py`, not a duplicate implementation of RPMDrate.
Original RPMDrate methodology: Suleimanov, Allen and Green, Computer Physics
Communications 184, 833 (2013), DOI 10.1016/j.cpc.2012.10.017. The restricted
quantum kernel is an additional research extension, not part of the original
method's validation.
