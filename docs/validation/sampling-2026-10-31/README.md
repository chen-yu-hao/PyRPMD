# Sampling implementation and validation

This change integrates the existing REUS implementation from
`afb04de875c40eec56b204df9b26a93dbd079c4e` (PR #21) with main
`76dd6ad24adc6c244f775857987cd2f0e119845f`, resolves its bilingual-documentation
conflicts, and adds OPES, an optional PLUMED bridge and transition-interface
path sampling. The methods reuse existing dynamics, CVs, potential adapters,
task scheduling and checkpoint facilities.

## Validation record

| Check | Local result |
| --- | --- |
| Full integration suite before the final targeted guard additions | 1,738 passed, 14 optional skips, 153 subtests passed; two existing i-PI tests failed because the sandbox prohibited socket creation |
| Retry of exactly those two i-PI tests with local loopback sockets enabled | 2 passed |
| Final combined sampling tests, including all review fixes | 243 passed, 1 skip for the unavailable local PLUMED runtime |
| English and Chinese manuals, HTML and singlehtml | All four strict Sphinx builds passed |
| Generated language-switcher targets | 86 checked, no missing targets |
| Native extension and source distribution | Built successfully; new sampler modules and examples are packaged |

These test counts overlap and must not be added as independent tests. The
full run's remaining optional skips require BKMP2, a configured Ni-Al-H EAM
potential, PyTorch or PLUMED. A local MPI runtime was unavailable.
`local-evidence.json` records environment versions and log hashes.

Reproduce the final sampling checks after installing `.[test]`:

```bash
python -m pytest -q \
  tests/test_replica_exchange.py tests/test_exchange_bootstrap.py \
  tests/test_fes_pile_l.py tests/test_fes_block_sampling.py \
  tests/test_opes.py tests/test_plumed.py \
  tests/test_transition_interface.py tests/test_sampling_public_api.py
```

GitHub Actions runs the complete CPU suite and actual four-rank MPI checks.
A dedicated workflow builds the official PLUMED 2.9.4 kernel and matching
Python wrapper at `52da2bb76d37dbad0d19f59c6fe0b2ab939e3ded`. It sets
`PYRPMD_REQUIRE_PLUMED=1`, so an absent runtime fails the job. It verifies live
restraint forces, update/deposition counts and the executable METAD example.
This is a configured remote gate; its success must be read from the PR's
Checks tab, separately from the local results above.

## Scientific and numerical checks

**REUS.** Exchange probabilities are compared with full ring Hamiltonian
ratios for different bead counts and with a finite-state detailed-balance
reference. Tests cover periodic CVs, whole-ensemble failure rollback, exact
resume, synchronized cross-window bootstrap and serial/process parity.
The integrated implementation also fixes checkpoint identities that otherwise
depended on copied Python CV objects' memory addresses.

**OPES.** Independent formulas check weighted KDE and kernel-center
normalization. Cartesian finite differences check full-ring forces for
centroid and bead-average CVs. Periodic kernels integrate to one and have
smooth boundary derivatives. Frozen histogram weights recover analytical
harmonic bin masses from deterministic inverse-CDF samples, including
independent +1000/-1000-hartree bias-gauge shifts. Tests also cover event
rollback, exact restart, external executor iterators, process parity, strict
checkpoint clocks and numerical tail stability.

**PLUMED.** Protocol tests check unit declarations, P=1/4/16 force scaling,
static-bias trajectory equivalence, Andersen/PILE-L/deterministic stepping,
read-only diagnostics, update clocks, ownership, stopping, failure poisoning
and synchronized cleanup. Local protocol tests are not a substitute for the
dedicated real-kernel gate.

**TIS.** Canonical two-way shooting selects uniformly over all frames,
including endpoints, with the old/new frame-count factor and canonical
proposal correction. Direct two-frame crossings remain represented.
Tests check time reversal, seed replay, rejected-path multiplicity, path
bounds, effective-flux recrossing counts and the analytical first crossing
probability of a confined classical double well. Rate combination requires
ordered interfaces, matching thermodynamic metadata and a common path-length
support. Imported malformed summaries are rejected.

## Examples and limits

- `examples/fes/replica_exchange_2d.inp` runs nine windows and two independent
  ensembles, with completed resume and saved reanalysis.
- `examples/opes/harmonic_2d.inp` runs P=4 and two independent OPES histories,
  retaining 200 frozen production samples and 20 kernels per history.
- `examples/plumed/centroid_metadynamics.py` runs the optional live-kernel
  centroid METAD bridge.
- `examples/sampling/transition_interface.py` runs four double-well interface
  ensembles. Its deliberately short default calculation is unconverged.

The native OPES method has fixed bandwidth and no kernel compression; its
force cost and memory grow with kernel count. Its smooth wrapped Gaussian
representation is not numerically identical to PLUMED's truncated kernels.
Frozen-bias relaxation still requires a system-specific convergence study.

The PLUMED bridge supports coordinate-dependent centroid biases and fixed
fully periodic or nonperiodic cells. Mixed-axis PBC, NPT coupling and
ENERGY-dependent actions are unsupported. Native PLUMED restart files remain
external to the generic PyRPMD checkpoint format.

TIS still requires basin definitions and an ordered scalar interface variable.
It does not require that variable to approximate the committor. The current
API uses unbiased deterministic two-way shooting, not stochastic path moves
or RETIS. Path-length truncation must be converged. Absolute first-order
rates require a separate compatible effective-flux trace; candidate rates
retain `rate_s_inv=null` and `converged=false`.

These checks establish implementation behavior on model systems. They do not
establish a converged molecular free-energy surface, physical rate, quantum
accuracy improvement or a production speedup.
