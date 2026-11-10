# Adiabatic-channel subspace validation (2026-11-10)

This validation records the implementation of issue #158: retain the path
coordinate `sigma` and a small number of transverse adiabatic states
`phi_n(Q; sigma)`.  The exact half uses the same dense Sylvester flux formula
as the existing subspace solver, now exposed through a general matrix
interface.  The zero-curvature RPMD half uses the same interpolated `V(sigma,Q)`.
Curved channel models expose the inverse metric `(1 + kappa Q)^-2`, but
`ring_polymer_rate` explicitly rejects them: a matching constrained
variable-mass sampler, thermodynamic-integration measure and flux prefactor
are not implemented. Curved-channel `C_S` is therefore not available.

The results in this directory are engineering and regression evidence.  They
are **not marked `physical_converged`**: the full curvature family and the
3D H + H2 200 K production calculation require the read-only workflow and the
stored data branch, respectively.

## Before/after acceptance table

| Check | Existing 1D/path projection | Channel implementation | Evidence/status |
|---|---|---|---|
| Curvature transfer error `Delta = ln C(H) - ln C(H_S)` | `~0.15 (kappa rho)^2` symmetric; `~0.11 (kappa rho)^2` asymmetric | Coupled channels retain `A_nm=<phi_n|d_sigma|phi_m>` and the projected `(1+kappa Q)^-2` metric; the expected trend is a decrease with `n_ch` | Unit-level matrix checks pass; full family run pending workflow |
| Separable/weak-curvature limit | 1D projection | `n_ch=1` reduces to the existing 1D sinc-DVR Hamiltonian; increasing `n_ch` retains transverse levels | `tests/test_subspace_channels.py` |
| Phase continuity | Not represented | Adjacent-sigma overlaps are phase aligned before finite differences | `tests/test_subspace_channels.py` |
| Exact rate normalization | Tensor-product `_dense_rate` only | `_dense_rate_matrix` accepts a general Hermitian matrix and the channel partition at the same reactant slice | `tests/test_subspace_correction.py` plus channel smoke check |
| H + H2 at 200 K, 1D correction | `C_S(1D) = 2.06` | Baseline only; no curved-channel correction factor is implemented | 2D reference `C_S(2D) = 2.66`, corrected rate `1.97e-18 cm^3/s`, QD `2.01e-18 cm^3/s` |
| DVR truncation | 2D reference used 11,165 retained points and reported margin about 0.096 Eh | Channel runs must report `truncation_margin` or an equivalent normal-cut margin at the reactant and dividing-surface lines | No channel result is called converged until the wall is at least six stiff transverse quanta above the valley floor |

## Reproduction

The small deterministic regression can be run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_subspace_correction.py tests/test_subspace_transfer.py tests/test_subspace_channels.py
python tools/check_workflow_policy.py
```

The transfer family is intentionally run by the read-only workflow so that the
aggregate job prints one `TABLE` row per point.  The workflow now accepts the
feature branch `codex/subspace-channels-158` as well as the earlier theory
branch; it has `contents: read` permissions and does not write repository
state.  Its matrix is `h=0, 0.3, 0.6, 0.9`, symmetric and asymmetric barriers,
and `T=600, 300, 200 K`.

For the existing 1D/2D baseline, restore the raw branch named in the H + H2
validation README and run (this command does not run the new channel API):

```bash
pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K.toml
```

The channel API must still be integrated with a validated matching curved
RPMD estimator before an `n_ch = 1, 2, 4, 6, 8` correction-factor ladder can
be run. No channel transfer factor is claimed here.

## Review fixes and supported boundary

The exact Hamiltonian now directly projects the full normal-grid kinetic
operator using overlaps between local channel states. Unlike a covariant
derivative built only inside the truncated basis, this retains the omitted
Born-Huang contribution. Keeping every normal DVR state is a unitary basis
change and reproduces the same two-dimensional DVR spectrum. Phase alignment
changes only phases, never mixes unequal-energy eigenstates.

Curvature must leave `1 + kappa Q` positive throughout the normal interval;
a zero between grid points is rejected. The path adapter uses the signed
curvature of its selected normal, so an inward Frenet normal has
`1 - |kappa| Q`. The curved matrix represents a flat-measure divergence-form
kinetic operator; it is not a validated Cartesian-coordinate transformation.
The asymptotic exact-rate normalization additionally requires a flat reactant
reference; curved matrices with non-flat reactant references can still be
inspected, but their rate calculation is rejected.

The former variable-mass RPMD branch was neither reversible nor volume
preserving and omitted mass terms from the mean-force and QTST prefactor.
It has been removed. A correct curved-channel rate ratio requires a separate
validated estimator. Zero-curvature RPMD remains available. Turning off
nonadiabatic coupling creates a different Hamiltonian, so that diagnostic
model also refuses to advertise a matching full-potential RPMD model.

## Scope and known limits

The channel Hamiltonian is dense and therefore intended for a few channels and
moderate sigma grids.  It uses a real sinc-DVR normal cut; a harmonic-basis
front end can be added without changing the block assembly.  Near a true
normal-state degeneracy, overlap projection remains invariant to basis phases;
a model with an avoided crossing still needs a channel-count convergence check.
The current validation deliberately does not claim that the 3D H + H2 value
has been reproduced, and no tolerance has been loosened to make a benchmark
pass.

## Channel-count ledger

The required production ladder is recorded explicitly here; entries remain
pending until the corresponding workflow or H + H2 data replay is run.  A
pending entry is not a zero correction and must not be interpreted as a
converged result.

| `n_ch` | `C_S` curvature family | `C_S` H + H2, 200 K | status |
|---:|---:|---:|---|
| 1 | pending | 2.06 (existing 1D reference) | available as baseline; misses cut-angle tunnelling |
| 2 | pending | pending | channel matrix available; curved RPMD unsupported |
| 4 | pending | pending | default implementation setting |
| 6 | pending | pending | production ladder point |
| 8 | pending | pending | production ladder point |

The H + H2 reference row for the established 2D plane is `C_S = 2.66`,
corrected rate `1.97e-18 cm^3/s`, against the QD reference
`2.01e-18 cm^3/s`.  A future channel correction ladder would need the missing curved RPMD
estimator before any comparison to that row is meaningful.

Review validation: `tests/test_subspace_channels.py
 tests/test_subspace_correction.py tests/test_subspace_transfer.py` completed
with **44 passed** using Python 3.11, NumPy 1.26.4 and SciPy 1.12.0 with one
BLAS thread. `tools/check_workflow_policy.py` and `git diff --check` passed.
The new tests include complete-basis spectral equivalence, truncated-basis
contributions, preservation of eigenstates during phase alignment, a metric
singularity between grid points, the signed circle metric, unsupported-rate
rejection and a working zero-curvature RPMD smoke calculation.
