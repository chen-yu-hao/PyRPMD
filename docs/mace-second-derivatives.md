---
orphan: true
---

# MACE analytic Hessians and matrix-free Hessian–vector products

Two physical Cartesian derivatives are available on the named MACE and
MACE-batched adapters:

- `potential.hessian(x)`: float64[B,3N,3N], energy Hessian in hartree/bohr².
- `potential.hessian_vector_product(x,v)`: float64[B,N,3], the Hessian
  acting on the supplied float64 direction `v` (same shape as `x`).

Coordinates and directions are in bohr; energies are in hartree.
The derivative excludes ring springs, mass weighting, bias potentials
and bead scaling. Existing `evaluate(x)` and scientific fingerprints
are unchanged.

## How the derivatives are computed

The `mace` ASE adapter calls upstream `MACECalculator.get_hessian`
for a full Hessian, and uses autograd of the model force graph for HVP,
without first constructing the Hessian. The `mace-batched` provider
uses its existing disconnected graph batch and one model forward:
HVP takes one force VJP; the dense Hessian takes 3N force VJPs, each
returning one Cartesian row for every independent frame.

Both require a supported conservative MACE energy model and float64.
The ASE path additionally requires one model, an uncompilated/unpadded
calculator and the upstream private MACE graph contract for HVP.
Unsupported combinations, nonfinite results, unsupported precision,
missing gradient connectivity and incorrect tensor shapes fail
explicitly. There is no silent numerical finite-difference fallback
inside these new interfaces. They preserve the explicitly selected
device and perform the public output transfer to NumPy.

For standard MACE energy units, the Hessian scale to PyRPMD is
EV_HARTREE * BOHR_ANGSTROM**2; nondefault upstream explicit energy
and length unit scales are included in the ASE adapter.

## Harmonic TST and instanton usage

HarmonicState.from_potential and the instanton physical Hessian
collector already prefer `hessian(x)` when present, with strict
shape/symmetry/precision checks. Therefore these callers can consume
MACE second derivatives without numerical force differencing.

`RingPolymerAction.hessian_vector_product(path,direction)` exposes
a matrix-free action-Hessian product suitable for iterative solvers.
It returns float64[P,N,3], matching the path and direction shapes; when
an active-coordinate mask is present, inactive components are exactly zero.
For dtau=beta/P and each active component at bead j it implements

```text
(H_action v)[j] = dtau * (H_physical(q[j]) @ v[j])
                + m/dtau * (2*v[j] - v[j-1] - v[j+1])
```

The direction must be float64[P,N,3] and vanish in inactive components.
Missing or invalid physical PES HVP raises an error. This addition does
not change the existing dense instanton optimizer or fluctuation
prefactor: large-system optimization still needs a separate iterative
solver and determinant treatment.

## Boundaries and independent validation

These Hessians differentiate a particular trained energy model at a
**fixed neighbor topology**. At neighbor cutoff changes the graph may
change discontinuously; finite differences straddling these events
can disagree with autograd. Numerical correctness does not validate
the model's reaction-region energy curvature, TS imaginary frequency,
or the physical rate.

The optional tests use actual locally created MACE modules, compare
independent ASE and batched Hessians, compare HVP against dense H @ v,
and compare both to central finite differences of forces. A separate
quadratic action test supplies an analytic reference independent of
any optional ML dependency. Missing optional dependencies cause a
pytest skip, not a successful MACE validation.

```bash
python -m pip install -e '.[mace-batched,test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_mace_second_derivatives.py tests/test_instanton_hvp.py \
  tests/test_batched_ml_mace.py tests/test_potential_adapters.py
```

Upstream API: MACE 0.3.17 MACECalculator.get_hessian and energy-model
compute_hessian path. Harmonic TST and ring-polymer instanton method
sources are already indexed in the project's method reference catalog;
this adapter introduces no new primary method citation.
