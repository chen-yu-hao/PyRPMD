# Nonadiabatic extension validation

This record covers the nonadiabatic/electronic-friction extension based on
`76dd6ad24adc6c244f775857987cd2f0e119845f`. The models are synthetic. Numerical
identities, numerical propagation and workflow behavior are tested; no
real-system chemical rate or surface diffusion coefficient is certified.

## Local results

The complete suite passed: **1,759 tests and 153 subtests**, with **23 optional
tests skipped** and **zero failures** in 176.90 seconds. Both existing local
socket protocol tests passed when local socket access was enabled. See
[pytest.txt](pytest.txt) for the exact summary and skip reasons, and
[checks.json](checks.json) for the local validation checklist.

The English/Chinese HTML and single-page manuals built with warnings treated
as errors. Wheel/sdist build, isolated installed imports, and all four new
examples completed successfully. Remote CI results are recorded on the PR;
this local record does not assert a remote CI result.

## Reproduction

Local runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pytest 9.2.0, and the
compiled PyRPMD nuclear extension with OpenMP. The following commands are run
from the repository root after installing `.[test,docs]` and `build`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m sphinx -q -W --keep-going -b html docs /tmp/pyrpmd-docs/html
python -m sphinx -q -W --keep-going -b html docs/zh /tmp/pyrpmd-docs/html/zh
python -m sphinx -q -W --keep-going -b singlehtml docs /tmp/pyrpmd-docs/singlehtml
python -m sphinx -q -W --keep-going -b singlehtml docs/zh /tmp/pyrpmd-docs/singlehtml/zh
python -m build --no-isolation
python examples/nonadiabatic/soc_and_rpsh.py
python examples/nonadiabatic/thermal_rates.py
python examples/nonadiabatic/method_models.py
pyrpmd validate examples/nonadiabatic/friction_surface.inp --check-pes
pyrpmd run examples/nonadiabatic/friction_surface.inp --output /tmp/pyrpmd-friction-surface
pyrpmd resume /tmp/pyrpmd-friction-surface
```

Use a fresh output directory for the run command. The full suite needs local
socket access for its existing i-PI protocol tests. Optional ASE, Torch, real
BKMP2, EAM and external reference-solver tests depend on separately installed
providers/data. The existing CI tests Python 3.11/3.12, ASE, MPI, Torch, docs
and package builds. Its documentation job also runs all four new examples.

The local wheel was installed with `--no-deps` into a separate virtual
environment. An isolated interpreter (`python -I`) running outside the checkout
imported all 39 public exports, including all 26 new APIs. All 59 runtime Python
files matched the checkout in both the wheel and sdist. All seven new modules
were present, and the four new guides and four examples were included in the
sdist. The wheel retains the existing runtime-only documentation/example policy.

## Independent numerical controls

| Component | Controls and regressions |
| --- | --- |
| Complex SOC-NRPMD | Independent matrix-exponential impulse quadrature; full Hamiltonian ODE convergence; complex unitary-basis covariance; imaginary-gradient force; degenerate energy gaps; inverse map and symplectic Jacobian; complex finite differences and lossless preparation JSON |
| Thermal NRPMD | Gaussian quadrature of the SEO trace and projector insertions for noncommuting bead matrices; independent five-bead harmonic stiffness; thermal nuclear/momentum/mapping moments; one-state RPMD reduction; analytic electronic and free-streaming nuclear flux; signed normalization and joint block-jackknife diagnostics |
| MV-RPMD | Signed Wigner Gaussian quadrature against direct electronic traces; every nuclear/mapping derivative by finite differences; implicit-midpoint convergence; negative populations retained; explicit node and solver failures |
| Original KC-RPMD | Enumeration of all `2**P` primitive electronic paths; analytic nuclear/auxiliary derivatives versus finite differences; auxiliary mass conventions; explicit calibration and primitive-propagator domain checks |
| Isomorphic Hamiltonian | Finite-bead trace matching against SciPy exponentials; whole-ring derivatives; electronic impulse quadrature; collapsed-ring limit; constant coupling down to `1e-30`; unresolved nonconstant weak coupling rejected |
| RPSH | One-surface RPMD reduction; known rotating electronic frame; probability integration and norm; centroid/ring rescaling energy conventions; frustrated-hop choices; reproducible stochastic continuation |
| Electronic friction | Unequal-mass correlated OU covariance versus independent SciPy exponentials and Gaussian cubature; centroid covariance `M/beta` at different bead counts; unchanged internal modes; bead-averaged spatial tensor; zero-friction ordinary-RPMD identity for all three integrators; fixed-atom projection and kinetic estimators |
| Transactions and integration | State/RNG/counter rollback on provider errors and interrupts; overflowed-clock rejection; configuration/alias guards; provider cleanup; public input/prepare validation; interrupted serial EF workflow continuation; deterministic helper rejection for stochastic drivers |

The added MPI materials smoke test checks EF-RPMD with two trajectory groups
and two bead ranks per group, output ownership, and a completed-run resume.
It does not itself establish interrupted MPI continuation or serial/MPI
bitwise equality. Interrupted stochastic continuation is covered separately
by the serial workflow regression.

## Example observations

[examples.json](examples.json) contains the finite model outputs with settings
and explicit convergence flags. Last-digit agreement can depend on numerical
libraries; the tests use tolerances tied to the mathematical identities.

| Example | Observed diagnostic |
| --- | --- |
| Complex SOC-NRPMD, 200 steps | Maximum absolute Hamiltonian drift `2.95227636e-6 hartree`; population sum `1.0000000000000253` |
| RPSH, 200 steps | One accepted hop; amplitude norm `1.0000000000000018`; centroid rescaling reports a full-ring energy jump of `-3.90406035e-6 hartree` |
| Thermal NRPMD, 64 parents / 2 beads | 420 MC sweeps; average sign `0.78125`; two finite-time curves; no selected plateau or rate constant |
| MV/KC/iso model APIs | 20 propagation steps and short sampler demonstrations; signed MV populations retained; no equilibrated or converged ensemble claim |
| Fixed-slab electronic friction | 2 replicas, 201 saved frames each; validate/run/resume completed; dissipative production labeled explicitly; diffusion coefficient withheld because the short fitted window is ballistic/drifting |

## Scientific scope

- **Electronic friction:** the Markovian centroid approximation of Bi and Dou.
  The physical tensor is bead averaged, mass weighted for the OU map, and its
  correlated momentum increment is shared across beads. It does not implement
  non-Markovian memory or bath-induced imaginary-time renormalization. The
  example's friction is illustrative, not fitted to copper.
- **Complex SOC:** a fixed-basis Hermitian NRPMD Hamiltonian and positive
  Hermitian gradients. The other new matrix methods currently require real
  symmetric inputs.
- **Thermal NRPMD:** samples the absolute SEO-projected thermal weight and keeps
  its sign. It does not sample a naive Boltzmann distribution of the generally
  unbounded mapping Hamiltonian. Rate curves require additional convergence,
  normalization and plateau analysis; canonical stationarity of approximate
  mapping dynamics is not guaranteed.
- **MV:** retains the sign problem and nodal singularities. Exact bead
  exponentials replace the paper's first-order electronic propagator; the
  projected Wigner weight and Hamiltonian are otherwise explicit.
- **KC:** the original 2014 primitive formulation, with explicit crossing
  calibration and finite restraint parameters. No later KC variants, automatic
  calibration or reproduction of the paper's tabulated rate is claimed.
- **Isomorphic:** a two-state trace-matching potential composed with explicitly
  named Ehrenfest dynamics. The trace identity alone does not validate thermal
  dynamics or reaction rates.
- **RPSH:** real-matrix centroid electronic dynamics with beadwise active-surface
  nuclear forces. The rescaling convention is explicit. No decoherence,
  Berry-force/DBOC treatment or universal detailed-balance correction is added.

Full equations, contracts and primary references are in
[electronic friction](../../electronic-friction.md),
[thermal NRPMD](../../thermal-nrpmd.md),
[SOC/RPSH](../../nonadiabatic-soc-rpsh.md), and
[KC/MV/isomorphic methods](../../nonadiabatic-methods.md).
