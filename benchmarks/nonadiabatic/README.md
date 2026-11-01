# Nonadiabatic population benchmark

`population_dynamics.py` compares a nonequilibrium NRPMD ensemble with an
independent one-dimensional quantum calculation. It produces population **time
series**, records numerical refinement differences and Monte Carlo standard
errors separately, and never assumes that NRPMD must reproduce quantum dynamics.

From the repository root, with NumPy/SciPy and PyRPMD's dependencies installed:

```bash
PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py --smoke --output population_smoke.json
```

The same JSON is printed to standard output and saved by `--output`. The small
smoke settings are 16 independent trajectories, 8 beads, 40 steps of 0.05 atomic
time, and 48 grid points on `[-8,8)`. This checks the workflow and reports its
limitations; it is not a converged benchmark. Without `--smoke`, the defaults
are 128 trajectories, 16 beads, 200 steps of 0.025, and 96 grid points.

Options `--trajectories`, `--beads`, `--steps`, `--dt`, `--grid-points`,
`--half-width`, `--sample-stride`, and `--seed` override these settings. For time
step comparisons, keep `steps * dt` fixed. A seed controls separate nuclear and
mapping-variable random streams, whose integer seeds are recorded in the JSON.
All trajectory propagation in this script uses `native=False`.

## Hamiltonian and initial condition

This is a synthetic one-mode, two-state linear-vibronic-coupling model in atomic
units, with mass and oscillator frequency both equal to one:

\[
V(x)=\frac{x^2}{2}I+
\begin{pmatrix}0&0.15\\0.15&0.3\end{pmatrix}
+x\begin{pmatrix}0.35&0.05\\0.05&-0.35\end{pmatrix}.
\]

Only atom 0's x coordinate is active; the unused y/z coordinates and momenta
are zero. The full serialized `LinearVibronicCoupling` identity is included in
the output. The starting density is

\[
\rho(0)=Z_0^{-1}e^{-\beta[p^2/2+x^2/2]}\otimes|0\rangle\langle0|,
\qquad\beta=2.
\]

This is a vertical switch from an uncoupled harmonic thermal nucleus to the
coupled Hamiltonian, rather than thermal equilibrium of the coupled system.

The quantum calculation diagonalizes the scalar harmonic Hamiltonian on its
finite Fourier grid, retains eigenstates up to a specified thermal tail, and
propagates the resulting mixed state with its Boltzmann weights. NRPMD samples
independent finite-P primitive harmonic rings with `sample_harmonic_ring` and
uses focused MMST electronic variables for state 0. These prepare the same
physical initial condition in their respective representations, with an
explicit finite-bead approximation on the NRPMD side.

The output gives the sampled bead-position second moment and its trajectory
SEM, the exact finite-P harmonic value, and the continuum quantum value:

\[
\langle x^2\rangle_P=\frac{1}{\beta m}
\sum_{k=0}^{P-1}\frac{1}{\omega^2+[2P\sin(\pi k/P)/\beta]^2},
\qquad
\langle x^2\rangle_Q=\frac{1}{2m\omega\tanh(\beta\omega/2)}.
\]

The difference between these two analytic moments diagnoses initial finite-P
discretization; it does not bound the population error at later times.

## Independent quantum reference

`pyrpmd.analysis.quantum_reference.SplitOperator1D` implements

\[
e^{-iV\Delta t/2}\;\mathcal F^{-1}
e^{-ik^2\Delta t/(2m)}\mathcal F\;e^{-iV\Delta t/2}.
\]

Each matrix potential exponential is evaluated by Hermitian diagonalization.
The Fourier kinetic step and potential steps are unitary; the symmetric
splitting has second-order time accuracy. The input is a uniform periodic grid
without a duplicated endpoint, a stationary Hermitian `V[grid,state,state]`,
and a scalar mass. No absorbing potential is supported. Pure wavefunctions
have shape `[grid,state]`; mixtures have shape `[component,grid,state]` and
explicit nonnegative weights. Each component has norm
`dx * sum(abs(psi)**2) = 1`. Electronic populations and densities include the
same `dx` integration factor.

The primary quantum time step is the NRPMD time step divided by eight. The
script also computes a reference at twice that time step, a grid with twice as
many points on the same interval, and a box twice as wide with the original
grid spacing. These changes have separate entries in `numerical_differences`.
The outer 5% of the points at **each** end of the box supply an additional
boundary-population diagnostic. The thermal omitted-weight field describes
truncation within the finite grid, not missing continuum states outside it.

This is a finite-domain numerical quantum reference. Periodic boundaries,
spatial discretization, the thermal-state cutoff, and the splitting time step
all need control before it can be called a converged representation of the
unbounded model.

## Reading the comparison

| Output | Meaning |
| --- | --- |
| `quantum_reference.populations` | Weighted quantum populations on the primary grid |
| `quantum_reference.numerical_differences` | Maximum population changes under time/grid/box refinements; sensitivity estimates, not rigorous error bounds |
| `nrpmd.mean_populations` | Ensemble mean of the signed bead-averaged MMST population estimator; negative values are retained |
| `nrpmd.population_sem` | Pointwise standard error across independent trajectories; neither beads nor time frames count as independent samples |
| `nrpmd.paired_halved_dt_mean_population_change` | Mean NRPMD time-step refinement change using the same nuclear parents and mapping random streams |
| `nrpmd.paired_halved_dt_change_sem` | Trajectory SEM of that paired change |
| `nrpmd_minus_quantum_population` | Observed difference, containing method error and remaining numerical, finite-P, and sampling contributions |

Reference refinement, NRPMD time-step refinement, bead convergence, and sampling
convergence answer different questions. A small energy or norm drift is an
integration check, not evidence that an approximate electronic dynamics method
is physically accurate. Every output retains `physical_converged=false`; no
statistical or numerical threshold silently changes it to true.

## Verification

The [2026-11-01 validation record](VALIDATION.md) gives the executed tests,
CLI examples, build checks and retained smoke JSON, including their limits.

```bash
PYTHONPATH=src python -m pytest tests/test_nonadiabatic_reference.py -q
```

The reference tests use analytic constant-matrix Rabi oscillations, analytic
free-Gaussian translation and spreading, and an independently constructed
dense Fourier Hamiltonian propagated by `scipy.linalg.expm`. The latter checks
second-order convergence for a spatially varying, noncommuting matrix
potential. Thermal mixtures are also checked against a dense canonical matrix
exponential. These tests do not compare NRPMD to quantum populations as a
correctness gate.

## References

- M. D. Feit, J. A. Fleck Jr., and A. Steiger, *J. Comput. Phys.* **47**, 412–433
  (1982), [doi:10.1016/0021-9991(82)90091-2](https://doi.org/10.1016/0021-9991(82)90091-2).
- G. Strang, *SIAM J. Numer. Anal.* **5**, 506–517 (1968),
  [doi:10.1137/0705041](https://doi.org/10.1137/0705041).
