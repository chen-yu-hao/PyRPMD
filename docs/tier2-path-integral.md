# Force-based path-integral sampling in the main engine

`task="pimd"` now accepts `[path_integral]` to select a Suzuki–Chin action,
bosonic exchange, or path-integral ground-state projection (PIGS). These
actions use the existing `Dynamics` integration and sampling machinery. They
are equilibrium sampling Hamiltonians; their trajectories are not physical
real-time dynamics and must not be used as RPMD spectra or rate trajectories.

The implementation uses atomic units and the existing convention
`H_P = spring + sum_j potential_j`, sampled at inverse temperature `beta/P`.
Each wrapper supplies only the correction needed in addition to the engine's
ordinary periodic springs. It never adds a second set of those springs.

## Run an example

```bash
pyrpmd validate examples/path_integral/suzuki_chin.inp --check-pes
pyrpmd run examples/path_integral/suzuki_chin.inp --output runs/sc
pyrpmd resume runs/sc
pyrpmd analyze runs/sc
```

The other examples in that directory select `bosonic` and `pigs`. They are
short harmonic-trap workflow checks, not equilibrated material simulations.
`validate --check-pes` evaluates the actual action and force on the initial
full ring without propagating it or creating an output directory.

All three workflows currently require serial execution, all atoms mobile,
nonperiodic coordinates, and a fixed cell. Use the main engine
(`sampling.engine="legacy"`, the default), with Andersen or PILE-L sampling.
Compiled free-ring substeps remain available, but the action calls use Python
step orchestration. Barostats, electronic friction and active-learning
collection are not combined with these workflows. The physical PES must be a
deterministic function of independent configurations; an RPC wrapper cannot
be nested inside one of these actions.

## Suzuki–Chin

```toml
task = "pimd"
beads = 16

[path_integral]
method = "suzuki_chin"
alpha = 0.0
fd_step = "1e-4 bohr"
```

P must be even. With zero-based even and odd beads, the modified potential is

```{math}
\widetilde V_j = w_j V(q_j)
 + w_j d_j (\beta/P)^2\sum_i |F_{ji}|^2/m_i,
\qquad
w_{\rm even}=2/3,\quad w_{\rm odd}=4/3,
```

where `d_even=alpha/6` and `d_odd=(1-alpha)/12`. The additional force uses a
symmetric directional finite difference along `F/m`. `fd_step` is the RMS
Cartesian displacement on each active bead, in bohr. Thus no analytic Hessian
interface is required. The implementation follows the force construction of
[Kapil, Behler and Ceriotti (2016)](https://doi.org/10.1063/1.4971438).

The Python `SuzukiChinPotential` accepts `0 <= alpha <= 1`. The input workflow
requires `alpha=0`, because its reported physical potential observable is
evaluated on the even beads. It does not average the modified potential and
label that result a physical energy. A total-energy, heat-capacity or
pressure estimator for this action is not supplied.

Converge bead count, MD timestep and finite-difference displacement
independently. A small displacement can amplify noise from a PES; a large one
introduces a force approximation error. An even bead count alone does not
establish fourth-order accuracy in a finite trajectory.

## Quadratic bosonic exchange

```toml
[path_integral]
method = "bosonic"
groups = [[0, 1, 2, 3]]
```

Each group is a disjoint set of zero-based atom indices with exactly equal
masses and identical element labels. Unlisted particles remain
distinguishable. If groups are omitted, all atoms form one group. The
physical PES must be invariant under exchange within each group; equal
element labels cannot establish this property for an arbitrary user model.

`bosonic_exchange(q, mass, beta)` implements the quadratic
[Feldman–Hirshberg algorithm (2023)](https://doi.org/10.1063/5.0173749).
The work is `O(N² + PN)` and memory is `O(N² + PN)` for a group of N particles
and P beads. Prefix/suffix log partition recurrences give the representative
cycle weights and endpoint connection probabilities without enumerating
permutations. Interior spring energies are factored out before the
recurrences, and logarithmic arithmetic avoids underflow in the weights.

The returned `ExchangeResult` contains an energy, full force array, and
`connections[i,j]`, the probability of connecting the last bead of particle i
to the first bead of particle j. `BosonicExchangePotential` subtracts the
ordinary springs and adds this exchange potential once. This is the
representative-permutation formulation in that paper: its pointwise
potential is not the factorial sum of every permutation at fixed labeled
coordinates. Symmetric observables of the sampled ensemble are the intended
use.

Periodic winding sectors and permutation-resolved particle observables are
not implemented. A minimum-image closing spring is not a substitute for the
missing winding sum. The workflow therefore rejects periodic systems and
reports only a symmetric bead-averaged physical potential observable.

## PIGS ground-state sampling

```toml
task = "pimd"
beads = 65

[path_integral]
method = "pigs"
projection_time = "100 fs"
trial = {alpha = 3.0, center = 0.0}
```

`projection_time` is an imaginary time, expressed using the normal time-unit
parser. P must be odd and at least three. There are L=P−1 open links, with
`tau=projection_time/L`. The sampled density is proportional to

```{math}
\Psi_T(q_0)\Psi_T(q_L)
\exp\left[-\sum_{j=0}^{L-1}\sum_i
 \frac{m_i|q_{j+1,i}-q_{j,i}|^2}{2\tau}
 -\tau\left(\tfrac12 V(q_0)+\sum_{j=1}^{L-1}V(q_j)
 +\tfrac12 V(q_L)\right)\right].
```

The closing ring link is removed, the endpoint potential weights are halved,
and trial-wavefunction log gradients enter the endpoint forces. The engine
uses an auxiliary inverse temperature `P*tau`; the ordinary input
`temperature` does not set the physical ground-state distribution. Preparation
labels this distinction, and the result has `temperature_K=null` and an
explicit `projection_time_au`.

The CLI trial is the positive Gaussian amplitude
`exp(-sum_i alpha_i |q_i-center_i|²/2)`. `alpha` is a positive scalar or
per-atom array in inverse bohr squared; `center` is in bohr and must broadcast
to `[N,3]`. The Python `PIGSPotential` also accepts a custom
`trial.evaluate(q)` returning a finite log amplitude `[B]` and its gradient
`[B,N,3]`. Only positive-trial, primitive-action projection is supported. This
is an open-chain projection construction associated with
[Sarsa, Schmidt and Magro (2000)](https://doi.org/10.1063/1.481926), sampled here
by thermostatted molecular dynamics.

Observables are measured at the midpoint. Endpoint averages have a different
trial-dependent distribution and are not relabeled as pure ground-state
averages. Converge the projection length, the imaginary-time spacing, the
sampling trajectory and the MD timestep. Bosonic exchange and PIGS are
separate action choices in this version.

## Checkpoints and results

Each independent replica writes one atomic, integrity-checked
`checkpoints/path_integral_NNNN.npz`. It contains the current full position and
momentum arrays, integer MD clock, physical time, RNG state and the complete
saved sample prefix. The model/source fingerprint is checked before restart.
Failed propagation rolls back both the state and RNG to the last accepted
step; incomplete or inconsistent sample clocks are rejected.

Saved positions and physical potential samples remain in memory and are
included in the checkpoint. This is explicitly `trajectory_storage="memory"`;
the workflow rejects streaming/chunk controls it cannot honor. Use a sample
interval appropriate for the available memory. A future streaming backend
can retain the same scientific contract without implying that this version
has bounded-memory trajectory storage.

`results/summary.json` identifies the action, observable, replica/sample
counts and mean potential energy in hartree. It does not provide a total
energy, heat capacity, pressure, physical time correlation, or error bar from
uncorrelated-sample assumptions. Saved configurations allow a separate
observable and correlated-uncertainty analysis.

## Validation

`tests/test_pimd_methods.py` compares Suzuki–Chin forces with finite
differences of a nonlinear potential, verifies fourth-order convergence of
an exactly integrated harmonic action, and compares the quadratic bosonic
recurrence with explicit representative-cycle sums. It also checks bosonic
force gradients, connection normalization and the absence of double-counted
springs in `Dynamics`.

PIGS tests compare the action and forces with an independently written open
chain and integrate a finite Gaussian action to verify midpoint convergence
to the harmonic ground state. `tests/test_pimd_method_workflows.py` covers
input preflight, actual CLI dispatch, malformed checkpoints and bitwise
restart after a deliberately interrupted stochastic step for all three
methods with both supported thermostats. These checks establish numerical
contracts; they do not replace validation of a molecular application.
