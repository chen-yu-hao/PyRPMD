# Force-based path-integral sampling in the main engine

`task="pimd"` now accepts `[path_integral]` to select a Suzuki–Chin action,
bosonic/fermionic exchange, or path-integral ground-state projection (PIGS). These
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

These workflows require serial execution and all atoms mobile. Use the main engine
(`sampling.engine="legacy"`, the default), with Andersen or PILE-L sampling.
Compiled free-ring substeps remain available, but the action calls use Python
step orchestration. Suzuki–Chin supports fixed fully periodic 3D cells and
isotropic BZP NPT. Exchange supports periodic image sums and isotropic all-bead
BZP NPT as detailed in [periodic exchange](#periodic-exchange-and-numerical-winding-sums);
PIGS retains nonperiodic fixed-cell scope.
Electronic friction and active-learning collection are not combined with these
workflows. The physical PES must be a
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
beta_fd_step = 1e-3
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
label that result a physical energy. The workflow also reports thermodynamic
total, kinetic and potential energies, a scaled-coordinate heat capacity,
and pressure when the physical provider supplies a configurational virial.
`SuzukiChinPotential.estimators(q)` and `.estimators_batch(paths)` expose the
same quantities for all supported alpha values. `suzuki_chin_thermodynamics`
aggregates a saved estimator series with conditional block uncertainty.

Writing `G_j = c_j |F_j|²/m`, where `c_j = w_j d_j (beta/P)²`, and using
`delta_j=q_j-q_centroid`, the thermodynamic centroid-virial energy is

```{math}
E = \frac{3N}{2\beta}
  + \frac1P\sum_j\left[w_jV_j+3G_j
                   -\tfrac12\delta_j\cdot F_{\mathrm{SC},j}\right].
```

The explicit beta dependence of the force-square term produces the factor of
three. For heat capacity the implementation holds the cell fixed, scales the
internal ring coordinates by `sqrt(beta_new/beta)`, and evaluates the first
and second beta derivatives of the potential action. Its `h` and `h/2`
curvature diagnostics expose finite-difference sensitivity; `beta_fd_step` is
the dimensionless coarse relative displacement and the result uses half that
step. NVT reports `Cv`;
NPT uses the enthalpy fluctuation and reports `Cp`. These estimators require
canonical SC sampling and cannot be applied to PIGLET trajectories.

For SC-NPT, add a fully periodic cell to the system and a legacy barostat:

```toml
[barostat]
method = "bzp"
pressure = "1 bar"
time_constant = "1000 fs"
```

The physical provider must support `set_cell(cell_bohr)` and return
`W=-V*stress`, including explicit cell derivatives. The SC wrapper differentiates
this virial consistently with its force-square correction; the piston uses
the resulting **action** pressure, while output uses a thermodynamic pressure
estimator with the analytical mean centroid kinetic term. Checkpoints retain
the cell, piston, thermostat RNG and every saved estimator frame. See
[Condensed-phase nuclear quantum effects](condensed-phase-nqe.md) for the
implemented estimator conventions, limitations and representative validation.

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

For periodic systems, choose the converged Gaussian image sum (`full_winding`)
or the explicit `minimum_image` approximation; see
[periodic exchange](#periodic-exchange-and-numerical-winding-sums). The latter
is not a winding sum. Permutation-resolved particle observables and a superfluid
estimator remain outside this workflow's symmetric-observable scope.

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
counts and mean potential energy in hartree. SC adds method-specific mean
energies and a `thermodynamics` report with Cv or Cp and conditional block
uncertainties when enough samples are available. Its checkpoint retains the
full estimator series and, for NPT, the cell history and piston. Exchange
summaries also retain sign diagnostics, and periodic exchange NPT records its
pressure/cell state; see [restart checks](#restart-and-independent-checks).
PIGS retains the midpoint-potential observable. None of these sampling
trajectories defines a physical real-time correlation function.

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

`tests/test_suzuki_chin_estimators.py` checks SC energy and heat capacity against
independent Gaussian partition derivatives and full cell virials against
affine-strain differences. `tests/test_condensed_phase_ensembles.py` tests
SC-NPT pressure and recovery, PIGLET centroid covariance and its ideal-gas NPT
measure. See [condensed-phase validation](condensed-phase-nqe.md) for the
coupled crystal and anharmonic FCC solid evidence, bead/timestep comparisons
and retained sampling uncertainties.

## Fermionic and periodic exchange, including isotropic NPT (T08)

This extension supersedes the nonperiodic-only exchange restriction above.
It adds fermionic estimators from positive bosonic sampling, periodic Gaussian
image sums, and a separate all-bead isotropic BZP propagator for the periodic
coordinate measure. The explicitly selected **minimum-image approximation**
remains available. Suzuki–Chin and PIGS behavior is unchanged.

### Selecting statistics and reporting uncertainty

```toml
[path_integral]
method = "fermionic"
groups = [[0, 1, 2]]
minimum_average_sign = 0.05
sign_sigma = 3.0
# Optional: choose only after examining correlation and block-length stability.
block_size = 20
```

`method="fermionic"` evolves the same positive bosonic sampling action as
`method="bosonic"`. Each group contains identical, equal-mass particles;
separate groups can represent different spin species. Spin multiplicities
are not inferred, and all listed groups use the selected statistics. Unlisted
particles remain distinguishable. The physical potential must be symmetric
within each group. The method estimates symmetric equilibrium observables,
not real-time fermionic dynamics or labeled-particle observables.

The configuration estimator `sign` is the continuous ratio `W_F / W_B`, not
an individually sampled permutation parity of exactly +1 or -1. Its bosonic
ensemble mean is `Z_F / Z_B`. Signed prefix recurrences are normalized by the
positive bosonic prefix weights and use compensated summation. The work and
memory remain `O(N² + PN)` for a group; no factorial permutation enumeration
is used in production. Multiple groups are combined using products that
never divide by a group's sign, including at fermionic nodes.

`BosonicExchangePotential(..., statistics="fermionic").estimators(q)` returns
`sign` and already signed potential, kinetic and internal-energy numerators.
For total signed spring numerator `M`, physical bead-average potential `V`,
and `s = W_F/W_B`, the primitive total-energy numerator is

```text
A_E = (3*N*P/(2*beta) + V)*s - M/P
E_F = mean_B(A_E) / mean_B(s)
```

`M` is the signed weighted quadratic spring energy, divided by the bosonic
weight. It includes interior springs and the derivative of the signed
exchange recurrence. Multiplying the bosonic primitive energy by `s` would
omit that derivative and is incorrect. No estimator divides by the sign of
an individual configuration. `ExchangeResult.energy` is a sampling free
energy, whereas `ExchangeResult.spring_energy` is the weighted quadratic
spring energy; neither alone is a physical total-energy estimator.

For a nonperiodic bound system, the API also provides the signed unshifted
virial energy `s*(V - sum(q*F_physical)/(2*P))`. This additional estimator
requires an integrable confined distribution; it is not provided for periodic
paths. All quantities use atomic units, and `PESResult.forces` retains the
physical force convention without bead scaling.

`pyrpmd.quantum_statistics.exchange_thermodynamics` accepts the saved rows or
matching one-dimensional columns. Each replica's `thermodynamics` report
contains the average sign and its block standard error, energy ratios and
their joint delete-one-block jackknife errors, and correlation diagnostics.
The joint errors retain covariance between numerator and denominator. The
same complete-block prefix is used for all means and errors, and discarded
trailing frames are reported. Automatic block selection examines the sign,
numerators and ratio influence functions; the resulting errors remain
conditional on adequate block length and mixing.

Physical ratios and their errors are `null` when fewer than four complete
blocks are available, the mean sign is nonpositive or below
`minimum_average_sign`, its `sign_sigma`-standard-error interval reaches zero,
or a jackknife denominator is nonpositive. Structured `warnings` explain the
failure. The workflow retains the unweighted potential only under
`bosonic_sampling_potential_energy_hartree` for fermionic runs; that value is
not presented as a fermionic energy. `status="complete"` describes the run
clock, and `physical_converged` is always false. Finite sampling, time step,
bead convergence and the sign problem require separate assessment.

The nonperiodic example is `examples/path_integral/fermionic.inp`. Its short
run illustrates configuration and reporting, not a converged physical result.

### Periodic exchange and numerical winding sums

For a periodic system, explicitly select `periodic_mode="full_winding"` or
`periodic_mode="minimum_image"` in the exchange table. Both statistics are
supported. The workflow requires
`dynamics.integrator="velocity-verlet"` with the Andersen thermostat: the
existing analytic normal-mode spring propagation assumes unwrapped quadratic
springs and is not invariant under independent bead-image changes.

```toml
[dynamics]
integrator = "velocity-verlet"
timestep = "0.01 au"

[sampling]
thermostat = "andersen"
andersen_interval = 10

[path_integral]
method = "bosonic"
groups = [[0, 1]]
periodic_mode = "full_winding"
winding_tolerance = 1e-12
winding_max_terms = 100000
```

In full-winding mode each interior and possible exchange-endpoint connection
uses the periodic Gaussian kernel

```text
K(d) = sum_{n in Z^k} exp(-a * |d + n*B|^2)
a = mass*P/(2*beta)
```

Here `B` contains the `k` periodic lattice vectors. The algorithm sums the
individual connection kernels before the exchange recursion, so the
permutation and winding weights remain coupled. The free energy of a link
is `-P/beta*log(K)`. Its conditional displacement and displacement-square
moments supply forces, affine virials and beta derivatives. In particular,
the quadratic spring moment is distinct from the link free energy.

Real-space Gaussian sums and their Poisson reciprocal representation are
selected according to the kernel width. Orthogonal periodic directions
factor into one-dimensional sums; skew cells use the multidimensional
lattice. Both full and partial periodicity are supported. Sums are chunked
to bound temporary memory instead of allocating all links times all images.
The work is quadratic in particle count at fixed lattice-sum cost.

`winding_tolerance` controls the per-link truncation error of the kernel and
scaled moments. It is a numerical summation tolerance, independent of bead,
timestep and sampling convergence, and excludes floating-point roundoff.
`winding_max_terms` limits the lattice-sum work per link. A cell or requested
tolerance that exceeds the budget raises an error; it never silently changes
to MIC. The two controls are accepted only in full-winding mode and enter
the scientific identity. Small cells use the reciprocal representation to
avoid a growing direct-image count. The construction follows
[Higer, Feldman and Hirshberg (2025)](https://doi.org/10.1063/5.0265906).

`ExchangeResult.winding_diagnostics` reports the representations, cutoffs,
term counts and the three link truncation bounds. Its representative-weight
bound combines the link errors over `N*P` connections. Preflight reports the
initial plans, and each replica summary retains its fixed-cell or explicitly
labeled final-cell plans. NPT validates the requested budget at every cell
update. The implementation caps `winding_max_terms` at 1,000,000 to bound
cached lattice data. Link batches target at most 262,144 link-term pairs;
one link with more terms is evaluated alone. The reported
`working_pair_budget` is the larger of that target and the actual term count,
and is therefore at most 1,000,000.

MIC retains only the exact nearest image for each displacement, including
skew cells and partial periodicity. Derivatives at an exact MIC image tie
are not defined. Periodic particles outside the exchange groups receive
the same selected periodic treatment with distinguishable connections.
For either mode the complete sampling action, forces and affine virial are
invariant under independent lattice translations of beads, up to the
declared numerical precision. The ordinary unwrapped engine springs are
subtracted once by `evaluate`; the wrapper correction alone need not be
image invariant. `evaluate_sampling` instead returns the complete sampling
action directly, including all springs, for the all-bead NPT propagator.

`examples/path_integral/bosonic_periodic.inp` demonstrates fixed-cell MIC;
`examples/path_integral/fermionic_winding.inp` demonstrates full periodic
fermionic reweighting. Full-winding mode sums the periodic free-density
matrix images to numerical tolerance. It does not introduce a superfluid
estimator or remove finite-bead errors of an interacting physical potential.

### Affine virial and thermodynamic pressure

`ExchangeResult.virial` is the negative affine strain derivative of the
bosonic spring free energy. When the physical provider supplies its own
configurational virial, the wrapper returns the physical virial plus the
exchange correction, with the subtracted ordinary spring contribution
handled consistently. A missing physical virial is never replaced by
Cartesian `q dot F`.

For a fully periodic current cell, `estimators(q, volume=Vbox,
require_pressure=True)` returns a signed primitive pressure numerator:

```text
A_p = (N*P/(beta*Vbox) + trace(sum(W_physical))/(3*P*Vbox))*s
      - 2*M/(3*P*Vbox)
p = mean_B(A_p) / mean_B(s)
```

It requires the physical provider's virial and the matching current-cell
volume. With optional pressure reporting, the field is absent if the provider
has no virial. Pressure is consistent with the selected periodic action;
MIC therefore retains its image approximation, whereas full-winding mode
uses all images to the specified tolerance. NPT requires physical virials
and explicit `set_cell` support from the physical provider.

### Exchange with isotropic all-bead BZP NPT

Add the ordinary pressure-control table to a fully periodic exchange input:

```toml
[barostat]
method = "bzp"
pressure = "0.04 au"
time_constant = "2 au"
```

The workflow selects
`pyrpmd.exchange_barostat.ExchangeBZPBarostat`. This class scales all bead
coordinates, all bead momenta, and the cell, using the full sampling-action
force and strain virial. Its target density is

```text
exp[-beta/P * (H_P + P*p_ext*Vbox + piston_momentum^2/(2*piston_mass))]
    * dq * dp * dVbox * d(piston_momentum)
```

All bead coordinates live on the periodic cell; in fractional coordinates
their Jacobian is `Vbox**(N*P)`. No center-of-mass constraint is imposed.
The piston drive is
`2*K_all + trace(W_sampling) - 3*P*p_ext*Vbox + 3*P/beta`.
The last term accounts for the `dVbox` measure. The corresponding
deterministic extended invariant is
`H_P + P*p_ext*Vbox + piston_momentum^2/(2*piston_mass)
- P/beta*log(Vbox)`; stochastic thermostat heat is not included.

The reversible step uses exact force/piston and drift/dilation subflows,
including the quadratic and cubic piston terms during a force kick. It
requires two full-action PES evaluations per step and becomes velocity
Verlet when the piston rate is zero. A Langevin piston and Andersen particle
collisions provide thermostatting. The piston mass is
`3*N*P*time_constant**2/beta`. Existing Dynamics constraints, specialized
operators or thermostats cannot be silently bypassed by this propagator.
Direct callers must use plain `Dynamics` with velocity-Verlet and the Python
loop; specialized subclasses and composed operators are rejected. The
workflow rejects `native="required"` during preflight. Andersen is the
supported particle thermostat; `thermostat=None` is a low-level deterministic
validation option when piston friction is also zero.
The construction adapts the BZP equations of
[Bussi, Zykova-Timan and Parrinello (2009)](https://doi.org/10.1063/1.3073889)
and the exact subflows of
[Ceriotti, More and Manolopoulos (2014)](https://doi.org/10.1016/j.cpc.2013.10.027)
to this unconstrained all-bead coordinate domain.

The older centroid BZP class remains incompatible with periodic exchange:
its `sum((q-q_centroid)*F)` correction depends on independent bead-image
choices. Calling `BosonicExchangePotential.set_cell` without the explicit
`barostat_coupling="all_beads"` contract still fails, so a direct caller
cannot accidentally select that old generator. The workflow chooses the
new class automatically and records `barostat_coordinate_scaling="all_beads"`.

NPT reports signed volume and enthalpy numerators in addition to energy and
pressure. Their ratios and uncertainties use the same sign gate and joint
jackknife. A fermionic run exposes an unweighted volume only as
`bosonic_sampling_mean_volume_bohr3`; `mean_volume_bohr3` is the gated
fermionic estimate. See `examples/path_integral/bosonic_winding_npt.inp`.

For a finite periodic free Bose gas the zero-momentum ground state gives
`Z(Vbox -> 0) -> 1`. Its NPT mean pressure consequently has a volume-boundary
term and need not equal the external pressure. Validation instead compares
the full `Z(Vbox)*exp(-beta*p_ext*Vbox)` distribution and moment identities.
This boundary effect is separate from integration and sampling errors.

### Restart and independent checks

Atomic checkpoints now also contain `exchange_estimators`, with signed
numerators, sign and any applicable optional estimator columns. NPT stores
the all-bead piston coupling tag and a matching cell for every saved frame;
volume and enthalpy numerators are checked against that history. The full
position, momentum, clock and RNG state remain present. Invalid histories or
changed physics are rejected. Legacy nonperiodic bosonic checkpoints without
the new history reconstruct it from saved paths without consuming RNG or
changing trajectories. Existing input fingerprints remain unchanged; new
statistics, groups, periodic approximation and reporting controls participate
in the existing scientific identity.

The new `tests/test_exchange_core.py`, `test_exchange_statistics.py`,
`test_exchange_workflows.py` and `test_exchange_harmonic_reference.py` cover
explicit representative-cycle sums, force and strain finite differences,
skew-cell image invariance, zero-sign nodes, multiple groups, pressure,
joint statistical errors and interrupted bitwise recovery. The harmonic
reference includes two and three particles, several inverse temperatures and
finite bead counts. Independent Gaussian determinants over all permutations
check the analytic partition functions. Independent Gaussian-mixture samples
from the positive recursive density then check the production estimators and
average sign within measured sampling errors.

`test_exchange_winding.py` independently checks lattice sums, truncation
bounds, forces, affine virials and periodic image invariance.
`test_exchange_periodic_reference.py` uses reciprocal-lattice occupation
energies and independent torus Brownian bridges for one, two and three
particles at multiple bead counts, plus an independent NPT volume integral.
`test_exchange_barostat.py` verifies the complete physical-plus-exchange
pressure, reversible subflows, second-order invariant errors and atomic
rollback. `test_exchange_npt_workflows.py` checks coupled cell/piston/RNG
restart, scientific identities, and signed volume/enthalpy reporting.

For an isotropic trap of frequency `omega`, the finite-`P` cycle factor is
`z_k = [2*sinh(k*P*asinh(beta*omega/(2*P)))]**(-3)`. A length-`k` cycle has
`k*P` beads at inverse temperature `k*beta`; using only `P` beads at that
temperature is a different discretization. These tests verify finite-`P`
estimators, not MD ergodicity or the infinite-bead limit. The fermionic
reweighting construction follows
[Hirshberg, Invernizzi and Parrinello (2020)](https://doi.org/10.1063/5.0008720).
