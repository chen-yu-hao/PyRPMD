# Two-state spin methods

The Python APIs implement **W-spin SM-NRPMD**, the **centroid approximation
to RP-MASH (CA-RP-MASH)** and the **bead approximation to RP-MASH
(BA-RP-MASH)**. They use ordinary atomic masses, Cartesian nuclear
coordinates `q[P,N,3]`, momenta with the same shape, and atomic units
(`hbar = 1`). A reduced model can place its coordinates in selected Cartesian
slots. Neither driver converts mass-weighted normal coordinates implicitly.

| API | Electronic variables | Nuclear propagation | Direct population estimator |
| --- | --- | --- | --- |
| `SMNRPMDDynamics` | One unit spin `spin[P,3]` per bead in a fixed diabatic basis | W-spin Hamiltonian, exact spin/impulse subflows and a free-ring map | Bead average of the signed W kernel in the fixed diabatic basis |
| `RPMASHDynamics` | One unit spin `spin[3]` for the entire ring, stored in a fixed diabatic basis | One common adiabatic surface, centroid electronic evolution, deterministic equator events | Indicator of the lower/upper active adiabatic surface, with the specified population preparation |
| `RPMASHBADynamics` | One common unit spin `spin[3]` in the adiabatic representation | One common adiabatic surface, bead-averaged electronic evolution, deterministic equator events in the full bead space | Indicator of the lower/upper active adiabatic surface, with population importance sampling |

These are two-state implementations. The SM driver accepts complex Hermitian
matrices and gradients. Both RP-MASH drivers require a real symmetric matrix PES
and a nonzero adiabatic gap; they raise an error at a gap below
`gap_tolerance`. Complex-SOC RP-MASH, multistate RP-MASH and SU(N) SM-NRPMD
are outside these APIs. The separate classical multistate MASH implementation
is described in [the IESH and multistate guide](nonadiabatic-iesh-multistate.md).

## Sources and implemented versions

SM-NRPMD follows the two-state W representation of Bossion, Chowdhury and
Huo, *J. Chem. Phys.* **154**, 184106 (2021),
[doi:10.1063/5.0051456](https://doi.org/10.1063/5.0051456)
([author manuscript](https://huogroup.github.io/pdf/jcp11.pdf)). Relevant
definitions are the W radius, Hamiltonian and equations of motion in
Eqs. (8), (45), (52)–(53), the final-time estimator in Eq. (51), and the focused
nonequilibrium condition discussed in Appendix B.

CA-RP-MASH and BA-RP-MASH follow Cao, Suarez and Kretchmer,
*A ring polymer molecular dynamics extension of the mapping approach to surface
hopping: Incorporation of nuclear quantum effects*,
[doi:10.26434/chemrxiv-2025-xps2g](https://doi.org/10.26434/chemrxiv-2025-xps2g),
2025 preprint,
[full manuscript](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/692ba382a10c9f5ca169b8de/original/a-ring-polymer-molecular-dynamics-extension-of-the-mapping-approach-to-surface-hopping-incorporation-of-nuclear-quantum-effects.pdf).
The CA implementation uses its energy function, centroid equations and common
hop direction, Eqs. (36), (44)–(50). The BA implementation uses the single-spin
bead equations and extended-space impulse, Eqs. (34)–(43). Population
preparation follows the operator-pair weight in Eq. (29).
The population importance sampling is also described in Lawrence, Mannouch
and Richardson, *J. Phys. Chem. Lett.* **15**, 707–716 (2024),
[doi:10.1021/acs.jpclett.3c03197](https://doi.org/10.1021/acs.jpclett.3c03197).

The implementations here were written from these equations. They do not
incorporate an unverified third-party implementation or claim to reproduce
the authors' complete production code and numerical datasets.

## W-spin SM-NRPMD

Write the supplied complete diabatic matrix as

\[
V(q)=h_0(q)I+\mathbf h(q)\cdot\boldsymbol\sigma,
\qquad r_W=\sqrt{3}/2.
\]

For the standard Pauli matrices, `hy = -imag(V[0,1])`. This sign matters for
complex coherence and SOC. The driver uses the sum-bead propagation energy

\[
H_W=\sum_b\left[\sum_a\frac{|p_{ba}|^2}{2m_a}
+\frac12\left(\frac{P}{\beta}\right)^2
  \sum_a m_a|q_{ba}-q_{b+1,a}|^2
+h_0(q_b)+2r_W\mathbf h(q_b)\cdot\mathbf u_b\right].
\]

Each spin has unit length. At fixed coordinates it precesses as
`du/dt = 2 h cross u`. The NumPy path integrates that rotation and its nuclear
momentum impulse analytically, including the zero-field limit. A symmetric
half spin/impulse, full free-ring, half spin/impulse step advances the coupled
system. `integrator` selects `cayley-bcb`, `normal-mode-bab`, or
`velocity-verlet` for the free-ring part. These are numerical integrators of
the same W-spin Hamiltonian; timestep convergence is still necessary.

`state.density_matrix()` and `driver.density_matrix(state)` return

\[
\rho_W=I/2+r_W\langle\mathbf u_b\rangle_b\cdot\boldsymbol\sigma.
\]

`populations()` returns its real diagonal, preserving negative values.
The matrix is Hermitian with unit trace but need not be positive semidefinite
for one trajectory. Its off-diagonal entries are the fixed-diabatic coherence
estimator. Neither beads nor saved frames count as independent trajectories
when estimating uncertainty.

### Focused preparation

`SMNRPMDState.from_nuclear(nuclear, initial_state=0, rng=0)` and
`driver.initialize(nuclear, initial_state=0, rng=0)` copy the nuclear state.
For an initial diabatic state 0 or 1 they choose
`uz = +1/sqrt(3)` or `-1/sqrt(3)`, with independent uniform azimuths on every
bead. Thus each bead initially estimates the selected population as one.
The transverse coherence averages to zero across the electronic ensemble;
it is generally nonzero in one sample.

This is a **focused nonequilibrium preparation**. It does not implement the
electronic thermal trace `Gamma_W`, its complex phase reweighting, or the
paper's equilibrium Kubo correlation sampling. In particular, sampling
ordinary harmonic nuclear rings and then focusing the spin does not prepare
the thermal distribution of the coupled electronic-nuclear Hamiltonian.

### Electronic backends

`electronic_backend="numpy"` retains the independent Bloch-vector reference
implementation. `"native"` and `"torch"` use the common exact mapping kernel
through the identity

\[
\rho_W=\tfrac12zz^\dagger-\tfrac12\gamma I,
\qquad \gamma=\sqrt3-1,\quad z^\dagger z=2\sqrt3.
\]

Two spinor charts handle both poles without a division by a vanishing
component. An arbitrary overall spinor phase has no observable effect.
`electronic_device="cpu"` is the default; the Torch backend also accepts a
supported `"cuda"` device. An unavailable requested backend raises an error.
The `native` argument separately controls the nuclear free-ring integrator.
The spin state remains a NumPy host array, so using a GPU through this driver
includes host/device transfers and makes no automatic speedup guarantee.

## CA-RP-MASH

All beads occupy `active_surface=0` (lower) or `1` (upper). Each bead uses its
own adiabatic energy and gradient. Electronic evolution uses only the
centroid matrix. Storing the spin in the fixed diabatic basis avoids
eigenvector-sign bookkeeping; the instantaneous upper-state axis is
`h(centroid)/|h(centroid)|`.

The conserved energy function is

\[
E=\sum_b\left[K_b+U_{\rm spring,b}
+h_0(q_b)+s\,|\mathbf h(q_b)|\right],\qquad
s=2\,\mathrm{active\_surface}-1.
\]

A spin crossing the centroid adiabatic equator attempts to change the common
surface. The driver locates this event during the step. It does not draw
FSSH probabilities, compare only endpoint populations, or hop individual
beads independently. The full ring energy difference enters the impulse.

For unweighted Cartesian momenta with possibly unequal masses, the impulse
is evaluated directly from energy conservation. Let `d` be the centroid NAC
direction repeated on all beads and write `p_new = p + a*d`. Define

\[
A=\tfrac12\sum_{b,a,c}d_{ac}^2/m_a,
\quad B=\sum_{b,a,c}p_{bac}d_{ac}/m_a.
\]

The accepted impulse solves `A*a*a + B*a + DeltaE = 0` with the root that
preserves the direction of the incoming mass-weighted NAC velocity. If the
discriminant is negative, the impulse uses `a = -B/A` and reflects that
velocity while retaining the old surface. This evaluates the energy and
NAC-direction constraints for general atomic masses. It reduces to the
manuscript's scalar momentum formula for equal masses; simply using an
unweighted momentum projection with an effective mass is insufficient for
general unequal-mass momenta. The same shift on every bead leaves internal
momenta unchanged.

### Numerical event integration

`RPMASHDynamics` uses SciPy DOP853 with event root finding. Its
`integrator_name` is `"adaptive-dop853"`; `native=False` is required.
`dt` is the interval advanced by one public `step()`. Internal accuracy is
controlled by `rtol`, `atol` and `max_step` (default `dt/4`). Reduce these
controls and the reporting interval to check convergence, particularly for
rapid recrossings. As for other event-based ODE integrations, a large
internal step can fail to resolve multiple crossings inside that step.

`last_hops` reports all accepted or frustrated events in the last successful
step, with absolute event time, attempted target, full ring gap and the
impulse's energy residual. `last_hop` is its last entry, or `None`. The driver
raises on zero-gap evaluations, an undefined grazing impulse, integration
failure or exceeding `max_events`; it does not discard a failed trajectory.
`step()` commits state and hop records only after the entire interval succeeds.

### Population preparation and observables

`driver.initialize(nuclear, active_surface=0, sampling="population")` draws
the initial adiabatic spin from the selected hemisphere with density
proportional to `abs(Sz)`. Equivalently, `abs(Sz)=sqrt(U)` with uniform
azimuth. A population initialized at the pole or drawn uniformly over the
hemisphere with no weight would define a different calculation.

For this preparation, average `driver.populations(state)` over trajectories
to obtain lower/upper **adiabatic active populations**. These values sum to
one on every trajectory. A single common spin describes the ring; there is
no bead count in this population estimator.

`spin_density_matrix()` is explicitly an **electronic propagation
diagnostic**, not a MASH observable estimator. Its diagonal cannot be used
as MASH diabatic populations. This distinction is why the MASH driver does
not expose a generic `density_matrix()` method. `electronic_norm(state)`
reports the unit-spin norm for numerical diagnostics.

### General operator correlations

For fixed-diabatic populations, coherences or other Hermitian operators,
use `sampling="uniform-sphere"`, save a copy of each initial state, and
evaluate `driver.correlation(initial, final, A, B, basis="diabatic")`.
This returns a single-trajectory contribution to the electronic trace
`Tr[A B(t)]`, including the operator-pair factors `W_CC=3`, `W_CP=W_PC=2`,
and `W_PP=2*abs(Sz_initial)`. The function separates each transformed operator
into its adiabatic population and coherence parts. It includes the factor
of two between normalized uniform-sphere probability and the spin trace
measure. Average these contributions directly; do not self-normalize by
the sum of sampled weights.

Both `A` and `B` must be finite Hermitian `2x2` matrices. A complex coherence
can be reconstructed from separate Hermitian real and imaginary coherence
operators. With `basis="adiabatic"`, matrices use the order **(upper, lower)**
so that `sigma_z=diag(+1,-1)`; this is distinct from the public population
array order **(lower, upper)**. The method rejects population-importance
samples for this general correlation API. Nuclear preparation and any
nuclear operator factors remain the caller's responsibility; this routine
does not establish an electronic-nuclear thermal trace.

## BA-RP-MASH

`RPMASHBADynamics` implements the **two-state bead approximation** of
Cao, Suarez and Kretchmer, Eqs. (34)–(43). Its workflow name is
`method="rp-mash-ba"`. The existing `method="rp-mash"` continues to mean
CA-RP-MASH, with its existing state representation and scientific identity.
This addition does not extend classical multistate MASH to a ring polymer.

### Electronic motion and bead forces

There is one spin for the entire ring, expressed in a common adiabatic
representation with `Sz > 0` corresponding to the upper state. Each bead
evaluates its own two-state matrix, gradients, gap and derivative coupling.
Using the manuscript's convention,

\[
d_{ba}=\langle\psi_+(q_b)|\nabla_a\psi_-(q_b)\rangle,
\quad V_z(q_b)=\tfrac12[E_+(q_b)-E_-(q_b)],
\]

define

\[
D=\frac1P\sum_{b,a}\frac{p_{ba}}{m_a}\cdot d_{ba},
\qquad G=\frac1P\sum_b V_z(q_b).
\]

The electronic equations are

\[
\dot S_x=2DS_z-2GS_y,\qquad
\dot S_y=2GS_x,\qquad
\dot S_z=-2DS_x.
\]

Thus the velocity–NAC contraction is averaged **after** contracting each
bead's own velocity with its own NAC. Evaluating a centroid NAC or multiplying
two separately averaged arrays would implement a different approximation.
The nuclear force on each bead is the negative gradient of its own active adiabatic
energy plus the ring spring force. The spring frequency is `P/beta` in
atomic units. A single active-surface index applies to every bead.

For a real matrix written as `h0*I + hx*sigma_x + hz*sigma_z`, the derivative
coupling is evaluated without diagonalization phase choices:

\[
d_{ba}=\frac{h_x\nabla_a h_z-h_z\nabla_a h_x}
                   {2(h_x^2+h_z^2)}\bigg|_{q_b}.
\]

The gap check applies to every evaluated bead. There is no additional
centroid PES evaluation in the BA electronic equations.

### Equator events and energy conservation

Crossing `Sz=0` attempts to switch the entire ring. The propagation energy is
the same sum-bead energy written above for CA-RP-MASH. In particular, the
energy paid by a successful switch is

\[
\Delta E=\sum_b[E_{\rm target}(q_b)-E_{\rm source}(q_b)],
\]

with no division by `P`. The impulse points along the full array `d[P,N,3]`.
Set `p_new = p + a*d` and use

\[
A=\tfrac12\sum_{b,a}\frac{|d_{ba}|^2}{m_a},\qquad
B=\sum_{b,a}\frac{p_{ba}\cdot d_{ba}}{m_a}.
\]

The accepted impulse solves `A*a*a + B*a + DeltaE = 0`, choosing the root
that retains the incoming NAC velocity direction. A frustrated hop reflects
that velocity with `a=-B/A` and retains the original surface. This inverse
mass formulation conserves energy for unequal atomic masses as well as for
the equal-mass examples of the manuscript. Unlike CA-RP-MASH, the impulse
can change internal bead momenta because the bead NACs can differ.

BA uses the same DOP853 event-resolution controls, failure semantics and CPU
requirements described above. It rejects `rtol` below
`100*np.finfo(float).eps` rather than allowing the solver to raise the requested
tolerance. For `P=1` its continuous equations reduce to
two-state MASH and agree with CA-RP-MASH after transforming the spin between
their adiabatic and fixed-diabatic representations. Raw spin arrays from
these two APIs must not be compared or substituted without that transform.

### Preparation, observables and restart

`driver.initialize(nuclear, active_surface=0, sampling="population")` samples
`abs(Sz)=sqrt(U)` and a uniform azimuth in the chosen adiabatic hemisphere.
This incorporates the population-pair weight `W_PP=2*abs(Sz_initial)` into
the sampling distribution. Average `driver.populations(state)` directly
across independent trajectories to estimate lower/upper adiabatic active
populations. No bead factor or additional weight belongs in that average.
Nuclear preparation remains a separate input; electronic importance sampling
does not establish the coupled electronic-nuclear thermal distribution.

`spin_density_matrix()` is an adiabatic **propagation diagnostic**, ordered
**(upper, lower)** so that `sigma_z=diag(+1,-1)`. This order differs from
`populations()`, which returns **(lower, upper)**. The diagnostic is not
a general MASH observable density and does not define fixed-diabatic
populations. The BA API does not implement the CA driver's general
`correlation()` estimator or uniform-sphere preparation: those require a
specified treatment of operators in the different bead adiabatic frames.
Unsupported requests fail explicitly. Complex-SOC Hamiltonians and more than
two electronic states are also rejected.

The restartable `rp-mash-ba` workflow stores `RPMASHBAState`, including its
adiabatic spin, active surface, preparation, nuclear arrays and clock, together
with the three existing RNG streams and observation history. The distinct
method and state class prevent interpreting a BA checkpoint as a CA state.
All scientific controls enter the run identity. Restart keeps the ordinary
atomic checkpoint writes and integrity checks; it does not infer or convert
between the two spin representations.

### Physical limitations

BA-RP-MASH is an approximate trajectory method, and a larger bead count does
not guarantee more accurate electronic dynamics. In the source manuscript's
Tully comparisons (Sec. V A–B), a ring spread comparable to the narrow NAC
region reduces the effective coupling in the bead average and can strongly
distort transition probabilities. Its spin-boson example (Sec. V D) reports
severely overestimated BA rate constants; a smaller effective NAC does not
imply that every observable is underestimated. Nuclear preparation also
matters: independent Wigner draws for bead positions can introduce excessive
internal-mode energy. Check the preparation, bead count and integration
controls against an independent quantum reference for the intended model.
Conserving the ring energy or passing software regressions does not establish
physical convergence.

## Minimal Python examples

The following deliberately uses a point nuclear state to demonstrate the
API. An ensemble calculation needs an independently specified nuclear
distribution and many independent electronic samples.

```python
import numpy as np
from pyrpmd.dynamics import State
from pyrpmd.potentials.nonadiabatic_models import TullyModelOne
from pyrpmd.spin_mapping import SMNRPMDDynamics
from pyrpmd.rp_mash import RPMASHDynamics

q = np.zeros((8, 1, 3))
q[:, 0, 0] = -5.0
p = np.zeros_like(q)
p[:, 0, 0] = 20.0
nuclear = State(q, p)
potential = TullyModelOne()

sm = SMNRPMDDynamics(potential, [2000.0], beta=100.0,
                    nbeads=8, dt=0.1, native=False)
sm_state = sm.initialize(nuclear, initial_state=0, rng=42)
sm.step(sm_state)
rho_diabatic = sm.density_matrix(sm_state)

mash = RPMASHDynamics(potential, [2000.0], beta=100.0,
                     nbeads=8, dt=0.1, seed=42)
mash_state = mash.initialize(nuclear, active_surface=0)
mash.step(mash_state)
population_adiabatic = mash.populations(mash_state)
```

A separate general-correlation sample uses full-sphere initialization:

```python
state = mash.initialize(nuclear, sampling="uniform-sphere")
initial = state.copy()
mash.step(state)
projector = np.diag([1.0, 0.0])  # Fixed diabatic state 0.
contribution = mash.correlation(initial, state, projector, projector)
```

The bead approximation uses a separate driver and state:

```python
from pyrpmd.rp_mash_ba import RPMASHBADynamics

ba = RPMASHBADynamics(potential, [2000.0], beta=100.0,
                     nbeads=8, dt=0.1, seed=42)
ba_state = ba.initialize(nuclear, active_surface=0)
ba.step(ba_state)
ba_population_adiabatic = ba.populations(ba_state)
```

Save each final state's arrays and clock, its MASH `active_surface` and
`sampling` label, and all driver parameters when constructing a restart.
Save the random generator state if further initial conditions will be drawn.
Propagation itself consumes no MASH random draws.

## Restartable ensemble workflow

The `task="nonadiabatic"` workflow accepts `method="sm-nrpmd"`,
`method="rp-mash"` or `method="rp-mash-ba"`, the same point/harmonic nuclear preparations, and serial
or process execution across independent trajectories. SM uses
`initial.initial_state`; MASH uses `initial.active_surface`. MASH explicitly
requires `dynamics.native=false` and `integrator="adaptive-dop853"`.

The output `results/nonadiabatic.npz` records every trajectory's saved energy,
electronic norm, centroid position/momentum and population contributions.
It includes the initial and final frames even when the final step is not a
multiple of the sampling stride. Population SEM is calculated across
independent trajectories with `ddof=1`; one trajectory produces no SEM array.
SM also saves the complex W-spin density. MASH saves active-surface counts
and marks coherence as unavailable in this population-only workflow.

Each checkpoint includes the complete electronic state, observation history,
physics identity and independent nuclear/electronic/dynamics RNG states.
Restart validation checks array types, normalization, estimator invariants,
the observation schedule and agreement between a saved terminal frame and
the live state. Failed trajectories cause the run to fail without silently
changing the statistical denominator. Changing process placement or block
size preserves trajectories on the same numerical backend. Electronic
backend/device are separately recorded execution settings; changing a
backend preserves the equations but does not promise bitwise equality.

## Validation included in the repository

`tests/test_spin_mapping.py` compares complex electronic propagation to
matrix exponentials, exact spin impulses to independent matrix quadrature,
the `P=1` coupled problem to a density-matrix ODE, and checks second-order
energy convergence, reversal and constant unitary covariance. Native/Torch
comparisons include both spinor poles, degenerate electronic matrices and
complex coupling. Torch tests skip when the optional dependency is absent.

`tests/test_rp_mash.py` uses a rotating-eigenvector, constant-eigenvalue model
with an independently calculable piecewise MASH solution. It verifies event
times, accepted and frustrated hops, `P=1,2,8`, unequal-mass impulses,
unchanged internal momenta, free-ring energy, pure-electronic evolution,
real-basis covariance and exact operator-trace quadrature. It also tests
transactional failures and explicit limits. These checks validate the
equations and implementation; they are not a claim that an approximate
trajectory method reproduces all exact molecular quantum dynamics.

### BA-RP-MASH independent checks

`tests/test_rp_mash_ba_reference.py` checks a nonuniform four-bead,
unequal-mass LVC model against **finite differences of aligned eigenvector
overlaps**, without using the implementation's analytical NAC. It separately
checks forces against finite differences of the adiabatic energies, full-ring
impulses, and accepted/downward/frustrated events against piecewise matrix
exponentials and scalar root finding. The one-bead comparison to CA transforms
the spin representation before comparing trajectories.

`tests/test_rp_mash_ba.py` covers exact free-ring normal-mode motion,
electronic precession, time reversal through a hop, real-basis covariance,
importance sampling, provider failure, event limits and explicit numerical
overflow. `tests/test_rp_mash_ba_workflow.py` covers CLI execution, distinct
scientific identities, bitwise interrupted continuation, all three RNGs,
process placement and malformed checkpoints. The original CA identity is
locked to a fingerprint recorded on main `ceee59c`; an additional validation
run resumes a checkpoint actually produced by that revision.

The finite-mass comparison is reproducible from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/rp_mash_ba_reference.py \
  --trajectories 128 --output ba-rp-mash-reference.json
```

This uses four noncoincident beads and
`V(x)=g*(cos(x)*sigma_z+sin(x)*sigma_x)`, with `g=0.02 hartree`,
mass `1000` electron masses and mean momentum `100` atomic units. The reference
propagates a Gaussian wavepacket with the one-dimensional Fourier solver and
projects onto the local upper state. A second, independent check transforms
the Hamiltonian to `(p-sigma_y/2)^2/(2m)+g*sigma_z` and integrates its exact
two-level transition probability over the Gaussian momentum distribution.

| Quantity | Recorded result |
| --- | --- |
| BA upper population, 128 trajectories | 0.453125 |
| Independent-trajectory SEM | 0.0441724 |
| Fourier upper population | 0.4830619533 |
| Absolute BA–Fourier difference | 0.0299369533 |
| Fourier–infinite-line analytical difference | 1.95e-7 |
| Fourier timestep refinement change | 5.80e-7 |
| Fourier grid refinement change | 6.59e-14 |
| Maximum final trajectory energy change | 1.43e-14 hartree |

The difference is about 0.68 SEM for this finite ensemble; SEM does not include
trajectory-method bias. The constant gap and constant NAC make internal ring
modes spectators in this comparison. The prescribed internal displacement is
not a coupled thermal distribution, and scalar nuclear Wigner sampling omits
position-dependent electronic Wigner corrections. The faster Fourier unit test
uses an explicitly small-gap, high-kinetic-energy limit. Neither comparison
establishes general nuclear quantum accuracy or physical convergence.

The short CLI example is `examples/nonadiabatic/extensions/rp_mash_ba.inp`.
Raw reference results, restart compatibility evidence and test logs are in
`docs/validation/t25b-ba-rp-mash-2026-11-03/`. Both the workflow and reference
script report `physical_converged=false`. Runtime in the reference JSON records
the thread settings and includes concurrently running regression work; it is
not an isolated throughput benchmark.
