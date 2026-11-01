# Nonequilibrium nonadiabatic ensembles

`run_nonadiabatic_ensemble` runs an unweighted set of NRPMD or centroid-RPSH
trajectories and retains the contribution of every trajectory at every requested
time. It is useful for electronic population dynamics and coherences when a
collision classifier or a thermal rate correlation is not the desired observable.
The propagation methods are the existing
[MMST NRPMD](nonadiabatic.md) and [centroid RPSH](nonadiabatic-soc-rpsh.md).

The API does not create a nuclear thermal distribution. Supply prepared nuclear
parents or fully initialized electronic/nuclear states. In particular, samples
from the signed SEO thermal sampler require its [thermal analysis](thermal-nrpmd.md),
not the unweighted means in this API. Correlated parent chains also require a
separate statistical analysis.

## Focused NRPMD from nuclear parents

```python
import numpy as np
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import NRPMDDynamics
from pyrpmd.nonadiabatic_ensemble import run_nonadiabatic_ensemble

# These are examples of specified, unthermalized nuclear configurations.
# For a physical calculation, replace them with a justified initial ensemble.
parents = [State(np.zeros((4, 1, 3)), np.zeros((4, 1, 3)))
           for _ in range(8)]

def dynamics_factory(trajectory_index, rng):
    # matrix_pes supplies full V[B,S,S] and dV/dq[B,S,S,N,3].
    # A fresh driver is required for each trajectory; the PES can be shared.
    return NRPMDDynamics(
        matrix_pes, masses=[1.0], beta=4.0, nbeads=4,
        dt=0.01, nstates=2, native=False,
    )

result = run_nonadiabatic_ensemble(
    parents, dynamics_factory, steps=100, sample_stride=8,
    initial_state=0, seed=731,
)

population_curve = result.mean_populations   # [frame, state]
coherence_curve = result.mean_density[:, 0, 1]  # complex, fixed diabatic basis
```

The focused initialization samples independent mapping phases for each bead and
electronic state. Passing already prepared `NRPMDState` objects preserves their
electronic initialization; omit `initial_state` in that case. An initialization
option cannot silently overwrite a prepared electronic state. This API adds no
new coherent-state or general density-matrix preparation prescription.

## Independent harmonic nuclear parents

`sample_harmonic_ring` provides an explicit nuclear preparation for independent
harmonic coordinates. It draws the finite-bead primitive path-integral Gaussian
directly, so there is no burn-in or Markov-chain correlation:

```python
from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring

parents = sample_harmonic_ring(
    masses=[1.0], beta=4.0, nbeads=4, frequencies=[0.7],
    centers=[0.0], count=500, seed=11,
)

result = run_nonadiabatic_ensemble(
    parents, dynamics_factory, steps=100, sample_stride=8,
    initial_state=0, seed=731, independent=True,
)
```

The frequency is a positive angular frequency in atomic units. The default
coordinate assignment uses atom 0 x for mode 0, atom 1 x for mode 1, and so on.
For other assignments, pass `coordinates=[...]`, one distinct flattened
Cartesian index per frequency: atom `a`'s x/y/z coordinates have indices
`3*a`, `3*a+1`, and `3*a+2`. For example, two independent coordinates on one
atom use `masses=[m]`, `frequencies=[omega_x, omega_z]`, and `coordinates=[0, 2]`.
Each coordinate uses its atom's mass. These are ordinary Cartesian coordinates,
not mass-weighted normal-mode coordinates. General coupled normal modes require
an explicit coordinate transformation outside this sampler.

For mode `j` with mass `m_j`, frequency `omega_j`, and center `c_j`, the sampled
position covariance is

\[
\operatorname{Cov}(q_j)=
\frac{P}{\beta m_j}
\left[\omega_j^2 I+\left(\frac{P}{\beta}\right)^2L\right]^{-1},
\qquad
L=2I-\operatorname{roll}(I,1)-\operatorname{roll}(I,-1).
\]

All bead position means are `c_j`. The periodic Laplacian is zero for one bead;
with two beads it contains both links between the same pair, giving an internal
eigenvalue of 4. The implementation applies the exact positive square root of
this covariance to independent real bead noise using Fourier modes. It does
not approximate the inverse with an integration timestep.

Each bead's momentum is an independent Gaussian with variance `m_j*P/beta`.
Optional `centroid_momenta=[...]` adds a common per-bead momentum mean for each
mode; the realized centroid momentum still fluctuates, with variance
`m_j/beta`. It is not fixed to the requested mean. With zero mean momenta the
joint distribution is the specified harmonic ring's canonical distribution;
a nonzero shift prepares a boosted distribution.

The returned value is a tuple of detached `State` objects with shape
`[nbeads, len(masses), 3]` and zero clocks. Inactive coordinates and their
momenta are initialized to exactly zero. This is an initial-condition choice,
not a constraint imposed on subsequent propagation. A reduced-coordinate PES
must also leave spectator coordinates uncoupled if they are to remain zero.

The finite-P ring distribution differs from its infinite-bead quantum limit.
Its auxiliary bead momenta are not sampled from a quantum Wigner momentum
distribution. The sampler supplies no electronic thermal trace, no electronic
state, and no electronic sign weights. Its nuclear temperature, masses and bead
count must match the later dynamics. `seed` accepts either a nonnegative integer
or a NumPy `Generator`; a supplied generator advances on successful sampling.
Invalid inputs are rejected before drawing, and a sampling failure restores
that generator's state.

## RPSH amplitudes and active surfaces

RPSH has two different electronic observables: amplitude populations and the
fractions of trajectories whose active surface has a given index. The API keeps
them separate. RPSH coefficients are stored in a fixed diabatic basis; its active
surface indices refer to sorted centroid-adiabatic eigenvalues.

```python
from pyrpmd.surface_hopping import RPSHDynamics

def dynamics_factory(trajectory_index, rng):
    return RPSHDynamics(
        real_matrix_pes, masses=[1.0], beta=4.0, nbeads=4,
        dt=0.01, nstates=2, seed=rng, native=False,
        rescaling="ring", frustrated_hops="keep",
    )

result = run_nonadiabatic_ensemble(
    parents, dynamics_factory, steps=100, sample_stride=8,
    active_surface=0, seed=731,
)

diabatic_amplitude_populations = result.mean_populations
adiabatic_amplitude_populations = result.mean_adiabatic_amplitude_populations
active_counts = result.active_surface_counts       # integer [frame, state]
active_fractions = result.active_surface_fractions
```

`active_surface` initializes a nuclear `State` as a pure centroid-adiabatic state.
For previously prepared `RPSHState` objects, omit this option. The selected
rescaling, frustrated-hop and optional decoherence policies remain the policies
of the supplied `RPSHDynamics`; the ensemble runner does not change them.
These policies and electronic/nuclear timestep choices can affect the result.

## Density and coherence conventions

The reusable `density_matrix(state)` function returns a fixed-diabatic electronic
estimator. For NRPMD, with `z = mapping_q + 1j*mapping_p`, it is

\[
\rho_{ij}^{\mathrm{MMST}}=
\frac{1}{P}\sum_{b=1}^{P}
\frac{z_{bi}z_{bj}^{*}-\delta_{ij}}{2}.
\]

Its diagonal is exactly the existing bead-averaged MMST population estimator.
Off-diagonal elements retain complex coherences, with the convention
`rho[i,j] = c[i] * conjugate(c[j])` for an ordinary pure-state amplitude vector.
The estimator is Hermitian, but a single MMST trajectory can have negative
populations and negative density-matrix eigenvalues. Neither populations nor
eigenvalues are clipped. A normalized focused initial ensemble has unit trace,
up to propagation roundoff; arbitrary user-prepared mapping norms need not.

For RPSH, the density is simply `outer(coefficients, coefficients.conj())`.
It is an amplitude density, separate from the active-surface indicator. The
additional adiabatic amplitude populations are evaluated at the centroid using
the existing RPSH population function.

For a coordinate-independent Hermitian electronic matrix and no decoherence,
both observations can be checked against
`rho(t) = exp(-1j*V*t) @ rho(0) @ exp(1j*V*t)`. This is a validation of the
electronic observation/propagation convention, not a benchmark of coupled
electron-nuclear dynamics on a realistic potential.

## Saved quantities and uncertainty

All internal energies and times use atomic units. `time` is relative to the
start of each supplied parent; `start_times` and `start_steps` preserve the
original clocks. Initial time zero and the final step are always present,
including when `steps` is not divisible by `sample_stride`. Zero propagation
steps produces a single initial frame. A forward ensemble requires a positive,
finite timestep even in that case.

| Quantity | Shape or meaning |
| --- | --- |
| `time`, `sample_steps` | Common relative time and step grids, `[frame]` |
| `start_times`, `start_steps` | Original parent clocks, `[trajectory]` |
| `diabatic_density` | Complex `[trajectory, frame, state, state]` |
| `populations` | Real diagonal of the density, `[trajectory, frame, state]` |
| `mean_density`, `mean_populations` | Arithmetic means over trajectories |
| `density_sem_real`, `density_sem_imag` | Separate component SEMs, or `None` |
| `population_sem` | Pointwise population SEM, or `None` |
| `energy`, `energy_drift` | Sum-bead Hamiltonian and change from each initial energy, `[trajectory, frame]` |
| `electronic_norm`, `norm_drift` | Squared electronic norms and their changes, `[trajectory, frame, P]` for NRPMD; final dimension 1 for RPSH |
| `adiabatic_amplitude_populations` | RPSH amplitude populations, `[trajectory, frame, state]`; otherwise `None` |
| `active_surfaces` | RPSH centroid-adiabatic index, `[trajectory, frame]`; otherwise `None` |
| `active_surface_counts`, `active_surface_fractions` | RPSH observed counts/fractions, `[frame, state]` |
| `adiabatic_amplitude_population_sem`, `active_surface_sem` | Separate RPSH sampling errors, or `None` |
| `final_states` | Detached final electronic/nuclear state per trajectory |
| `rng_states` | Deep-copied `initialization` and `dynamics` bit-generator states per trajectory |

The default `independent=False` reports means and retains all trajectory samples,
but returns `None` for every SEM. Set `independent=True` only when the supplied
trajectory draws can be treated as independent samples of the intended initial
ensemble. Merely drawing different electronic random numbers does not make
correlated nuclear parents independent. With at least two independent
trajectories, the pointwise estimator is

\[
\mathrm{SEM}(t)=
\frac{\mathrm{std}_{\rm trajectory}(A(t),\mathrm{ddof}=1)}{\sqrt{N}}.
\]

With only one trajectory, the SEM remains `None`; an unestimable error is not
reported as zero. `sem_unavailable_reason` explains why an SEM is absent.
Beads and saved time frames never increase the statistical sample count.
Coherence SEMs describe the real and imaginary components separately; they do
not provide their covariance or a simultaneous confidence region. The metadata
retains `physical_converged=False`, because sampling error does not establish
timestep/bead convergence or the accuracy of the approximate dynamics.

Energy and norm drift are measured at saved frames. To inspect every integration
step, set `sample_stride=1`. RPSH's `rescaling="centroid"` policy does not generally
conserve the full ring Hamiltonian reported by `energy`; interpret its drift with
that convention. Optional EDC changes the amplitudes while preserving their
normalization, and that change is not a Hamiltonian integration error.

## Random streams, continuation and failure ownership

`dynamics_factory(index, rng)` is called once per trajectory, with a fresh
propagation generator. A `SeedSequence` rooted at the explicit nonnegative
integer `seed` gives each trajectory separate initialization and propagation
streams. Repeating the same input, factory and seed reproduces these streams.
The factory must return a fresh `NRPMDDynamics` or `RPSHDynamics` object each time;
reusing a mutable driver is rejected. Method, masses, temperature, bead count,
state count, timestep, integrator and RPSH policies must agree across trajectories.
The caller must ensure that all factory-created providers represent the same
Hamiltonian; distinct provider objects cannot be compared scientifically by the
runner.

For RPSH, pass the exact supplied generator as `seed=rng`. The runner verifies
its identity and does not replace or reset it. This also supports continuing
previous final states while restoring each trajectory's propagation stream:

```python
import copy

previous = result

def continuation_factory(index, rng):
    rng.bit_generator.state = copy.deepcopy(previous.rng_states[index]["dynamics"])
    return RPSHDynamics(
        real_matrix_pes, masses=[1.0], beta=4.0, nbeads=4,
        dt=0.01, nstates=2, seed=rng, native=False,
        rescaling="ring", frustrated_hops="keep",
    )

continued = run_nonadiabatic_ensemble(
    previous.final_states, continuation_factory, steps=100, sample_stride=8,
)
```

Continuation requires the same potential and physical/numerical configuration.
The code above exposes the in-memory pieces; it is not a versioned disk checkpoint
format. The newly reported relative grid begins at zero, and `start_times`
records where the continuation begins.

Parent states are copied before factory calls and are not advanced in place.
Potential objects are borrowed and never closed by the runner. A provider or
propagation failure raises `EnsemblePropagationError`, carrying
`trajectory_index`, `relative_step`, `stage`, and the original exception as its
cause. No reduced ensemble is returned by silently dropping failed trajectories.
The runner cannot undo side effects internal to a user-supplied provider or
factory.

## Validation

```bash
pytest tests/test_nonadiabatic_ensemble.py tests/test_nonadiabatic_sampling.py -q
```

The tests compare fixed-diabatic densities with independent matrix exponentials,
check complex-coherence phase and unitary covariance, verify free nuclear motion,
separate active-surface counts from amplitude populations, check trajectory-based
SEMs, validate reproducible independent random streams and continuation, and
exercise failure ownership without excluding failed trajectories.
The harmonic sampler's covariance transform is checked against independent dense
periodic stiffness matrices, including one- and two-bead limits; additional tests
check actual Gaussian centroid moments, momentum independence, spectator zeros,
array ownership and reproducible draws.
