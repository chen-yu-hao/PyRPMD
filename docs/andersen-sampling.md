# Andersen collision timing for umbrella sampling

Umbrella FES and gas-phase `rate` calculations can randomize the time between
global momentum refreshes. This option addresses slow relaxation caused by a
fixed refresh period nearly repeating an internal ring mode's phase. It applies
to equilibrium umbrella streams and to the constrained parent chains used to
launch recrossing trajectories.

```toml
[sampling]
thermostat = "andersen"
andersen_interval = 50
andersen_schedule = "geometric"
```

With a timestep of `0.1 fs`, this requests a **mean** interval of `5 fs`.
`andersen_interval` remains a positive integer number of MD steps. Omitting
`andersen_schedule` retains the historical fixed interval and its random-number
sequence. An explicit `andersen_schedule="fixed"` also uses that fixed schedule.

## Distribution and scope

For `geometric`, a refresh replaces all unconstrained momenta with Maxwell draws,
then chooses an independent positive integer wait $N$:

$$
\Pr(N=n) = p(1-p)^{n-1},\qquad p=1/M,\qquad
n=1,2,\ldots,\qquad E[N]=M,
$$

where $M$ is `andersen_interval`. This is the discrete waiting-time distribution
for a constant collision probability per MD step, using
[NumPy's geometric generator](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.geometric.html).
The implementation draws one waiting time per refresh and propagates to the
next scheduled collision. Waits can exceed $M$; they are not truncated or rounded
from continuous exponential draws. The first refresh occurs at step zero. If
$M=1$, a refresh occurs before every MD step.

The refresh acts on the whole ring polymer. In constrained rate parents, the new
momenta are projected onto the dividing surface's tangent space using the same
conditional Gaussian projection as the fixed schedule. Changing collision
timing preserves this momentum distribution. Timestep convergence of the
chosen integrator and equilibration of the configuration distribution still
need to be checked.

With $P$ beads and $\beta=1/(k_B T)$, PyRPMD's ring-polymer convention gives each
unconstrained bead momentum component variance $m_iP/\beta$. The corresponding
centroid momentum variance is $m_i/\beta$.

The supported combinations are:

| Calculation | Geometric global refresh |
| --- | --- |
| `fes` with fixed umbrellas | Supported |
| `fes` with adaptive umbrella pilots and subsequent production | Supported in both stages |
| Gas-phase `rate` umbrella streams and constrained parents | Supported |
| `fes` with replica exchange | Rejected during preflight |
| PILE-L sampling | Rejected; it has its own stochastic integration scheme |
| Materials, surface/site rates, direct collisions, metadynamics, OPES, network and PCET workflows | This setting is rejected during preflight |

The same support limits apply when the setting is explicitly `fixed`; omit the
new setting when using a workflow that already has its own thermostat schedule.
The option does not add a thermostat to physical recrossing child trajectories.
It supports the existing Cayley, normal-mode and velocity-Verlet umbrella
integrators through their current Python or native propagation paths.

## Why a fixed period can relax slowly

For a free ring mode of frequency $\omega$, one Cayley step rotates the
mass-scaled position and momentum through

$$
\theta = 2\arctan(\omega\Delta t/2).
$$

After $M$ deterministic steps, replacing the momentum leaves a position
autoregression with coefficient $a=\cos(M\theta)$. Starting with collapsed beads,
the position variance after $n$ refresh intervals is

$$
\frac{\operatorname{Var}(q_n)}{\operatorname{Var}_{\rm eq}(q)}
=1-a^{2n}.
$$

If $M\theta$ nearly equals an integer multiple of $\pi$, $a^2$ is nearly one and
configuration relaxation becomes slow even though each refresh supplies the
correct kinetic temperature. Random intervals break this persistent phase
alignment. This derivation concerns the composition of many deterministic steps
between global refreshes; the general Cayley integrator and its stability are
discussed by [Korol, Bou-Rabee and Miller (2019)](https://arxiv.org/abs/1907.07941).
Random waiting times do not fix an unstable timestep or a single-step map that
is itself resonant. Those require an appropriate integrator and timestep.

The repository's free-particle control uses 128 beads at 300 K, hydrogen mass,
`dt=0.1 fs` and a fixed 50-step refresh interval. For mode 37, the analytic
variance relaxation time is about 4.17 ps. Starting from coincident beads, its
predicted variance after 0.5 ps is only 11.3% of equilibrium. The recorded
`Dynamics` control gave 10.9%; after 1.5 ps it gave 30.2%. Randomized intervals
and a PILE-L control recovered the free-mode variance within the finite-ensemble
uncertainty. `tests/test_andersen_schedule.py` separately exercises the new
geometric option through the actual umbrella workflow.

These results establish a sampler mechanism and a regression control. They do
not establish the size of an error in a molecular rate: physical forces can
shift and couple the modes, and one-bead simulations have no internal ring
modes. Monitor internal-mode or ring-size equilibration along with coordinate
histograms, independent replicas, overlap and block uncertainties. A well mixed
thermostat cannot repair an inappropriate PMF estimator or missing barrier
support.

## Restart and parallel reproducibility

Each umbrella replica and parent chain keeps its existing task-specific random
stream. Geometric waiting times use that chain's stream immediately after the
momentum draw. A checkpoint saves both the generator state and the absolute step
of the next collision. Changing native block size, checkpoint cadence or worker
assignment does not redraw a pending waiting time.

Native blocks end at collision and checkpoint boundaries. A failed block retains
the accepted prefix. If no step following a new momentum refresh succeeds, the
workflow restores the pre-refresh momenta, generator state and collision
deadline before saving; restart therefore attempts that same refresh once.

An omitted schedule adds no fields to the scientific configuration or the
checkpoint payload. Pre-existing fixed-schedule checkpoints retain their old
fingerprints and remain readable. A geometric checkpoint must contain a matching
schedule name and a valid future or current collision deadline; missing or
inconsistent clock state is an error. Changing the schedule in an existing run
changes its scientific fingerprint, so use a fresh output directory for a new
protocol.

For bitwise replay, retain the numerical software and backend used to create the
checkpoint. The tests compare interrupted and uninterrupted execution exactly
within a backend, and native/Python execution within a floating-point tolerance.
