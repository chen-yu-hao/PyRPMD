# Transition-interface path sampling

`pyrpmd.transition_interface` provides canonical two-way transition-interface
sampling (TIS) for the existing deterministic `Dynamics` engine, with either one
classical bead or a real-mass ring polymer. `pyrpmd.retis` adds replica exchange
and an optional minus ensemble that supplies the effective flux internally.
`pyrpmd.pptis` provides partial-path TIS and REPPTIS, while `pyrpmd.ffs` provides
direct forward-flux sampling for an explicitly supplied Markov process.

These Python APIs accept an existing scalar collective variable or a callable
on the full `q[P,N,3]` array; they do not need coordinate derivatives. Runnable
examples are `examples/sampling/transition_interface.py`,
`examples/sampling/pptis.py`, and `examples/sampling/ffs.py`.

The original TIS method is due to
[van Erp, Moroni and Bolhuis (2003)](https://doi.org/10.1063/1.1562614),
with the path measures and algorithms elaborated by
[van Erp and Bolhuis (2005)](https://doi.org/10.1016/j.jcp.2004.11.003).
Their original 2003 application was a diatomic molecule isomerizing in a
simple fluid. The supplied separable double well is an implementation check,
and extending the path sampler to RPMD does not establish exact quantum rates.

TIS samples a **Markov chain of complete dynamical paths**. It can help when a
plausible order parameter distinguishes reactant and product basins but does
not locate a useful transition-state dividing surface. Basin definitions and
nested interfaces are still necessary. A poor order parameter can still cause
very long, correlated paths or trap sampling in one reaction channel. This API
does not infer the basins, establish their metastability, or find every channel.

## Minimal use

```python
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.transition_interface import TransitionInterfaceSampler

# potential, masses, beta and the seed's q/p use PyRPMD atomic units.
dyn = Dynamics(potential, masses, beta, nbeads, dt, native=False)
sampler = TransitionInterfaceSampler(
    dyn, Cartesian(0, axis="x"),
    basin_a=-0.95, basin_b=0.95,
    interfaces=[-0.8, -0.4, 0.0, 0.4],
    max_path_steps=10000, seed=321,
)
# An actual phase point outside both basins, with a backward path reaching A.
seed_path = sampler.generate_path(State(q, p), interface_index=3)
results = sampler.sample_interfaces(
    [seed_path] * 4, n_samples=1000, burn_in=200,
)
print([result.summary() for result in results])
```

Here `A = {s <= basin_a}`, `B = {s >= basin_b}`, and an interface crossing is
`s > lambda_i`. All interfaces satisfy
`basin_a <= lambda_0 < ... < lambda_last < basin_b`. Periodic scalar CVs must
first be replaced by a continuous, consistently ordered basin coordinate.
Distances and time use the same atomic units as the numerical `Dynamics` API.
The same samplers are available through `task="path_sampling"`, described below.

A seed must begin with exactly one frame in A, end with exactly one frame in
A or B, have all intermediate frames outside both basins, and cross its
ensemble's interface. `sample` replays a supplied seed with the configured
integrator to reject interpolated, biased, stale or otherwise incompatible
trajectories. `generate_path` is an initialization helper; its successful
outputs alone are **not** equilibrium path samples. Discard burn-in and check
independent chains, including chains initialized in different mechanisms.

## Input-file workflows

`task="path_sampling"` runs RETIS, REPPTIS or FFS through the ordinary
`pyrpmd run`, `pyrpmd resume` and report commands. The method is explicit in
`[path_sampling]`; all interfaces, seeds, physical dynamics and sampling
budgets enter the scientific identity. Existing inputs and sampler checkpoint
formats retain their identities.

```toml
task = "path_sampling"
temperature = "315775.02480398706 K" # beta = 1 hartree^-1 for this model
beads = 1
seed = 981
native = false

[system]
symbols = ["H"]
masses = [1.0]                     # electron masses, not amu
geometry = [[0.0, 0.0, 0.0]]
geometry_unit = "bohr"

[potential]
provider = "double_well"
params = {height = 1.0, width = 1.0, confinement = 1.0}

[dynamics]
timestep = 0.04                    # atomic time units

[[cvs]]
name = "x"
kind = "cartesian"
atom = 0
axis = "x"

[path_sampling]
method = "retis"
cv = "x"
basin_a = -1.0
basin_b = 1.0
interfaces = [-1.0, -0.5]
minus = true
samples = 2400
burn_in = 300
max_path_steps = 600
block_size = 120
seeds = [{q = [[-0.3, 0.0, 0.0]], p = [[1.4, 0.0, 0.0]]}]
minus_seed = {q = [[-1.1, 0.0, 0.0]], p = [[-0.7, 0.0, 0.0]]}

[output]
directory = "runs/path-retis"
checkpoint_steps = 100
```

Exactly one named continuous scalar CV is selected by `path_sampling.cv`.
Numeric basin and interface values follow its declared input unit; unit-bearing
quantities are converted through the ordinary unit parser. Seed `q` and `p`
arrays explicitly use atomic units. They can be `[N,3]`, broadcasting to all
beads, or full `[P,N,3]` arrays. These are **shooting phase points**, from which
the workflow integrates and checks genuine seed trajectories. A single RETIS
phase point may initialize all positive interfaces if it generates a valid
path in each. REPPTIS requires one phase point per local window. The ordinary
`system.geometry` remains the system reference, including fixed positions.

| Method | Required setup and budget | Rate normalization |
| --- | --- | --- |
| `method="retis"` | `seeds`, `samples`, `burn_in`, `max_path_steps`; `minus=true` also requires `minus_seed` | Internal minus/zero-plus residence times, with first interface equal to A; positive-only RETIS needs matching `external_flux` |
| `method="repptis"` | One seed per strictly interior center, `samples`, `burn_in`, `max_path_steps` | `external_flux` at the first **center**, with the same basins and timestep |
| `method="ffs"` | `initial_state`, `flux_steps`, `trials_per_interface`, `max_trial_steps`, explicit `dynamics` | Effective crossings and A-history time from the workflow's own flux phase |

An external flux table contains `flux_au`, `basin_a`, `basin_b`,
`first_interface` and `dt_au`; these metadata must match the path calculation.
`flux_au` and `dt_au` use atomic units; coordinate metadata follows the same
input units as the declared basins and interfaces. It must also describe the same physical
model and stationary ensemble, which matching coordinate metadata alone cannot
prove. REPPTIS reports its memory-loss approximation, and errors conditional
on an external flux exclude that flux's uncertainty.

FFS selects `dynamics="hamiltonian"` or `dynamics="basin-andersen"` explicitly.
The second choice requires `collision_rate`, a positive frequency, and applies
a full Maxwell momentum refresh with probability `1-exp(-collision_rate*dt)`
after an NVE step only when the resulting coordinate lies in A or B. It leaves
the outside-basin dynamics unchanged; it does not refresh momenta upon an FFS
interface crossing. `equilibration_steps` advances this same physical process
before flux collection. Initialization and relaxation still require a
stationarity check. A Hamiltonian flux trajectory conserves its energy and
need not explore a canonical distribution; choosing it explicitly does not
solve this sampling issue. Arbitrary Markov callbacks remain available through
the Python FFS API.

There is no `[sampling]` table for this task. Budgets and stochastic-dynamics
choices belong in `[path_sampling]`; unsupported thermostats, biases, barostats,
constraints and execution combinations fail explicitly. The deterministic
RETIS and REPPTIS restrictions in the shooting section still apply.

The four complete examples are:

```bash
pyrpmd run examples/sampling/path_sampling_retis.inp
pyrpmd run examples/sampling/path_sampling_repptis.inp
pyrpmd run examples/sampling/path_sampling_ffs.inp
pyrpmd run examples/sampling/path_sampling_infinite_retis.inp
pyrpmd resume examples/sampling/runs/path-retis
pyrpmd report examples/sampling/runs/path-retis
```

Paths in input files are resolved relative to the input file. Each run uses
the common manifest and saved resolved input, writes the detailed summary to
`results/path_sampling.json`, the standard summary to `results/summary.json`,
and raw production observations to `samples/path_sampling.npz`. The atomic
checkpoint is `checkpoints/path_sampling.npz`. Raw observations include
rejections and the complete cross-interface measurement history; FFS retains
its flux trace and stage outcomes/ancestry. The report remains a first-order
rate **estimate**: `converged` and `physical_converged` stay false.

`output.checkpoint_steps` counts completed workflow operations: a ladder sweep
for RETIS/REPPTIS, a flux step or completed branch trial for FFS, and a committed
event for infinite RETIS. A restart restores the last atomic checkpoint and
replays any uncommitted work. Burn-in is tracked and excluded from production.
FFS commits its first checkpoint after equilibration; interruption before that
point replays the same equilibration from the original state and RNG seed.
Resume keeps the
original total budget; it does not append a second full run. The persisted RNG,
paths, ancestry, counters and scheduler state support bitwise continuation with
the same implementation and numerical backend. This is not a cross-platform
or cross-version floating-point guarantee.

## Shooting, canonical weights and path length

Every Monte Carlo attempt chooses a fresh, uniformly random snapshot
from the current path. It redraws all mobile bead momenta with variance
`m * P / beta`, matching `Dynamics.sample_momenta`. Fixed atoms retain zero
momenta and their fixed reference positions. It then integrates backward by
momentum reversal and forward with unbiased, unthermostatted `Dynamics.step`.
Each half stops at its first encounter with A or B. When an endpoint is selected,
its endpoint role is preserved and that half is empty; the other half is
regenerated. This includes even a direct two-frame A-to-B crossing in the
ensemble instead of silently excluding very fast paths. A backward endpoint in B,
a missed interface or an overlong path rejects the whole trial.

Let `M = number_of_frames` be the number of selectable snapshots,
`H_P` PyRPMD's sum-over-beads Hamiltonian, and `K` the kinetic energy at the
shooting snapshot. For a valid proposal the acceptance ratio is

\[
 \log R = \log\frac{M_{\rm old}}{M_{\rm new}}
 -\frac{\beta}{P}\left[
 H_P(x^{\rm new}_0)-H_P(x^{\rm old}_0)
 -K(p^{\rm new}_{s})+K(p^{\rm old}_{s})\right].
\]

The acceptance probability is `min(1, exp(log R))`. The kinetic terms are the
reverse/forward Maxwell proposal-density ratio. For energy-conserving
dynamics, the energy terms cancel and only `M_old/M_new` remains. Retaining
them accounts for the canonical weight at the first frame of a path when a
reversible finite-step map has residual energy drift. The implementation
assumes a deterministic, reversible, volume-preserving map. Bare built-in
`Dynamics` and its supported real-time integrators are accepted. Subclasses,
patched stepping methods, declared history-dependent potentials, active
thermostats/barostats, force-splitting plans, and bond constraints are rejected.
Fixed atoms remain supported with zero momenta. The guard also checks the
resolved backend, thread setting, cell and periodic-boundary configuration for
changes after sampler construction. An arbitrary user potential must be a
pure, time-independent energy and force function. Timestep convergence of the
paths is still required. These restrictions also apply to RETIS and PPTIS;
the separate FFS callback contract is described below.

The current path is counted **after every attempt, including rejections**.
Dropping repeated paths or accepting every valid variable-length proposal
would bias the path ensemble. The implementation uses fresh shooting indices;
it is not the history-based “aimless shooting” algorithm. The distinction
and path-length selection factor are discussed by
[Falkner et al. (2025)](https://doi.org/10.1063/5.0261744).

`max_path_steps` bounds the entire path, including both halves. Hitting the
bound rejects the trial instead of treating its unfinished endpoint as a
product or reactant. A finite bound truncates the target ensemble. The output
reports all such rejections, including burn-in; increase the bound and repeat
until results are insensitive to it. Potential and CV exceptions are surfaced,
because converting numerical failures into ordinary Monte Carlo rejections
can create an undisclosed sampling bias.

## Crossing probabilities and rates

For ensemble `i`, the recorded binary observable is whether the retained path
also crosses `lambda_(i+1)`. For the last ensemble it is whether the endpoint
is in B. The mean gives the corresponding conditional crossing probability:

\[
 P_A(B\mid\lambda_0)
 =\left[\prod_{i=0}^{m-2}P_A(\lambda_{i+1}\mid\lambda_i)\right]
   P_A(B\mid\lambda_{m-1}).
\]

The crossing probabilities do not supply an absolute rate by themselves.
Ordinary TIS and positive-only RETIS need a matching effective positive flux;
RETIS with a minus ensemble can estimate it internally:

\[
 k_{AB}=\frac{\langle\phi_{\lambda_0,A}\rangle}
 {\langle h_{\mathcal A}\rangle} P_A(B\mid\lambda_0).
\]

`estimate_effective_flux` analyzes an unbiased coordinate trace saved at every
step. It counts only the first outward crossing of `lambda_0` since the last
visit to A. Returning through the interface without revisiting A does not
create another event. The denominator is the time for which the **most
recently visited stable basin is A**, including intermediate excursions. It
is not just the geometric time inside A. The initial unknown-history portion
is discarded until the trace first visits A or B.

```python
from pyrpmd.transition_interface import estimate_effective_flux, combine_tis_rate

flux = estimate_effective_flux(
    unbiased_coordinate_trace, basin_a=-0.95, basin_b=0.95,
    first_interface=-0.8, dt=dt,
)
candidate = combine_tis_rate(results, flux)
```

Use a stationary ensemble and the same Hamiltonian, bead count, temperature,
basins, timestep and physical dynamics for the flux and path calculations.
For deterministic canonical RPMD, independent canonical initial states with
unthermostatted trajectories can supply the flux evidence. Adding a sampling
thermostat to the flux trajectory changes its dynamics and must not be silently
mixed with these deterministic paths. The helper validates the available
basin/interface/timestep metadata; callers remain responsible for matching
the potential and preparing stationary states.

`combine_tis_rate` requires a complete ordered chain of interface results with
the same `max_path_steps` bound, so all conditional probabilities use the same
truncated path support. It validates imported summary metadata and
returns `rate_estimate_au` and `rate_estimate_s_inv`. This is a **first-order
rate estimate**, not a bimolecular rate coefficient. `rate_s_inv` remains
`None` and `converged` remains `False`: an API call cannot certify equilibrium,
stable basins, bead/timestep convergence, channel mixing or rate uncertainty.
An unvisited event is not evidence that the physical rate is zero.

## RETIS with a minus ensemble and internal flux

`ReplicaExchangeTIS(..., minus=False)` preserves the positive-ensemble mode.
Set `minus=True` to add `[0-]` and `[0-] <-> [0+]` exchanges. This mode requires
`interfaces[0] == basin_a`; a gap between the first interface and A is rejected
because its residence-time normalization would differ.

A minus path has one initial and one final frame with `s > basin_a` and at
least one interior frame with `s <= basin_a`. It is a first-return excursion
through A. Both minus and positive shooting use the same full-bead Maxwell
variance `m*P/beta`, momentum-reversed backward propagation, canonical
endpoint correction and selectable-frame ratio described above. A scalar CV
such as `Cartesian(0)` acts on the ring-polymer centroid; the stored states
and refreshed momenta still contain every bead.

```python
from pyrpmd.retis import ReplicaExchangeTIS

retis = ReplicaExchangeTIS(
    dyn, Cartesian(0), basin_a=-1.0, basin_b=1.0,
    interfaces=[-1.0, -0.5], max_path_steps=10000,
    minus=True, seed=123,
    identity={"potential": "my-model-and-parameters-v1", "coordinate": "centroid-x0-v1"},
)
# Supply genuine phase points: q_plus outside A/B with backward history A,
# and q_minus inside A. generate_* integrates and validates real trajectories.
positive_seed = retis.generate_path(State(q_plus, p_plus), interface_index=1)
minus_seed = retis.generate_minus_path(State(q_minus, p_minus))
result = retis.sample_interfaces(
    [positive_seed, positive_seed], minus_path=minus_seed,
    n_samples=2000, burn_in=300,
)
report = result.summary(block_size=100)
print(report["flux_au"], report["rate_estimate_au"],
      report["rate_standard_error_block_au"])
```

The residence-time construction follows the minus-ensemble RETIS idea of
[van Erp (2007)](https://doi.org/10.1103/PhysRevLett.98.268301). The precise
discrete-time bookkeeping here uses **interior frames**, not the total stored
timestamp span. Let `M_minus` and `M_plus` be the stored frame counts:

\[
 t_-=(M_- -2)\,\Delta t,\qquad
 t_+=(M_+ -2)\,\Delta t,\qquad
 f_A=\frac{1}{\langle t_-\rangle+\langle t_+\rangle}.
\]

Each interior A frame of a minus path owns the following observation interval,
including its final outward crossing interval. Each interior frame of the
positive path owns the subsequent A-history interval, up to the first return
to A or arrival in B. Stored endpoints are shared between neighboring
excursions; assigning `(M-1)*dt` to both paths would count crossing intervals
twice. A direct two-frame A-to-B positive path therefore contributes zero
additional positive residence time: its outbound interval already belongs to
the minus excursion. Time whose last visited basin is B is excluded. The
helper `estimate_retis_flux(minus_frames, plus_frames, dt=..., basin_a=...,
basin_b=..., first_interface=...)` uses the same convention as the independent
trace-based `estimate_effective_flux`.

Positive-to-positive exchanges accept when both retained paths belong to the
opposite ensemble. A zero-interface exchange instead shares the outgoing
crossing edges and propagates the old positive start backward into A and the
old minus end forward toward A/B. This deterministic continuation is an
involution. Its canonical Metropolis ratio is

\[
 \log R_0=-\frac{\beta}{P}
 \left[H_P(x_{-,0}^{\rm new})+H_P(x_{+,0}^{\rm new})
       -H_P(x_{-,0}^{\rm old})-H_P(x_{+,0}^{\rm old})\right].
\]

There is no uniformly chosen shooting frame in this move, so there is no
shooting-length factor. Valid proposals accept with probability one in the
energy-conserving limit; the endpoint correction remains for finite-step
Hamiltonian drift. A path-length or membership failure rejects the proposal;
potential and coordinate exceptions propagate. `zero_swap_attempts`,
`zero_swap_accepts`, and `zero_swap_length_rejections` expose these exchanges.

Every production sweep records the entire retained ladder after swaps,
including repeats. `summary(block_size=...)` estimates crossing probability,
flux and their product within the **same contiguous blocks**, preserving
their cross-correlations. It returns `crossing_probability_standard_error_block`,
`flux_standard_error_block_au`, `rate_standard_error_block_au`, and
`block_diagnostics` with block estimates, observable-mean covariance,
statistical inefficiencies and discarded-tail counts. Check several block
sizes and independently initialized chains. Missing variation yields `None`
for the corresponding error instead of a zero-error claim. A finite path
bound truncates both ensembles and still requires convergence checks.

## PPTIS and REPPTIS

`PPTISSampler` samples short local trajectories around each interior center.
For center `c_i`, the outer boundaries are `c_(i-1)` and `c_(i+1)`, using A or B
at the ends. Both left- and right-origin paths are retained. Every path must
cross its center, start at an outer boundary, and end at the first encounter
with either outer boundary. The local measurements are

\[
 p_i^{LR}=P(\text{exit right}\mid\text{enter left, cross }c_i),\qquad
 p_i^{RL}=P(\text{exit left}\mid\text{enter right, cross }c_i).
\]

PPTIS assumes that the arrival direction contains the relevant memory beyond
neighboring interfaces. Its direction-dependent recurrence is generally an
approximation, so multiply ordinary TIS conditionals only for full TIS/RETIS.
For local PPTIS probabilities, `pptis_crossing_probability(p_lr, p_rl)` starts
with `P_plus=P_minus=1` and applies

\[
 D_i=p_i^{LR}+(1-p_i^{LR})P_-^{\rm old},\qquad
 P_+^{\rm new}=\frac{p_i^{LR}P_+^{\rm old}}{D_i},\qquad
 P_-^{\rm new}=\frac{p_i^{RL}P_-^{\rm old}}{D_i}.
\]

The final `P_plus` estimates reaching B after the first center crossing from A.
This follows the memory reduction introduced by
[Moroni, Bolhuis and van Erp (2004)](https://doi.org/10.1063/1.1644537) and the
REPPTIS formulation of
[Vervust et al. (2023)](https://doi.org/10.1016/j.bpj.2023.02.021).
Interface-spacing and full-TIS comparisons are necessary to assess the
approximation; a numerically well-sampled local chain alone does not establish
memory loss. Missing either origin population or a singular recurrence leaves
the corresponding summary unsupported; it is not regularized into a rate.

`ReplicaExchangePPTIS` adds neighboring overlap-regrowth swaps. Independent
orientation choices are made before checking compatibility, so incompatible
directions reject rather than being removed from the proposal distribution.
The terminal overlap is retained and extended into its neighboring local
window. Canonical endpoint-energy corrections account for finite-step drift.
`swap_length_rejections` is reported separately from shooting-only local
`length_rejections`.

```python
from pyrpmd.pptis import ReplicaExchangePPTIS, combine_pptis_rate

repptis = ReplicaExchangePPTIS(
    dyn, Cartesian(0), basin_a=-1.0, basin_b=1.0,
    interfaces=[-0.8, -0.4], max_path_steps=10000, seed=325,
    identity={"potential": "my-model-and-parameters-v1", "coordinate": "centroid-x0-v1"},
)
# One genuine seed phase point inside each local window.
seeds = [repptis.generate_path(phase_point, i)
         for i, phase_point in enumerate(seed_states)]
local_result = repptis.sample_interfaces(seeds, n_samples=2000, burn_in=300)
flux = estimate_effective_flux(
    unbiased_coordinate_trace, basin_a=-1.0, basin_b=1.0,
    first_interface=-0.8, dt=dyn.dt,
)
candidate = combine_pptis_rate(local_result, flux, block_size=100)
```

Use `PPTISSampler` in the same example to disable replica exchange. The supplied
effective flux must be measured at the **first center**, here `-0.8`, with the
same basins and dynamics. That flux already includes the probability of
reaching the first center from A; multiplying another entrance factor would
count it twice. `combine_pptis_rate` validates this convention. Its
`rate_standard_error_block_conditional_on_flux_au` uses joint ladder blocks
and excludes uncertainty in the supplied flux. Blocks without both origin
populations are reported as unsupported; they are not silently discarded to
create an error bar. `memory_loss_approximation=True` and
`physical_converged=False` remain explicit in the result.

## Direct forward-flux sampling

`ForwardFluxSampler` requires an explicit `step(state, rng)` callback and needs
no reverse dynamics. The callback mutates its owned `State` or returns a new
one after exactly one step of duration `dt`. The sampler advances the clocks;
the callback may leave clocks untouched or advance them by exactly that step.
The transition law must be stationary and time-homogeneous. All Markov memory
must live in `State.q/p`, and all random draws must use the supplied generator.
Hidden solver memory, global RNGs and time-dependent protocols are unsupported.
FFS does not select a thermostat, change a device, or substitute another
physical process. The stochastic-splitting construction is reviewed by
[Allen, Valeriani and ten Wolde (2009)](https://doi.org/10.1088/0953-8984/21/46/463102).

```python
from pyrpmd.ffs import ForwardFluxSampler

ffs = ForwardFluxSampler(
    step, Cartesian(0), dt=dt, basin_a=-1.0, basin_b=1.0,
    interfaces=[-0.8, -0.4, 0.2],
    dynamics_id="complete-Markov-model-parameters-and-stepper-version",
    coordinate_id="centroid-x0-v1", flux_steps=20000,
    trials_per_interface=[1000, 1000, 1000],
    max_trial_steps=10000, seed=5821,
)
# initial_state must be prepared for the same stationary physical process.
ffs_result = ffs.run(initial_state)
report = ffs_result.summary(block_size=100)
```

The initial flux phase stores every first outward crossing of the first
interface since a visit to A and accumulates A-history time. Each interface
stage draws uniformly, with replacement, from the preceding successful bank
and runs forward until A or the next target. Every complete state is retained,
including all bead momenta; no Maxwell refresh occurs at an interface. A state
that already overshot its next target is a zero-step success. The final target
is B. Parent and original flux-crossing indices preserve the branching
genealogy.

`max_trial_steps` is a censoring bound. An unresolved trial raises
`FFSUnresolvedError`, preserving the diagnostic terminal state and rolling back
that operation's RNG and observations. It is not counted as a return to A.
Numerical errors also escape with the operation rolled back. Changing the
horizon requires a separately identified calculation. `advance(n_operations)`
allows interruption between individual flux steps or completed trials; an
FFS checkpoint does not interrupt the middle of a trial.

The rate estimate is the measured effective flux times the product of stage
success fractions. Successful descendants share earlier ancestors, so
`conditional_bank_standard_error_diagnostic` is not a full rate error.
`summary` supplies a flux ratio block-jackknife diagnostic and ancestry counts.
Use `combine_ffs_replicates(results, block_size=...)` on independently
initialized complete runs for an error on the full rate, including finite-bank
and flux uncertainty. Its blocks contain whole runs, never individual
descendants. Distinct run identities are checked; the caller must still
establish that starting states and random streams are independent.

A terminal `no_successes` run contributes an empirical zero rate, with
downstream conditional probabilities left unavailable. A `no_flux_crossings`
run also contributes an empirical zero when A-history observation time exists.
These zero observations must stay in replicate averages; excluding them would
bias the rate upward. Neither establishes a zero physical rate. A run with no
known A-history time is incomplete, and the combination helper rejects it
rather than dropping it. A single run reports no full rate error.

## Diagnostics, streaming and continuation

Each `TISResult` holds crossing indicators, acceptance flags, path lengths and
rejection reasons for every retained production sample. Only the final full
path is retained by default. Use
`sample(..., on_sample=lambda index, path, move: ...)` to stream complete
trajectories or compute mechanism observables without retaining every frame.
Callbacks receive repeated paths too; account for their multiplicity.

The summary reuses PyRPMD's autocorrelation diagnostic on the crossing
indicator and nonconstant path lengths. Its effective sample count and
standard error are diagnostics, not a convergence certificate. A constant
crossing indicator gives `standard_error_diagnostic=None`, not a falsely
precise zero error. Channel observables can mix much more slowly than either
of these diagnostics and should also be monitored.

Continue the same chain with its final path and generator state:

```python
sampler.rng.bit_generator.state = result.rng_state
continued = sampler.sample(result.final_path, result.interface_index,
                           n_samples=1000)
```

Store `final_path.q`, `final_path.p`, `final_path.values`, `final_path.dt` and
`rng_state` together with the unchanged potential, sampler and dynamics
configuration. Reconstruct `TISPath(q, p, values, dt)` on restart. A generator
state must be restored into the same NumPy bit-generator type. Saving these
objects does not establish a general cross-version checkpoint guarantee.

RETIS, PPTIS/REPPTIS and FFS additionally provide persistent checkpoints:

| Method | Save | Restore and continue |
| --- | --- | --- |
| RETIS | `retis.save_checkpoint(path)` | `retis.load_checkpoint(path)` returns the sampler; then `retis.sample_interfaces(n_samples=1000)` |
| PPTIS/REPPTIS | `repptis.save_checkpoint(path, result)` | `previous = repptis.load_checkpoint(path)` returns a result; then `repptis.sample_interfaces(n_samples=1000, resume=previous)` |
| FFS | `ffs.save_checkpoint(path)` | `ffs.load_checkpoint(path)` returns the sampler; then `ffs.advance()` or `ffs.advance(n_operations)` |

Reconstruct the sampler with matching scientific settings before loading.
RETIS and PPTIS require a nonempty JSON `identity` describing the potential
and coordinate for persistent checkpoints. Include model/version identifiers,
parameter values or parameter-file hashes, and all otherwise opaque callable
configuration. FFS requires `dynamics_id` and `coordinate_id` at construction;
the dynamics identifier must also identify its stochastic scheme, masses,
temperature, bead conventions, and all hidden-in-code parameters. Automatic
sampler metadata cannot infer these scientific facts from a Python class name.
Numerical options, interface definitions, path bounds, and RNG type are included
in the new identities. Existing workflow identities and legacy checkpoint
formats are unchanged.

The common NPZ writer writes to a temporary file, flushes it and atomically
replaces the destination. Each payload array has a checksum; loading forbids
pickle, verifies integrity and validates scientific metadata before resuming.
Saved state includes the RNG and complete retained path/crossing bank, labels
or genealogy, counters and accumulated observations. RETIS and PPTIS replay
saved final paths against the configured deterministic dynamics; FFS validates
its states, clocks, event trace and ancestry. A wrong configuration, damaged
payload or incompatible RNG fails explicitly. Checkpoints preserve exact
continuation under the same implementation, NumPy bit-generator type and
numerical backend; they do not certify interchangeability across platforms or
arbitrary future algorithm versions. Continued RETIS/PPTIS results contain
cumulative observations and must not be concatenated with their earlier prefix.

## Infinite RETIS with asynchronous workers

For RETIS, `exchange="infinite"` selects exact infinite exchanges among the
currently free positive ensembles. The central coordinator uses the
asynchronous process executor under `pyrpmd.parallel`; independent proposals
run concurrently and each ensemble is locked while its proposal is outstanding.
The minus ensemble forms a separate one-element block. A minus/zero-plus
endpoint-continuation move locks both affected ensembles.

```toml
[path_sampling]
method = "retis"
exchange = "infinite"
workers = 2
# ... basins, interfaces, seeds and the other RETIS options above ...

[parallel]
backend = "process"
mode = "serial"
scheduling = "dynamic"
trajectory_workers = 2
threads_per_rank = 1
```

The scientific algorithm follows
[Roet, Zhang and van Erp (2022)](https://doi.org/10.1021/acs.jpca.2c06004).
The later [Zhang et al. (2024) PNAS paper](https://doi.org/10.1073/pnas.2318731121)
combines asynchronous infinite exchanges with high-acceptance subtrajectory
moves. This implementation retains the existing full-Maxwell, two-way shooting
and its canonical acceptance correction. It does not implement those later
subtrajectory moves or claim their acceptance rate or speedup.

For free paths `X_i` and free positive ensembles `E_j`, the membership weight
is `W_ij = 1[X_i belongs to E_j]`. All ensembles share the same unmodified
canonical path density, so that common row factor cancels between assignments.
The exact marginal assignment probability is

\[
 P_{ij}=\frac{W_{ij}\,\operatorname{perm}(W_{-i,-j})}
                   {\operatorname{perm}(W)}.
\]

Because the interfaces are nested, each path's admissible columns form a
prefix. Sorting paths by the number of compatible free ensembles gives an
exact quadratic-cost recurrence for these marginals, without an exponential
general-purpose permanent calculation or a finite number of approximate swaps.
Tests compare the marginals and compatible assignments against independent
permutation enumeration for small ladders. Arbitrary path biases would change
these weights and are unsupported.

Each committed production event records **only free ensembles**. A free
ensemble's crossing observation is the fractional sum over its possible path
assignments, weighted by `P_ij`; its observation count increases by one.
Locked ensembles contribute neither an observation nor a count. The crossing
probability is the sum of its fractional observations divided by its own
count. The minus and zero-plus residence times use the same free-ensemble
bookkeeping. Their separate mean durations give the effective flux, and its
product with the interface conditionals gives the rate estimate. Joint
contiguous event blocks recompute these ratios using the per-ensemble counts;
a block lacking an ensemble cannot supply an error estimate. Counting locked
paths as repeated samples or taking an unweighted average across events would
change the estimator.

Proposal execution is asynchronous, while **commits follow deterministic
submission tickets**. This avoids a global batch barrier and makes the
scientific schedule independent of operating-system completion order. A fast
worker's result can wait behind an earlier slow ticket; this is a deliberate
throughput tradeoff for exact restart, and is not the first-finished policy
used in performance claims for other implementations. The logical worker
count affects which ensembles are locked and is part of the scientific
identity. It must match `parallel.trajectory_workers`; it cannot be changed
when resuming the same chain. MPI and bead/GPU worker pools are not currently
supported by this scheduler.

Checkpoints retain the coordinator RNG, logical ticket counters, ensemble
locks, immutable pending-task paths and seeds, retained paths, and weighted
observation histories. Restart resubmits pending proposals from those recorded
inputs and commits each ticket once. No checkpoint depends on serializing a
live process or accepting stale results from a prior worker. The numerical
backend, worker count and scientific settings must still match. For this mode,
`samples` and `burn_in` count committed events, rather than full-ladder sweeps;
compare effective observations and uncertainty when choosing budgets.

## Validation and scope

Tests cover real reversible path generation for one and several beads,
canonical momentum variance, fixed atoms, shooting and exchange acceptance
ratios, rejected-path multiplicity, seed replay, bounded rejection, malformed
checkpoints and exact split-run continuation. Stitched A/B-history traces check
the minus residence-time normalization independently, including direct
two-frame product crossings. Exact finite absorbing chains check the PPTIS
direction-memory recurrence, and overlap proposals are checked against their
inverse moves.

For classical `U(x)=(x*x-1)**2` with separable harmonic spectator coordinates
and basins at its two minima, equilibrium quadrature and the positive-flux
Boltzmann factor supply independent deterministic references. RETIS with
minus and REPPTIS are compared against that rate and independent full TIS
with a matching external flux. These finite-chain statistical tests allow
sampling uncertainty and, where specified, a timestep allowance; they do not
certify molecular-rate convergence or exact quantum dynamics.

FFS is independently checked on one-dimensional and nonseparable
two-dimensional finite-state double-well Metropolis processes. Their stationary
distributions and committors are solved from transition matrices, producing
exact rate and flux references for those **same stochastic processes**. An
irreversible directed cycle checks that reverse dynamics is unnecessary.
These discrete stochastic rates must not be equated to the Hamiltonian
double-well rates: changing the physical dynamics changes the kinetic problem.
The executable overdamped FFS example likewise identifies its own dynamics
and does not claim an RPMD rate.

`examples/sampling/ffs_quartic_reference.py` supplies a further continuous
quartic comparison. It starts independent canonical states, adds Maxwell
collisions only inside A/B to make the flux trajectory explore energy, and
keeps the outside-basin excursions Hamiltonian. Its in-basin dynamics differs
from the RETIS calculation. Canonical density and the same outside-basin
excursion law give a common continuous-time flux/rate reference; finite-step
stationary bias remains a separate convergence question. The script reports
whole-run errors, fixed statistical tolerances and execution timing, and does
not label this short reference check physically converged. A runnable minus
example with checkpoint/resume is `examples/sampling/retis_minus.py`.

`examples/sampling/path_sampling_quartic.py` runs all four input workflows and
reports the quadrature reference, complete-run error bars and execution timing.
The exact classical unit-mass, beta-one reference is
`k_AB = 0.1487158864662166 au^-1`. Its RETIS flux at A is
`0.4042516917842950 au^-1`; REPPTIS uses the first-center flux instead. With the
two REPPTIS centers on the uphill side of A, every right-origin local path
returns left, and this special separable model has an exact directional
recurrence. That check does not establish the general memory-loss assumption.
The workflow reference tests recompute errors from raw observations, rather
than treating branch descendants or exchanged replicas as independent runs.

All current interfaces are ordered coordinate-based boundaries. Phase-space
basin predicates, stochastic one-way TIS shooting, MPI infinite exchanges and
the 2024 high-acceptance subtrajectory moves remain outside this implementation.
Use the dedicated Python modules for the low-level APIs; no top-level
`pyrpmd` export is added. Model initialization, stationarity, mechanism mixing,
interface spacing, timestep, bead count, path-length support and independent-run
convergence remain scientific checks for each application.

## References

- T. S. van Erp, D. Moroni and P. G. Bolhuis, *A novel path sampling method for
  the calculation of rate constants*, Journal of Chemical Physics **118**,
  7762–7774 (2003),
  [doi:10.1063/1.1562614](https://doi.org/10.1063/1.1562614);
  [author preprint](https://arxiv.org/abs/cond-mat/0210614).
- T. S. van Erp and P. G. Bolhuis, *Elaborating transition interface sampling
  methods*, Journal of Computational Physics 205, 157–181 (2005),
  [doi:10.1016/j.jcp.2004.11.003](https://doi.org/10.1016/j.jcp.2004.11.003);
  [author preprint](https://arxiv.org/abs/cond-mat/0405116). Sections 3.1–3.4
  and Appendix B define the conditional probabilities, flux and path measures.
- D. W. H. Swenson et al., *OpenPathSampling: A Python Framework for Path
  Sampling Simulations. 1. Basics*, JCTC 15, 813–836 (2019),
  [doi:10.1021/acs.jctc.8b00626](https://doi.org/10.1021/acs.jctc.8b00626).
- T. S. van Erp, *Reaction rate calculation by parallel path swapping*,
  Physical Review Letters **98**, 268301 (2007),
  [doi:10.1103/PhysRevLett.98.268301](https://doi.org/10.1103/PhysRevLett.98.268301).
- D. Moroni, P. G. Bolhuis and T. S. van Erp, *Rate constants for diffusive
  processes by partial path sampling*, Journal of Chemical Physics **120**,
  4055–4065 (2004),
  [doi:10.1063/1.1644537](https://doi.org/10.1063/1.1644537);
  [author preprint](https://arxiv.org/abs/cond-mat/0310466).
- W. Vervust, D. T. Zhang, T. S. van Erp and A. Ghysels, *Path sampling with
  memory reduction and replica exchange to reach long permeation timescales*,
  Biophysical Journal **122**, 2960–2972 (2023),
  [doi:10.1016/j.bpj.2023.02.021](https://doi.org/10.1016/j.bpj.2023.02.021);
  [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10398259/).
- R. J. Allen, C. Valeriani and P. R. ten Wolde, *Forward flux sampling for rare
  event simulations*, Journal of Physics: Condensed Matter **21**, 463102 (2009),
  [doi:10.1088/0953-8984/21/46/463102](https://doi.org/10.1088/0953-8984/21/46/463102);
  [author preprint](https://arxiv.org/abs/0906.4758).
- S. Roet, D. T. Zhang and T. S. van Erp, *Exchanging Replicas with Unequal Cost,
  Infinitely and Permanently*, Journal of Physical Chemistry A **126**,
  8878–8886 (2022),
  [doi:10.1021/acs.jpca.2c06004](https://doi.org/10.1021/acs.jpca.2c06004);
  [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9720720/).
- D. T. Zhang, L. Baldauf, S. Roet, A. Lervik and T. S. van Erp,
  *Highly parallelizable path sampling with minimal rejections using
  asynchronous replica exchange and infinite swaps*, Proceedings of the
  National Academy of Sciences **121**, e2318731121 (2024),
  [doi:10.1073/pnas.2318731121](https://doi.org/10.1073/pnas.2318731121);
  [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10873605/).
- S. Falkner, A. Coretti, B. Peters, P. G. Bolhuis and C. Dellago,
  *Revisiting shooting point Monte Carlo methods for transition path sampling*,
  Journal of Chemical Physics **163**, 034105 (2025),
  [doi:10.1063/5.0261744](https://doi.org/10.1063/5.0261744);
  [author preprint, version 3](https://arxiv.org/html/2408.03054v3), especially
  Section II.5 on fresh uniform shooting indices and the path-length ratio.

The implementation is independent; no source from these projects is copied.
