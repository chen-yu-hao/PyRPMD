# Transition-interface path sampling

`pyrpmd.transition_interface` provides canonical two-way transition-interface
sampling (TIS) for the existing deterministic `Dynamics` engine, with either one
classical bead or a real-mass ring polymer. The Python API accepts an existing
scalar collective variable or a callable on the full `q[P,N,3]` array; it does
not need coordinate derivatives. An executable model example is
`examples/sampling/transition_interface.py`.

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
No new `.inp` task is introduced in this version.

A seed must begin with exactly one frame in A, end with exactly one frame in
A or B, have all intermediate frames outside both basins, and cross its
ensemble's interface. `sample` replays a supplied seed with the configured
integrator to reject interpolated, biased, stale or otherwise incompatible
trajectories. `generate_path` is an initialization helper; its successful
outputs alone are **not** equilibrium path samples. Discard burn-in and check
independent chains, including chains initialized in different mechanisms.

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
`Dynamics` and its supported real-time integrators are accepted; subclasses,
patched stepping methods and declared history-dependent potentials are
rejected. An arbitrary user potential must be a pure, time-independent energy
and force function. Timestep convergence of the paths is still required.

The current path is counted **after every attempt, including rejections**.
Dropping repeated paths or accepting every valid variable-length proposal
would bias the path ensemble. The implementation uses fresh shooting indices;
it is not the history-based “aimless shooting” algorithm.

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

The crossing probabilities do not supply an absolute rate by themselves. An
independently measured effective positive flux is needed:

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

## Validation and scope

Tests cover real reversible path generation for one and several beads,
canonical momentum variance, fixed atoms, the selectable-frame length ratio,
canonical endpoint/proposal weights, rejected-path multiplicity, seed replay,
bounded rejection, reproducible continuation and effective-flux normalization.
A separable classical double-well test compares an actually sampled conditional
probability against its analytical positive-flux Boltzmann factor. This model
check is not a molecular-rate validation or proof of quantum accuracy. TIS
samples the dynamics supplied to it; using RPMD does not remove RPMD's own
physical approximations.

This first implementation uses two-way shooting and coordinate-defined basins.
It does not yet implement replica-exchange TIS, minus ensembles, stochastic
one-way shooting, phase-space basin predicates or an automated flux run.

## References

- T. S. van Erp and P. G. Bolhuis, *Elaborating transition interface sampling
  methods*, Journal of Computational Physics 205, 157–181 (2005),
  [doi:10.1016/j.jcp.2004.11.003](https://doi.org/10.1016/j.jcp.2004.11.003);
  [author preprint](https://arxiv.org/abs/cond-mat/0405116). Sections 3.1–3.4
  and Appendix B define the conditional probabilities, flux and path measures.
- D. W. H. Swenson et al., *OpenPathSampling: A Python Framework for Path
  Sampling Simulations. 1. Basics*, JCTC 15, 813–836 (2019),
  [doi:10.1021/acs.jctc.8b00626](https://doi.org/10.1021/acs.jctc.8b00626).
- S. Falkner et al., *Revisiting Shooting Point Monte Carlo Methods for
  Transition Path Sampling*,
  [author preprint, version 3](https://arxiv.org/html/2408.03054v3), especially
  Section II.5 on fresh uniform shooting indices and the path-length ratio.

The implementation is independent; no source from these projects is copied.
