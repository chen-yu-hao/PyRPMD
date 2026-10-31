# RPMD rate from a transition state

Starting from nothing but a TS structure and an analytic potential, this example runs the
complete RPMDrate-style Bennett–Chandler calculation in about 15 seconds.

```bash
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/rate
```

`prepare` evaluates the PES, forms a finite-difference mass-weighted Hessian, follows its unique
unstable mode in both directions and identifies the equivalent A + BC / AB + C channels. `run`
then performs umbrella sampling, the flux-metric PMF, constrained parent chains, paired recrossing
trajectories and parent-cluster/block uncertainty estimates. `minimal.inp` shows the smallest
valid input: the TS structure and the PES selection.

> **Scope.** `toy_exchange.py` is an analytic teaching potential with Morse reactant asymptotes.
> The H labels supply convenient isotope masses and connectivity radii; these are not fitted H3
> parameters or a physical H + H₂ benchmark (see [`benchmarks/h3`](../../benchmarks/h3/README.md)
> for that). Pilot durations are short by design; converging beads, time step, Rinf, sampling and
> the plateau is the next step for a real system, and the output reports which checks were made.

## How the rate is defined

The rate follows RPMDrate: umbrella forces include the flux metric, a constrained canonical
parent samples the dividing surface, and unconstrained Maxwell momenta generate paired ±p
children. An unsuccessful child trajectory is never replaced by an assumed transmission.

| Summary field | Meaning |
| --- | --- |
| `rate_estimate_cm3_s` | Finite-time estimate |
| `rate_cm3_s` | The rate, reported when window overlap, the time-plateau check and the endpoint energy-drift check all pass |
