# RPMD rate from endpoints or a transition state

Start with reactant/product structures or a TS guess and an analytic potential, then run the
complete RPMDrate-style Bennett–Chandler calculation. The endpoint example first finds a saddle
with CI-NEB and refines it with P-RFO:

```bash
python -m pyrpmd prepare examples/rate/from_endpoints.inp --pilot --output preparation.json
python -m pyrpmd run examples/rate/from_endpoints.inp --output runs/rate-from-endpoints
```

`from_endpoints.inp` uses `reactant.xyz` and `product.xyz`, which contain the complete system in
the same atom order and Cartesian frame. CI-NEB keeps these endpoints fixed. See
[transition-state search](../../docs/transition-state-search.md) for refinement controls, path
initialization, units, connectivity-radius overrides, and limitations.

The existing TS example remains available:

```bash
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/rate
```

`prepare --pilot` evaluates the PES, refines an imperfect TS when needed, forms a finite-difference
mass-weighted Hessian, follows its unique unstable mode in both directions and identifies the
equivalent A + BC / AB + C channels. After successful preparation, `run`
performs umbrella sampling, the flux-metric PMF, constrained parent chains, paired recrossing
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
