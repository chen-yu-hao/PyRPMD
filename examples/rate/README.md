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
obtains the flux-metric PMF, then runs constrained parent chains and paired recrossing
trajectories. The examples above use the default umbrella route and its paired block
uncertainty estimates. `minimal.inp` shows the smallest valid input: the TS structure
and the PES selection.

## Choose the PMF method

Use `[rate.pmf]` to select a sampler and its compatible estimator, with PMF-specific
`dynamics`, `sampling` and `analysis` tables. The root dynamics still control recrossing.
The complete short pilots are:

| Input | PMF route |
| --- | --- |
| [`pmf_umbrella.inp`](pmf_umbrella.inp) | Umbrella + MBAR, with PMF PILE-L and a separate timestep |
| [`pmf_metadynamics.inp`](pmf_metadynamics.inp) | Well-tempered MetaD, followed by frozen-bias relaxation and reweighting |
| [`pmf_opes.inp`](pmf_opes.inp) | OPES, followed by frozen-bias relaxation and reweighting |

```bash
pyrpmd validate examples/rate/pmf_metadynamics.inp --check-pes
pyrpmd run examples/rate/pmf_metadynamics.inp --output runs/rate-meta
```

See [PMF selection and phase budgets](../../docs/rate-pmf-selection.md) /
[中文指南](../../docs/zh/rate-pmf-selection.md) for the compatibility table, external
PMF sources, inheritance rules and uncertainty limits. These short pilots can correctly
report unsupported PMF bins; they do not establish a converged rate. Existing inputs
without `[rate.pmf]` retain the original umbrella workflow.

> **Scope.** `toy_exchange.py` is an analytic teaching potential with Morse reactant asymptotes.
> The H labels supply convenient isotope masses and connectivity radii; these are not fitted H3
> parameters or a physical H + H₂ benchmark (see [`benchmarks/h3`](../../benchmarks/h3/README.md)
> for that). Pilot durations are short by design; converging beads, time step, Rinf, sampling and
> the plateau is the next step for a real system, and the output reports which checks were made.

## How the rate is defined

The rate follows RPMDrate: PMF sampling forces include the flux metric, a constrained canonical
parent samples the dividing surface, and unconstrained Maxwell momenta generate paired ±p
children. An unsuccessful child trajectory is never replaced by an assumed transmission.

| Summary field | Meaning |
| --- | --- |
| `rate_estimate_cm3_s` | Finite-time estimate |
| `rate_cm3_s` | The rate, reported when PMF support, the time-plateau check and the endpoint energy-drift check all pass |
