# RPMD rate inputs

Start a rate study with [`research.inp`](research.inp). It explicitly declares the reactant
fragments, product partition and forming/breaking bonds, then independently requests a TS
validation and a downhill channel check. Its regional table gives 16 entrance windows and
26 barrier windows with different force constants. Umbrella sampling, constrained parents
and recrossing children have separate settings.

```bash
pyrpmd validate examples/rate/research.inp --check-pes
pyrpmd prepare examples/rate/research.inp --pilot --output preparation.json
pyrpmd run examples/rate/research.inp --output runs/rate
pyrpmd analyze runs/rate
pyrpmd resume runs/rate
```

The new `force_constant` defaults to the RPMDrate temperature-normalized coefficient `k/T`:
`0.00016` at 1000 K means a physical spring of 0.16 hartree for dimensionless xi. Explicit
physical units remain supported. See the complete [input guide](../../docs/rate-input-v2.md)
([中文](../../docs/zh/rate-input-v2.md)) for the regional boundaries, units and migration rules.

`minimal.inp` explicitly selects `reaction.definition="infer"` as a convenience mode.
It uses the unstable TS mode and two downhill traces to infer the channel. `pilot.inp`
retains the old schema-1 syntax for compatibility. In the new research layout,
`ts_action="validate"` checks coordinates without changing them; use `"refine"` when
P-RFO refinement is intended. Defining a reaction never substitutes for a requested TS check.

The separate legacy endpoint example first finds a saddle with CI-NEB and refines it with P-RFO:

```bash
python -m pyrpmd prepare examples/rate/from_endpoints.inp --pilot --output preparation.json
python -m pyrpmd run examples/rate/from_endpoints.inp --output runs/rate-from-endpoints
```

`from_endpoints.inp` uses `reactant.xyz` and `product.xyz`, which contain the complete system in
the same atom order and Cartesian frame. CI-NEB keeps these endpoints fixed. See
[transition-state search](../../docs/transition-state-search.md) for refinement controls, path
initialization, units, connectivity-radius overrides, and limitations.

After successful preparation, `run` performs umbrella sampling, the flux-metric PMF,
constrained parent chains, paired recrossing trajectories and parent-cluster/block uncertainty
estimates. The preparation report separates `saddle_valid`, `channel_consistent` and
`refinement_performed`, and previews the resolved window and recrossing budgets.

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
