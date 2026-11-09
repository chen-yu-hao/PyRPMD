# RPMD rate inputs

## Physical BKMP2 H + H₂ exchange

For a **real BKMP2 H + H₂ rate calculation**, use
[`bkmp2_pilot.inp`](bkmp2_pilot.inp) for a functional smoke test or
[`bkmp2_production.inp`](bkmp2_production.inp) for a production-scale starting
input. [`BKMP2_README.md`](BKMP2_README.md) gives the compiler, pinned PES
source, pre-run numerical verification, command sequence, and **mandatory
convergence** checks. Both inputs use the existing validated physical BKMP2
implementation in `benchmarks/h3/pes.py`, with two equivalent exchange
channels and explicit H masses; neither automatically certifies a converged rate.

## Analytic teaching and workflow regression inputs

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

## Choose the PMF method

Use `[rate.pmf]` to select a sampler and its compatible estimator, with PMF-specific
`dynamics`, `sampling` and `analysis` tables. The root dynamics still control recrossing.
The umbrella sampler records and checkpoints trajectories; the **rate analysis**
then selects `estimator="ui"` or `"mbar"` and performs its own paired-endpoint
bootstrap. Do not configure a generic `analysis.method="mbar"` to request UI.
UI retains MBAR overlap/connectivity checks without running a separate FES
bootstrap. `umbrella/fes.*` exports are point-estimate diagnostics; the selected
curve and uncertainty live in `rate_pmf.*`. The complete short pilots are:

| Input | PMF route |
| --- | --- |
| [`pmf_umbrella.inp`](pmf_umbrella.inp) | Umbrella + **UI** (Gaussian integration), with PMF PILE-L and a separate timestep |
| [`pmf_umbrella_mbar.inp`](pmf_umbrella_mbar.inp) | Same umbrella setup with explicit **MBAR** rate estimation and optional UI comparison |
| [`pmf_metadynamics.inp`](pmf_metadynamics.inp) | Well-tempered MetaD, followed by frozen-bias relaxation and reweighting |
| [`pmf_opes.inp`](pmf_opes.inp) | OPES, followed by frozen-bias relaxation and reweighting |

```bash
pyrpmd validate examples/rate/pmf_umbrella.inp --check-pes
pyrpmd validate examples/rate/pmf_umbrella_mbar.inp --check-pes
pyrpmd validate examples/rate/pmf_metadynamics.inp --check-pes
pyrpmd validate examples/rate/pmf_opes.inp --check-pes
pyrpmd run examples/rate/pmf_umbrella.inp --output runs/rate-ui
```

`minimal.inp` deliberately exercises the implicit umbrella + UI defaults.
`research.inp` shows canonical `[rate.pmf]` estimator selection while retaining
regional `[umbrella]` windows. MetaD and OPES use their frozen-bias
`reweighted_histogram` rate estimator, not UI/MBAR. See
[`pmf_umbrella_mbar.inp`](pmf_umbrella_mbar.inp) for a direct estimator comparison.

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
