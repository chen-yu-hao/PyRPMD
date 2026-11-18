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

Start a rate study with [`research.inp`](research.inp). It uses `force_constant=0.1` (the RPMDrate H + H2 example coefficient) with 0.005 window spacing; the resulting dense grid is costly and must be convergence-checked. It explicitly declares the reactant
fragments, product partition and forming/breaking bonds, then independently requests a TS
validation and a downhill channel check. Its regional table gives 160 entrance windows and
51 barrier windows with the same normalized force constant. Umbrella sampling, constrained parents
and recrossing children have separate settings.

The example leaves `rate.dividing_surface` unset so the supported PMF maximum
selects `xi*` for constrained parents and recrossing rather than fixing it to 1.0.

```bash
pyrpmd validate examples/rate/research.inp --check-pes
pyrpmd prepare examples/rate/research.inp --pilot --output preparation.json
pyrpmd run examples/rate/research.inp --output runs/rate
pyrpmd analyze runs/rate
pyrpmd resume runs/rate
```

The new `force_constant` defaults to the RPMDrate temperature-normalized coefficient `k/T`:
`0.1` at 1000 K means a physical spring of 100 hartree for dimensionless xi. This follows the RPMDrate H + H2 example, but is not a universal convergence recommendation. Explicit
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

## Subspace quantum rate correction

[`subspace.inp`](subspace.inp) adds `[rate.subspace_correction]` to the schema-2 rate
layout in a low-temperature tunnelling regime ($T = 200\,\mathrm{K} \approx 0.55\,T_c$,
$P = 24$, `toy_exchange.py` tuned to a $7.5\,\mathrm{kcal/mol}$ barrier with
$T_c \approx 363\,\mathrm{K}$) so that quantum corner cutting and tunnelling yield a
substantial correction factor ($C_S(\mathrm{2D}) \approx 3$, versus $C_S \approx 1$ near the
classical limit) while still completing a full pilot run in under a minute on a multi-core
machine:

```bash
# 1. Validate the input and analytic teaching PES:
pyrpmd validate examples/rate/subspace.inp --check-pes

# 2. Run preparation, umbrella PMF, recrossing, and 1D + 2D RPH subspace correction:
pyrpmd run examples/rate/subspace.inp --output runs/rate-with-subspace

# 3. Inspect the saved summary (including subspace_correction factors and corrected estimate):
pyrpmd analyze runs/rate-with-subspace

# 4. Re-run ONLY the subspace stage after editing [rate.subspace_correction.*]
#    (for example refining quantum.spacing_2d, changing rpmd.sweeps, or toggling anchor):
pyrpmd run examples/rate/subspace.inp --output runs/rate-with-subspace --stage subspace

# 5. Or postprocess any saved rate run standalone via a TOML specification:
pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K.toml
```

### Parallelism, low-temperature DVR truncation, and stage artifacts

- **Multi-core execution (`[parallel]` and `[rate.subspace_correction.execution]`):**
  `[parallel]` runs umbrella and recrossing trajectories across worker processes, and
  `[rate.subspace_correction.execution] cpus = "auto"` reuses that CPU allocation to overlap
  the 1D/2D quantum Sylvester and compiled C++/OpenMP subspace RPMD tasks.
- **Low-temperature DVR wall margin (`quantum.truncation_min`):** At $200\,\mathrm{K}$,
  $30\,k_B T \approx 0.019\,\mathrm{E_h}$ is smaller than two quanta of a stiff stretch mode,
  so energy-truncated 2D DVR walls would artificially confine transverse zero-point motion.
  Set `quantum.truncation_min` (bare $\mathrm{E_h}$) to about six quanta of the stiffest
  transverse mode (`0.08` $\mathrm{E_h}$ here) and size `quantum.dense_limit` from
  `solver_plans` in `runs/rate-with-subspace/subspace/status.json`.
- **Inspecting full-run vs `--stage subspace` outputs:** `pyrpmd analyze runs/rate-with-subspace`
  prints `results/summary.json` from the full `pyrpmd run` (including `rate_estimate_cm3_s`
  and its `subspace_correction` summary). A subsequent `--stage subspace` recomputation never
  mutates `summary.json` or `results/summary.json`; it prints the updated `subspace_correction`
  JSON to stdout and writes authoritative postprocessing artifacts under
  `runs/rate-with-subspace/subspace/` (`status.json`, `subspace_pipeline.json`, `C_S_1d.json`,
  `C_S_2d.json`, and `rate_quantum_corrected.json`).
- **Umbrella-free $C_S$ from TS + PES only:** If you only want $C_S$ and the finite-bead
  factor $B_P$ without running umbrella windows first, see
  [`examples/subspace_correction/molecular_ts_to_cs.py`](../subspace_correction/molecular_ts_to_cs.py)
  and [`examples/subspace_correction/README.md`](../subspace_correction/README.md).

All dimensional numerical controls under `[rate.subspace_correction.*]` take bare atomic
units (`bohr * sqrt(m_e)` for `step`, `spacing_1d`, `spacing_2d`, `padding` and
`dvr_half_width`; `Eh` for `truncation_min` and `soft_mode_frequency`; `hbar / Eh` for
`hmc_dt`, `md_dt` and `md_time`; `bohr` for `hessian_displacement`; `K` for
`crossover_temperature`; unit strings are rejected). See
the full user guide in [`docs/subspace-correction.md`](../../docs/subspace-correction.md)
([中文指南](../../docs/zh/subspace-correction.md)).
