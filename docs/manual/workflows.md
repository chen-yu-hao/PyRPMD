# Gas-phase reactions, free energies and enhanced sampling

This chapter covers `rate`, `drpmd`, `fes`, `network` and `metadynamics`. Run the commands from the
repository root and give each calculation a new output directory. Periodic materials and fixed
surfaces are in [Materials and surfaces](periodic.md), local electrochemistry in [PCET](pcet.md),
and units and normalization in [Scientific concepts](science.md).

## Gas-phase RPMD rate: `rate`

For two distant reactant fragments and a transition-state reference structure, `rate` runs the
Bennett–Chandler procedure: it prepares the reaction coordinate, samples an umbrella PMF, generates
constrained parent configurations and launches pairs of unbiased recrossing children. The rate is
$k_{\mathrm{RPMD}}=k_{\mathrm{QTST}}\kappa$.

### Run the example

```bash
pyrpmd validate examples/rate/pilot.inp --check-pes
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/manual-rate
pyrpmd analyze runs/manual-rate
pyrpmd resume runs/manual-rate
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/rate/pilot.inp) uses the
analytic potential `toy_exchange.py`, starts from `ts.xyz`, and uses 4 beads with short sampling so
it finishes in about 15 s. `prepare --pilot` runs the numerical preparation.

Starting from a TS alone, the preparation checks the forces, finds the unstable mode of a numerical
Hessian and descends in both directions to identify the two entrance fragments. When the channel is
ambiguous, give `reaction.fragments`, `forming_bonds` and `breaking_bonds` explicitly.

### Setting up your own system

| Input part | Determines | Check |
| --- | --- | --- |
| `system.transition_state` | TS reference geometry and atom order | Consistency with the PES, isotopes and bonds formed/broken |
| `reaction.fragments`, bonds | The entrance fragments and the reaction coordinate | 0-based indices, atom conservation, channel definition |
| `reaction.rinf` | Asymptotic normalization distance | Fragment decoupling and long-range potential as it varies |
| `umbrella` | Window centers, range, stiffness | Connected support from the reactant reference to the dividing surface |
| `sampling` | Equilibration, production, save interval and replicas per window | Distribution shapes, effective sample size, independent repeats |
| `rate.pmf_method` | `ui` or `mbar` | UI's Gaussian-window assumption; MBAR bins and support |
| `recrossing` | Parent count, sampling interval, child pairs, propagation time | Independent parents, a stable plateau, energy errors |

The rate PMF uses the `rate_flux` measure, which includes the mass-weighted flux metric; a generic
`fes` uses `cv_density`. `rate.pmf_method="mbar"` switches the PMF estimator within the same
measure. Set a fixed dividing surface with `recrossing.dividing_surface`; otherwise the workflow
places it at the PMF maximum within `rate.pmf_search_range` (default: the umbrella-center range).

Parents are sampled with a constraint on the dividing surface. Each child pair draws unconstrained
Maxwell momenta and launches with `+p` and `-p`, without thermostat or umbrella.

### Output

| File / summary field | Meaning |
| --- | --- |
| `preparation.json` | TS preparation: Hessian eigenvalues, descent endpoints, identified reaction and diagnostics |
| `rate_pmf.csv`, `rate_pmf.npz` | Rate PMF, support region and related arrays |
| `rate_pmf.json` | PMF barrier and error, $k_{\mathrm{QTST}}$, dividing surface `xi_star` and diagnostics |
| `recrossing.csv`, `recrossing.npz` | Full $\kappa(t)$ curve, errors and bootstrap information |
| `umbrella/` | Per-window sampling, checkpoints and intermediate analysis |
| `results/summary.json` | Run summary, warnings and diagnostics |
| `kappa`, `plateau_detected` | Final transmission coefficient and whether a plateau was found |
| `rate_estimate_cm3_s` | Finite-time rate estimate |
| `rate_cm3_s` | The rate, when the plateau and energy diagnostics pass; otherwise `null` |
| `rate_standard_error_cm3_s`, `rate_percentile_95_ci_cm3_s` | Joint sampling error and percentile interval from the PMF and recrossing |
| `conditional_rate_standard_error_cm3_s` | Recrossing error with the QTST factor held fixed |
| `rate_uncertainty` | Error availability, valid bootstrap counts and qualifying notes |

Rates are in `cm^3 molecule^-1 s^-1`. If the PMF overlap is disconnected or the dividing surface
lacks support, add windows or sampling before reading a rate.

The mathematics is in the
[rate conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md) *(Chinese)* and
the [UI / MBAR notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md) *(Chinese)*.
The [H₃/BKMP2 benchmark](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md) *(partly Chinese)*
runs the full workflow on a published reactive potential.

## Direct collisions: `drpmd`

`drpmd` samples parent chains of two reactants and launches independent collisions from them, then
classifies the products. The launch distribution decides whether the result is a thermal rate or a
fixed-energy cross section. For collisions from a selected incoming diabatic electronic state on a
coupled matrix potential, use the experimental [NRPMD extension](nonadiabatic.md).

```bash
pyrpmd validate examples/drpmd/free_scattering.inp --check-pes
pyrpmd run examples/drpmd/free_scattering.inp --output runs/manual-scattering
pyrpmd analyze runs/manual-scattering
pyrpmd resume runs/manual-scattering
```

The [free-scattering example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/free_scattering.inp)
uses `V=0`, so every collision is nonreactive; it exercises launching, propagation, completion and
restart.

`collision.trajectories` is the total number of collisions across all parent chains; the example
splits five collisions over two parents as 3+2. `sampling.equilibration` is the first
thermalization of each parent, and `sampling.interval` is how far a parent advances between two
launches. Set `sampling.parents` explicitly to keep the task identity when the number of worker
processes changes.

| `collision.sampling` | Launch distribution | Result |
| --- | --- | --- |
| `thermal_flux` | Relative energy $E\sim\mathrm{Gamma}(2,k_BT)$ | Thermal flux-averaged rate |
| `maxwell` | Maxwell distribution of relative velocities | Thermal rate weighted by each trajectory's speed |
| `fixed_energy` | Explicit `collision.energy` | Cross section at that collision energy |

`system.separation` is the initial radial COM distance R0, including the impact-parameter
contribution; `collision.bmax` must be smaller than R0. Impact parameters are sampled uniformly in
$b^2$. Converge R0, bmax and the maximum propagation time separately.

The built-in connectivity classifier uses bond cutoffs, a separation condition and a persistence
time. Isomers with the same formula and exchange channels between identical elements need a custom
classifier through `outcomes.factory`. Without explicit rules, PyRPMD infers cutoffs from covalent
radii, labels the outcomes as candidates and withholds chemical rates and cross sections.

In `results/summary.json`, read `counts`, `all_trajectories_resolved` and `classification`, then
`total_reaction` and `channels`. Channel thermal rates are in `rate_cm3_molecule_s`, fixed-energy
cross sections in `cross_section_bohr2`. Unresolved and failed trajectories are counted separately
and give observed lower and possible upper bounds. `parent_cluster_bootstrap95` accounts for the
correlation between trajectories from one parent.

`results/channels.csv` lists each channel's count, point estimate and bounds. The run directory also
keeps `states/initial-*.npz`, `trajectories/` and checkpoints, so every trajectory's initial state
and classification can be inspected. See the
[dRPMD example notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/README.md).

Without a `[collision]` section, a pilot runs 4 `thermal_flux` trajectories with
`max_time="100 fs"` and `bmax=R0/4` and reports no thermal rate; see the defaults in
[Input](input.md#defaults).

## One- and two-dimensional free energies: `fes`

`fes` runs umbrella sampling on one or two CVs and unbiases the samples with MBAR, giving the
probability-density free energy of those coordinates.

```bash
pyrpmd validate examples/fes/harmonic2d.inp --check-pes
pyrpmd run examples/fes/harmonic2d.inp --output runs/manual-fes
pyrpmd analyze runs/manual-fes --recompute --bins 20 20 --bootstrap 50
```

The [2D harmonic input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic2d.inp)
defines centroid x/y CVs, a 3×3 umbrella grid and two replicas per window. `umbrella.stiffness` is a
scalar, a diagonal array or a full symmetric positive-definite matrix. Place windows over the whole
region you want to analyze.

Results are in `fes.csv`, `fes.npz` and `fes.h5`, with `free_energy`, `probability_mass`, `density`,
bin edges, counts and `mask`. Check that the overlap is connected and look at the effective sample
size of each time series and the weighted ESS. Cells without support are masked (NaN), and
`overlap_connected=false` means some groups of windows share no common normalization.

`analyze --recompute` rebuilds the analysis from the saved window checkpoints with new bins, count
thresholds or bootstrap settings, without calling the PES.

### PILE-L and replica-exchange umbrellas

FES keeps Andersen as its default thermostat. Set `sampling.thermostat="pile-l"`
to use the existing BCOCB/BAOAB integrator. For fixed windows, add
`[replica_exchange]`, `enabled=true`, and `interval_steps=50` to exchange
complete ring configurations and momenta at the same temperature.

Here, `sampling.replicas` counts independent exchange ensembles. Windows in
one ensemble propagate sequentially, and analysis resamples their time blocks
synchronously. Checkpoints save and restore the whole ensemble together.
Exchange trajectories do not represent physical reaction times. Adaptive
window insertion and the `rate_flux` measure cannot be combined with REUS.
See [replica exchange](../replica-exchange.md) for the accepted input and
method limits.

```bash
pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/manual-reus
pyrpmd resume runs/manual-reus
```

### Adaptive umbrella sampling

The adaptive workflow inserts windows and lengthens pilots according to the pilot overlap and
effective sample sizes, then freezes the windows and runs production on fresh independent streams:

```bash
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/manual-adaptive
pyrpmd resume runs/manual-adaptive
pyrpmd analyze runs/manual-adaptive --recompute --bins 30 --bootstrap 20
```

`adaptive-plan.json` records each round's decisions, windows, seeds and checksums. Production uses
only its own samples. If `max_rounds` or `max_windows` runs out before the targets are met, the task
stops without starting production and records `failure_reason` in the plan.

Adaptive umbrellas work for `fes`, surface `fes` and the `rate` PMF, and refine windows inside the
declared domain; `surface_rate` and `site_rate` use fixed windows. Parameters are in the
[enhanced-sampling specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md).

## Several entrances and TSs: `network`

`task="network"` runs several gas-phase bimolecular rate problems as one restartable calculation.
Each path runs its own PMF and recrossing and keeps its own entrance normalization, PES, TS geometry
and errors.

```bash
pyrpmd prepare examples/network/two_entrances.py --pilot
pyrpmd run examples/network/two_entrances.py --output runs/manual-network
pyrpmd analyze runs/manual-network
pyrpmd resume runs/manual-network
```

The [example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/network/two_entrances.py) has
two entrances with two analytic TS/PES each. All paths share the ordered elements, isotope masses,
charge and multiplicity. Each entrance can set its own fragments and `rinf`; each TS its own
structure, bonds formed and broken, PES and symmetry factor. Different stoichiometries or isotope
inventories go in separate tasks.

The PES is inherited global → entrance → TS, and a more specific `potential` table replaces the one
above it entirely. Sampling and rate tables can be overridden at each level. `parallel` and block
settings belong to the shared configuration.

Results are indexed as `matrix[entrance][channel]["transition_states"][ts_id]`. Each path runs in
`paths/<entrance-id>/<ts-id>/`, and `network-progress.json` records the finished paths; a restart
reuses them and continues the rest.

Fluxes through different dividing surfaces can overlap, so PyRPMD reports each TS separately
(`aggregation="separate"`) and leaves `total_rate_cm3_s` and `summed_rate_cm3_s` as `null`. See the
[reaction network definition](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/reaction-networks.md).

## Metadynamics

`task="metadynamics"` supports standard and well-tempered modes on 1D/2D CVs, with several replicas
that each build their own bias history. CVs may declare a period. The workflow runs on molecular
systems without PBC or fixed atoms.

```bash
pyrpmd validate examples/metadynamics/well_tempered_2d.inp
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/manual-meta
pyrpmd analyze runs/manual-meta --recompute --bins 30 30
pyrpmd resume runs/manual-meta
```

| Parameter | Meaning |
| --- | --- |
| `metadynamics.cvs` | Names of one or two declared CVs |
| `mode` | `standard` or `well_tempered` |
| `sigma` | Gaussian width per CV, with units |
| `height` | Hill height |
| `pace` | Completed production steps between depositions |
| `bias_factor` | $\gamma>1$ in WT mode; omitted in standard mode |
| `analysis.method` | `bias_estimate` |
| `analysis.bootstrap` | 0 |

Hills start with production: the first is deposited after step `pace` completes, and all force
evaluations within a step see the same history. On restart, the bias history in
`checkpoints/metadynamics_XXXX.npz` is authoritative; `replicas/XXXX/HILLS.csv` is rebuilt from it.

The free-energy estimate is $F\approx -V+C$ in standard mode and $F\approx-\gamma V/(\gamma-1)+C$ in
WT mode. `fes.npz` / `fes.csv` hold the finite-time estimate and the support mask. Replicas are
aligned on their common support and compared there; `replica_spread` measures the spread between
replicas.

The history-dependent bias runs in a Python time-step loop with compiled substeps, so
`loop_backend="native"` is rejected; the cost grows with the number of hills. Metadynamics outputs
free-energy estimates only. Methods and restart timing are in the
[enhanced-sampling specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md).
