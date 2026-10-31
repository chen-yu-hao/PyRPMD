# Gas-phase reactions, free energies and enhanced sampling

This chapter introduces `rate`, `drpmd`, `fes`, `network` and `metadynamics` task by task. Run all
commands from the root of a repository where PyRPMD is installed, and use a new output directory
for every new calculation. The examples demonstrate the workflow with analytic or simplified
models; before switching to a real chemical system, define the potential, the reaction and the
convergence parameters again.

For periodic materials and fixed surfaces go to [Periodic tasks](periodic.md); for local
electrochemical models go to [PCET](pcet.md). The units, statistical ensembles and normalization of
each method are in [Scientific concepts](science.md).

## Gas-phase RPMD rate: `rate`

When the problem has two well-defined distant reactant fragments and a transition-state reference
structure, `rate` runs the Bennett–Chandler procedure: prepare the reaction coordinate, sample an
umbrella PMF, generate constrained parent configurations, then launch pairs of unbiased recrossing
children. The result combines as $k_{\mathrm{RPMD}}=k_{\mathrm{QTST}}\kappa$.

### Run the complete teaching example first

```bash
pyrpmd validate examples/rate/pilot.inp --check-pes
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/manual-rate
pyrpmd analyze runs/manual-rate
pyrpmd resume runs/manual-rate
```

The [complete input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/rate/pilot.inp) uses
the analytic potential `toy_exchange.py` and starts the preparation from `ts.xyz`. Its 4 beads,
short thermalization, short PMF and short recrossing are meant for learning the file organization;
they do not represent a real H + H₂ rate. `prepare --pilot` performs numerical preparation rather
than just printing the input.

When only a TS is given, the preparation checks the forces, the unstable mode of a numerical
Hessian and the connectivity after descending in both directions. It continues only when the
entrance is uniquely identified; if channels or modes are ambiguous, give `reaction.fragments`,
`forming_bonds` and `breaking_bonds` explicitly. This step is a preparation tool, not a full IRC
replacement from a mature quantum-chemistry program.

### From a trial run to your system

| Input part | What it determines | What to check |
| --- | --- | --- |
| `system.transition_state` | TS reference geometry and atom order | Consistency with the PES, isotopes and bonds formed/broken |
| `reaction.fragments`, bonds | The two entrance fragments and the reaction coordinate | 0-based indices, atom conservation, channel definition |
| `reaction.rinf` | Asymptotic normalization distance | Vary it; check fragment decoupling and long-range potential |
| `umbrella` | Window centers, range, stiffness | Connected support from the reactant reference to the dividing surface |
| `sampling` | Equilibration, production, save interval and replicas per window | Distribution shapes, effective sample size, independent repeats |
| `rate.pmf_method` | `ui` or `mbar` | UI's Gaussian window assumption; MBAR bins and support |
| `recrossing` | Parent count, sampling interval, child pairs, propagation time | Independent parents, a stable plateau, energy errors |

The gas-phase rate uses the `rate_flux` measure, which contains the mass-weighted flux metric. It
differs from the `cv_density` of an ordinary `fes`. `rate.pmf_method="mbar"` changes the PMF
estimator under the same rate measure; it does not plug an arbitrary distance FES into the
bimolecular formula. A fixed dividing surface is set with `recrossing.dividing_surface`; otherwise
the workflow selects it from the PMF, and bin width and peak-selection bias must still be
considered.

Parents are sampled under a constraint on the dividing surface. Each child pair redraws
unconstrained Maxwell momenta and launches with `+p` and `-p`; children have no thermostat or
umbrella. Projecting child momenta onto the tangent space would cancel the initial flux, so do not
copy the parents' momentum constraint.

### Reading the PMF, κ and the rate

| File / summary field | Meaning |
| --- | --- |
| `preparation.json` | TS preparation: Hessian eigenvalues, descent endpoints, identified reaction definition and diagnostics |
| `rate_pmf.csv`, `rate_pmf.npz` | Dedicated rate PMF, support region and related arrays |
| `rate_pmf.json` | PMF barrier and its error, $k_{\mathrm{QTST}}$, chosen dividing surface `xi_star` and diagnostics |
| `recrossing.csv`, `recrossing.npz` | Full $\kappa(t)$ curve, errors and bootstrap information |
| `umbrella/` | Per-window sampling, checkpoints and intermediate analysis |
| `results/summary.json` | Public run summary, warnings and diagnostics |
| `kappa`, `plateau_detected` | Final transmission coefficient and whether a plateau was detected |
| `rate_estimate_cm3_s` | Finite-time candidate estimate for review |
| `rate_cm3_s` | Filled in when the current plateau and energy diagnostics pass, otherwise `null` |
| `rate_standard_error_cm3_s`, `rate_percentile_95_ci_cm3_s` | Joint sampling error and percentile interval when both PMF and recrossing uncertainties are available |
| `conditional_rate_standard_error_cm3_s` | Recrossing error conditional on a fixed QTST factor |
| `rate_uncertainty` | Whether errors are available, valid bootstrap counts and qualifying reasons |

Rates are in `cm^3 molecule^-1 s^-1`. A detected plateau does not mean that the bead number, time
step, asymptotic distance and statistics are all converged. When the PMF overlap is disconnected or
the dividing surface lacks support, do not manufacture a rate by smoothing, filling holes or
ignoring warnings.

The full mathematical definitions are in the
[rate conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md) *(Chinese)* and
the [UI / MBAR notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md) *(Chinese)*.
A real potential interface with finite-sampling benchmarks is in
[H₃/BKMP2](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md) *(partly Chinese)*.

## Direct collisions: `drpmd`

`drpmd` starts from two reactant structures, keeps sampling parent chains and launches independent
relative collision motions. It is the task for explicitly inspecting collisions, scattering and
product classification. The initial distribution decides whether the output is a thermal rate or
a fixed-energy cross section.

To select the incoming diabatic electronic state and use a full coupled matrix potential, use the
experimental [NRPMD extension](nonadiabatic.md). It is still `task="drpmd"`, its electronic-state
contributions are signed, and it does not turn the scalar-potential procedure below into an
electronic-equilibrium PMF or a nonadiabatic Bennett–Chandler rate.

```bash
pyrpmd validate examples/drpmd/free_scattering.inp --check-pes
pyrpmd run examples/drpmd/free_scattering.inp --output runs/manual-scattering
pyrpmd analyze runs/manual-scattering
pyrpmd resume runs/manual-scattering
```

The [free-scattering example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/free_scattering.inp)
uses `V=0` to test launching, propagation, completion conditions and restart. There is no reactive
potential, so every completed collision should be nonreactive.

`collision.trajectories` is the total number of collisions shared by all parent chains. The example
creates five tasks from two parents, split 3+2. `sampling.equilibration` is the first
thermalization of a parent; `sampling.interval` is how far the same parent advances between two
draws. Fixing `sampling.parents` keeps the scientific task identity when the number of execution
processes changes.

| `collision.sampling` | Launch distribution | Meaning of the output |
| --- | --- | --- |
| `thermal_flux` | Relative energy sampled as $E\sim\mathrm{Gamma}(2,k_BT)$ | Thermal flux-averaged rate |
| `maxwell` | Ordinary Maxwell distribution of relative velocities | Thermal rate weighted by each trajectory's speed |
| `fixed_energy` | Explicit `collision.energy` | Cross section at that collision energy; no thermal rate |

`system.separation` is the initial radial COM distance R0, including the impact-parameter b
contribution; `collision.bmax` must be smaller than R0. Uniform-area sampling means uniform
sampling of $b^2$. For a formal conclusion, enlarge R0, bmax and the maximum propagation time
separately to check truncation errors.

Product assignment needs explicit rules. The built-in connectivity classifier uses bond cutoffs, a
separation condition and a persistence time; isomers with the same formula or exchange channels
with identical elements usually need a custom classifier via `outcomes.factory`. Default inferred
cutoffs only give tentative candidate labels and do not replace a checked chemical channel
definition.

In `results/summary.json`, look at `counts`, `all_trajectories_resolved` and `classification`
first, then at `total_reaction` / `channels`. The thermal-rate field of a channel is
`rate_cm3_molecule_s` and the fixed-energy cross-section field is `cross_section_bohr2`. Unresolved
or failed trajectories are kept separately; the corresponding observed lower bound and possible
upper bound are not confidence intervals. Trajectories from the same parent are correlated; judge
errors with `parent_cluster_bootstrap95` and more independent parents.

`results/channels.csv` lists the count, point estimate and observed lower / possible upper bound for
each channel. The run directory also stores `states/initial-*.npz`, `trajectories/` and checkpoints
so the initial state and classification of every trajectory can be audited. See the
[dRPMD example notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/README.md).

Without a `[collision]` section, the pilot runs 4 `thermal_flux` trajectories with
`max_time="100 fs"` and `bmax=R0/4`, and therefore reports no thermal rate; all defaults are listed
in [Input](input.md#defaults-for-omitted-fields).

## One- and two-dimensional free energies: `fes`

`fes` performs umbrella sampling on one or two CVs and unbiases with MBAR. It computes the
probability-density free energy of the chosen coordinates; it does not produce a reaction rate.

```bash
pyrpmd validate examples/fes/harmonic2d.inp --check-pes
pyrpmd run examples/fes/harmonic2d.inp --output runs/manual-fes
pyrpmd analyze runs/manual-fes --recompute --bins 20 20 --bootstrap 50
```

The [2D harmonic input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic2d.inp)
defines centroid x/y CVs, a 3×3 umbrella grid and two independent replicas per window.
`umbrella.stiffness` may be a scalar, a diagonal array or a full symmetric positive-definite
matrix. The windows must cover the region of interest; a wider analysis range does not create the
missing samples.

Results are saved in `fes.csv`, `fes.npz` and `fes.h5` with `free_energy`, `probability_mass`,
`density`, bin edges, counts and `mask`. Check that the overlap is connected, the effective sample
size of each time series and the weighted ESS. Unsupported regions stay masked/NaN; the free-energy
constant between disconnected groups of windows cannot be determined from the data.

`analyze --recompute` rebuilds the analysis from the saved window checkpoints and can change bins,
count thresholds and bootstrap without calling the PES again. Fewer bins sometimes raise the count
per bin, but they do not repair an unsampled basin or missing window overlap.

### Adaptive umbrella sampling

Within a given domain, the adaptive workflow inserts windows and lengthens pilot runs based on the
pilot overlap and effective sample sizes, then freezes the windows and re-equilibrates and samples
production with new independent random streams:

```bash
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/manual-adaptive
pyrpmd resume runs/manual-adaptive
pyrpmd analyze runs/manual-adaptive --recompute --bins 30 --bootstrap 20
```

`adaptive-plan.json` records every round's decision, windows, random seeds and checksums. Pilot
samples never enter the final production estimate. When `max_rounds` or `max_windows` is exhausted
without meeting the targets, the task stops explicitly; it does not promote a failing pilot to a
production result.

Adaptive umbrellas work for ordinary `fes`, surface `fes` and the gas-phase `rate` PMF. They refine
a given region and do not discover arbitrary distant unknown basins; `surface_rate` / `site_rate`
currently need fixed windows declared in advance. All parameters are in the
[enhanced-sampling specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md).

## Several entrances and TSs: `network`

`task="network"` organizes several explicit gas-phase bimolecular rate problems into one restartable
calculation. Each path still runs the full PMF and recrossing and keeps its own entrance
normalization, PES, TS geometry and errors.

```bash
pyrpmd prepare examples/network/two_entrances.py --pilot
pyrpmd run examples/network/two_entrances.py --output runs/manual-network
pyrpmd analyze runs/manual-network
pyrpmd resume runs/manual-network
```

The [complete example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/network/two_entrances.py)
has two entrances with two independent analytic TS/PES each. All paths share the ordered elements,
isotope masses, charge and multiplicity; each entrance may have its own fragment partition and
`rinf`, and each TS its own structure, bonds formed/broken, PES and symmetry factor. Different
stoichiometries or isotope inventories need separate tasks.

The PES is inherited global → entrance → TS; a more specific `potential` table replaces the one
above it completely and never mixes leftover fields from another provider. Tables such as sampling
and rate parameters can be overridden level by level. `parallel` and execution block settings
belong to the shared configuration.

Results are indexed as `matrix[entrance][channel]["transition_states"][ts_id]`. Each path lives in
`paths/<entrance-id>/<ts-id>/`, and `network-progress.json` records the finished paths. A restart
reuses finished results and continues the unfinished paths.

`aggregation="separate"` is the only mode; `total_rate_cm3_s` and `summed_rate_cm3_s` stay `null`.
Channel names are only labels and do not prove that fluxes through different dividing surfaces do
not overlap; adding multi-TS estimates directly may double count. This is not kMC, microkinetics or
a multi-PES nonadiabatic hopping simulation. See the
[reaction network definition](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/reaction-networks.md).

## Metadynamics

`task="metadynamics"` supports standard and well-tempered modes, 1D/2D CVs, and several replicas
with independent bias histories. A CV may declare its own period; that is a different concept from
the periodic cell of the atomic system. The workflow currently does not accept a System with PBC or
fixed atoms.

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
| `sigma` | Gaussian width per CV; explicit units recommended |
| `height` | Physical hill height |
| `pace` | Completed production steps between two depositions |
| `bias_factor` | $\gamma>1$ in WT mode; omit in standard mode |
| `analysis.method` | `bias_estimate` |
| `analysis.bootstrap` | Must be 0 |

There are no hills during the initial equilibration. The first hill is added after production step
`pace` completes successfully; every force evaluation within one step sees the same history. On
restart the bias history in `checkpoints/metadynamics_XXXX.npz` is authoritative, and
`replicas/XXXX/HILLS.csv` can be rebuilt from the checkpoint.

The standard estimate is $F\approx -V+C$ and the WT estimate $F\approx-\gamma V/(\gamma-1)+C$.
`fes.npz` / `fes.csv` hold the finite-time bias estimate and the support mask. Several replicas are
first aligned on their common support and then compared there; do not concatenate independent
HILLS files to fake one shared bias trajectory. `replica_spread` is the dispersion between replicas,
not a calibrated confidence interval.

The history-dependent bias uses a Python time-step loop with compatible compiled substeps; forcing
`loop_backend="native"` is rejected. Sampling cost also grows with the number of directly summed
hills. Time-dependent reweighting, infrequent-metadynamics rates and automatic physical convergence
certification are not implemented, so a biased barrier-crossing time is not a reaction rate.
Methods and restart timing are in the
[enhanced-sampling specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md).
