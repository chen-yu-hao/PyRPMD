# Nonadiabatic population dynamics from one input

`task="nonadiabatic"` runs an independent, unweighted ensemble of focused NRPMD
or centroid RPSH trajectories on a declared analytic model. It records fixed
diabatic populations and complex coherences throughout the simulation, with
separate RPSH adiabatic amplitude populations and active-surface occupations.
It uses the existing `validate`, `run`, `analyze` and `resume` commands.

## Run the examples

From the repository root after installation:

```bash
pyrpmd validate examples/nonadiabatic/population.inp --check-pes
pyrpmd run examples/nonadiabatic/population.inp --workers 2 --output runs/population
pyrpmd analyze runs/population
pyrpmd analyze runs/population --recompute
pyrpmd resume runs/population --workers 1 --block-steps 64

pyrpmd validate examples/nonadiabatic/rpsh_population.inp --check-pes
pyrpmd run examples/nonadiabatic/rpsh_population.inp --output runs/rpsh-population
pyrpmd resume runs/rpsh-population
```

The first example is a small finite, one-mode spin-boson calculation. The second
is a short nuclear-packet calculation on Tully I, with optional centroid EDC.
These small ensembles demonstrate the workflow; neither establishes a continuum
bath limit, a scattering probability, a thermal rate or physical convergence.

## Input contract

Numeric model parameters use atomic units: bohr, hartree, electron mass,
atomic time, and angular frequencies with hbar=1. `dynamics.dt` also accepts
unit strings such as `"0.1 fs"`. Supply exactly one of `beta` (inverse hartree)
and `temperature` (kelvin). The beta parameter controls the ring springs; it
does not turn an arbitrary initial condition into an equilibrium distribution.

| Section | Required choices and useful fields |
| --- | --- |
| Root | `task="nonadiabatic"`, `method="nrpmd"` or `"rpsh"`, `beta` or `temperature`, `beads`, `seed` |
| `model` | `name`: `tully1`, `tully2`, `tully3`, `lvc` or `spin_boson`; `params` contains that model's constructor parameters |
| `system` | Explicit `masses[N]` and optional `positions[N,3]`; spin-boson can use its declared nuclear masses |
| `initial` | `kind="point"` or `"harmonic_ring"`; `initial_state` for NRPMD, `active_surface` for RPSH |
| `ensemble` | `trajectories`, required `steps`, `sample_stride` |
| `dynamics` | `dt`, `integrator`, `native`, `loop_backend="auto"` or `"python"`, `block_steps` |
| `parallel` | Local `trajectory_workers` and `threads_per_rank`; serial or process backend |
| `output` | `directory` |

See [model formulas and parameters](nonadiabatic-models.md). A serialized model
identity contains all parameters, including real and imaginary parts of complex
matrices. The task constructs a dummy-atom `System` with explicit masses; it does
not interpret these reduced coordinates as a chemical structure. The general
[ensemble Python API](nonadiabatic-ensembles.md) also accepts user-supplied matrix
providers and electronic/nuclear states.

For RPSH, `dynamics` additionally accepts `electronic_substeps`, `rescaling`,
`frustrated_hops`, `gap_tolerance`, `decoherence` and `decoherence_constant`.
`decoherence="none"` is the default. `"edc"` applies the documented
[Granucci–Persico correction with centroid quantities](nonadiabatic-soc-rpsh.md).
Its default constant is 0.1 hartree. Converge both nuclear and electronic time
steps. RPSH requires real matrices and nondegenerate adiabatic states; complex
Hermitian SOC matrices are supported by NRPMD instead.

## What is sampled

`point` repeats `system.positions` and `initial.momenta[N,3]` on every bead.
Positions and momenta then have no thermal spread. Focused NRPMD independently
samples mapping phases; RPSH independently samples hopping events. A collapsed
ring does not by itself demonstrate a nuclear quantum effect.

`harmonic_ring` uses exact independent Gaussian draws of the finite-P primitive
harmonic path-integral nuclear distribution. Declare `frequencies[D]`, optional
flattened `coordinates[D]`, `centers[D]` and `centroid_momenta[D]`. Coordinate
`3*atom+axis` uses that atom's mass. Undeclared coordinates and momenta remain
zero. The momentum field shifts the distribution's mean centroid momentum;
individual draws still fluctuate. The initial harmonic reference can differ
from the propagated diabatic surfaces, defining a nonequilibrium preparation.

NRPMD uses focused MMST initialization in the selected fixed diabatic state,
factorized from this nuclear distribution. RPSH starts on the selected centroid
adiabatic eigenstate for each nuclear sample; this electronic vector generally
depends on that sample's geometry. Using the same integer
for these two methods generally specifies different physical electronic states.
The Python API supports explicitly prepared states when a controlled comparison
requires a different initialization.

This task does not consume signed thermal SEO samples or correlated Monte Carlo
parents. Continue to use [thermal NRPMD](thermal-nrpmd.md) for its signed weights
and ratio/jackknife estimators. It does not silently reinterpret those data as
independent unweighted trajectories.

## Output and uncertainty

| File | Contents |
| --- | --- |
| `results/summary.json` | Model/method, final estimates, pointwise SEM, saved-frame norm/energy drift, explicit convergence status |
| `results/populations.csv` | Common time grid, mean diabatic populations and SEM; RPSH also has active-surface fractions and adiabatic amplitude populations |
| `results/nonadiabatic.npz` | Per-trajectory complex density estimator, energies, norms, nuclear centroids, means and available real/imaginary coherence SEMs |
| `checkpoints/nonadiabatic_XXXXXX.npz` | One trajectory's complete state, hopping RNG, observations, physics identity and integrity envelope |
| `logs/nonadiabatic_XXXXXX_failure.json` | Failure context, when a trajectory stops unsuccessfully |

Both the initial and final observation are always retained, even when the final
step is not a multiple of `sample_stride`. NPZ arrays are read with
`pyrpmd.checkpoint.load_checkpoint(..., require_integrity=True)`; no pickle or
object arrays are used.

NRPMD stores the signed MMST density estimator; populations can be negative and
are not clipped. RPSH amplitude density and active-surface occupation are
different observables. Coherences are reported in the fixed diabatic basis,
avoiding arbitrary instantaneous adiabatic eigenvector phases.

Each independent trajectory supplies one sample at each saved time. Beads and
frames never increase the number of independent samples. SEM uses sample
variance with `ddof=1`; one trajectory has no estimated SEM. These are pointwise
sampling errors, not simultaneous confidence bands, and exclude bead, timestep,
model and method errors. Energy diagnostics describe saved frames; centroid-hop
rescaling can change the full-ring energy under its documented convention.

## Restart and validation boundaries

Each trajectory has random streams indexed by the global seed, trajectory ID
and sampling role. Worker placement and checkpoint block size do not select
different streams. A checkpoint atomically stores its state, RNG and history.
On interruption, `resume` restarts from the last complete checkpoint. A failure
never publishes a successful reduced ensemble with the failed sample omitted.

Changing output location or process count does not change the physics identity;
changing the model, masses, preparation, method, timestep, total steps or
electronic options requires a new calculation. A completed resume verifies and
reduces the stored trajectories without new propagation. `analyze --recompute`
rebuilds output from all complete, checked checkpoints without propagating.

Converge bead number, nuclear/electronic timesteps, trajectory count, preparation
and observation duration for the physical quantity of interest. Compare to the
independent [quantum-reference benchmark](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/nonadiabatic)
where applicable. The software leaves `physical_converged=false`; passing
numerical implementation tests does not certify a chemical prediction.
