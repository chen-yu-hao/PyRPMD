# Nonadiabatic population dynamics from one input

`task="nonadiabatic"` runs independent nonequilibrium trajectories on a declared
analytic model or the published molecular NaH2 5G diabatic potential. Supported
methods are NRPMD, two-state W-spin SM-NRPMD, CA-RP-MASH, centroid RPSH,
RPSH-BA, RPSH-DM and complex SOC RPSH. It uses the existing `validate`, `run`,
`analyze` and `resume` commands. Each method retains its own estimator:
mapping methods have signed fixed-diabatic density estimators, hopping methods
report amplitude densities and active-surface occupations separately, and the
MASH population workflow reports adiabatic active occupations without inventing
a diabatic coherence estimator.

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

pyrpmd run examples/nonadiabatic/extensions/rp_mash.inp --output runs/rp-mash
pyrpmd run examples/nonadiabatic/extensions/nah2_sm_nrpmd.inp --output runs/nah2-sm
```

The first example is a small finite, one-mode spin-boson calculation. The second
is a short nuclear-packet calculation on Tully I, with optional centroid EDC.
These small ensembles demonstrate the workflow; neither establishes a continuum
bath limit, a scattering probability, a thermal rate or physical convergence.

The [extension examples](https://github.com/chen-yu-hao/PyRPMD/tree/main/examples/nonadiabatic/extensions)
cover all new methods, DBOC and complex SOC. See [spin methods](nonadiabatic-spin-methods.md),
[hopping extensions](nonadiabatic-hopping-extensions.md),
[molecular potentials](molecular-nonadiabatic.md) and
[electronic acceleration](nonadiabatic-performance.md) for exact variants.

## Input contract

Numeric model parameters use atomic units: bohr, hartree, electron mass,
atomic time, and angular frequencies with hbar=1. `dynamics.dt` also accepts
unit strings such as `"0.1 fs"`. Supply exactly one of `beta` (inverse hartree)
and `temperature` (kelvin). The beta parameter controls the ring springs; it
does not turn an arbitrary initial condition into an equilibrium distribution.

| Section | Required choices and useful fields |
| --- | --- |
| Root | `task="nonadiabatic"`; `method`: `nrpmd`, `sm-nrpmd`, `rp-mash`, `rpsh`, `rpsh-ca`, `rpsh-ba`, `rpsh-dm`, `soc-rpsh`; `beta` or `temperature`, `beads`, `seed` |
| `model` | `name`: `tully1`, `tully2`, `tully3`, `lvc`, `spin_boson` or `nah2_leps_5g_1992`; `params` contains that model's constructor parameters |
| `system` | `masses[N]`, `positions[N,3]`; molecular `symbols` and `unit="bohr"` or `"angstrom"`; NaH2 defaults to isotope masses, spin-boson to its declared masses |
| `initial` | `kind="point"` or `"harmonic_ring"`; `initial_state` for NRPMD/SM-NRPMD, `active_surface` for hopping/MASH |
| `ensemble` | `trajectories`, required `steps`, `sample_stride` |
| `dynamics` | `dt`, `integrator`, `native`, `loop_backend="auto"` or `"python"`, `block_steps` |
| `parallel` | Local `trajectory_workers` and `threads_per_rank`; serial or process backend |
| `output` | `directory` |

See [model formulas and parameters](nonadiabatic-models.md). A serialized model
identity contains all parameters, including real and imaginary parts of complex
matrices. Reduced analytic models use dummy atoms with explicit masses. NaH2
uses actual Na/H symbols, atom order, isotope masses, neutral charge and doublet
multiplicity from the fixed molecular PES identity. Its Cartesian matrix and
gradients are in bohr/hartree units. The general
[ensemble Python API](nonadiabatic-ensembles.md) also accepts user-supplied matrix
providers and electronic/nuclear states.

For RPSH, `dynamics` additionally accepts `electronic_substeps`, `rescaling`,
`frustrated_hops`, `gap_tolerance`, `decoherence` and `decoherence_constant`.
`decoherence="none"` is the default. `"edc"` applies the documented
[Granucci–Persico correction with centroid quantities](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-soc-rpsh.md).
Its default constant is 0.1 hartree. Converge both nuclear and electronic time
steps. Legacy `rpsh` requires real matrices. `rpsh-ca` and `soc-rpsh` support
complex Hermitian matrices with a phase-invariant real momentum direction and
the documented local geometric force. `rpsh-ba` retains common adiabatic
amplitudes with bead energies/connections; `rpsh-dm` retains bead wavefunctions
and constructs a centroid density. These hopping methods require nondegenerate
states. Advanced hopping accepts `dboc=true` and `dboc_step`; DM rejects EDC
because it uses its density-matrix mechanism. DBOC is a scalar correction from
the supplied states, with a finite-difference gradient and a consistent energy
ledger. It is not a regularization of a conical intersection.

SM-NRPMD and NRPMD accept `electronic_backend="numpy"|"native"|"torch"`
and `electronic_device="cpu"|"cuda[:index]"`. CUDA requires Torch; a requested
unavailable backend raises an error. RP-MASH uses `native=false` and
`integrator="adaptive-dop853"`, with `rtol`, `atol`, `max_step`, `max_events`
and `gap_tolerance` for event-resolving integration. The ordinary native
free-ring splitting is not substituted for this event integrator.

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

NRPMD uses focused MMST initialization and SM-NRPMD uses a focused W-spin
latitude in the selected fixed diabatic state, factorized from this nuclear
distribution. MASH uses its population hemisphere importance distribution.
RPSH starts on the selected centroid
adiabatic eigenstate for each nuclear sample; this electronic vector generally
depends on that sample's geometry. Using the same integer
for these two methods generally specifies different physical electronic states.
The Python API supports explicitly prepared states when a controlled comparison
requires a different initialization.

This task does not consume signed thermal SEO samples or correlated Monte Carlo
parents. Continue to use [thermal NRPMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/thermal-nrpmd.md) for its signed weights
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

For `rp-mash`, the population basis is `centroid_adiabatic`, and the NPZ omits
`diabatic_density`, coherence and amplitude-population fields. Its separate
Python correlation API uses full-sphere sampling and the original operator
weights. BA checkpoints also contain transported `basis_vectors[P,S,S]`; DM
checkpoints contain bead coefficients `[P,S]`. Restarts validate these method
specific shapes and their physical interpretation.

Both the initial and final observation are always retained, even when the final
step is not a multiple of `sample_stride`. NPZ arrays are read with
`pyrpmd.checkpoint.load_checkpoint(..., require_integrity=True)`; no pickle or
object arrays are used.

NRPMD stores the signed MMST density estimator and SM-NRPMD its signed W kernel;
populations can be negative and
are not clipped. RPSH amplitude density and active-surface occupation are
different observables. Coherences are reported in the fixed diabatic basis,
avoiding arbitrary instantaneous adiabatic eigenvector phases.

Each independent trajectory supplies one sample at each saved time. Beads and
frames never increase the number of independent samples. SEM uses sample
variance with `ddof=1`; one trajectory has no estimated SEM. These are pointwise
sampling errors, not simultaneous confidence bands, and exclude bead, timestep,
model and method errors. Energy diagnostics describe saved frames. Legacy
`rpsh` centroid-hop rescaling can change the full-ring energy under its
documented convention; the new BA/DM/SOC/advanced-CA branches and RP-MASH solve
the full-ring energy constraint for each accepted hop.

## Restart and validation boundaries

Each trajectory has random streams indexed by the global seed, trajectory ID
and sampling role. Worker placement and checkpoint block size do not select
different streams. A checkpoint atomically stores its state, RNG and history.
On interruption, `resume` restarts from the last complete checkpoint. A failure
never publishes a successful reduced ensemble with the failed sample omitted.

Changing output location or process count does not change the physics identity;
changing the model, masses, preparation, method, timestep, total steps or
physical electronic options requires a new calculation. Electronic backend and
device are execution choices. The Python API accepts
`resume_run(path, execution={"electronic_backend": "native", "electronic_device": "cpu"})`
for NRPMD/SM-NRPMD. Cross-backend continuation may change floating-point rounding;
it is not promised to be bitwise identical. A completed resume verifies and
reduces the stored trajectories without new propagation. `analyze --recompute`
rebuilds output from all complete, checked checkpoints without propagating.

Converge bead number, nuclear/electronic timesteps, trajectory count, preparation
and observation duration for the physical quantity of interest. Compare to the
independent [quantum-reference benchmark](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/nonadiabatic)
where applicable. The software leaves `physical_converged=false`; passing
numerical implementation tests does not certify a chemical prediction.
