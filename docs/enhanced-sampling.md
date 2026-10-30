# Adaptive umbrellas and metadynamics

These workflows sample ring-polymer configuration distributions. Biased
trajectory time is not a reaction rate, and no infrequent-metadynamics rate
estimator is provided. Production rate coefficients still use the rate
workflow with its prescribed PMF measure and unbiased recrossing dynamics.

## Adaptive umbrella placement

Add `[adaptive_umbrella]` to a one- or two-dimensional `task="fes"` job, or to
a `task="rate"` PMF job:

```toml
[adaptive_umbrella]
enabled = true
pilot_steps = 1000
pilot_equilibration_steps = 200
max_rounds = 4
max_windows = 64
target_overlap = 0.03
min_effective_samples = 20.0
min_spacing = 0.000001
```

The initial `[umbrella]` specification defines the domain and positive definite
stiffness matrix. Each pilot round independently equilibrates every window,
then samples its fixed bias. The pilot-only MBAR calculation constructs an
overlap graph with an edge when `sqrt(O_ij * O_ji) >= target_overlap`. Replicas
at the same center are pooled for the *overlap solver*, while each trajectory
retains its own autocorrelation diagnostic. Window effective sample counts
sum `N/g` across independent replicas. The slowest coordinate or centered
coordinate-square trace sets `g`, so rapid sign oscillations do not conceal
slow amplitude relaxation. Periodic CVs use first and second sine/cosine
harmonics, avoiding false decorrelation at the coordinate branch cut.
The production FES/PMF effective sample counts and bootstrap correlation blocks
use the same diagnostics; constant derived observables are ignored. This does
not alter the MBAR weights or free-energy point estimate.

If the graph is disconnected at the requested threshold, the algorithm
inserts midpoints along the shortest links between components. Distances use
the dimensionless metric `sqrt(beta * delta.T @ K @ delta)`; `min_spacing`
uses this metric too. Periodic displacements follow the shortest wrapped
arc. The stiffness stays fixed. The next pilot doubles its sampling duration
and uses a separate deterministic seed. This also gives a connected set with
insufficient effective samples another chance to meet the ESS target.

For a declared domain, optionally add a spatial coverage target to the same
table (the example below has two Cartesian CVs):

```toml
coverage_bins = [4, 8]
coverage_ranges = [["-0.5 bohr", "0.5 bohr"], ["-1 bohr", "1 bohr"]]
min_bin_effective_samples = 5.0
target_coverage = 0.95
```

`coverage_bins` and `coverage_ranges` must be supplied together. Numeric
bounds use canonical CV units. This pilot grid is independent of the final
`analysis.bins`: choose enough cells to resolve the domain's sampling holes
without requiring an impractical number of pilot samples. The default omits
this additional target; graph connectivity alone does not test domain coverage.
For each cell, the diagnostic is `(sum w)^2 / sum(g_stream * w^2)`, using
unbiased MBAR weights and the per-stream correlation factor. Empty cells count
as unsupported. Deficient cells propose additional windows at their centers,
subject to `max_windows` and `min_spacing`. Existing centers are not duplicated;
longer pilots can improve their support. `adaptive-plan.json` retains the full
cell counts, effective sample estimates, coverage fraction and decisions.

Once all enabled criteria pass, the window set is frozen. **Fresh production streams
start from newly equilibrated states and a different seed. No pilot samples
enter the production MBAR or UI estimate.** The saved `adaptive-plan.json`
contains a checksum, initial/final windows, every decision, diagnostics and
seeds. `adaptation/round_*/checkpoints/` contains the pilot states;
`checkpoints/window_*.npz` contains only final production. Interruptions resume
the current pilot or the frozen production without redesigning an already
committed stage. Changing analysis bins/bootstrap does not resample pilots.

Exhausting `max_rounds`, or finding no admissible midpoint while the overlap
graph is disconnected, stops with `adaptation_incomplete`; production is not
silently launched. `max_windows` limits insertion: a connected set may still
pass ESS/coverage targets after longer pilots without adding more windows.
A reached pilot
target is a local overlap/ESS/coverage diagnostic, not proof of equilibration,
adequate CVs, final histogram coverage, or physical convergence. The local ESS
does not include uncertainty in fitted MBAR offsets or unseen slow variables.
The algorithm refines a supplied
domain and does not automatically discover unspecified distant basins.

Previously frozen plans keep their saved decisions on restart; their summaries
identify legacy coordinate-only ESS diagnostics. Use a fresh output directory
to reevaluate them under the new moment and optional coverage criteria.

The rate workflow retains its `rate_flux` metric bias during both stages;
the downstream PMF and recrossing preparation use the actual frozen window
centers from the sampled streams. Generic FES retains `cv_density`.

```bash
pyrpmd validate examples/fes/adaptive_harmonic.inp
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/adaptive
pyrpmd resume runs/adaptive
pyrpmd analyze runs/adaptive --recompute --bins 30 --bootstrap 20
```

## Standard and well-tempered metadynamics

Use `task="metadynamics"`, ordinary CV specifications and an explicit hill
height/width. One or two CV dimensions are supported, including custom Python
CVs with declared positive periodicities and full ring-polymer Jacobians.

```toml
[metadynamics]
cvs = ["x", "y"]
mode = "well_tempered"  # or "standard"
sigma = ["0.2 bohr", "0.2 bohr"]
height = "0.02 kcal/mol"
pace = 50               # completed production steps per hill
bias_factor = 8.0       # gamma > 1, omit for standard metadynamics
```

Numeric sigma values use the corresponding CV input units; `sigma_au` uses
canonical units. Height is a physical energy in hartree or an explicit energy
quantity, and `height_au` is also accepted. Supplying both forms is rejected.
The CV can be a function of the centroid, the average of a bead-wise function,
or a custom full-ring function. These definitions are distinct; the declared
value and full Jacobian are used without substituting one for another.

For a CV vector `s`, the physical bias is a sum of Gaussian hills. Periodic
dimensions use a smooth **sum over periodic Gaussian images**, normalized so
the value at a hill center equals its deposited height. An equivalent Fourier
sum is used for broad periodic kernels. This avoids a force discontinuity at
half a period. The ordinary nonperiodic Gaussian convention is
`exp(-delta**2 / (2*sigma**2))`.

The Python `MetadynamicsBias.hills` property returns a detached, serializable
history snapshot. Modify the running bias through `deposit()` or
`truncate_history()`, or construct a new bias with `hills=modified_history`
for reanalysis. Editing a snapshot does not change the running forces. This
keeps the cached Gaussian arrays consistent with the deposited history.

PyRPMD uses the sum-beads Hamiltonian sampled at `beta/P`. Therefore:

\[
H_P^{\rm biased}=H_P+P V(s,t),\qquad
F_{jac}^{\rm bias}=-P\sum_d\frac{\partial V}{\partial s_d}
                       \frac{\partial s_d}{\partial q_{jac}}.
\]

Standard metadynamics deposits a constant physical height `w0`.
Well-tempered metadynamics evaluates the **pre-deposition physical bias** at
the current center and deposits

\[
w(t)=w_0\exp\left[-\frac{\beta V(s(t),t^-)}{\gamma-1}\right],
\qquad \gamma=\frac{T+\Delta T}{T}>1.
\]

There is no extra factor of P in this exponent. The full-ring force and
Hamiltonian P factors are independently checked against finite differences.

### Sampling and event semantics

The ordinary `[sampling]` equilibration, production, save interval and Andersen
settings apply. Initial equilibration uses zero hills. The first hill is
deposited at production step `pace`, then at `2*pace`, and so on. Within each
integration step, both force evaluations see exactly the same bias history.
Only after the complete step succeeds is its hill committed. Saved sample
energies and physical bias values are evaluated before a coincident deposition.

History-dependent biases use the Python propagation loop, with a compiled
free-ring substep when the selected integrator supports it. `loop_backend="native"`
is rejected before simulation. An apparently fast frozen native bias must not
silently replace an evolving history. This initial implementation evaluates
the hill sum directly, so force cost grows with the number of hills; a grid
acceleration is future work, not part of the performance claim.

`sampling.replicas` creates independent metadynamics runs that can use ordinary
task parallelism. They do not share hills. A checkpoint atomically saves q/p,
RNG, absolute and production clocks, accumulated hills, observations and
propagation counts, with the common checkpoint checksum envelope. Failed steps
restore their previous q/p and RNG before saving the valid prefix.

Each `replicas/0000/HILLS.csv` is a human-readable view regenerated from its
authoritative `checkpoints/metadynamics_0000.npz`. A stale or missing CSV cannot
cause a duplicated hill during resume. HILLS files from different replicas
must not be concatenated to pretend they came from one shared-bias trajectory.

### Free-energy estimate and limits

`analysis.method="bias_estimate"` exports `fes.npz` and `fes.csv`:

\[
\begin{aligned}
F_{\rm standard}(s)&\approx -V(s,t)+C,\\
F_{\rm WT}(s)&\approx -\frac{\gamma}{\gamma-1}V(s,t)+C.
\end{aligned}
\]

These are finite-time, finite-width bias estimates. Standard metadynamics
continues to deposit constant hills and can oscillate about the free energy.
Neither formula proves convergence. All result summaries keep `converged=false`.
Unsampled bins are masked. Independent-replica surfaces are aligned using one
commonly sampled reference bin; the aggregate is restricted to common sampled
support. If no common reference exists, the aggregate is entirely masked.
`replica_spread` is descriptive, not a calibrated error bar.

`analysis.bootstrap` must be zero; fixed-umbrella MBAR and ordinary equilibrium
bootstrap are not applied to these evolving-bias samples. Time-dependent
trajectory reweighting, well-tempered time-averaged estimators, automatic
convergence certification and infrequent-metadynamics kinetics are not
implemented. A periodic analysis range must span exactly one period.

```bash
pyrpmd run examples/metadynamics/standard.inp --output runs/meta1
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/meta2
pyrpmd resume runs/meta2
pyrpmd analyze runs/meta2 --recompute --bins 30 30
```

Python users can call `run_metadynamics`, `reanalyze_metadynamics`,
`MetadynamicsBias`, and `analysis.metadynamics.bias_fes` directly. The normal
`pyrpmd.run` / `JobSpec` entry points remain available.

## Method references

- A. Laio and M. Parrinello, *Escaping free-energy minima*, PNAS **99**, 12562–12566
  (2002), [doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399).
- A. Barducci, G. Bussi and M. Parrinello, *Well-Tempered Metadynamics: A Smoothly
  Converging and Tunable Free-Energy Method*, PRL **100**, 020603 (2008),
  [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).
- M. R. Shirts and J. D. Chodera, *Statistically optimal analysis of samples
  from multiple equilibrium states*, JCP **129**, 124105 (2008),
  [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177).
- [PLUMED's official metadynamics tutorial](https://www.plumed.org/doc-v2.9/user-doc/html/master-_i_s_d_d-2.html)
  gives the physical hill-height, bias-factor and asymptotic FES conventions
  used for an independent formula cross-check. Code here is an independent
  implementation; no PLUMED code has been copied.
