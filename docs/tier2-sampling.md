# Extended collective variables and sampling methods

This page describes the tested reference implementations added for Tier 2.
The existing production workflow still distinguishes finite-time numerical
results from physical convergence. In particular, a finite biased trajectory
does not by itself determine a reaction rate.

## Collective variables

All CVs return `CVResult(value[D], jacobian[D,P,N,3])`; coordinates are in bohr
and Jacobians differentiate the complete ring estimator. The new factories can
be used in the same `[[cvs]]` sections as distances and angles.

| `kind` | Required fields | Optional fields | Output |
| --- | --- | --- | --- |
| `dihedral` | `atoms` (four distinct, zero-based indices) | `estimator` | Signed torsion, radians, period 2π |
| `periodic_dihedral` | `atoms` | `estimator` | Same torsion using minimum-image bond vectors |
| `coordination` | `group_a`, `r0` | `group_b`, `exponent=6`, `estimator` | Sum of rational switching functions |
| `periodic_coordination` | `group_a`, `r0` | Same as above | Same sum using minimum-image distances |
| `rmsd` | `reference` | `atoms`, `weights`, `align`, `estimator` | Weighted RMSD in bohr |
| `path` | `references`, `path_lambda` | `scale`, `component`, `atoms`, `weights`, `align`, `estimator` | Dimensionless path coordinates `s,z` |
| `torchscript` | `model` | `dimension=1`, `units`, `periods`, `estimator`, `device="cpu"` | Model-defined canonical outputs |

Python classes live in `pyrpmd.advanced_cv`: `Dihedral`, `CoordinationNumber`,
`RMSD`, `PathCV`, and `TorchScriptCV`. The geometric classes default to
`estimator="centroid"`; `"bead_average"` averages the complete scalar mapping
over beads, with the required 1/P Jacobian factor. Dihedrals instead use a
circular bead mean, so angles close to −π and +π stay near that seam. A zero
circular resultant and collinear torsions are explicitly undefined.

Periodic variants require a periodic `System`. Its continuous unwrapped bead
positions are averaged before applying minimum-image bond geometry; independently
wrapping beads before taking a centroid changes the estimator. RMSD, paths and
TorchScript models never guess molecular images. Supply reconstructed, unwrapped
coordinates and an appropriate model for periodic systems.

### Coordination

For an even integer n≥2, the switch is

```{math}
c(r)=\frac{1}{1+(r/r_0)^n}.
```

This is the rational n:2n switch, including its finite limit at r=r₀. The
implementation uses logarithms and a logistic function to avoid overflow.
The value at a coincident pair is one and its Cartesian gradient is zero.
With one group, each unordered pair is counted once. With two groups, they
must be disjoint and every cross-group pair is counted once. No neighbor list
or discontinuous cutoff is introduced.

```toml
[[cvs]]
name = "contacts"
kind = "coordination"
group_a = [0]
group_b = [1, 2, 3]
r0 = "2.5 angstrom"
exponent = 6
```

### Aligned RMSD and paths

`reference` is an `[N,3]` array or a `.npy` file in bohr. `references` is an
`[M,N,3]` array/file with M≥2. Files are loaded with `allow_pickle=False`.
An atom subset selects rows of the full reference. Positive weights have one
entry per selected atom and are normalized internally. `align="optimal"`
removes translation and fits a proper Kabsch rotation; `"center"` removes only
translation, and `"none"` retains the supplied frame. Alignment and displacement
use the same weights. Reflections are never allowed. The optimal-fit envelope
theorem gives the analytic gradient without differentiating an SVD.

RMSD itself is a norm and has no unique gradient at an exact reference match.
That case raises a clear error; `RMSD.squared_value_gradient()` remains smooth
there and is the metric used by `PathCV`. Reflection-corrected degenerate
alignment optima also raise rather than selecting an arbitrary force. A
collinear or collapsed geometry fitted to a reference with higher rank can
likewise have a nonunique gradient and raises. Straight or single-point
references remain supported when their rotational symmetry leaves the
gradient unique.

The path definition follows the smooth reference-path construction of
[Branduardi, Gervasio and Parrinello (2007)](https://doi.org/10.1063/1.2432340).
Let dₖ be the weighted aligned mean-square displacement to reference k divided
by `scale**2`. Then

```{math}
w_k=\frac{e^{-\lambda d_k}}{\sum_j e^{-\lambda d_j}},\qquad
s=\sum_{k=1}^{M}k w_k,\qquad
z=-\lambda^{-1}\log\sum_k e^{-\lambda d_k}.
```

Indices are one-based. Both outputs are dimensionless, and `path_lambda` is
dimensionless because the squared displacement was normalized by the explicit
length `scale` (default 1 bohr). The sum in z is intentionally not divided by
M, so z can be negative near overlapping reference kernels. `component="s"`
or `"z"` selects a scalar; `"both"` returns `[s,z]`. Log-sum-exp evaluation
prevents underflow when a configuration is far from every reference.

```toml
[[cvs]]
name = "reaction_path"
kind = "path"
references = "path_nodes.npy"
path_lambda = 5.0
scale = "1 angstrom"
component = "both"
align = "optimal"
```

### TorchScript CV contract

The optional PyTorch dependency is loaded only when constructing the model.
The model is loaded in evaluation mode, in float64, and receives bohr positions.
Its output must be a finite real floating tensor of shape `[D]` with an autograd
connection to positions; detached outputs are rejected. Derivatives are
obtained with real PyTorch autograd, including under an outer `torch.no_grad()`.

| Estimator | Input to model | Ring result |
| --- | --- | --- |
| `centroid` | `[N,3]` centroid | Model output with chain-rule 1/P gradients |
| `bead_average` | Each `[N,3]` bead | Mean of bead outputs |
| `full_ring` | `[P,N,3]` | Exact model-defined ring map |

Metadata declares `dimension`, canonical `units` (`bohr`, `hartree`, `radian`
or `dimensionless`) and optional positive `periods`. A periodic bead average
requires an explicit circular model and is rejected by the generic arithmetic
wrapper. The model file's SHA256 participates in the CV identity used when
comparing bias histories. No training, descriptor conversion, atomic species
inference, or implicit unit conversion is performed.

```toml
[[cvs]]
name = "learned"
kind = "torchscript"
model = "cv.pt"
dimension = 2
units = ["dimensionless", "dimensionless"]
estimator = "centroid"
```

## OPES-explore and adaptive widths

The existing `task="opes"` workflow now accepts these additional `[opes]`
fields: `mode="metad"|"explore"`, `adaptive_sigma=true|false`, and
`sigma_min` (or `sigma_min_au`). Existing inputs retain fixed-bandwidth OPES.
The modes have different estimators:

| Mode | KDE weights | Bias prefactor multiplying log(p/Z+ε) |
| --- | --- | --- |
| `metad` | exp(β V before deposition) | (1−1/γ)/β |
| `explore` | One for every kernel | (γ−1)/β |

These are the OPES and OPES-explore density/bias equations documented by
[PLUMED OPES_METAD](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d.html)
and [OPES_METAD_EXPLORE](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d__e_x_p_l_o_r_e.html).
The explore KDE estimates the sampled target density. Its direct FES output
uses `-gamma/beta*log(p_target)` and is meaningful as free energy only after
convergence to the well-tempered target. Frozen-bias histogram reweighting is
available for quantitative analysis in either mode.

`sigma` is the literal initial kernel width in the CV units. PLUMED's explore
interface multiplies its input SIGMA by sqrt(γ); an equivalent fixed physical
width therefore needs `sigma/sqrt(γ)` as PLUMED's input. Kernels are normalized wrapped Gaussians for periodic
CVs, and no truncation or compression is applied.

The empty-density limit is `V=prefactor*log(epsilon)=-barrier`, with zero force.
The first metad kernel therefore has log-weight `-beta*barrier`. Although
the initial bias is spatially constant, discarding it changes the first
weight relative to later kernels and changes the resulting bias. New runs
persist this convention as OPES bias state schema 2 with `initial_bias`.
Recognized schema-1 checkpoints resume their original zero initial bias and
weights, including empty histories; they remain schema 1 when saved again.
Analysis reports each history's version and initial bias and warns about
legacy histories. Unknown versions or inconsistent initialization fields
are rejected. This preserves old trajectories without silently rewriting
their algorithm or labeling them as new-initialization runs.
When a partially started job has only schema-1 checkpoints, its never-started
replicas also use schema 1, preserving whole-job restart behavior. If existing
histories mix schemas and replicas are missing, continuation rejects the
ambiguous initialization choice before constructing a potential. Existing
histories can still be reanalyzed with their individual versions.

The adaptive option uses the following **explicit reference rule**, which is
different from PLUMED's diffusion-window algorithm:

1. At each deposition event, update the unweighted mean and Welford second
   moment of the observed CV. Periodic observations are unwrapped against the
   preceding deposited value using the shortest signed displacement.
2. Keep the initial `sigma` for the first two kernels. Starting at event n=3,
   calculate the unbiased sample variance of these observations.
3. Multiply its square root by
   `[4 / ((D+2) N_eff)]**(1/(D+4))`, where `N_eff` is the importance-weight ESS
   for metad mode and n for explore mode.
4. Apply the componentwise `sigma_min` floor. If omitted, the floor is one
   percent of the initial sigma.
5. Store the resulting width on the new kernel. Existing kernel widths are
   unchanged, and each Gaussian retains its own normalization factor.

The normalization is the mean KDE density at its deposited centers. Variable
widths make the kernel matrix asymmetric, so its column sums are maintained
explicitly; treating them as symmetric would give a wrong Z. Restart retains
all widths and recurrence state and validates them against the center/weight
history before continuing. A finite floor leaves finite-bandwidth smoothing
error. An uncompressed history costs O(n) memory and O(n) work per point; it
is a transparent numerical reference, not a substitute for PLUMED's compressed
large-scale implementation.

```toml
[opes]
cvs = ["x", "y"]
mode = "explore"
sigma = ["0.3 bohr", "0.3 bohr"]
adaptive_sigma = true
sigma_min = ["0.03 bohr", "0.03 bohr"]
barrier = "2 kcal/mol"
bias_factor = 8.0
pace = 20
```

Run the complete example with
`pyrpmd run examples/opes/adaptive_explore.inp --output runs/adaptive-explore`.

## Temperature and Hamiltonian replica exchange

`pyrpmd.replica_exchange.ReplicaExchange` is a serial Python engine accepting
one `Dynamics` and one `State` per slot. Potentials and temperatures may differ;
bead count, masses, timestep and fixed-atom constraints must match. Static
biases are optional. Time-dependent/history-dependent Hamiltonians are rejected.
The existing same-temperature umbrella workflow remains available separately.

For each slot i, the configurational reduced potential is

```{math}
u_i(q)=\frac{\beta_i}{P}\left[\sum_b V_i(q_b)
  +\frac{1}{2}\left(\frac{P}{\beta_i}\right)^2
   \sum_{ba}m_a|q_{ba}-q_{b+1,a}|^2+B_i(q)\right].
```

The swap log ratio is `u_i(q_i)+u_j(q_j)-u_i(q_j)-u_j(q_i)`. Temperature
exchange therefore includes the target-temperature spring action, even when
the physical PES is shared. It does not use a classical potential-only formula
for a quantum ring. Accepted swaps rescale momenta by
`p_i <- sqrt(beta_j/beta_i)*p_j` and the inverse factor in j. The combined reduced
kinetic energy and paired transformation Jacobian are preserved. RNGs and
clocks stay in their temperature/Hamiltonian slots; walker labels track movement.

`advance(steps, gamma_centroid=...)` samples every slot with PILE-L then attempts
disjoint nearest-neighbor swaps at a random parity. A late force failure rolls
back the whole event, including every slot's state, RNG and propagation counters.
`attempt(i,j)` exposes an individual exchange for existing Python workflows.
This new engine currently has no standalone CLI task, persistent checkpoint
format or MPI orchestration. Its trajectory time is sampling time, not physical
reaction time.

## Kinetics from audited biased trajectories

`pyrpmd.biased_kinetics.rescale_biased_time` implements the acceleration clock
used by [infrequent metadynamics](https://doi.org/10.1103/PhysRevLett.111.230602)
and applicable to a correctly constructed
[OPES-flooding](https://doi.org/10.1021/acs.jctc.2c00806) run:

```{math}
t_{\mathrm{physical}}=\sum_n\Delta t_n
  \exp\{\beta[V_n-V_{\mathrm{TS},n}]\}.
```

Supply every completed integration interval's duration and the physical bias
held during that interval. The absolute bias gauge is not interchangeable with
the transition-region reference: `transition_bias` is required, and shifting
both by the same value leaves the clock unchanged. For an OPES bias whose
asymptote is `-barrier`, using zero as the transition reference changes the
recovered clock. Sparse saved frames are insufficient to reconstruct an
arbitrarily varying bias history.

The caller must explicitly provide `transition_unbiased=True`, after checking
that the transition region is unperturbed, and must not report a deposition
inside it. These are eligibility inputs, not properties that the numerical
integral can establish. This module does not implement a native flooding
exclusion bias or automatically convert ordinary MetaD/OPES runs to kinetics.
It can analyze an audited external/PLUMED history. A physically valid rate also
requires appropriate basins, initial equilibrium, independent first-passage
experiments and a tested rare-event/Poisson regime.

`estimate_biased_rate` consumes one rescaled first-passage time or right-censored
exposure per independent experiment. The exponential MLE is the number of events
divided by total exposure; unsuccessful runs must not be discarded. The function
returns zero events explicitly. For complete datasets of at least three events,
it reports a KS statistic and optionally a parametric-bootstrap p-value, refitting
the exponential scale in every bootstrap draw. Censored samples are not silently
treated as complete first passages. Outputs always retain `converged=False`.

## Replica-exchange TIS

`pyrpmd.retis.ReplicaExchangeTIS` extends the existing two-way canonical
transition-interface sampler with exchanges between its positive ensembles.
It follows the path-swapping principle of
[van Erp (2007)](https://arxiv.org/abs/0704.3317).
Each sweep attempts one shooting move in each ensemble, then neighboring path
swaps. Since every ensemble has the same underlying dynamics and canonical
path measure, a swap is accepted when both trajectories meet the destination
interface indicators. Validated seed paths must be actual trajectories of the
configured reversible dynamics.

Every sweep contributes an observation, including rejected shooting moves and
repeated paths. Results retain per-interface crossing sequences, path lengths,
shooting acceptance, swap counts and the full walker-label history. Statistics
across interface slots are correlated and should be resampled jointly.

The implementation covers **positive-ensemble RETIS with a separate flux
estimate**. The minus-interface flux ensemble, asynchronous/infinite swapping,
MPI execution and persistent ensemble restart are not implemented. Use
`combine_tis_rate(result.interfaces, matching_flux)` with a separately validated
effective flux. The existing finite path-length cutoff also applies: increasing
it until truncation is negligible is part of a physical convergence study.

## Verification

The new tests independently finite-difference complete ring Jacobians and
bias forces; check translation/rotation/periodic-image invariance and singular
geometry handling; execute actual CPU TorchScript autograd; compare OPES densities,
forces and Z with direct normalized Gaussian sums; test exact continuation from
adaptive checkpoints; verify the temperature-dependent spring exchange ratio and
momentum map; and test biased-clock gauge invariance, censored exposure and actual
RETIS trajectory replay. They do not establish a converged molecular rate or
physical parity with another engine.
