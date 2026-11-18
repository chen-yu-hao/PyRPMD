# Subspace quantum correction of RPMD rates

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/subspace-correction.md)

> **Which quantum-correction workflow do I need?**
> - **Thermal reaction rate constants $k \approx k_{\mathrm{RPMD}} \times C_S$ (this page):** Computes a multiplicative quantum/RPMD correction factor $C_S = k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$ in a 1–3D reaction subspace via `[rate.subspace_correction]` in `.inp`, `pyrpmd subspace-correct correct.toml`, or the `pyrpmd.analysis` Python API.
> - **Time-correlation functions, spectral responses and online Ehrenfest trajectories:** Go to **[Quantum correction](quantum-correction.md)** for frequency-domain response defect transfers (`run_quantum_correction`), online coupled ring-polymer + 1D wavefunction propagation (`OnlineQuantumRPMD`), and time-dependent flux–side curve transfers (`hybrid_flux_side_correction`).

RPMD treats the environment of a reaction well (zero-point energy, adiabatic following,
recrossing) but misses deep tunnelling, resonances and coherence along the reaction
coordinate. The subspace correction keeps the full-dimensional RPMD rate and fixes only
that defect with an exact quantum calculation in one to three coordinates:

$$
k \approx k_{\mathrm{RPMD}}(\text{full system}) \times C_S, \qquad
C_S = \frac{k_{\mathrm{QM}}(H_S)}{k_{\mathrm{RPMD}}(H_S)}
$$

Numerator and denominator use the same subspace Hamiltonian $H_S$ and the same reactant
normalization, so $C_S$ contains only the RPMD method error inside $S$. The transfer is
multiplicative: the absolute rate of $H_S$ may differ from the full system's by orders of
magnitude, but a shared prefactor cancels in the ratio. $C_S$ is exact when $S$ is the
whole system and an approximation otherwise.

---

## 1. Choose your entry point

Pick one of the three workflows below according to whether you are starting a new `rate`
run, postprocessing a saved `rate` directory, or computing $C_S$ directly from a PES in
Python without umbrella sampling:

| Scenario | Recommended entry point | Requires umbrella sampling? | Key command / API |
| --- | --- | --- | --- |
| **A. Run `rate` + subspace correction in one step** | `[rate.subspace_correction]` in `.inp` | Yes (local `umbrella` PMF; auto-enables `output.centroid_path = true`) | `pyrpmd run calculation.inp` |
| **B. Postprocess or re-tune an existing `rate` run** | `--stage subspace` on `.inp`, or `pyrpmd subspace-correct correct.toml` | Saved `rate` run (`output.centroid_path = true` for centroid paths or `--stage subspace`; `path.source = "mep"` also works from the saved saddle + PES) | `pyrpmd run calculation.inp --stage subspace` or `pyrpmd subspace-correct correct.toml` |
| **C. Compute $C_S$ without umbrella sampling (or for `surface_rate` / custom models)** | `pyrpmd.analysis` Python API | No (builds $H_S$ directly from $V$, $\nabla V$, and saddle/path) | `project_linear_subspace`, `project_path_subspace`, `subspace_correction`, `correct_rate_run` |

(optional-native-rate-stage)=
### Route A: Optional native `rate` stage (one `.inp`)

Use the normal rate input and runner; a second correction input is not needed:

```toml
# All ordinary system, potential, reaction, umbrella and recrossing sections
# remain in this same file (see examples/rate/subspace.inp).
task = "rate"

[rate.subspace_correction]
enabled = true
mode = "apply"             # default; alternatively "factor_only"
dimension = "auto"         # alternatively 1, 2, or [1, 2]
environment = "adiabatic"  # also "free_energy" or "frozen"
model_2d = "rph"           # default MEP reaction-path Hamiltonian; or "plane"
anchor = "none"            # or "skeleton" for low-T instanton anchoring
threshold = 0.5
reuse = true

[rate.subspace_correction.path]
source = "mep"             # default IRC from refined saddle; or "centroid", "instanton", ...
step = 10.0                # bare atomic units: bohr * sqrt(m_e)

[rate.subspace_correction.quantum]
refinement = 3
truncation_kt = 30.0
# Converge truncation_min (hartree) for stiff modes, especially at low T (~6 quanta).
# The retained 2D grid depends on temperature, masses and the PES; the stage
# plans it before solving and reports the dense_limit it needs (solver_plans).
truncation_min = 0.0
dense_limit = 8000

[rate.subspace_correction.rpmd]
walkers = 32
sweeps = 800
trajectories = 1024
seed = 17

# Validity gates (defaults shown). "allow" is an explicit, recorded override.
[rate.subspace_correction.validity]
unconverged_path = "reject"   # both path continuations must end at an accepted terminus
unstable_bath = "reject"      # the transverse Hessian of the path model must be stable
outside_grid = "reject"       # the RPMD estimator must stay on the V_S grid
max_outside_fraction = 0.01
submerged_surface = "reject"  # the RPMD dividing surface must be the free-energy bottleneck
max_submerged_kt = 1.0
folded_ribbon = "reject"      # an RPH 2D path may not bend more tightly than its transverse mode is wide
max_curvature_width = 0.5
on_invalid = "stop"           # or "diagnose": solve a rejected model anyway, never apply it
```

```bash
pyrpmd run calculation.inp
# Re-run only the postprocessing using the same input/output directory:
pyrpmd run calculation.inp --stage subspace
# Ordinary checkpoint restart also runs the enabled postprocessing at the end:
pyrpmd run calculation.inp --resume
```

A complete teaching input is [`examples/rate/subspace.inp`](../examples/rate/subspace.inp).
`enabled` defaults to `false`. When `true`, input resolution automatically enables
`output.centroid_path` **before** umbrella sampling. PMF and recrossing still use
the existing native rate executor and its ordinary backends. The subsequent
subspace solve runs once on the root rank, using the SciPy/NumPy quantum
and subspace RPMD solvers.

`mode = "apply"` computes the factor and multiplies the saved RPMD estimate.
`mode = "factor_only"` computes the same quantum/RPMD ratio without requiring a
full-system rate estimate or writing a corrected rate. It still needs the saved
system/PES and complete centroid moments: it is not a TS-only or sampling-free
calculation. In a full run, this mode does not skip the normal rate stages; use
`--stage subspace` to avoid rerunning those stages on existing data.

Temperature, bead count, masses, PES and output location are inherited from the
rate run. `[rate.subspace_correction.path]`, `.quantum` and `.rpmd` accept the
pipeline numerical controls; dimensional values use **bare atomic units**, not
unit-bearing strings. Native `.rpmd.windows` is an integer count (at least four).
The subspace section rejects unknown fields and does not accept `source`,
`output`, or a separate `nbeads`. Path environments are `frozen`, `adiabatic` and
`free_energy`; `relaxed` is only available for linear projections in the API.

The native integration supports gas-phase `task = "rate"` with **local
umbrella PMF sampling**, including the independent `[rate.pmf]` umbrella stage.
It rejects external PMFs, metadynamics and OPES because they lack the required
window centroid moments. `surface_rate` and rate-extension generations are not
supported by this native stage; use Route C (the Python API) instead.

#### Outputs, restart and failure handling

All postprocessing artifacts live in `RUN/subspace/`:
- `path.npy`: assembled reaction path in Cartesian coordinates (`bohr`).
- `C_S_1d.json` and/or `C_S_2d.json`: serialized `SubspaceCorrection` records with quantum/RPMD component rates, diagnostics and validity gates.
- `subspace_pipeline.json`: full pipeline report (path termination, sampled and static curvature $\times$ width, `solver_plans`, `models`, `validity`, `nested_sensitivity`, `instanton_anchor`).
- `request.json` and `status.json`: resolved subspace options, `reuse_key`, execution budget and final stage status (`"complete"`, `"invalid_model"`, `"solver_capacity_exceeded"`, or `"failed"`).
- `rate_quantum_corrected.json` (in `"apply"` mode): corrected rate estimate, publishable rate (when source rate is publishable), combined standard error and SHA-256 hashes of source files.

The full-run return value and `results/summary.json` include a `subspace_correction` status
record, without replacing the original rate fields. The root `summary.json`,
PMF, recrossing files, manifest and saved inputs are not rewritten by this stage.
A stage-only run writes only under `RUN/subspace`; its `status.json` is authoritative
if a previous full-run summary contains an older stage status.

Changing subspace grid, dimension or sampling controls does not change the RPMD
physics fingerprint. Editing the `.inp`/`.toml` to change these controls is allowed
when resolved physics still matches; PES/model file hashes remain enforced.
`--stage subspace` checks saved input/manifest identity before evaluating the PES
and refuses changed temperature, beads, reaction, sampling or system settings.
It is incompatible with `--resume` and warm-start options. No sampling or TS
preparation is performed on this path. Missing or incomplete centroid moments
must be regenerated by sampling; enabling the flag after a checkpoint cannot
reconstruct earlier statistics.

`reuse = true` for the native stage requires matching source data, numerical
settings, implementation hashes, bead count, temperature and saved factor checksums.
A changed request or damaged record is recomputed, never reused on temperature
alone. This is completed-result reuse, not an in-solver checkpoint. Reusing a factor
still rebuilds/checks the path and applies it to the current saved rate estimate.

An optional-stage exception in a full rate run is recorded as `status = "failed"`
and emits a warning, while the uncorrected rate remains available. An explicit
`--stage subspace` failure records the diagnostic and returns a nonzero CLI exit.
Always check the latest `status.json`: an earlier corrected artifact can remain
on disk after a failed retry or a switch to factor-only mode (if no model passes
its gates, stale `rate_quantum_corrected.json` files are removed automatically).

(one-command-from-a-rate-run)=
### Route B: Standalone postprocessing of a saved rate run (`pyrpmd subspace-correct`)

```bash
pyrpmd subspace-correct correct.toml
```

`correct.toml` names a finished rate run (`[source] rate_run = "..."`) and optional
`[path]`, `[subspace]`, `[quantum]`, `[rpmd]`, `[validity]`, `[execution]` and `[output]`
settings (`benchmarks/clhbr_subspace/correct.toml` is a complete example):

```toml
[source]
rate_run = "../../runs/clhbr-300K-production"

[path]
source = "mep"           # default IRC from refined saddle; or "centroid", "instanton", ...
step = 10.0              # mass-weighted continuation step (bohr * sqrt(m_e))
spread_factor = 2.0      # orientation-locked windows: spread <= 2 x TS spread
smoothing = 1.5          # local-quadratic fit of xbar(xi), bandwidth in window spacings

[subspace]
dimension = "auto"       # 1D always; 2D added when curvature x width > threshold
threshold = 0.5
environment = "adiabatic"
model_2d = "rph"         # "rph" (default) or "plane"
anchor = "none"          # "none" or "skeleton"

[quantum]
refinement = 3           # RPMD spline grid 3x finer than the DVR grid (same V_S)
truncation_kt = 30.0     # energy-truncated DVR: keep V <= floor + max(30 kT, truncation_min)
truncation_min = 0.0     # set to ~6 transverse quanta (Eh) at low temperature
threshold_margin = 1.0   # flux operator restricted to E >= channel threshold - 1 kT
reference_kt = 0.3       # reference slice: reactant channel within 0.3 kT of its asymptote
product_drop_kt = 15.0   # absorb products once the path is 15 kT below its maximum
dense_limit = 8000

[rpmd]
windows = 48
walkers = 32
sweeps = 800
trajectories = 1024
seed = 11

[output]
directory = "../../runs/clhbr-300K-production/subspace"
reuse = false
```

When using window-sampled centroid moments (`path.source` other than standalone `"mep"`, or
sampled curvature $\times$ width), the rate run must have been made with:

```toml
[output]
centroid_path = true
```

which makes every umbrella window of the rate PMF stage accumulate, for each recorded ring
polymer aligned onto the transition-state geometry (mass-weighted Kabsch, after assigning
the nearest equivalent reaction path): the mean centroid, the centroid covariance, the
bead-about-centroid covariance $\Sigma(\xi)$ and the mean interatomic distances. The moments
are additive, so replicas and checkpoint resumes combine exactly; they are written to
`centroid_path.npz` and read back with `mean_path_from_rate_run(run_dir)`. The switch is
analysis-only (it is not part of the physics fingerprint and does not change the
trajectories) but disables one fused native fast path, so it is off by default when
`[rate.subspace_correction]` is not enabled. Extend the umbrella windows past the dividing
surface (e.g. `stop = 1.45`) if the reaction path bends after the transition state.

(workflow)=
### Route C: Pure Python API without umbrella sampling (`pyrpmd.analysis`)

When you want to compute $C_S$ directly from a potential energy surface and saddle geometry
without running full-system umbrella windows (or to apply a correction to `surface_rate` or
an external rate), construct `H_S` and solve it directly in Python:

1. **Build `H_S`** (`pyrpmd.analysis.subspace_model`). Coordinates are mass weighted
   ($\mathrm{bohr}\cdot\sqrt{m_e}$) and axis 0 is the reaction coordinate; the dividing
   surface is a value on axis 0.
   - `model_from_function(V, axes, masses=..., dividing_surface=...)` — the Hamiltonian
     already has 1–3 coordinates.
   - `project_linear_subspace(V, grad, reference, masses, basis, axes, environment=...)`
     — a linear subspace of a larger system. The complement is `"frozen"`,
     `"relaxed"` (minimized), `"adiabatic"` (relaxed plus its harmonic zero-point
     energy, the default), or `"free_energy"` (relaxed plus harmonic quantum free energy at
     `beta`).
   - `project_path_subspace(V, grad, path, masses, axis, ...)` — a curved 1D path, for
     example an IRC/MEP or the ring-polymer mean path from constrained-centroid sampling
     (`ring_polymer_mean_path`), with a normal-cut bath and arc-length coordinate.
     `zero_point = "harmonic"` retains the historical projected-Hessian frequencies;
     `zero_point = "rph"` uses the quadratic reaction-path potential Hessian `N.T @ H @ N`
     on the linear normal cut; and `zero_point = "dvr"` solves independent sinc-DVR cuts
     along the local normal modes (using the thermal cut free energy when
     `environment = "free_energy"`). `dvr_points`, `dvr_half_width`, and `dvr_spacing` are
     recorded in the model provenance.
2. **Exact quantum rate** `exact_quantum_rate(model, beta)`
   (`pyrpmd.analysis.subspace_quantum`). Thermal flux–flux formula with an absorbing
   potential on a sinc-DVR grid:
   - `method = "sylvester"` (default up to `dense_limit = 1600` in the low-level API,
     `8000` in the pipeline) solves $H^\dagger X - X H = i F$ once; there is no time cutoff,
     so long-lived resonances are exact.
   - `method = "flux_eigenstates"` (larger 2D/3D grids without dense-only options) finds the
     significant thermal flux eigenstates by block Lanczos/subspace iteration with a
     Chebyshev Boltzmann operator and propagates them with a split operator. Norm still
     trapped at `max_time` is assigned by the absorption branching ratio of the last half of
     the run; the estimates at $T/2$, $3T/4$ and $T$ are reported as
     `diagnostics["extrapolation"]`.
3. **RPMD rate of the same `H_S`** `ring_polymer_rate(model, beta, nbeads)`
   (`pyrpmd.analysis.subspace_rpmd`): centroid thermodynamic integration (exact mean
   force, HMC with exact free ring-polymer propagation and Metropolis test) plus
   recrossing trajectories. The reported error combines walker statistics, a TI
   quadrature estimate and the transmission bootstrap.
4. **Correction factor** `subspace_correction(model, beta, nbeads)` returns a
   `SubspaceCorrection` (`factor`, `log_standard_error`, `valid`, `validity`, both
   sub-results) that can be saved with `.write(path)`. Use the bead number of the full RPMD
   run.
5. **Apply it.** `apply_subspace_correction(rate, correction, standard_error=..., beta=..., bead_factor=..., anchor_log_shift=...)`
   for a rate in hand, or `correct_rate_run(run_dir, correction)` for a saved PyRPMD
   `rate`/`surface_rate` run. The latter checks `beta` against `input/resolved.json`,
   never touches the source files, records their SHA-256 and writes
   `rate_quantum_corrected.json`. A run whose rate is not publishable keeps
   `corrected_rate_*` empty and reports only the corrected estimate.

```python
import numpy as np
from pyrpmd.analysis import (
    project_linear_subspace,
    subspace_correction,
    harmonic_bead_factor,
    apply_subspace_correction,
    correct_rate_run,
)

# 1. Project a 1D or 2D subspace around the saddle x_ts (axes in bohr * sqrt(m_e))
sub = project_linear_subspace(
    V, grad, x_ts, masses, basis_mw,
    axes=[np.linspace(-150.0, 150.0, 301)],
    environment="adiabatic",
    dividing_surface=0.0,
    rigid_body=True,
)

# 2. Compute C_S = k_QM(H_S) / k_RPMD(H_S) at matched beta and bead count P
corr = subspace_correction(sub, beta, nbeads=32)
corr.write("C_S_300K.json")

# 3a. Apply to a saved rate or surface_rate directory:
correct_rate_run("runs/rate-300K", corr)

# 3b. Or apply to a numerical rate in memory (optionally with finite-bead factor B_P):
bp = harmonic_bead_factor(omega_ts_complement, omega_reactant_complement, beta, nbeads=32)
res = apply_subspace_correction(k_rpmd, corr, standard_error=se_rpmd, beta=beta, bead_factor=bp["factor"])
```

---

## 2. Complete parameter and unit reference

> **Critical unit rules for subspace inputs (`input_subspace.py` & `subspace_pipeline.py`)**
> 1. **No unit strings are accepted:** Every numerical parameter in `[rate.subspace_correction.*]` and `correct.toml` must be a bare number (`allow_strings = False`). Writing `"10.0 bohr"` or `"300 K"` inside the subspace tables raises an `InputError`.
> 2. **Mass-weighted lengths use electron mass ($m_e = 1$), not amu:** Path steps and grid lengths (`step`, `mep_displacement`, `spacing_1d`, `spacing_2d`, `padding`, `dvr_half_width`, `dvr_spacing`) are in **$\mathrm{bohr}\cdot\sqrt{m_e}$**. Since $1\,\mathrm{amu} \approx 1822.89\,m_e$, displacing a $1\,\mathrm{H}$ atom by $0.23\,\mathrm{bohr}$ equals $0.23 \times \sqrt{1836} \approx 10\,\mathrm{bohr}\cdot\sqrt{m_e}$.
> 3. **Energies and frequencies are in $\mathrm{E_h}$ (hartree, $\hbar = 1$):** `truncation_min`, `asymptote_tolerance`, and `soft_mode_frequency` ($2.3\times 10^{-4}\,\mathrm{E_h} \approx 50\,\mathrm{cm}^{-1}$) are in $\mathrm{E_h}$.
> 4. **Cartesian Hessian step is in $\mathrm{bohr}$, and crossover temperature is in $\mathrm{K}$:** `hessian_displacement` ($5\times 10^{-3}\,\mathrm{bohr}$) is the physical Cartesian displacement of the lightest atom; `crossover_temperature` is in kelvin ($\mathrm{K}$).

### 2.1 Top-level, subspace model, source and output controls

| `.inp` key (`[rate.subspace_correction]`) | `correct.toml` key | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| `enabled` | *(not used)* | `bool` | `false` | `true`, `false` | — (enables native stage and sets `output.centroid_path = true` before sampling) |
| `mode` | *(API `apply=`)* | `str` | `"apply"` | `"apply"`, `"factor_only"` | — (`"apply"` writes `rate_quantum_corrected.json`; `"factor_only"` writes only $C_S$) |
| `reuse` | `[output] reuse` | `bool` | `true` (`.inp`) / `false` (`correct.toml`) | `true`, `false` | — (reuse content-verified saved $C_S$ when no `[execution]` table is active) |
| *(inherited from run)* | `[source] rate_run` | `str` | `None` (required in `correct.toml`) | Path to saved `rate` run | — (rejected in `.inp`) |
| *(always `<run>/subspace`)* | `[output] directory` | `str` | `None` (`<rate_run>/subspace`) | Output directory path | — (rejected in `.inp`) |
| `dimension` | `[subspace] dimension` | `str`, `int`, or `list[int]` | `"auto"` | `"auto"`, `1`, `2`, `[1]`, `[2]`, `[1, 2]` | Dimensionless (`"auto"` runs 1D and adds 2D if $\kappa\rho > \text{threshold}$ or 1D `bath` is unstable) |
| `threshold` | `[subspace] threshold` | `float` | `0.5` | $> 0$ | Dimensionless curvature $\times$ width threshold ($\kappa\rho$) for adding 2D in `"auto"` mode |
| `environment` | `[subspace] environment` | `str` | `"adiabatic"` | `"frozen"`, `"adiabatic"`, `"free_energy"` | — (transverse complement treatment; `"relaxed"` is available only in Python `project_linear_subspace`) |
| `model_2d` | `[subspace] model_2d` | `str` | `"rph"` | `"rph"`, `"plane"` | — (`"rph"`: unrolled MEP reaction-path Hamiltonian; `"plane"`: fixed TS plane) |
| `anchor` | `[subspace] anchor` | `str` | `"none"` | `"none"`, `"skeleton"` | — (`"skeleton"` adds full-vs-skeleton instanton steepest-descent shift $A_H - A_{S^0}$ below $T_c$) |
| `curvature` | `[subspace] curvature` | `str` | `"none"` | `"none"`, `"small"` | — (1D path effective mass: `"none"` unit mass, or `"small"` SCT effective mass $\mu_{\mathrm{eff}}(\sigma)$) |
| `zero_point` | `[subspace] zero_point` | `str` | `"harmonic"` | `"harmonic"`, `"rph"`, `"dvr"` | — (1D normal-cut transverse energy: projected Hessian, RPH normal Hessian, or 1D sinc-DVR cuts) |
| `dvr_points` | `[subspace] dvr_points` | `int` | `81` | $\ge 8$ (even values incremented to odd) | Grid points per 1D normal-mode cut when `zero_point = "dvr"` |
| `dvr_half_width` | `[subspace] dvr_half_width` | `float` | `None` (auto: $8/\sqrt{\omega}$) | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — half-width of each 1D normal-cut DVR box (`zero_point = "dvr"`) |
| `dvr_spacing` | `[subspace] dvr_spacing` | `float` | `None` (uses `dvr_points`) | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — grid spacing of each 1D normal-cut DVR (overrides `dvr_points` if set, min 8 points) |
| `hessian_displacement` | `[subspace] hessian_displacement` | `float` | `5e-3` | $> 0$ | $\mathrm{bohr}$ — Cartesian finite-difference step of the lightest atom ($h_{\mathrm{mw}} = d \sqrt{\min m_i}$) |

### 2.2 Reaction path construction and continuation (`path`)

| `.inp` key (`[rate.subspace_correction.path]`) | `correct.toml` key (`[path]`) | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| `source` | `source` | `str` | `"mep"` | `"mep"`, `"pmf_minimum"`, `"centroid"`, `"centroid_all"`, `"bead_weighted"`, `"least_action"`, `"instanton"` | — (reaction-path core construction; `"mep"` refines TS saddle and follows IRC) |
| `step` | `step` | `float` | `10.0` | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — steepest-descent continuation arc-length step |
| `max_steps` | `max_steps` | `int` | `None` (no limit) | $\ge 1$ (if set) | Steps — optional hard cap on continuation steps; exhausting it fails the `path` gate (`max_steps_exhausted`) |
| `spread_factor` | `spread_factor` | `float` | `2.0` | $> 0$ | Dimensionless — keeps umbrella windows whose Kabsch alignment spread is $\le \text{spread\_factor} \times$ TS spread |
| `asymptote_tolerance` | `asymptote_tolerance` | `float` | `1e-6` | $> 0$ | $\mathrm{E_h}$ — energy range threshold over trailing $10 \times \text{step}$ arc length to declare `asymptote_converged` |
| `asymptote_patience` | `asymptote_patience` | `int` | `50` | $\ge 1$ | Windows — consecutive windows without a 10% drop in trailing energy range before `asymptote_not_approached` |
| `smoothing` | `smoothing` | `float` | `1.5` | $\ge 0$ | Dimensionless (multiples of $2\,\mathrm{median}(\Delta\xi)$) — local quadratic smoothing bandwidth for centroid cores (`0` disables) |
| `reactant_xi` | `reactant_xi` | `float` | `0.0` | $< 1.0$ | Dimensionless $\xi$ — target reaction coordinate where reactant climb out of a pre-reaction complex stops ($\xi=0$ is $R_\infty$) |
| `mep_displacement` | `mep_displacement` | `float` | `None` (`0.1 * step`) | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — initial $\pm$ displacement along the TS imaginary mode for `source = "mep"` |
| `ts_gradient_tolerance` | `ts_gradient_tolerance` | `float` | `1e-5` | $> 0$ | $\mathrm{E_h}/\mathrm{bohr}$ — max Cartesian vibrational gradient for P-RFO saddle refinement (`source = "mep"` and `anchor = "skeleton"`) |
| `pmf_constraint_tolerance` | `pmf_constraint_tolerance` | `float` | `1e-8` | $> 0$ | Dimensionless $\xi$ — max $\lvert\xi(x) - \xi_{\mathrm{window}}\rvert$ residual in `source = "pmf_minimum"` |
| `least_action_points` | `least_action_points` | `int` | `9` | $\ge 2$ | Count — number of interpolation scan points $\alpha \in [0, 1]$ for `source = "least_action"` |
| `instanton_beads` | `instanton_beads` | `int` | `None` (`max(8, nbeads)` or `max(nbeads, 64)` for anchor) | $\ge 1$ (rounded up to even $\ge 8$) | Beads — full ring-polymer bead count for `source = "instanton"` and `anchor = "skeleton"` |
| `instanton_max_iterations` | `instanton_max_iterations` | `int` | `200` (`300` for anchor) | $\ge 1$ | Iterations — optimizer iteration limit for `source = "instanton"` and `anchor = "skeleton"` |
| `crossover_temperature` | `crossover_temperature` | `float` | `None` (from TS $\omega^\ddagger$) | $> 0$ | $\mathrm{K}$ — fallback crossover temperature $T_c$ if automatic TS Hessian evaluation fails in `source = "instanton"` |

### 2.3 Exact quantum solver and DVR grid controls (`quantum`)

| `.inp` key (`[rate.subspace_correction.quantum]`) | `correct.toml` key (`[quantum]`) | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| `spacing_1d` | `spacing_1d` | `float` | `None` (auto de Broglie / `refinement`) | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 1D sinc-DVR grid spacing along arc length |
| `spacing_2d` | `spacing_2d` | `float` or `[float, float]` | `None` (auto from $\sigma_\perp/3$ and $\lambda_{\mathrm{dB}}$) | $> 0$ (scalar or 2-element list) | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 2D coarse DVR grid spacing (fine RPMD spline grid uses `spacing_2d / refinement`) |
| `refinement` | `refinement` | `int` | `3` | $\ge 1$ | Dimensionless ratio — fine projection/RPMD-spline grid is `refinement` times finer than the 2D DVR grid |
| `absorber_fraction` | `absorber_fraction` | `float` | `0.2` | $(0, 0.5)$ (strictly $0 < x < 0.5$) | Dimensionless fraction — default share of 1D box length occupied by absorbing potential at each end |
| `threshold_margin` | `threshold_margin` | `float` | `1.0` | $> 0$ | $k_B T$ — sets thermal flux energy floor $E_{\mathrm{floor}} = \max(E_{\mathrm{thresh},R}, E_{\mathrm{thresh},P}) - \text{threshold\_margin}\cdot k_B T$ |
| `truncation_kt` | `truncation_kt` | `float` | `30.0` | $> 0$ | $k_B T$ — thermal DVR truncation offset above $E_{\mathrm{floor}}$ |
| `truncation_min` | `truncation_min` | `float` | `0.0` | $\ge 0$ | $\mathrm{E_h}$ — temperature-independent minimum truncation offset above $E_{\mathrm{floor}}$: $\Delta E_{\mathrm{trunc}} = \max(\text{truncation\_kt}\cdot k_B T,\ \text{truncation\_min})$ |
| `padding` | `padding` | `float` | `None` (`40.0`) | $\ge 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — transverse box half-width / padding around the path in 2D (`rph` uses $\max(\text{padding}, 6\sigma_{\perp,\max})$) |
| `dense_limit` | `dense_limit` | `int` | `8000` (pipeline) / `1600` (API) | $\ge 1$ | Grid points — max retained DVR points allowed on the dense Sylvester route ($\approx 192 N^2$ bytes) |
| `reference_kt` | `reference_kt` | `float` | `0.3` | $> 0$ | $k_B T$ — max bare-potential offset from the asymptote allowed for the free reactant channel (and high-$T$ product asymptote) |
| `reference_density_tolerance` | `reference_density_tolerance` | `float` | `1e-2` | $> 0$ | Dimensionless — max Wigner–Kirkwood density mismatch $\beta^2\lvert V''\rvert/12 + \beta^3(V')^2/24$ at the reactant reference slice |
| `product_drop_kt` | `product_drop_kt` | `float` | `15.0` | $> 0$ | $k_B T$ — energy drop below the barrier top at which products are committed (for product wells and 2D absorber start) |

### 2.4 Subspace RPMD solver controls (`rpmd`)

| `.inp` key (`[rate.subspace_correction.rpmd]`) | `correct.toml` key (`[rpmd]`) | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| *(inherited from `beads`)* | `nbeads` | `int` | `None` (inherits `run.nbeads`) | $\ge 1$ | Beads — ring-polymer bead count $P$ (rejected in `.inp`, which always uses top-level `beads`) |
| `windows` | `windows` | `int` | `48` | $\ge 4$ | Count — number of constrained-centroid TI windows from `reactant_reference` to `dividing_surface` |
| `walkers` | `walkers` | `int` | `32` | $\ge 2$ | Count — parallel HMC ring-polymer walkers per TI window |
| `burn_in` | `burn_in` | `int` | `200` | $\ge 0$ | Sweeps — equilibration HMC sweeps per walker before recording mean force |
| `sweeps` | `sweeps` | `int` | `400` | $\ge 2$ | Sweeps — production HMC sweeps per walker |
| `leapfrog_steps` | `leapfrog_steps` | `int` | `8` | $\ge 1$ | Steps — mean leapfrog steps per HMC proposal (randomized in $[0.5, 1.5] \times \text{leapfrog\_steps}$) |
| `hmc_dt` | `hmc_dt` | `float` | `None` ($0.3/\omega_{\mathrm{th}}$) | $> 0$ | $\hbar/\mathrm{E_h}$ (atomic time, $\approx 0.024189\,\mathrm{fs}$) — HMC proposal time step |
| `trajectories` | `trajectories` | `int` | `512` | $\ge 2$ | Pairs — dividing-surface parent configurations launching paired $\pm p$ recrossing trajectories |
| `md_dt` | `md_dt` | `float` | `None` ($0.1/\omega_{\mathrm{th}}$) | $> 0$ | $\hbar/\mathrm{E_h}$ (atomic time) — unconstrained RPMD recrossing time step |
| `md_time` | `md_time` | `float` | `None` ($0.3 L_0 \sqrt{\beta m_0}$) | $> 0$ | $\hbar/\mathrm{E_h}$ (atomic time) — total propagation duration of recrossing trajectories |
| `plateau_fraction` | `plateau_fraction` | `float` | `0.3` | $(0, 1]$ ($0 < x \le 1$) | Dimensionless fraction — trailing share of `[0, md_time]` averaged for $\kappa$ |
| `bootstrap` | `bootstrap` | `int` | `400` | $\ge 2$ | Replicates — paired-trajectory bootstrap draws for transmission standard error |
| `seed` | `seed` | `int` | `0` | $\ge 0$ | Dimensionless integer — RNG seed for subspace HMC and recrossing |
| *(auto)* | `backend` | `str` | `"auto"` | `"auto"`, `"native"`, `"python"` | — (selects compiled C++17 B-spline extension or Python reference in `correct.toml` / API) |

### 2.5 Validity gates and policies (`validity`)

| `.inp` key (`[rate.subspace_correction.validity]`) | `correct.toml` key (`[validity]`) | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| `unconverged_path` | `unconverged_path` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the `path` gate: both path ends must reach an accepted terminus and 1D reference slice must be free) |
| `unstable_bath` | `unstable_bath` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the `bath` gate: 1D normal-plane Hessian must have no imaginary mode above `soft_mode_frequency`) |
| `outside_grid` | `outside_grid` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the `grid` gate: RPMD estimator outside-grid share $\le \text{max\_outside\_fraction}$) |
| `max_outside_fraction` | `max_outside_fraction` | `float` | `0.01` | $[0, 1]$ | Dimensionless fraction — upper bound on `grid_exit.estimator_outside_fraction` |
| `submerged_surface` | `submerged_surface` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the `surface` gate: RPMD dividing surface must not lie $> \text{max\_submerged\_kt}$ below upstream centroid PMF peak) |
| `max_submerged_kt` | `max_submerged_kt` | `float` | `1.0` | $\ge 0$ | $k_B T$ — maximum allowed submergence of the RPMD dividing surface below the upstream free-energy bottleneck |
| `folded_ribbon` | `folded_ribbon` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the 2D RPH `ribbon` gate: $\lvert\kappa\rvert\sigma_\perp \le \text{max\_curvature\_width}$) |
| `max_curvature_width` | `max_curvature_width` | `float` | `0.5` | $> 0$ | Dimensionless — upper bound on $\max \lvert\kappa\rvert/\sqrt{2\omega_n}$ between reference slice and product absorber in 2D RPH |
| `tight_truncation` | `tight_truncation` | `str` | `"reject"` | `"reject"`, `"allow"` | — (policy for the 2D `truncation` gate: energy-truncated DVR walls must leave at least `min_truncation_quanta` transverse quanta) |
| `min_truncation_quanta` | `min_truncation_quanta` | `float` | `2.5` | $> 0$ | Dimensionless ($\hbar\omega_\perp$ units) — minimum required truncation margin above the valley floor at reference and dividing-surface slices |
| `soft_mode_frequency` | `soft_mode_frequency` | `float` | `2.3e-4` | $\ge 0$ | $\mathrm{E_h}$ ($\approx 50\,\mathrm{cm}^{-1}$) — imaginary transverse frequency cutoff below which modes are treated as harmless soft rotations |
| `on_invalid` | `on_invalid` | `str` | `"stop"` | `"stop"`, `"diagnose"` | — (`"stop"` skips solving models that fail pre-solve gates; `"diagnose"` solves them for `diagnostic_factor` but never applies them) |

### 2.6 Execution resources and parallelism (`execution`)

| `.inp` key (`[rate.subspace_correction.execution]`) | `correct.toml` key (`[execution]`) | Type | Default | Allowed values / range | Physical unit & meaning |
| --- | --- | --- | --- | --- | --- |
| `cpus` | `cpus` | `"auto"` or `int` | `"auto"` (`.inp`) / `1` (`correct.toml`) | `"auto"` or $\ge 1$ | CPU cores — total CPU budget (`"auto"` derives from `[parallel]` in `.inp` or uses process CPU affinity in `correct.toml` / MPI), capped by OS CPU affinity |
| `workers` | `workers` | `int` | Omitted ($\min(\text{tasks}, \text{cpus})$) | $\ge 1$ | Processes — max concurrent task processes in `"process"` backend; the dynamic scheduler overlaps light stages alongside heavy 2D tasks from the idle CPU budget |
| `blas_threads` | `blas_threads` | `int` | `8` ($\min(8, \text{cpus})$) | $\ge 1$ | Threads — max BLAS threads per task for dense eigensolvers, Schur and Sylvester |
| `memory_limit_mb` | `memory_limit_mb` | `float` | `75%` of RAM/cgroup (`.inp`) / `4096.0` (`correct.toml`) | $> 0$ | $\mathrm{MiB}$ ($2^{20}$ bytes) — working memory budget (per-node under MPI) checked before admitting tasks |
| *(derived from `[parallel]`)* | `backend` | `str` | `"process"` | `"serial"`, `"process"`, `"mpi"` | — (`correct.toml` / Python API only; derived automatically in `.inp`) |
| *(always enabled)* | `resume` | `bool` | `true` | `true`, `false` | — (`correct.toml` / Python API only; reuse atomic per-task checkpoints in `checkpoint_dir`) |
| *(always `<subspace>/tasks`)* | `checkpoint_dir` | `str` | `<output.directory>/tasks` | Directory path | — (`correct.toml` / Python API only; directory for `<fingerprint>.json` task checkpoints) |

---

(validity-gates)=
## 3. Validity gates and troubleshooting

### Validity gates

A positive $k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$ is not yet a correction factor: the ratio
is only meaningful if $H_S$ is a valid model. Six model gates (`path`, `bath`, `truncation`, `grid`,
`ribbon`, `surface`) plus the pre-solve capacity gate (`solver`) are evaluated, and a
model that fails one is recorded but never applied.

| Gate | Policy / threshold key | Passes when | Evidence in `subspace_pipeline.json` |
| --- | --- | --- | --- |
| `path` | `unconverged_path` (`"reject"`) | Both continuations ended at an accepted terminus — `asymptote_converged`, the reactant reference $\xi \le \text{reactant\_xi}$ after climbing out of a complex, or on the product side a well `product_drop_kt` below the barrier or a shallow asymptotic van der Waals exit settled within `reference_kt` (`product_asymptote`) — and the 1D reactant reference slice lies in the free reactant channel | `path.reactant_termination`, `path.product_termination` (`status`, `accepted`, `reason`, `steps`, `arc_length`, `tail_energy_range`, `end_energy`, `drop_below_top`, `minima`, `exit_coordinate`); `models.1.model_construction.reactant_reference` |
| `bath` | `unstable_bath` (`"reject"`), `soft_mode_frequency` (`2.3e-4` $\mathrm{E_h}$) | No complement curvature below $-\text{soft\_mode\_frequency}^2$ along the 1D/RPH path or inside the active region of the 2D `plane` model | `models.D.model_construction.significant_unstable` (arc length or grid coordinates, imaginary frequency, character) |
| `truncation` | `tight_truncation` (`"reject"`), `min_truncation_quanta` (`2.5`) | (Energy-truncated 2D/3D DVR) The truncation margin at the reactant reference and dividing-surface slices is at least `min_truncation_quanta` $\hbar\omega_\perp$ | `solver_plans.2.truncation_confinement` (`truncation_quanta`, `required_truncation_min`, `recommended_truncation_min`, `recommended_dense_limit`) |
| `grid` | `outside_grid` (`"reject"`), `max_outside_fraction` (`0.01`) | At most `max_outside_fraction` of the ring-polymer estimator saw $V_S$ outside its grid | `corrections.D.rpmd.diagnostics.grid_exit` |
| `ribbon` | `folded_ribbon` (`"reject"`), `max_curvature_width` (`0.5`) | (RPH 2D model) The path's curvature times the transverse width, $\lvert\kappa\rvert / \sqrt{2 \omega_n}$, stays at most `max_curvature_width` from the reference slice to the product absorber | `models.2.model_construction.ribbon_fold` |
| `surface` | `submerged_surface` (`"reject"`), `max_submerged_kt` (`1.0`) | The RPMD dividing surface lies at most `max_submerged_kt` below the centroid free-energy maximum between the reactant reference and the surface | `corrections.D.rpmd.diagnostics.free_energy_bottleneck` |
| `solver` | `quantum.dense_limit`, `execution.memory_limit_mb` *(cannot be overridden)* | The planned quantum route accepts the model's options and size, and the tasks fit the execution memory budget | `solver_plans.D` (also `models.D.solver_plan`) |

### Quick troubleshooting table

| Symptom / gate failure | What happened | Actionable fix |
| --- | --- | --- |
| `solver_plans.2.status = "dense_limit_exceeded"` (`status = "solver_capacity_exceeded"`) | Retained 2D DVR grid points after energy truncation exceed `quantum.dense_limit` (common at higher $T$ or when raising `truncation_min`) | Read `solver_plans.2.required_dense_limit` and `dense_memory_mb` in `subspace_pipeline.json` / `status.json`. If RAM suffices ($\approx 192 N^2$ bytes), raise `quantum.dense_limit`; otherwise coarsen `quantum.spacing_2d` or shrink `quantum.padding` and verify convergence |
| `solver_plans.D.status = "memory_limit_exceeded"` | Estimated dense quantum or RPMD task memory exceeds `execution.memory_limit_mb` | Increase `execution.memory_limit_mb` (or reduce `execution.workers` so fewer tasks share node memory), or shrink the 2D grid |
| `path` fails with `asymptote_not_approached` or `step_collapsed` | Steepest-descent continuation wandered in a flat/noisy long-range tail, left the fitted PES domain, or took too large/small a step | Check PES behaviour at large separation; tune `path.step`, `path.asymptote_tolerance`, or `path.asymptote_patience`; for noisy umbrella centroids use `path.source = "mep"` (the default) |
| `path` fails with `reactant reference slice ... rule = "not_free"` | The reactant channel is not flat to `quantum.reference_kt` ($0.3\,k_B T$) and Wigner–Kirkwood tolerance `reference_density_tolerance` ($10^{-2}$) over at least 4 grid spacings | Extend the path further into the asymptotic channel (smaller `path.asymptote_tolerance` or smaller `path.reactant_xi` $< 0$ when climbing out of a pre-reaction complex) |
| `bath` fails on 1D model (`tangent_misaligned` or `ridge`) | A transverse vibrational mode has imaginary frequency $> \text{soft\_mode\_frequency}$ ($50\,\mathrm{cm}^{-1}$) along the 1D path | Keep `dimension = "auto"` (or set `dimension = 2` / `[1, 2]`): the pipeline automatically escalates to 2D (`dimension_escalation`) so the unstable transverse mode enters $S$. If caused by single-precision PES noise on H atoms, increase `hessian_displacement` (e.g. `1e-2` bohr) |
| `ribbon` fails on 2D RPH model (`kappa * sigma > 0.5`) | Tight reaction-path bend folds the RPH ribbon coordinates $x = w(s) + q n(s)$ inside the transverse ground-state width | Switch to the fixed-plane 2D model: set `model_2d = "plane"` |
| `surface` fails (`surface_below_maximum_kt > 1.0`) | Submerged barrier behind an entrance complex lies $> 1\,k_B T$ below the upstream centroid free-energy bottleneck; short-time RPMD traps in the complex | Place the RPMD dividing surface at the free-energy bottleneck reported in `free_energy_bottleneck.maximum_position` (for pure Python models, see `capture_surface()`) |
| `grid` fails with `dominant: ti` or `dominant: surface` | Constrained TI or dividing-surface ring polymers sampled outside the transverse edge of the 2D grid | Increase `quantum.padding` (default `40.0` $\mathrm{bohr}\cdot\sqrt{m_e}$) to widen the 2D box across the channel |
| `grid` fails with `dominant: recrossing` | Recrossing trajectories exited where $V_S$ still rises outwards ($> 0.1\,k_B T$ per thermal length) or recrossed the dividing surface after leaving the grid | Extend the box/absorber further along the channel (increase `quantum.product_drop_kt` or decrease `quantum.reference_kt`) |
| `InputError: ... requires locally sampled umbrella windows` or incomplete `centroid_path.npz` | `[rate.subspace_correction]` was enabled after a checkpoint, or with MetaD/OPES/external PMF | Run umbrella sampling from the start with `rate.subspace_correction.enabled = true` (or `output.centroid_path = true`), or use `pyrpmd subspace-correct` with `path.source = "mep"` / Python API |

### Detailed gate mechanics

**Ribbon.** The RPH coordinates $x = w(s) + q n(s)$ fold at $q = 1 / \kappa$. When the
radius of curvature is comparable to the width $\sigma = 1 / \sqrt{2 \omega_n}$ of the transverse
ground state, the fold lies inside the wavefunction and the reaction-path Hamiltonian no
longer describes the system (the large-curvature regime). The validated curved valleys have
$\kappa \sigma$ of 0.21 to 0.25; on a model with deep complexes and a tight bend at the saddle
(0.56) the RPH quantum rate came out 66% too high. Use `model_2d = "plane"` there.

**Surface.** The Bennett–Chandler rate does not depend on the dividing surface only if the
transmission coefficient is converged to its long-time value. A barrier submerged below the
reactant asymptote, behind an ion–dipole complex, is not the bottleneck: the centroid free
energy at it lies below that of the entrance channel, so QTST through it exceeds the rate the
entrance channel allows by up to $\exp(\beta \cdot \text{depth})$, and trajectories trapped in the complex
keep recrossing far longer than any recrossing run. On a 6D submerged-barrier model the RPMD
rate through the barrier came out 2300 times the free-particle flux with a transmission
coefficient still at 0.79. The gate refuses such a factor and reports where the free-energy
maximum is; the RPMD surface belongs there. The quantum rate, computed with absorbers, does
not depend on the surface.

**Path.** The continuations have no step budget. How far an asymptote lies is a
property of the potential — an ion–dipole tail needs far more steps than a
neutral one — and a path that stops at a count is not a path to the asymptote.
A continuation ends when the energy varies by less than `asymptote_tolerance`
over the last `10 * step` of arc length. The window is an arc length rather than
ten points, so halved steps at a bend cannot imitate a flat tail. It ends
unconverged with `asymptote_not_approached` when that trailing energy range has
not fallen by 10% during `asymptote_patience` (50) consecutive windows — a bound
coordinate, a potential leaving its fitted domain, a path that wanders — with
`step_collapsed` when the tangent cannot be resolved at the smallest step, and
with `potential_failure` on a failed or non-finite evaluation. `path.max_steps`
is optional; if set, reaching it is `max_steps_exhausted` and fails the gate.

A descending continuation that runs into a local minimum — a pre- or
post-reaction complex, typical of ion–molecule reactions with a submerged
barrier — stops there with `minimum_reached` (issue #191). Steepest descent
cannot leave a minimum, and climbing on along the incoming tangent only climbs
a wall: such paths used to wander for thousands of steps and end with
`asymptote_not_approached`, or pass through the well and reach the asymptote on
its far side. A minimum is judged per side. On the product side it is accepted
(`accepted: "product_well"`) when it lies at least `quantum.product_drop_kt`
($15\,k_B T$) below the highest energy on the path: products are committed there,
which is all the correction needs, and the 1D model then absorbs products only
inside the committed part of the well (`models.1.product_absorber`).

On the reactant side a minimum is not an end: both rates are normalized on a
slice of the *free* reactant channel, which lies beyond the complex. The
reactant continuation records the complex (`minima`) and climbs out of it along
decreasing rate coordinate $\xi$: each point is the minimum of $V$ on the
hyperplane one step further along $-\nabla\xi$ (mass-weighted, rigid-body modes
removed). These are, to first order, the minima of $V$ on the $\xi = \mathrm{const}$
surfaces of the umbrella windows; $\xi$ decreases monotonically, so the path
cannot turn back into the well. It ends with `coordinate_target_reached`
(`accepted: "reactant_reference"`) at $\xi \le \text{path.reactant\_xi}$ — by default
$\xi = 0$, where the fragments are $R_\infty$ apart, the reactant reference of the
rate itself — or earlier with `asymptote_converged` if the energy is already
flat. $\xi$ is defined for every rate run, so the same rule serves neutral and
ionic reactions and needs no model of the long-range tail. Called without a
reaction coordinate (the library API), a reactant continuation that stops in a
complex still fails the gate with the reason `bound pre-reaction complex`.

The 1D model puts its reactant reference slice `absorber_fraction + 0.05` of
the path length from the reactant end, behind the reactant absorber. As for the
2D model, the slice must lie in the free reactant channel: the potential along
the path there within `quantum.reference_kt` ($0.3\,k_B T$) of its value at the
reactant end, and its Wigner–Kirkwood density correction within
`quantum.reference_density_tolerance` ($10^{-2}$). (The bare potential is used rather than $V_S$: the zero-point
energy of soft fragment rotations is noisy at that scale.) If the default slice
is not free, it moves to the last point of the flat channel that starts at the
reactant end and the reactant absorber shrinks in front of it
(`models.1.model_construction.reactant_reference.rule = "free_channel"`); if
there is no such channel (`"not_free"`), the path gate fails.

**Bath.** A vibrationally adiabatic model needs a bath that is bound in the plane
normal to the path. A negative normal-plane curvature has no zero-point energy;
the term is omitted, which lowers $V_S$ exactly where the model has failed.
Imaginary frequencies below `soft_mode_frequency` ($2.3\times 10^{-4}\,\mathrm{E_h}$, $50\,\mathrm{cm}^{-1}$) are soft
modes — free relative rotations of separating fragments, whose curvature tends
to zero with either sign; they are counted (`soft_complement_points`) and
`dropped_zero_point_bound` bounds the omitted energy. Larger ones fail the gate
and are classified from the vibrational Hessian in the basis of the tangent and
the normal plane: `tangent_misaligned` (one unstable vibrational mode, the
reaction mode, leaking into the bath because the path tangent is tilted away
from it by `tangent_to_lowest_mode_degrees`) or `ridge` (two or more: a
transverse coordinate is not a bath mode). Either way that coordinate belongs in
$S$, so `dimension = "auto"` then also builds the 2D model
(`dimension_escalation`). The finite-difference Hessian displaces the lightest
atom by `hessian_displacement` ($5\times 10^{-3}\,\mathrm{bohr}$); the former fixed mass-weighted step
moved hydrogen by $2\times 10^{-5}\,\mathrm{bohr}$, where the gradient difference of a soft mode is
comparable to the evaluation noise of a fitted potential.

**Grid.** Outside its grid a gridded $V_S$ is a constant with zero force.
`spline_outside_fraction` counts every gradient evaluation — rejected HMC
proposals, equilibration, and recrossing trajectories that left through a flat
channel end and are simply committed — so it cannot say how much of the
estimator was affected. `grid_exit` resolves it: per-stage counts (`stages`),
the bead points of the accepted TI and dividing-surface samples outside the grid
(`ti_sample_outside_fraction`, `surface_sample_outside_fraction`), and for the
recrossing trajectories the flux share that crossed the dividing surface again
after leaving, or whose centroid left where $V_S$ still rises outwards by more
than $0.1\,k_B T$ per thermal length (`recrossing_contaminated_flux_fraction`). The
gate uses the largest of the three (`estimator_outside_fraction`); its reason
lists all three and names the dominant one (`dominant_component`), because the
remedy differs: TI or dividing-surface samples outside the grid need a wider box
across the channel (`padding`), contaminated recrossing flux a box or absorber
further along it.

**Solver.** Before any rate is solved, each model gets a solver plan
(`solver_plans`): grid shape, points retained after energy truncation, the
route the exact quantum solver will take, the dense and route memory
estimates, and, with an `[execution]` table, the RPMD task estimate against
`execution.memory_limit_mb`. The energy truncation, energy floor and slice
normalization of the 2D model exist only on the dense route, so a 2D grid
that retains more than `dense_limit` points cannot be solved; it is refused
here with `status = "dense_limit_exceeded"`, `required_dense_limit` and
`recommendations` (the `dense_limit` it needs and the memory that takes, or
the keys that shrink the grid), instead of failing with a generic error after
the rest of the stage has run. The gate cannot be overridden and also applies
under `on_invalid = "diagnose"`. The retained grid grows with temperature at
fixed truncation, so a `dense_limit` sized at one temperature (11500 for the
200 K H + H₂ benchmark, 11165 retained points) is not a safe value at another;
read it from the plan. The exact solver raises the same plan as
`DenseRouteUnavailable` (a `ValueError`), and `quantum_solver_plan()` computes
it without solving.

Each dimension is solved independently: an exception in one is recorded in
`solve_failures` and `validity.failed_to_solve`, and the other dimensions are
still solved and gated. (With an `[execution]` table the task scheduler still
stops at the first failed task; its checkpoints keep the completed ones.)

When a gate fails, `C_S_Dd.json` carries `valid: false`, `factor: null`, the
ratio as `diagnostic_factor`, the gate record in `validity` and both component
rate results; `apply_subspace_correction` refuses it. With `on_invalid = "stop"`
a model whose `path`, `bath` or `ribbon` gate fails is not solved at all. If no model passes,
no corrected rate is written (a corrected rate left by an earlier run is
removed), `status.json` of the native stage carries the report status, and
`pyrpmd subspace-correct` exits nonzero. The status says why:
`solver_capacity_exceeded` if a solver plan refused a model, `failed` if every
solve that started raised (an explicit `--stage subspace` then also exits
nonzero), otherwise `invalid_model`. When the applied dimension is lower than
the curvature $\times$ width verdict because the higher one did not pass, the report
says so in `warnings`.

---

(cpus-for-the-stage)=
## 4. Multi-core performance and memory planning

### CPUs for the native `rate` stage

The native stage uses the CPU allocation the same `.inp` already gives the rate run;
no extra setting is needed:

| Global `[parallel]` | CPUs used by the subspace stage |
| --- | --- |
| absent / defaults (1 worker, 1 thread) | 1: the unchanged sequential route |
| `backend = "serial"` or `"process"` | `trajectory_workers × bead_workers × threads_per_rank` |
| `backend = "mpi"` | `threads_per_rank` of rank zero, the only rank running the stage |

The count is capped by this process's CPU affinity (for example `taskset`, or a
Slurm/cgroup CPU set). With more than one CPU the QM and RPMD tasks of each
dimension run concurrently in separate processes (threads only, under MPI),
each with OpenMP RPMD threads and at most eight BLAS threads. The memory budget defaults to 75%
of the smaller of physical memory and a cgroup v2 limit. Spawned workers get
`OMP_WAIT_POLICY=PASSIVE` unless it is already set. Optional overrides in `.inp`:

```toml
[rate.subspace_correction.execution]
cpus = 16                 # default "auto": derived from [parallel] as above; 1 = sequential
# workers = 4             # concurrent tasks; default: number of unfinished tasks
# blas_threads = 8        # default min(8, cpus)
# memory_limit_mb = 32000 # set explicitly when the job's memory limit is not a cgroup v2 limit
```

These settings never change factors or the factor-cache identity: native RPMD
results are independent of the thread count, and a later `--stage subspace` on
a different allocation reuses completed tasks from `subspace/tasks/`.
`subspace/status.json` records the budget that was used.

### Memory estimation and sizing `dense_limit`

Before allocating any solver arrays, `quantum_solver_plan()` and the task scheduler estimate
peak working memory for each task:

- **Dense Sylvester quantum route (`method = "sylvester"`):**
  $$
  \mathrm{Memory}_{\mathrm{dense}} \approx 64\,\mathrm{MiB} + \frac{192\,N_{\mathrm{retained}}^2 + 64\,N_{\mathrm{grid}} + 32\,N_\perp^2}{2^{20}}\,\mathrm{MiB}
  $$
  where $N_{\mathrm{retained}}$ is the number of DVR grid points with $V_S \le E_{\mathrm{trunc}}$
  (four complex $N_{\mathrm{retained}} \times N_{\mathrm{retained}}$ matrices live simultaneously
  during the Schur and triangular Sylvester steps: $\sim 192\,N_{\mathrm{retained}}^2$ bytes).
  For example:
  - $N_{\mathrm{retained}} = 4000 \implies \sim 3.0\,\mathrm{GiB}$
  - $N_{\mathrm{retained}} = 8000 \text{ (default } \texttt{dense\_limit}\text{)} \implies \sim 11.8\,\mathrm{GiB}$
  - $N_{\mathrm{retained}} = 11500 \text{ (200 K H + H}_2\text{ benchmark)} \implies \sim 24.3\,\mathrm{GiB}$
- **Iterative flux-eigenstate quantum route (`method = "flux_eigenstates"`, 1D/2D/3D without dense-only options):**
  $$
  \mathrm{Memory}_{\mathrm{iter}} \approx 64\,\mathrm{MiB} + \frac{256\,N_{\mathrm{grid}}\,N_{\mathrm{states}} + 64\sum_a n_a^2}{2^{20}}\,\mathrm{MiB}
  $$
- **Subspace RPMD task (`ring_polymer_rate`):**
  $$
  \mathrm{Memory}_{\mathrm{RPMD}} \approx 64\,\mathrm{MiB} + \frac{192\max(N_{\mathrm{win}} N_{\mathrm{walk}},\, 2 N_{\mathrm{traj}})\,P\,D + 48 N_{\mathrm{traj}}(N_{\mathrm{steps}} + 1) + 64 N_{\mathrm{grid}}}{2^{20}}\,\mathrm{MiB}
  $$

Set `execution.memory_limit_mb` below the job's physical or Slurm memory allocation to leave
headroom for Python, loaded libraries and the coordinator process. When running multiple concurrent
`workers` on a single node, the scheduler admits a new task from the queue only while the sum of
active task memory estimates stays within `memory_limit_mb`.

(hybrid-cpu-and-mpi-execution)=
### Hybrid CPU and MPI execution

The gridded RPMD solver uses a C++17 extension by default when available. It
copies SciPy's cubic B-spline knots and coefficients and evaluates the same
potential and analytic gradient in C++. HMC proposal trajectories and paired
recrossing trajectories release the GIL and run without Python callbacks.
OpenMP distributes independent ring polymers across threads; SIMD covers the
normal-mode updates. Free ring propagation uses the existing FFT implementation
(direct Fourier transforms for non-power-of-two bead counts).

Python still owns the NumPy random stream, randomized proposal lengths, the
Metropolis test, shared burn-in step-size adaptation, walker statistics, TI,
and paired-trajectory bootstrap. Thread count does not change random draws or
reduction order. The FFT and spline arithmetic can differ slightly from the
Python reference; long chaotic trajectories need statistical comparison as
well as short trajectory comparisons. The physical definitions, error formulas
and existing result fields are retained.

For a single calculation in Python, pass `threads=32` to `ring_polymer_rate`. Select
`backend="python"` for the reference implementation or `backend="native"` to
require the compiled kernel. `backend="auto"` reports a fallback reason if the
extension is missing or a model has Python potential callbacks. Callbacks are
never silently replaced by an interpolated potential: use an explicit
`SubspaceModel` containing the intended grid to enable native propagation.

The quantum solver retains SciPy eigensolvers and complex Schur decomposition.
Dense DVR assembly avoids full Kronecker temporaries. The Sylvester equation is
solved in triangular blocks with LAPACK `ztrsyl` and BLAS matrix updates;
LAPACK scaling or near-degeneracy triggers the original full solve. A dense
memory estimate is checked before allocation. This changes neither the DVR
grid nor the flux operator, absorbing potential or reactant normalization.

Add an `[execution]` section to a standalone `correct.toml` to overlap QM/RPMD and, when both
dimensions are requested, all four stages of $C_S(\mathrm{1D})$ and $C_S(\mathrm{2D})$:

```toml
[execution]
backend = "process"
cpus = "auto"        # or an explicit positive integer, e.g. 32
# workers = 4        # optional cap on concurrent tasks
blas_threads = 8
memory_limit_mb = 8192
resume = true
# checkpoint_dir = "subspace-tasks"  # default: output directory / tasks

[rpmd]
backend = "native"
```

`cpus` is the total CPU budget (`"auto"` uses this process's CPU affinity, or the
node affinity union under MPI) and `workers` caps the maximum number of concurrent
tasks. The local `process` scheduler dynamically allocates threads from the idle
CPU budget and overlaps lighter stages alongside heavy 2D tasks based on each
task's workload and scaling cap, avoiding stranding heavy tasks at
$\lfloor\text{cpus} / \text{workers}\rfloor$ threads. BLAS uses at most
`blas_threads` (default 8) and never exceeds that process's CPU allocation.
Large thread teams can slow the sequential stages of complex Schur decomposition;
the cap is independently adjustable for the installed BLAS and matrix size. Set
`OMP_WAIT_POLICY=PASSIVE` before launching mixed Python/OpenMP workloads so idle
dynamics threads sleep during sampling and analysis. BLAS thread limits are scoped
inside separate processes, avoiding conflicting thread limits in Python thread pools.

The same scheduler is available through Python:

```python
from pyrpmd.analysis import subspace_correction, subspace_corrections

# Guard process-spawning code with if __name__ == "__main__" in a script.
corr = subspace_correction(model, beta, 32,
    execution={"backend": "process", "cpus": 32, "workers": 2})
corrections = subspace_corrections({
    "1d": {"model": model1, "beta": beta, "nbeads": 32},
    "2d": {"model": model2, "beta": beta, "nbeads": 32},
}, execution={"cpus": 32, "workers": 4, "memory_limit_mb": 8192})
```

For multiple nodes, install the existing `mpi` extra and set
`execution.backend = "mpi"`. Launch the same command on all ranks, for example
with two ranks per node on two nodes:

```bash
mpiexec -n 4 --map-by ppr:2:node:PE=16 pyrpmd subspace-correct correct.toml
```

The launcher example uses Open MPI; use the equivalent binding options for
your MPI implementation and scheduler. `cpus` and `memory_limit_mb` are
**per-node** budgets in MPI mode. Each shared-memory node divides them among
its ranks; `workers` applies only to the local process backend. Rank zero
builds models and writes the pipeline report, dynamically dispatches tasks,
and also computes. Only gridded models and quantum grid arrays travel between
ranks; no live PES object or pipeline closure is sent. All ranks must call
the Python API. Rank zero alone supplies `cases` to `subspace_corrections`;
other ranks may pass `None`. There are no MPI collectives inside time steps.
A single correction currently supplies two distributed tasks; two dimensions
supply four. This is task-level scaling, not distributed dense linear algebra.

Each completed quantum or RPMD result is atomically checkpointed, including
the RPMD time series. Identity includes the grids, masses, temperature,
beads, potential metadata, sampled absorber/truncation arrays, numerical and
sampling settings, seed, solver implementation and NumPy/SciPy versions.
Changing thread counts or rank placement can reuse results. Changed physical
or sampling inputs cannot. With an `[execution]` section the task cache replaces
the older `output.reuse` shortcut, which checked temperature alone. A failed
or interrupted task reruns from its original seed; completed tasks survive a
job/launcher restart. This is restart between tasks, not continuation within
an unfinished HMC chain or recovery of a failed MPI rank inside a live job.

Diagnostics include actual OpenMP team size, native/Python polymer-step
counts, Python potential callbacks, fallback reasons, per-task CPU/wall time,
BLAS limits, memory estimates and checkpoint reuse. The reproducible H + H₂
parallel benchmark is `benchmarks/h3_subspace/parallel.py`.

---

## 5. Model theory, low-temperature essentials, and convergence criteria

### 5.1 Six-step pipeline mechanics

Whether invoked via `[rate.subspace_correction]` or `pyrpmd subspace-correct`, the pipeline
executes six stages and writes every intermediate quantity to `subspace_pipeline.json`:

1. **Path source (`path.source`).** `source = "mep"` (the default) is the minimum-energy path
   described below; `"centroid"` is the historical mean-centroid path (the default before
   0.2.x). `"bead_weighted"` shifts each locked-window centroid along the transverse principal
   eigenvector of the bead covariance $\Sigma(\xi)$ by its bead RMS $\sqrt{\lambda_{\max}}$, with
   the sign chosen from the local curvature. `"instanton"` uses the folded half-ring from
   `pyrpmd.instanton.optimize_instanton` below the crossover temperature and records a
   centroid fallback at or above $T_c$ (or when the orbit fails to converge). `"least_action"`
   scans a one-parameter interpolation between the sampled core and its endpoint chord and
   selects the smallest one-dimensional WKB action. `"centroid_all"` uses the mean centroids
   of every umbrella window, over the full sampled $\xi$ range of the PMF.

   A centroid is a thermal average, not a geometry: a librating or ring-opening coordinate
   has a centroid shorter than any bond the ring polymer visits, and freely rotating fragments
   have no mean geometry. Two sources are built on the potential instead
   (`pyrpmd.analysis.subspace_paths`):

   - `"mep"`: the intrinsic reaction coordinate. The rate run's transition-state geometry is
     refined to a first-order saddle (partitioned rational-function optimization with a
     finite-difference Hessian, rigid-body modes removed; `ts_gradient_tolerance`,
     $10^{-5}\,\mathrm{E_h}/\mathrm{bohr}$), displaced by `mep_displacement` (default `step / 10`,
     mass-weighted) along its imaginary mode, and continued by steepest descent on both sides.
     It is a minimum-energy path everywhere, which is what the normal-cut adiabatic model
     assumes; the gradient across it is a chord error of order the step. A geometry that does
     not refine to a saddle with exactly one negative curvature is refused.
   - `"pmf_minimum"`: for every umbrella window, the minimum of $V$ on that window's
     $\xi = \mathrm{const}$ surface, marched out from the window nearest the transition state so that
     the points stay in one basin. It covers the full sampled range of the PMF with physical
     bond lengths, and its point on $\xi = 1$ is the saddle; it is a distinguished-coordinate
     path ($\nabla V$ parallel to $\nabla \xi$, not in general to the tangent).

   Both are tested on a LEPS H + H₂ surface, with and without ion–induced-dipole complexes
   (`tests/test_subspace_path_cores.py`). All sources retain their selection and fallback
   diagnostics in `subspace_pipeline.json`.
2. **Mean path and path to both asymptotes.** For the centroid source, windows where the
   aligned centroids stay compact (spread at most `spread_factor = 2.0` times that near the transition state) are
   kept; near the reactant asymptote the fragments rotate freely and a mean geometry has no
   meaning. $\bar x(\xi)$ is fitted by local quadratic regression in $\xi$ (window means carry
   transverse noise). For a linear reference the mean path is built from mean bond lengths
   along the axis: the Kabsch rotation about the axis is undetermined, and axial projections
   of librating light atoms are shorter than the bonds. Every selected core is then continued
   by steepest-descent path following (minimum on a sphere around each point, forward hemisphere, rigid-body modes
   excluded; uphill segments use the hyperplane minimum), with step halving at sharp turns.
   The continuation runs until the energy is flat over ten step lengths of arc. A product
   continuation may stop in a deep post-reaction complex; a reactant continuation climbs out
   of a pre-reaction complex along decreasing $\xi$ to $\xi = \text{reactant\_xi}$ (`0.0`). There is no step budget, and
   each end records how it terminated and whether that end is accepted (see
   [Validity gates](#validity-gates)).
3. **Curvature $\times$ width ($\kappa\rho$).** From $\Sigma(\xi)$ along the sampled windows
   ($\rho = \sqrt{2\,n^{\mathsf T} \Sigma n}$ along the curvature normal) and, along the whole path, with
   the harmonic bead width of the mode along the normal. Above `threshold` (default `0.5`)
   a 2D subspace is added.
4. **1D model** along the path (normal-cut vibrationally adiabatic, thermal free-energy, or frozen, with rigid-body modes
   removed) and its $C_S(\mathrm{1D})$. With large curvature the normal cuts cross inside the bend and
   the adiabatic potential develops a cusp; $C_S(\mathrm{1D})$ is then not trustworthy, which is
   what the curvature test says.
5. **2D model.** `model_2d` selects it:

   - `"rph"` (default) is the reaction-path Hamiltonian on the MEP
     (`pyrpmd.analysis.subspace_rph`). Axis 0 is the arc length $s$ of the path, axis 1 the
     displacement $q$ along a normal $n(s)$; the rest of the bath stays adiabatic as in the
     1D model. With $x(s, q) = w(s) + q n(s)$ the kinetic energy is
     $p_s^2 / [2 (1 - \kappa q)^2] + p_q^2 / 2$, $\kappa = t' \cdot n$: on the inner side of a bend
     the path is shorter, which is corner cutting. $n$ is transported along the path without
     rotating about it (a Bishop frame), so the ribbon has exactly this metric; it is the
     transport of the curvature direction that carries most of the curvature weighted by
     $\exp[-\beta (V_{\mathrm{top}} - V)]$ (`captured_curvature` in the report). A ribbon with this metric
     is isometric to a strip of the plane: it is unrolled along a plane curve with the same
     signed curvature, where the Hamiltonian is the ordinary flat
     $-(\partial_X^2 + \partial_Y^2)/2 + V_S$, so the 2D solvers apply unchanged; each grid
     point is mapped back to $(s, q)$ by its nearest point on the developed curve. For a
     two-dimensional system the development is congruent to the system itself. A path that
     twists (torsion) keeps only the component of its curvature along $n$; the rest couples
     to adiabatic modes as in 1D. The box runs from the reactant reference slice (the end of
     the flat channel that starts at the reactant end, as in 1D) to two de Broglie
     wavelengths past the product absorber; axis 0 bisects the path direction at the
     reference slice and at the barrier, and the path must advance along it through the box.
     The default spacing is a third of the narrowest width of the $n$ mode on both axes.
     Validated against exact full-dimensional quantum and RPMD rates on 2D, 3D and rotated 8D
     model potentials ([docs/validation/2026-11-13-subspace-rph-2d](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-13-subspace-rph-2d/README.md)): both 2D
     models are within 3.3% of the exact correction, the 1D model 4–14% below it.
   - `"plane"` is the plane through the transition state spanned by the rate dividing-surface
     normal (mass-weighted $\nabla \xi$, so the 2D dividing line is the rate's $\xi = 1$ surface)
     and the principal direction of the path within the complement; for a collinear atom
     transfer this is exactly the collinear plane. The curvature-normal candidate is reported
     with its overlap with that plane. Absorbing regions are keyed to arc length along the
     path, so hairpin channels that leave the grid on the same side are both absorbed; the
     rate is normalized by the quantum line density on the reference slice, matching RPMD
     constrained TI along axis 0; an energy-truncated DVR keeps the dense exact route
     affordable. Both halves of $C_S$ share one $V_S$: it is projected on a grid
     `refinement` (3) times finer than the DVR; the DVR uses every third node and RPMD the
     spline of the fine grid (a spline through DVR nodes alone rings across narrow valleys).
     The box only extends to a reference slice where the reactant channel is within
     `reference_kt` ($0.3\,k_B T$) of its asymptote — $C_S$ needs both rates on the same slice,
     not at infinity — and products are absorbed once the path is `product_drop_kt` ($15\,k_B T$)
     below its maximum or, when the product channel cannot drop that far (a
     thermoneutral or endothermic reaction at high temperature: H + H₂ above about 320 K),
     once it has settled within `reference_kt` of its asymptote. The threshold
     $\max(V_{\mathrm{top}} - \text{product\_drop\_kt}\cdot k_B T,\ V_{\mathrm{asymptote}} + \text{reference\_kt}\cdot k_B T)$ is continuous in
     temperature; the former fallback, the last path point, put the box (and the dense
     grid) at the end of a path whose length is set by `asymptote_tolerance`.
     `models.2.product_absorber` records the criterion used. The default grid is anisotropic: a third of the narrowest
     in-plane vibrational width along the transfer axis, a quarter of the shortest de
     Broglie wavelength along the path axis.

   **Reference slice.** All models normalize both rates per unit length on a reactant
   reference slice: the quantum rate by the quantum (1D: classical) density, the RPMD rate
   by the centroid density. These agree only in the free channel. On a slope they differ
   at order $\hbar^2$ (Wigner–Kirkwood: $\rho_{\mathrm{Q}}/\rho_{\mathrm{cl}} = 1 - \beta^2 V''/12 + \beta^3 (V')^2/24$,
   the centroid density has $-\beta^2 V''/24$, unit mass), which does not cancel in $C_S$.
   The slice is therefore the end of the channel, starting at the reactant end, along
   which the path potential is within `reference_kt` ($0.3\,k_B T$) of the reactant end **and**
   $\beta^2 |V''| / 12 + \beta^3 (V')^2 / 24 \le \text{quantum.reference\_density\_tolerance}$ ($10^{-2}$, a
   bound on the relative error it introduces). The energy rule alone put the 300 K slice
   of the curved-valley test model where this was $\sim 0.15$, and $C_S$ came out 5% high.

   In both 2D models the reactant reference slice is moved onto a node of the DVR grid:
   the quantum rate is normalized on the grid line nearest the reference and the RPMD TI
   starts exactly at the reference, so off a node the two rates were normalized on
   different slices, which matters wherever the channel is only flat to `reference_kt`.
6. **Nested sensitivity** $\ln(C_{\mathrm{2D}}/C_{\mathrm{1D}})$ selects the dimension, and
   `rate_quantum_corrected.json` is written.

**Exo- or endothermic reactions.** The box holds deep states of the lower channel whose net
flux vanishes but whose Boltzmann weight is enormous ($\exp(25)$ for Cl + HBr at 300 K); any
absorber reflection then swamps the reactive flux. Both models therefore restrict the
thermal flux operator to energies above the higher channel threshold minus `threshold_margin` ($1\,k_B T$)
(`exact_quantum_rate(..., energy_floor=...)`), which is exact in the continuum limit because
the cumulative reaction probability vanishes below that threshold. Without it the Cl + HBr
1D rate changes by 10% and the 2D rate by orders of magnitude when the absorber changes.

**Cl + HBr (`BrHCl_LEPS1`, 300 K)** is worked through in
[docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md):
curvature $\times$ width is 1–12 at the H-transfer corner, the 1D model develops a cusp there
with the first, kinked path (normal cuts cross; with the corrected path $C_S(\mathrm{1D}) = 0.99$), and
in the collinear plane RPMD matches the exact rate within its statistical error
($C_S(\mathrm{2D}) = 1.02 \pm 0.29$ in the pipeline run, $0.9 \pm 0.2$ with a second RPMD run), with the 2D
quantum rate converged to 2.5% in the grid. The corrected rate is $\mathrm{RPMD} \times C_S(\mathrm{2D})$.

(choosing-the-subspace)=
### 5.2 Choosing the subspace and low-temperature essentials

$S$ must contain the RPMD defect, not merely the strongly coupled modes. Static coupling
strength is the wrong criterion: a mode that follows a smooth path adiabatically is handled
by RPMD itself, so adding it to $S$ changes $C_S$ by only a few percent. Two checks decide:

- `curvature_width_criterion(curvature, width)` — the path curvature in mass-weighted
  coordinates times the transverse quantum width ($\sqrt{2} \times \text{bead RMS}$). Values $\ll 1$ let a 1D path
  carry the coupling; values near 1 mean the vibration straddles the bend (collinear H + H₂:
  0.80–1.01) and the transverse coordinate belongs in $S$. The 0.5 threshold is provisional.
- `nested_subspace_sensitivity([C_1D, C_2D, ...])` — the change of $\ln C_S$ when a coordinate
  is added. It is the final verdict and costs one larger quantum calculation.

[Subspace transfer theory](subspace-transfer-theory.md) derives why: static environment effects
(zero-point energy, imaginary-time corner cutting), which curvature $\times$ width measures, cancel from
the transfer error; what remains is the environment's influence on RPMD's dynamical error.
[The instanton anchor](subspace-instanton-anchor.md) derives the closed-form steepest-descent part
$\varepsilon_{\mathrm{SD}}=2\pi/(\beta\hbar\eta_1)$ of the RPMD defect and anchors $C_S$ to the
full-system instanton relative to the bare-potential skeleton $S^0$ (`SubspaceModel.skeleton()`,
`subspace_instanton_defect`, `instanton_anchor`, and `anchor = "skeleton"` in the pipeline).

#### Low-temperature essentials (avoiding the three classic pitfalls)

1. **Energy-truncated DVR walls (`truncation_min`) and `dense_limit` memory scaling.**
   In 2D models, grid points with $V_S > E_{\mathrm{floor}} + \max(\text{truncation\_kt}\cdot k_B T,\ \text{truncation\_min})$
   are removed to keep the dense Sylvester matrix affordable. Removed DVR points act as **infinite
   hard walls** confining the transverse vibration. The quantum rate converges only once the energy
   margin above the local valley floor (including the barrier ridge) exceeds **about six quanta of
   the stiffest transverse mode ($6\hbar\omega_\perp$, walls at $\sim 5$ ground-state widths)**:
   - At low temperature, $30\,k_B T$ is much smaller than $6\hbar\omega_\perp$ for stiff stretches
     (at $200\,\mathrm{K}$, $30\,k_B T \approx 0.019\,\mathrm{E_h}$, whereas a transverse mode with
     $\omega_\perp = 0.010\,\mathrm{E_h}$ needs $6\hbar\omega_\perp \approx 0.060\,\mathrm{E_h}$).
     On a separable 2D Eckart model ($\omega_\perp = 0.010\,\mathrm{E_h}$) at $200\,\mathrm{K}$, a margin of
     $0.016\,\mathrm{E_h}$ above barrier + ZPE underestimates the rate by 12%, $0.03\,\mathrm{E_h}$ by 1%,
     and $0.07\,\mathrm{E_h}$ is exact to $1 - 4\times 10^{-6}$.
   - Always inspect `diagnostics.truncation_margin` in `C_S_2d.json` and set
     `quantum.truncation_min` (in $\mathrm{E_h}$) to $\sim 6\hbar\omega_\perp$.
   - Raising `truncation_min` (or raising $T$ at fixed `truncation_kt`) retains more 2D grid points
     $N_{\mathrm{retained}}$. If $N_{\mathrm{retained}} > \text{dense\_limit}$, the `solver` gate
     stops before solving (`status = "dense_limit_exceeded"`) and reports `required_dense_limit` and
     `dense_memory_mb` ($\approx 192\,N_{\mathrm{retained}}^2$ bytes) in `solver_plans.2`. Increase
     `quantum.dense_limit` and `execution.memory_limit_mb` accordingly.
2. **Finite-bead environment factor `harmonic_bead_factor` ($B_P$): 1D vs 2D physical difference.**
   The `"adiabatic"` and `"free_energy"` projections put the converged ($P \to \infty$) quantum
   zero-point or free energy of the orthogonal environment modes $E = S^\perp$ into $V_S$. However,
   the full-dimensional $P$-bead RPMD rate carries a finite-bead partition-function error in those
   environment modes:
   $$
   B_P = \frac{\prod_{k \in E^\ddagger} Z_P(u_k^\ddagger) / Z_\infty(u_k^\ddagger)}{\prod_{k \in E_r} Z_P(u_{k,r}) / Z_\infty(u_{k,r})}
   \approx \exp\!\left[-\frac{\sum_{k \in E_r} u_{k,r}^3 - \sum_{k \in E^\ddagger} (u_k^\ddagger)^3}{48 P^2}\right],
   \qquad u_k = \beta\hbar\omega_k
   $$
   Keeping $|\ln B_P| < \delta$ requires $P \gtrsim \sqrt{(\sum_r u_r^3 - \sum_\ddagger u_\ddagger^3)/(48\delta)}$,
   which grows as $u^{3/2}$ at low $T$ (for $\mathrm{H + H_2}$ at $200\,\mathrm{K}$, $u_r = 31.7$,
   $u_\ddagger = 14.8$, so 5% accuracy requires $P \approx 100$). When $P$ is not fully converged,
   divide the full RPMD rate by $B_P$ (`harmonic_bead_factor` + `apply_subspace_correction(..., bead_factor=B_P)`).
   - **Crucial 1D vs 2D distinction:** Inside the subspace $S$, both $k_{\mathrm{QM}}(H_S)$ and
     $k_{\mathrm{RPMD},P}(H_S)$ are evaluated on the same $H_S$ at the same bead count $P$, so
     finite-$P$ errors of coordinates **inside $S$** are already corrected by $C_S$ itself.
     Therefore $B_P$ must include **only the environment modes orthogonal to $S$ ($E = S^\perp$)**,
     never modes that lie inside $S$:
     - For a **1D path model**, all $3N - 7$ (or $3N - 6$) transverse normal modes at the TS
       (and all reactant vibrational modes) belong to $E = S^\perp$, including the in-plane stretch
       that couples to path curvature.
     - For a **2D model (`"rph"` or `"plane"`)**, the second subspace coordinate (the curvature-normal /
       symmetric-stretch mode at the TS and the reactant bond stretch in collinear $\mathrm{A + BC}$ transfer)
       is **explicitly inside $S$**. Its finite-$P$ error is already carried by $C_S(\mathrm{2D})$.
       Thus $B_P(\mathrm{2D})$ must be computed over only the remaining $3N - 8$ out-of-plane complement
       modes (for a 3-atom collinear model in 2D, only the degenerate out-of-plane bends remain in $E$).
       Applying a 1D $B_P$ (which includes the reactant stretch) on top of $C_S(\mathrm{2D})$
       double-counts that mode's finite-bead error.
3. **Thermally excited bath modes (`environment = "free_energy"`) and instanton anchoring (`anchor = "skeleton"`).**
   - Whenever an environment mode has $\beta\hbar\omega \lesssim 4$, use `environment = "free_energy"`
     so the thermal average of the environment coupling is first-order exact
     ([Subspace transfer theory](subspace-transfer-theory.md)).
   - Below the crossover temperature $T_c$, slow transverse modes ($\omega_\perp \lesssim \lambda^\ddagger$)
     or 1D path reductions can shift the steepest-descent instanton defect
     $A = \ln[2\pi / (\beta\hbar\eta_1)]$. Setting `anchor = "skeleton"` adds the closed-form shift
     $A_H - A_{S^0}$ relative to the bare-potential skeleton $S^0$ without double-counting the
     transverse zero-point dressing ([The instanton anchor](subspace-instanton-anchor.md)).

(numerical-parameters-to-converge)=
### 5.3 Numerical parameters to converge

| Parameter | Where | Symptom when too small |
| --- | --- | --- |
| Box length and absorbing-potential width/strength | `exact_quantum_rate(cap_width=, cap_strength=)`, pipeline `absorber_fraction`, `product_drop_kt`, `reference_kt` | Reflection from the absorber; at 200 K a 16 bohr Eckart box is 0.7% high, 32 bohr 0.06% |
| Grid spacing | model axes, pipeline `spacing_1d`, `spacing_2d`, `refinement` | Rate drifts when refined |
| Dense route capacity | pipeline `dense_limit` | Not a convergence parameter: the 2D model is refused before solving (`solver_plans.2.status = "dense_limit_exceeded"`) when it retains more points; the plan gives `required_dense_limit` and the dense memory ($\approx 192\,N^2$ bytes) |
| Truncation energy (energy-truncated DVR) | `exact_quantum_rate(truncation_energy=)`, pipeline `truncation_kt` and `truncation_min` ($\mathrm{E_h}$) | The removed points are hard walls. The rate converges only once the margin above the local valley floor exceeds about six quanta of the stiffest transverse mode (walls at $\sim 5$ ground-state widths). Check `diagnostics.truncation_margin` and set `truncation_min` to $\sim 6\hbar\omega_\perp$ at low $T$ |
| Propagation time and step | `max_time`, `dt` (iterative route) | `relative_extrapolation_spread` large |
| TI windows | `ring_polymer_rate(windows=)`, pipeline `rpmd.windows` | `ti_quadrature_log_error` comparable to the statistical error |
| Beads | `nbeads`, `harmonic_bead_factor` ($B_P$) | $C_S$ should correct the same finite-$P$ estimator as the full run; divide by $B_P$ over complement modes $E = S^\perp$ or converge $P$ |
| Sampling | `walkers`, `sweeps`, `trajectories` | `log_standard_error` |
| Grid box extent for RPMD | `quantum.padding`, `grid_exit.estimator_outside_fraction` | Ring polymers leaving the grid sample a constant spline continuation; `grid_exit.estimator_outside_fraction` must stay below `max_outside_fraction` (`0.01`) |

(validation)=
### 5.4 Validation and relation to other quantum-correction tools

- **2D Eckart + softening bath (`examples/subspace_correction/eckart_bath_2d.py`):** Symmetric Eckart
  barrier ($V_0 = 0.425\,\mathrm{eV}$, $a = 0.66\,\mathrm{bohr}$, $m = 1060$) with a vibration that
  softens by 30% and shifts by $0.2\,\mathrm{bohr}$ at the barrier:
  [docs/validation/2026-11-07-subspace-correction](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-07-subspace-correction/README.md).
- **Cl + HBr collinear heavy–light–heavy transfer (300 K):**
  [docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md).
- **Subspace transfer theory benchmarks (600, 300, 200 K):**
  [docs/validation/2026-11-09-subspace-transfer](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-09-subspace-transfer/README.md)
  and [Subspace transfer theory](subspace-transfer-theory.md).
- **2D RPH vs plane models on 2D, 3D and rotated 8D curved valleys:**
  [docs/validation/2026-11-13-subspace-rph-2d](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-13-subspace-rph-2d/README.md).
- **Instanton skeleton anchor on 2D valleys (M1–M4) and 6D–10D complex PESs:**
  [docs/validation/2026-11-14-subspace-instanton-anchor](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-14-subspace-instanton-anchor/README.md)
  and [The instanton anchor](subspace-instanton-anchor.md).
- **Unit tests (`tests/test_subspace_*.py`, `tests/test_rate_subspace.py`):** Check the exact solver
  against the analytic Eckart transmission, the iterative route against the dense route, a separable
  2D rate against its 1D rate, single-bead RPMD against classical TST, the projections against
  analytic adiabatic potentials, all six validity gates, and the provenance and temperature checks of
  `correct_rate_run`.

[Quantum correction](quantum-correction.md) covers response-level (spectral) corrections, the
online Ehrenfest propagator and flux–side curve transfers. For thermal rate constants use this
module. `hybrid_flux_side_correction(..., mode="ratio")` and
`adaptive_quantum.matched_rate_defect(..., mode="ratio")` apply the same multiplicative
transfer to curves or plateau values computed elsewhere; their additive default is not
invariant to the subspace model's absolute rate.
