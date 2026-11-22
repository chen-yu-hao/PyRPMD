# Subspace quantum rate correction examples (Python API)

These standalone Python scripts compute the multiplicative quantum rate correction
$C_S = k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$ directly from a potential energy
surface and saddle/path geometry using `pyrpmd.analysis`, without requiring full-system
umbrella-sampling windows.

For native `.inp` rate calculations with `[rate.subspace_correction]` and `--stage subspace`,
see [`examples/rate/subspace.inp`](../rate/subspace.inp) and
[`examples/rate/README.md`](../rate/README.md). Full theory, validity gates, and parameter
units are documented in [`docs/subspace-correction.md`](../../docs/subspace-correction.md)
([中文](../../docs/zh/subspace-correction.md)) and
[`docs/subspace-instanton-anchor.md`](../../docs/subspace-instanton-anchor.md).

## 1. Coupled 2D Eckart barrier vs exact 2D quantum reference (`eckart_bath_2d.py`)

### Model

`eckart_bath_2d.py` benchmarks a 1D vibrationally adiabatic subspace reduction against the
exact 2D quantum and 2D RPMD thermal rates on an analytic coupled valley potential
($m_s = m_y = 1060\,m_e$, $V_0 = 0.425\,\mathrm{eV} \approx 0.01562\,\mathrm{E_h}$,
$a = 0.66\,\mathrm{bohr}$):

$$
V(s, y) = V_0\,\mathrm{sech}^2(s / a) + \frac{1}{2} m \omega(s)^2 \bigl(y - c(s)\bigr)^2,
$$

where $\omega(s) = \omega_0 \bigl(1 - 0.3\,\mathrm{sech}^2(s / a)\bigr)$ ($\omega_0 = 0.010\,\mathrm{E_h}$)
softens by 30% across the barrier (zero-point energy drop) and the valley equilibrium
$c(s) = 0.2\,\mathrm{sech}^2(s / a)$ bends across the saddle (reaction-path curvature), so
projecting onto mass-weighted $s$ (`project_linear_subspace(..., environment="adiabatic")`)
is a non-trivial reduction rather than a separable model.

### Running

```bash
# Default scan at 300 K (P = 16) and 200 K (P = 24):
python examples/subspace_correction/eckart_bath_2d.py --out eckart_result.json

# Quick single-temperature smoke run (~20 s):
python examples/subspace_correction/eckart_bath_2d.py \
    --temperatures 200 --beads 24 --sweeps 200 --trajectories 256 --out eckart_quick.json
```

### Output fields (`--out result.json`)

The JSON output contains `model` (potential constants and grid ranges), `subspace`
(`SubspaceModel.describe()` provenance of the 1D adiabatic projection), and `rows`
(one record per `(temperature, beads)` pair):

| Field | Meaning |
| --- | --- |
| `temperature_K`, `beads` | Temperature $T$ ($\mathrm{K}$) and ring-polymer bead number $P$ |
| `exact_2d` | Exact 2D thermal quantum rate $k_{\mathrm{QM}}(\text{2D})$ (`exact_quantum_rate`) |
| `rpmd_2d`, `rpmd_2d_relative_error` | Full 2D RPMD rate $k_{\mathrm{RPMD}}(\text{2D})$ and standard error of $\ln k_{\mathrm{RPMD}}(\text{2D})$ |
| `C_S_1d`, `C_S_log_error` | 1D adiabatic subspace factor $C_S = k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$ and standard error of $\ln C_S$ |
| `corrected`, `corrected_relative_error` | Corrected 2D rate $k_{\mathrm{RPMD}}(\text{2D}) \times C_S$ and combined relative standard error (`apply_subspace_correction`) |
| `rpmd_over_exact` | Uncorrected ratio $k_{\mathrm{RPMD}}(\text{2D}) / k_{\mathrm{QM}}(\text{2D})$ (e.g. $\sim 0.72$ at $300\,\mathrm{K}$, $\sim 0.55$ at $200\,\mathrm{K}$) |
| `required_C` | Exact target defect $k_{\mathrm{QM}}(\text{2D}) / k_{\mathrm{RPMD}}(\text{2D})$ |
| `corrected_over_exact` | Transfer accuracy $\bigl[k_{\mathrm{RPMD}}(\text{2D}) \times C_S\bigr] / k_{\mathrm{QM}}(\text{2D})$ ($\approx 1.00$ when $C_S$ captures the defect) |
| `exact_method`, `exact_extrapolation_spread` | Quantum solver route (`"sylvester"` or `"flux_eigenstates"`) and long-time extrapolation spread |
| `seconds` | Per-stage wall times (`exact_2d`, `rpmd_2d`, `correction_1d`) |

---

## 2. Full-dimensional 3D/9D molecular $C_S$ and $B_P$ directly from TS + PES (`molecular_ts_to_cs.py`)

When you already have (or compute separately) a full-dimensional classical/RPMD rate and only
want the quantum tunnelling correction $C_S$ and finite-bead environment factor $B_P$ **without
running dozens of umbrella-sampling windows**, `molecular_ts_to_cs.py` runs the complete
Cartesian molecular subspace workflow from an approximate 3D transition-state geometry and
analytic $\mathrm{A + BC \leftrightarrow AB + C}$ PES:

1. **9D saddle refinement & rigid-body removal (`refine_saddle`, `rigid_modes`, `crossover_temperature`):**
   Refines a slightly bent, displaced 3-atom Cartesian TS guess in the vibrational subspace
   (projecting out the 5 collinear rigid-body translation/rotation modes, `rigid_body=True`),
   determines the crossover temperature $T_c = \hbar\lambda^\ddagger / (2\pi k_B)$, and extracts
   both TS and reactant vibrational frequencies.
2. **Umbrella-free MEP + 1D & 2D RPH + 9D skeleton instanton anchor (`RateRun`, `run_subspace_pipeline`):**
   Constructs an in-memory `RateRun(root=None, ...)` with `RPMDrateCoordinate` and runs
   `run_subspace_pipeline(..., apply=False, run=run)` using `path.source = "mep"`,
   `dimension = [1, 2]`, `model_2d = "rph"`, and `anchor = "skeleton"`. Below $T_c$, the 9D
   Cartesian instanton defect $A_H = \ln\varepsilon_{\mathrm{SD}}(H)$ shifts each subspace
   factor by $A_H - A_{S^0}$ relative to its bare-potential skeleton $S^0$:
   - On the **1D MEP**, the path cannot cut the corner, so $A_H - A_{1\mathrm{D}^0} > 0$
     provides a substantial corner-cutting steepest-descent correction.
   - On the **2D RPH** unrolled ribbon, corner cutting is already inside $S_{2\mathrm{D}}$, so
     $A_H - A_{2\mathrm{D}^0} \approx 0$ while `nested_subspace_sensitivity` selects 2D when
     $\kappa\rho > 0.5$.
3. **Complement-matched finite-bead factor $B_P$ (`harmonic_bead_factor`, `apply_subspace_correction`):**
   Computes $B_P = k_{\mathrm{RPMD}}(P) / k_{\mathrm{RPMD}}(\infty)$ over **only** the
   orthogonal environment modes $E = S^\perp$ (all 3 transverse TS modes vs 1 reactant stretch
   for 1D; only the 2 out-of-plane bends for 2D RPH, avoiding double-counting the in-plane
   stretch already inside $S_{2\mathrm{D}}$).

> **Multiprocessing note:** Scripts that pass `execution = {"backend": "process", ...}` must
> guard their entry point with `if __name__ == "__main__":` because worker processes use the
> `"spawn"` start method.

### Running

```bash
# Compute 1D MEP + 2D RPH C_S (with 9D skeleton anchor) and B_P at 200 K, P = 24:
python examples/subspace_correction/molecular_ts_to_cs.py --out cs_9d.json

# Also apply C_S / B_P to an external full-dimensional RPMD rate and uncertainty:
python examples/subspace_correction/molecular_ts_to_cs.py \
    --rpmd-rate 2.5e-16 --rpmd-se 2.0e-17 --out cs_9d_applied.json
```

### Key JSON output fields (`--out cs_9d.json`)

| Field | Meaning |
| --- | --- |
| `temperature_K`, `crossover_temperature_K`, `beads` | Target $T$, saddle crossover $T_c$, and matched bead count $P$ |
| `saddle` | Refined 3D geometry (`bohr`), `rigid_modes_removed`, classical barrier (`Eh`), and TS/reactant frequencies ($\mathrm{cm}^{-1}$) |
| `curvature_width_max`, `applied_dimension` | Peak $\kappa\rho$ along the MEP and selected subspace dimension (`1` or `2`) |
| `validity` | Pre- and post-solve gate status (`path`, `bath`, `truncation`, `ribbon`, `grid`, `surface`, `solver`) |
| `instanton_anchor` | Full 9D instanton defect $\varepsilon_{\mathrm{SD}}(H)$ and per-dimension skeleton shifts $A_H - A_{S^0}$ |
| `corrections.1d`, `corrections.2d_rph` | `unanchored_factor`, `anchored_factor`, `anchor_log_shift`, `log_standard_error`, complement `bead_factor_B_P`, and `effective_multiplier_C_over_B_P` ($C_S / B_P$) |
| `applied_rate` | Optional `apply_subspace_correction` result when `--rpmd-rate` is supplied |

## Reaction path past a conical intersection (`conical_intersection_path.py`, research)

A 2D model whose diabats cross at $x = 0$ with a coupling $\lambda(y - y_c)$, so a conical
intersection sits 2.5 transverse widths from the ground-state path. The subspace pipeline
computes $C_S$ on the ground state, diabatizes scrambled adiabatic data from a Python source
along the same path, and applies $C_S\,\eta_\mathrm{NA}$ (about $1.67 \times 0.21$ at 300 K):

```bash
python examples/subspace_correction/conical_intersection_path.py --temperature 300
python examples/subspace_correction/conical_intersection_path.py --coupling-scale 4   # wider gap, eta -> 1
```

The script's docstring shows the `[electronic]` table for an excited-state backend. See
[`docs/multistate-subspace.md`](../../docs/multistate-subspace.md) ([中文](../../docs/zh/multistate-subspace.md)).

