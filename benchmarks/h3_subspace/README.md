# H + H₂ (BKMP2) subspace quantum correction benchmarks

This directory contains two complementary benchmarks on the published Boothroyd–Keogh–Martin–Peterson
(BKMP2) $\mathrm{H_3}$ Born–Oppenheimer potential energy surface ($T_c \approx 346\,\mathrm{K}$,
$\omega^\ddagger = 1511i\,\mathrm{cm}^{-1}$, classical barrier $15.35\,\mathrm{mE_h} = 9.63\,\mathrm{kcal/mol}$):

| Suite | Files | Dimensionality & physics | What it tests |
| --- | --- | --- | --- |
| **Collinear 2D exact benchmark & parallel scaling** | `collinear.py`, `run.py`, `campaign.txt`, `parallel.py` | 2D centre-of-mass-free collinear coordinates $(s, t)$ with **no bath projection** ($S$ is the whole collinear system) | Direct RPMD method defect $C = k_{\mathrm{QM}} / k_{\mathrm{RPMD}}$ across $150\text{–}600\,\mathrm{K}$, plus OpenMP / multi-process / MPI task-scheduler scaling (`parallel.py`) |
| **Full-dimensional 3D/9D molecular rate + subspace correction** | `rate_200K.inp`, `correct_200K.toml`, `correct_200K_curvature.toml` | 3 atoms in 3D space (**9D Cartesian**, 5 rigid-body modes removed, 2 degenerate bends in the adiabatic bath) at $200\,\mathrm{K}$ ($0.58\,T_c$, $P = 128$) | End-to-end bimolecular RPMD rate ($47$ umbrella windows + MBAR + paired recrossing) postprocessed with `pyrpmd subspace-correct` using 1D MEP, 1D SCT (`curvature = "small"`), and 2D RPH (`model_2d = "rph"`) |

---

## 1. Prerequisites: build and verify the BKMP2 Fortran PES

No third-party BKMP2 binary is distributed in the repository. Compile and verify the checksum-pinned
Fortran library once from the repository root (requires `gfortran` and PyRPMD's native C++ extension):

```bash
python -m pip install -e '.[test]'
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check
```

See [`benchmarks/h3/SOURCE.md`](../h3/SOURCE.md) and [`benchmarks/h3/README.md`](../h3/README.md)
for source provenance, SHA-256 verification, and offline `--source` builds.

---

## 2. Collinear 2D benchmark (`run.py`, `campaign.txt`, `parallel.py`)

In `collinear.py`, all three $^{1}\mathrm{H}$ atoms move along a line in orthonormal mass-weighted
symmetric coordinates $s = (r_2 - r_1)\sqrt{m / 6}$ and $t = (r_1 + r_2)\sqrt{m / 2}$. Because there is
no orthogonal bath, $k_{\mathrm{QM}} / k_{\mathrm{RPMD}}$ isolates the intrinsic collinear RPMD error:

```bash
# Inspect crossover temperature, barrier height, and collinear H2 ground-state energy:
python benchmarks/h3_subspace/run.py info

# Exact 2D Sylvester quantum rate at 200 K (converged walls >= 0.10 Eh above H2 v=0):
python benchmarks/h3_subspace/run.py quantum --temperature 200 --spacing 2.0 --truncation 0.10

# Collinear 2D RPMD rate at 200 K:
python benchmarks/h3_subspace/run.py rpmd --temperature 200 --beads 64 --seed 1
```

`campaign.txt` lists the temperature/bead sweep from $600\,\mathrm{K}$ down to $100\,\mathrm{K}$
(note that at $100\,\mathrm{K}$ near-threshold box-absorber reflection prevents quantum convergence;
use $T \ge 150\,\mathrm{K}$).

`parallel.py` benchmarks hybrid C++/OpenMP, dynamic multi-process (`--mode tasks`), and MPI
(`--mode mpi`) execution on fixed 300 K / $P = 32$ 1D + 2D gridded BKMP2 Hamiltonians:

```bash
OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
    python benchmarks/h3_subspace/parallel.py --mode tasks --cpus 8 --output runs/h3-parallel-tasks8.json
```

---

## 3. Full-dimensional 3D/9D $\mathrm{H + H_2}$ at 200 K (`rate_200K.inp` + `correct_200K*.toml`)

### Running end to end

Run from the repository root **without overriding `--output`** on `rate_200K.inp`, so the rate directory
`benchmarks/h3_subspace/runs/h3-200K-P128-b20` matches `[source] rate_run` in `correct_200K.toml` and
`correct_200K_curvature.toml`:

```bash
# 1. Validate and run full-dimensional 3D/9D RPMD rate at 200 K, P = 128:
pyrpmd validate benchmarks/h3_subspace/rate_200K.inp --check-pes
pyrpmd run benchmarks/h3_subspace/rate_200K.inp --workers 8

# 2. Compute 1D MEP + 2D RPH subspace correction and write rate_quantum_corrected.json:
pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K.toml

# 3. Optional 1D small-curvature tunnelling (SCT) effective-mass recomputation:
pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K_curvature.toml
```

### RPH grid, validity-gate, and execution notes

- **2D RPH grid & DVR wall margin (`[quantum]`):** At $200\,\mathrm{K}$, $30\,k_B T \approx 0.019\,\mathrm{E_h}$
  is less than two quanta of the $\mathrm{H_2}$ stretch ($4404\,\mathrm{cm}^{-1} \approx 0.0201\,\mathrm{E_h}$).
  `correct_200K.toml` sets `truncation_min = 0.10` $\mathrm{E_h}$ ($\sim 5\text{–}6$ stretch quanta above
  the channel floor) and `spacing_2d = [2.2, 2.2]`, retaining $\sim 10{,}273$ DVR points within
  `dense_limit = 11500` ($\sim 18.9\,\mathrm{GiB}$ peak dense Sylvester memory). For finer grid-convergence
  checks, set `spacing_2d = [2.0, 2.0]`, `dense_limit = 13000`, and `execution.memory_limit_mb = 32768`
  ($12{,}445$ retained points, $\sim 27.8\,\mathrm{GiB}$).
- **Physical BKMP2 gate thresholds (`[validity]`):**
  - `soft_mode_frequency = 4.5e-4` ($\approx 99\,\mathrm{cm}^{-1}$): in the long-range collinear
    entrance/exit van der Waals region ($r \approx 3.0\text{–}3.8\,\mathrm{bohr}$), T-shaped
    $\mathrm{H\cdots H_2}$ lies slightly below collinear, producing a weak doubly degenerate
    hindered-rotation curvature down to $\sim 85i\,\mathrm{cm}^{-1}$ ($< 0.4\,\mathrm{mE_h}$ ZPE).
  - `max_curvature_width = 0.60`: equal-mass $\mathrm{H + H_2}$ turns through $120^\circ$ ($60^\circ$
    skew angle) with $\max \lvert\kappa\rvert\sigma_n \approx 0.53$, just above the generic `0.50`
    default while the RPH ribbon remains unfolded ($1/\lvert\kappa\rvert \approx 13.6 > \sigma_n \approx 7.1$).
- **Multi-core execution (`[execution]`):** Both `.toml` files configure `backend = "process"`,
  `cpus = "auto"`, and `blas_threads = 8` with `rpmd.backend = "native"` so QM and $P = 128$ subspace
  RPMD tasks run in parallel with compiled C++/OpenMP B-spline propagation rather than falling back to
  single-core execution.

---

## 4. Resource requirements and reference results

### Expected wall times and memory (8-core x86\_64 VM / workstation)

| Task | Peak RAM | Typical wall time (8 cores) |
| --- | --- | --- |
| `run.py quantum --temperature 200 --spacing 2.0 --truncation 0.10` | ~6.5 GiB | ~3–4 min |
| `run.py rpmd --temperature 200 --beads 64` (exact Python BKMP2 callback, 1 thread) | < 0.5 GiB | ~20–25 min (~4 min with `--sweeps 100 --trajectories 128`) |
| `parallel.py --mode tasks --cpus 8` (300 K, $P = 32$, gridded 1D + 2D C++/OpenMP) | ~3.5 GiB | ~1–2 min |
| `rate_200K.inp` ($P = 128$, 47 windows $\times$ 4 replicas $\times$ 5 ps + recrossing) | ~1.5 GiB | ~18–25 min |
| `correct_200K_curvature.toml` (1D SCT, $P = 128$) | < 0.5 GiB | ~0.5–1 min |
| `correct_200K.toml` (1D MEP + 2D RPH, `spacing_2d = [2.2, 2.2]`) | ~19–22 GiB (`memory_limit_mb = 24576`) | ~30–35 min |

### Reference ranges ($200\,\mathrm{K}$, $P = 128$, BKMP2)

- **Collinear 2D exact benchmark (`run.py`, converged walls $\ge 0.10\,\mathrm{E_h}$):**
  - $k_{\mathrm{QM}}(\text{2D}) \approx 7.67 \times 10^{-9}\text{–}7.77 \times 10^{-9}$ (model units),
    $k_{\mathrm{RPMD}}(\text{2D}, P=128) \approx (5.0 \pm 0.4) \times 10^{-9}$,
    collinear defect $C = k_{\mathrm{QM}} / k_{\mathrm{RPMD}} \approx 1.54 \pm 0.12$
    (harmonic bead factor $B_{128} \approx 0.965$).
- **Full-dimensional 3D/9D rate + subspace correction (`rate_200K.inp` + `correct_200K*.toml`):**
  - Uncorrected 3D RPMD rate ($P = 128$): $k_{\mathrm{RPMD}} \approx (7.4 \pm 2.2) \times 10^{-19}\,\mathrm{cm^3\,molecule^{-1}\,s^{-1}}$
    ($\kappa \approx 0.66$, $\xi^\ast \approx 1.005$, $\Delta F^\ddagger \approx 8.5\,\mathrm{kcal/mol}$).
  - Path curvature $\times$ width: $\max(\kappa\rho) \approx 0.84 > 0.50 \implies$ 2D subspace required.
  - 1D MEP (no corner cutting): $C_S(\mathrm{1D}) \approx 1.03\text{–}1.06$; with SCT curvature mass
    (`correct_200K_curvature.toml`): $C_S(\mathrm{1D, SCT}) \approx 1.43\text{–}1.50$.
  - 2D RPH (`model_2d = "rph"`, corner cutting + adiabatic bend): $C_S(\mathrm{2D, RPH}) \approx 1.94\text{–}1.97$
    (historical fixed-plane 2D centroid projection: $2.66 \pm 0.20$), nested sensitivity
    $\ln(C_{\mathrm{2D}} / C_{\mathrm{1D}}) \approx +0.63 \implies$ 2D applied.
  - Corrected 3D rate $k_{\mathrm{RPMD}} \times C_S(\mathrm{2D})$:
    $\sim (1.4\text{–}2.0) \times 10^{-18}\,\mathrm{cm^3\,molecule^{-1}\,s^{-1}}$
    (3D quantum dynamics reference: $\sim 2.01 \times 10^{-18}\,\mathrm{cm^3\,molecule^{-1}\,s^{-1}}$;
    statistical uncertainty is dominated by the finite-sample umbrella PMF bootstrap of `rate_200K.inp`).

See [`docs/validation/2026-11-08-h3-collinear-lowT`](../../docs/validation/2026-11-08-h3-collinear-lowT/README.md),
[`docs/validation/2026-11-09-subspace-transfer`](../../docs/validation/2026-11-09-subspace-transfer/README.md),
[`docs/validation/2026-11-10-subspace-curvature`](../../docs/validation/2026-11-10-subspace-curvature/README.md), and
[`docs/validation/2026-11-10-subspace-hybrid`](../../docs/validation/2026-11-10-subspace-hybrid/README.md)
for full validation records.
