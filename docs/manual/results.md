# Output, analysis and interpreting results

## Summary and run identity

```python
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
print(summary["task"])
print(summary.get("converged"))
print(summary.get("warnings", []))
print(summary.get("files", {}))
```

Every workflow writes its final summary to `results/summary.json`. The high-level entry points add
the software version, output path, preset and input diagnostics; the task adds its physical results,
statistics and file index. Report a result together with its units, unresolved fraction, fitting
window and normalization.

| Path | Contents |
| --- | --- |
| `manifest.json` | Software and dependencies, task identity, hashes of referenced files, native capabilities |
| `input/resolved.json` | Configuration with units and defaults resolved |
| `input/system.json` | Molecular or periodic system; PCET uses `input/model.json` |
| `input/original.*` | Copy of the original input file |
| `checkpoints/` | Task state, random streams and stage checkpoints |
| `samples/`, workflow data directories | Saved trajectories, windows and other task data |
| `summary.json` (top of the run directory) | Summary written by the workflow; an unfinished network writes `network-progress.json` |
| `results/summary.json` | The workflow summary plus software version, output path, preset and input notes; `analyze` reads it first |

Check `status`, sample sizes and failed or unresolved counts in the summary, and resume interrupted
runs with `pyrpmd resume`.

## FES: density, probability mass and free energy

FES writes `fes.npz`, `fes.csv` and `fes.h5`; `files` in the summary gives the paths. The CSV has one coordinate column per CV and these columns:

| Column | Meaning |
| --- | --- |
| `cv0_bohr`, … | Grid centers; the column name carries the unit |
| `free_energy_hartree` | Relative free energy from the probability density |
| `standard_error_hartree` | Bootstrap standard error, when computed |
| `probability_mass` | Probability mass of the cell |
| `density` | Probability density per unit CV volume |
| `masked` | Cells without sampling support or excluded by diagnostics |
| `raw_count` | Number of samples in the cell |

Probability mass and density differ by the cell volume. Compare FES curves on the same CV, binning
and measure; gas-phase rates use the separate `rate_flux` PMF.

```python
import numpy as np
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
table = np.genfromtxt(summary["files"]["table"], delimiter=",", names=True)
supported = table["masked"] == 0
print("supported bins:", int(supported.sum()))
print(table["free_energy_hartree"][supported])
```

When the window overlap graph is disconnected, add bridging windows or sampling to join the
segments into one free-energy surface.

Rebinning reuses the saved samples without calling the PES:

```bash
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 20
```

The bootstrap gives the statistical error of the saved samples. When no valid error exists, the
field is `null`.

## Reaction rates: PMF and recrossing

Saved gas-phase rate runs can be reanalyzed with `pyrpmd rate-postprocess RUN --bootstrap 200`.
[2D rate postprocessing](../rate-postprocessing.md) describes dividing-curve integration,
reference normalization and signed channel fluxes. Results are written separately while preserving
the original energy and convergence diagnostics.

Gas-phase `rate` follows the Bennett–Chandler procedure: the static PMF/QTST factor and the dynamic
transmission coefficient together give the rate. Check the PMF measure, dividing surface, reactant
normalization, constrained parents and recrossing plateau. `site_rate` and `surface_rate` use an
explicit reactant basin and one directed boundary and give first-order rates in `s^-1`.

Summaries report finite-time estimates, signed κ, plateau detection, failed bootstrap draws and
missing joint intervals as they occur. In basin normalization, each PMF bootstrap draw computes the
density and the basin integral together to keep their covariance. The joint error combines the
PMF/QTST and κ bootstraps as independent samples, with the parent burn-in and independence
assumptions recorded in `rate_uncertainty`. Field names for each workflow are listed in
[Workflows](workflows.md) and [Materials and surfaces](periodic.md).

For dRPMD, read the product, nonreactive, unresolved and failed counts in `counts` together with
`results/channels.csv` and the classification rules. Unresolved and failed trajectories are counted
separately from nonreactive ones, and chemical rates and cross sections need explicit classification
rules. `network` reports each entrance, channel and TS separately.

With NRPMD enabled, also read `summary["nonadiabatic"]` and `results/electronic_channels.csv`.
Electronic populations are signed MMST estimators and can fall outside [0,1]. When trajectories
failed or remain unresolved, the final state-resolved rates or cross sections are empty and the
observed contributions are reported. See [Nonadiabatic NRPMD](nonadiabatic.md).

## PIMD and diffusion

The PIMD primitive and centroid-virial estimators give physical thermodynamic energies; the ring
Hamiltonian is a propagation diagnostic. PIMD runs at the fixed cell volume.

Diffusion saves continuous, unwrapped centroid positions and velocities. The MSD fit needs the
long-time diffusive regime, and `analysis_status` flags ballistic motion and unstable fits. The VACF
integral and the between-replica error give two further checks.

```bash
pyrpmd analyze runs/diffusion --recompute \
  --fit-range '10 fs' '100 fs' --axes 0 1 --atoms 0 --max-lag 100
```

This template reanalyzes a diffusion run with enough data; the fit range must lie within the saved
lag times, and the axes and atoms must exist. Reanalysis can reselect the saved atoms, axes, fit
window and bootstrap; the time resolution is set by the saved interval. (`--bins` applies to FES
and metadynamics.) Fields and statuses are described in the [materials chapter](periodic.md).

## PCET and metadynamics

PCET saves the proton spectra, wavefunctions, channel contributions and scan results. For rates that
underflow to 0, read the log rate. The spectral grid, number of states and tail checks measure the
numerical convergence of the local model.

Metadynamics saves the HILLS, the bias and the finite-time bias FES. `analyze --recompute`
re-evaluates the saved bias; bootstrap must stay 0.

## A reproducible report

Record the code commit, installed dependencies, potential source and hash, input and units, run
identity, bead number and time step, equilibration and production lengths, independent replicas,
window or fitting scheme, every diagnostic and every missing quantity. When comparing two results,
state which scientific parameters changed and which were reanalysis settings. Run the convergence
series in [Scientific conventions](science.md) before reporting physical predictions.
