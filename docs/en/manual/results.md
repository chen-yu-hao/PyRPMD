# Output, analysis and interpreting results

## Start from the summary and the run identity

```python
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
print(summary["task"])
print(summary.get("converged"))
print(summary.get("warnings", []))
print(summary.get("files", {}))
```

For most workflows the final summary is `results/summary.json`. The high-level entry points add the
software version, output path, preset and input diagnostics; the task itself keeps the physical
quantities, statistical qualifications and file index. Never keep only a scalar and discard its
units, unresolved fraction, fitting window or normalization convention.

| Path | Purpose |
| --- | --- |
| `manifest.json` | Software/dependencies, task identity, hashes of referenced files and native capabilities |
| `input/resolved.json` | Configuration after unit and default resolution |
| `input/system.json` | Molecular/periodic system; PCET uses `input/model.json` |
| `input/original.*` | Copy of the original input when it came from a file |
| `checkpoints/` | Task state, random streams and stage checkpoints |
| `samples/`, workflow data directories | Saved trajectories, windows or other task data |
| `summary.json` (top of the run directory) | Summary written by the workflow itself; an unfinished network writes `network-progress.json` |
| `results/summary.json` | Read this first: task results and diagnostics, plus software version, output path, preset and input notes on top of the workflow summary; `analyze` reads it first |

The presence of files does not mean the whole task finished. Check the status, sample sizes, failed
trajectories and unresolved results in the summary; after an interruption use `resume` instead of
patching a partial result by hand into "success".

## FES: density, probability mass and free energy

FES saves `fes.npz`, `fes.csv` and, when available, `fes.h5`; the exact paths are in the summary's
`files`. The CSV has the following columns, with the number of CV columns depending on the dimension:

| Column | Meaning |
| --- | --- |
| `cv0_bohr`, … | Grid centers; the column name states the coordinate unit |
| `free_energy_hartree` | Relative free energy based on the probability density |
| `standard_error_hartree` | Bootstrap standard error when available |
| `probability_mass` | Probability mass of the cell |
| `density` | Probability density per unit CV volume |
| `masked` | Regions without sampling support or where diagnostics forbid an estimate |
| `raw_count` | Raw number of samples in the cell |

Probability mass and density differ by the cell volume. Curves with different binning, CV metrics or
Jacobians cannot be compared without conversion. A `cv_density` free energy cannot replace the
`rate_flux` measure required by gas-phase rates either.

```python
import numpy as np
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
table = np.genfromtxt(summary["files"]["table"], delimiter=",", names=True)
supported = table["masked"] == 0
print("supported bins:", int(supported.sum()))
print(table["free_energy_hartree"][supported])
```

Keep unsampled regions masked. When the window overlap graph is disconnected, the global relative
free energy is not identifiable; add bridging windows or sampling instead of shifting segments until
the plot looks continuous.

Rebinning never calls the PES:

```bash
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 20
```

The bootstrap treats the statistical uncertainty of the saved samples; it cannot repair a wrong
potential, an unconverged bead number or a missing reaction channel. When there is no valid error
bar, a `null`/missing value must not be read as a zero error.

## Reaction rates: read the PMF and recrossing together

Gas-phase `rate` reports the Bennett–Chandler procedure: a static part and the dynamic transmission
coefficient together form the rate. Check the PMF measure, dividing surface, reactant normalization,
constrained parents and the recrossing plateau. `site_rate` / `surface_rate` use an explicit reactant
basin and one directed boundary and output a first-order channel rate in `s^-1`; they must not be
multiplied by a gas-phase bimolecular factor and do not become per-area, per-site or
coverage-normalized experimental rates by themselves.

Keep finite-time estimates, negative signed recrossing estimates, missing plateaus, failed bootstrap
draws and missing joint intervals. In basin normalization, the density and the basin integral are
computed in the same PMF bootstrap draw to keep their covariance. The joint error of the static
PMF/QTST and the dynamic κ is combined under an independent-sampling assumption, with the parent
burn-in and independence qualifications preserved. What these checks are for, and the actual fields of
each workflow, are described in [Workflows](workflows.md) and [Materials and surfaces](periodic.md).

dRPMD needs the product, nonreactive, unresolved and failed events in `counts`, `results/channels.csv`
and the classification rules read together; missing trajectories cannot all be counted as
nonreactive. Without a validated classifier or collision parameters, geometric event counts are not a
formal chemical cross section. `network` currently keeps separate results per entrance/channel/TS
and never sums them implicitly into a total rate.

With experimental NRPMD enabled, also read `summary["nonadiabatic"]` and
`results/electronic_channels.csv`. Electronic populations are signed MMST estimators and may be
negative or larger than one; do not clip them to [0,1] or use argmax to assign one electronic surface
to a trajectory. When failed/unresolved trajectories exist, the final state-resolved rates/cross
sections stay empty while the observed contributions remain available for diagnosis. See
[Nonadiabatic NRPMD](nonadiabatic.md).

## PIMD and diffusion

The PIMD primitive and centroid-virial estimators describe physical thermodynamic energies; the ring
Hamiltonian is an integration/propagation diagnostic. They are not the same energy. A fixed-cell
PIMD does not give the NPT equilibrium lattice constant.

Diffusion saves continuous unwrapped centroid positions and velocities. An MSD fit needs a genuine
long-time diffusive regime; short-time ballistic motion also gives a positive slope, but that is no
reason to accept a diffusion coefficient. The VACF integral is another diagnostic; check its
finite-time plateau and the between-replica error.

```bash
pyrpmd analyze runs/diffusion --recompute \
  --fit-range '10 fs' '100 fs' --axes 0 1 --atoms 0 --max-lag 100
```

This is a reanalysis template for an existing diffusion run with enough data. The fitting range must
lie within the saved lag times, and the selected axes and atoms must exist; the short teaching example
does not necessarily support this range. You can reselect saved atoms, axes, fitting windows and the
bootstrap, but you cannot recover time resolution that was never saved. Diffusion analysis does not
use `--bins`. Full fields and the accept/reject statuses are in the [materials chapter](periodic.md).

## PCET and metadynamics

PCET saves the quantum proton spectra, wavefunctions, channel contributions and scan results. When an
ordinary rate underflows, also read the finite log rate; the spectral grid, number of kept states and
tail checks only judge the numerical adequacy of the fixed local model. Electrode structure, pH and
current-density normalization outside the model cannot be derived from this output.

The HILLS, bias and finite-time bias FES of metadynamics describe a biased sampling history.
`analyze --recompute` can re-evaluate the saved bias; a non-zero bootstrap request is currently
rejected. Biased trajectory time is not a physical reaction time.

## Saving a reproducible report

Record at least the code commit, installed dependencies, potential source/hash, input and units, run
identity, bead number and time step, equilibration/production lengths, independent replicas, window
or fitting scheme, all diagnostics and every missing quantity. When comparing two results, state
which scientific parameters changed and which were only reanalysis settings. Discuss physical
predictions of the system only after completing the convergence series described in the
[scientific conventions](science.md).
