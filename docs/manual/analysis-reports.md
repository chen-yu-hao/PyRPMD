# Automatic reports and figures

Generate a report from saved results with one command:

```bash
python -m pip install -e '.[analysis]'
pyrpmd report runs/my-run --output reports/my-run
```

The optional `analysis` dependencies include Matplotlib for plotting. Open
`reports/my-run/index.html` in a browser. The report reads native PyRPMD result
files and preserves the original run. It does not call the PES, restart dynamics,
or execute a saved Python input.

## One run, several runs, or a run-and-report shortcut

From a source checkout, this short harmonic example exercises the full workflow:

```bash
pyrpmd run examples/fes/harmonic2d.inp --output runs/report-harmonic
pyrpmd report runs/report-harmonic --output reports/harmonic \
  --formats png pdf svg --dpi 600 --energy-unit kcal/mol --length-unit angstrom
```

This is a sampling pilot of an analytic harmonic model. Its success checks the
workflow; it is not a converged chemical-reaction calculation. Unsupported cells
remain masked on the two-dimensional surface.

For existing runs:

```bash
pyrpmd analyze runs/my-run --report --report-output reports/my-run
```

For a new calculation, `pyrpmd run INPUT --report` creates the default report after
the run. With one run and no report output specified, the report directory is
`RUN/analysis-report`. For several runs, specify `--output` explicitly:

```bash
pyrpmd report runs/rate-300 runs/rate-500 runs/rate-700 \
  --output reports/temperature-series \
  --labels '300 K' '500 K' '700 K' --series RPMD RPMD RPMD
```

Each label identifies an individual run. Each series name identifies runs to be
considered together for temperature plots, KIEs or parameter comparisons. Supply
one label and one series value per input when using those options. Without an
explicit series assignment, metadata-based grouping is deliberately conservative.
For example, temperature calculations using different bead numbers may need an
explicit shared series after you have established bead convergence.

An explicit series name is your assertion that the runs form a meaningful
comparison. Check the reaction, potential, isotope, CV, normalization and
sampling protocol. Compatible units alone do not establish comparability.

## What is written

| Output | Purpose |
| --- | --- |
| `index.html` | Local overview, figures, result tables and diagnostics |
| `report.md` | Portable text report |
| `report.json` | Structured report, source provenance, diagnostics and artifact index |
| `summary.csv` | Run-level quantities for further analysis |
| `figures/` | PNG/PDF by default; SVG can also be requested |
| `data/` | One CSV per figure with the numerical plot data, including units |

Use the artifact index in `report.json` to locate individual figures and tables.
The report includes SHA-256 hashes of its source files and software versions.
Missing values are blank in exported CSV tables. Two-dimensional plot tables
retain grid-center coordinates and a mask column.
Keep the original runs as well: a plotting table is not a replacement for raw
samples, input metadata or a trajectory archive.

Output directories are protected. To regenerate an existing report, use
`--overwrite`; the directory must contain a valid PyRPMD report marker. This
option does not authorize replacement of an arbitrary directory or a source run.

```bash
pyrpmd report runs/my-run --output reports/my-run --overwrite
```

## Which figures appear?

The module selects figures from the task and files present. Missing optional
data are reported; they are not filled with invented values.

| Saved results | Analysis and figure families |
| --- | --- |
| One-dimensional PMF/FES | Profile, supported points, available uncertainty, extrema and metadata |
| Two-dimensional FES | Free-energy map and available sampling/mask information |
| Umbrella sampling | Window distributions, overlap and restraint/sampling diagnostics when saved |
| Gas, site or surface rates | PMF, transmission coefficient and source rate diagnostics |
| Several compatible runs | PMF/recrossing overlays and temperature-dependent rate comparisons |
| Metadynamics | Saved bias/FES and available history diagnostics |
| dRPMD | Outcome counts and supported channel statistics, with unresolved/failed outcomes visible |
| Materials and diffusion | Saved observable or MSD/VACF data and the source fitting diagnostics |
| PCET | Available model, channel and scan results |

The module reads native PyRPMD output. Legacy RPMDrate `.dat` files and executable
`input.py` parsing from Post-RPMDrate are not included in this release.

## Units, masks and reported uncertainty

`--energy-unit` and `--length-unit` control presentation of convertible quantities.
The defaults are `kcal/mol` and `angstrom`; they do not change the simulation or
turn an angular/dimensionless CV into a distance. Rate dimensions are kept
separate: bimolecular rates use `cm3 molecule^-1 s^-1`, and first-order rates use
`s^-1`.

A two-dimensional FES is not a two-dimensional potential-energy cut. Its axes
are the CVs actually sampled, and its values depend on their probability-density
measure. A surface originally sampled in other CVs cannot be relabeled as two
bond lengths. Masked cells remain unsupported; the report does not interpolate a
path through them. A masked region by itself is not evidence of a high barrier.

Saved standard errors and intervals retain their labels. No error band is
created when the source has no usable uncertainty. A standard error is different
from a confidence interval; variation among bead counts or time steps is a
sensitivity comparison. Existing warnings about overlap, limited samples,
failed trajectories or plateau tests remain relevant after plotting.

The report does not certify physical convergence. A plausible PMF, apparently
flat recrossing tail or successful pipeline test still needs the convergence
studies described in [Scientific conventions](science.md).

## Published rates and diagnostic estimates

By default the report plots published source rate fields. If a workflow withholds
a rate because one of its acceptance checks failed, the report keeps it withheld.
To inspect available estimates explicitly:

```bash
pyrpmd report runs/rate-pilot --output reports/rate-diagnostics --include-estimates
```

These estimates are labeled as diagnostic results and kept separate from
published rates. They are never used in an Arrhenius fit. A published source field
still does not imply that all physical convergence questions have been resolved.

### Optional Arrhenius fit

```bash
pyrpmd report runs/rate-300 runs/rate-500 runs/rate-700 \
  --output reports/arrhenius --series RPMD RPMD RPMD --fit-arrhenius
```

Fitting is opt-in and needs at least three distinct temperatures with accepted,
positive rates in a compatible series. The unweighted regression describes
`ln(k)` against `1/T`; its activation energy and goodness of fit describe those
data over that temperature interval. It does not add rate uncertainties or
establish a temperature-independent mechanism. Inspect curvature before using
one activation energy, especially across a tunneling regime.

### Kinetic isotope effects

```bash
pyrpmd report runs/H-300 runs/H-500 runs/D-300 runs/D-500 \
  --output reports/kie --series H H D D --kie H D
```

The order is numerator then denominator: this command requests `k_H/k_D`.
Temperatures must match, rate units must be compatible, and the underlying
reactions and normalizations must be comparable. Missing temperatures are not
interpolated. Where both rates have saved standard errors, the first-order ratio
error assumes independent estimates:

$$
\operatorname{SE}(R) = |R|\sqrt{
  \left(\frac{\operatorname{SE}(k_H)}{k_H}\right)^2 +
  \left(\frac{\operatorname{SE}(k_D)}{k_D}\right)^2},
\qquad R = k_H/k_D.
$$

Paired simulations can have covariance; this expression does not estimate it.
Without both saved standard errors, no propagated ratio error is invented.

### Parameter comparisons

```bash
pyrpmd report runs/P8 runs/P16 runs/P32 --output reports/beads \
  --series bead-check bead-check bead-check --convergence beads
```

The supported comparison variables are `beads`, `dt_fs`, `rinf_bohr`,
`bmax_bohr` and `samples`. Only available metadata can be plotted. Hold other
scientific settings fixed before interpreting a change as sensitivity to the
selected variable. An axis named "convergence" does not establish convergence
without sufficient range, independent sampling and uncertainty analysis.

## Reference data with explicit provenance

Add your own comparison data with `--reference reference.csv`. Required columns:

| Column | Meaning |
| --- | --- |
| `series` | Reference-data series label |
| `temperature_K` | Temperature in kelvin |
| `rate` | Rate in the declared unit |
| `rate_unit` | `cm3 molecule^-1 s^-1` or `s^-1` |
| `source` | Citation, DOI, dataset identifier or an explicit synthetic label |
| `standard_error` | Optional standard error, in the same rate unit |

Reference observations are labeled as comparison data and retain their source.
Do not put a 95% interval width, an unspecified experimental error or the spread
between independent methods into `standard_error`.

The repository's
[reference-rates.csv](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/analysis/reference-rates.csv)
contains **synthetic example values**, including synthetic error values. It
demonstrates the format and must not be cited as experiment or used to validate
a PES:

```bash
pyrpmd report runs/rate-pilot --output reports/reference-format \
  --reference examples/analysis/reference-rates.csv
```

## Python API

```python
from pyrpmd.analysis.reporting import generate_report

report = generate_report(
    ["runs/rate-300", "runs/rate-500", "runs/rate-700"],
    output="reports/rates",
    formats=("png", "pdf", "svg"),
    dpi=600,
    energy_unit="kcal/mol",
    length_unit="angstrom",
    labels=["300 K", "500 K", "700 K"],
    series=["RPMD", "RPMD", "RPMD"],
    arrhenius_fit=True,
)
```

The entry point also accepts `reference`, `include_estimates`, `kie`,
`convergence` and `overwrite`, corresponding to the CLI options above. Set
`kie=("H", "D")` to request a ratio of two named series. `arrhenius_fit=False`
and `include_estimates=False` are the defaults.

## Literature and figure style

The PMF, recrossing, isotope and rate comparisons follow the scientific figure
types in [Gui et al., JCTC 2022](https://doi.org/10.1021/acs.jctc.2c00522).
Channel comparisons are motivated by the
[H + MgH study](https://arxiv.org/abs/2007.07059); sampling comparisons also draw
on the [NO/Pd(111) study](https://doi.org/10.1039/D2SC06559B).
[Post-RPMDrate](https://github.com/chen-yu-hao/Post-RPMDrate) is a functional
reference for displaying sampling diagnostics alongside final results. Its code
is not imported or copied.

The profile uses 3.33 in compact plots or 7 in larger figures and at least 8 pt
type. Default outputs are 300 dpi color PNG and vector PDF; SVG and other DPI
values are explicit options. The
[current JCTC graphics guidance](https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce)
specifies 300 dpi for color, 600 dpi for grayscale and 1200 dpi for black-and-white
line art. Inspect exported figures at their intended publication size.
