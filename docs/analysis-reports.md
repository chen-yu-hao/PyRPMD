# Automatic analysis and publication figures

`pyrpmd report` turns saved PyRPMD results into a local HTML report, machine-readable
diagnostics, tables and figures. It reads the results already produced by the
workflows; it does not rerun dynamics or invoke the potential. The
[English manual](manual/analysis-reports.md) and
[中文手册](zh/manual/analysis-reports.md) give command-line and Python examples.

## Design references and figure selection

The figure selection follows the scientific questions addressed in RPMD studies,
with sampling diagnostics inspired by
[Post-RPMDrate](https://github.com/wbfan95/Post-RPMDrate) and the
[maintained user fork](https://github.com/chen-yu-hao/Post-RPMDrate).
The implementation is independent. No license file was found at either repository
root when inspected; no Post-RPMDrate source is copied or executed. This release
reads native PyRPMD results, not legacy RPMDrate `.dat` files or `input.py`.

| Reference | Relevant published figures | Report design |
| --- | --- | --- |
| [Gui et al., JCTC 2022, DOI 10.1021/acs.jctc.2c00522](https://doi.org/10.1021/acs.jctc.2c00522), [author manuscript](https://arxiv.org/abs/2205.09331) | Rate comparisons; PMF and transmission coefficient panels; classical/RPMD and isotope comparisons; sampling, integrator and bead checks in SI | PMF and recrossing views, temperature series, KIE and parameter comparisons |
| [Yang et al., H + MgH roaming study](https://arxiv.org/abs/2007.07059) | Channel PMFs, transmission coefficients and channel/total rate comparison | Keep channel identity and normalization explicit |
| [Chemical Science 2023, DOI 10.1039/D2SC06559B](https://doi.org/10.1039/D2SC06559B) | PMF tests against window spacing, overlap, trajectory duration/count and beads; frozen/relaxed surface comparisons | Show available sampling and parameter diagnostics with their limits |
| [Post-RPMDrate](https://github.com/chen-yu-hao/Post-RPMDrate) | Window overlap, variance, restraint constants, PMF, recrossing and sampling evolution | Include sampling quality beside the final observable |

This matrix is a design reference, not a promise that every figure in each paper
can be reconstructed from every run. For example, a minimum-energy path requires
path data, while an energy-conservation trace requires saved energies. The report
records missing inputs and available diagnostics instead of manufacturing curves.

## Scientific meaning of the outputs

### A PES cut is not a free-energy surface

A two-dimensional potential-energy cut fixes or optimizes the coordinates not on
its axes. A two-dimensional FES is a probability-density-derived free energy in
specified collective variables after sampling the other degrees of freedom. Use
the saved quantity and CV metadata when labeling a plot. Bond-length coordinates
such as `r(H1,H2)` and `r(H2,H3)` are appropriate only when those are the actual
sampled CVs; renaming axes does not transform an existing surface.

Unsupported FES cells remain masked. They are not interpolated into a reaction
path, treated as infinite barriers, or assigned arbitrary high finite energies.
An attractive contour plot does not establish that windows share a global
normalization or that a reaction channel has been sampled. Probability density,
cell probability mass and flux-weighted rate PMF have different meanings.

### The workflow decides whether a rate is published

By default, rate plots use published rate fields in the saved workflow result.
When a workflow withholds a rate after a failed diagnostic, the report retains
that decision. `--include-estimates` exposes available finite-time estimates as
diagnostics, separately from published rate series, and excludes them from fits.
Publication of a field is not a claim of physical convergence or agreement with
experiment. The report does not turn a smooth PMF or a flat tail into such a claim.

In particular, a gas-phase bimolecular rate has units
`cm3 molecule^-1 s^-1`, whereas a basin-normalized site rate has units `s^-1`.
They cannot share one rate series. Electronic degeneracy, channel multiplicity,
reactant normalization and dynamical corrections remain properties of the source
workflow; reporting adds no reaction-specific correction factors.

### Uncertainty has an identified source

Saved standard errors, intervals and diagnostic spreads retain their distinct
meaning. A standard-error band is not automatically a 95% confidence interval.
Variation across beads or time steps is parameter sensitivity, not statistical
error. A short recrossing tail can pass a numerical flatness check while sampling
or systematic uncertainty remains substantial.

The optional Arrhenius fit is an unweighted descriptive regression of accepted,
positive rates at at least three distinct temperatures. It reports the fitted
activation energy and goodness of fit; it does not create uncertainty on the
underlying rates or establish that one activation energy describes a tunneling
regime. KIEs use matched temperatures without interpolation. If both saved rate
standard errors exist, the reported ratio error uses independent first-order
propagation; this does not account for covariance between paired simulations.

dRPMD outcome counts distinguish products, nonreactive, unresolved and failed
trajectories. A count fraction alone is not a cross section or thermal rate.
Those quantities require the source workflow's impact-parameter sampling,
weights, classification rules and normalization.

## Graphics profile

The [JCTC author guidelines](https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce)
specify widths up to 3.33 in for a single column and 4.167–7 in for a double
column, lines at least 0.5 pt wide, and lettering at least 4.5 pt at publication
size. Raster minima are 300 dpi for color, 600 dpi for grayscale and 1200 dpi for
black-and-white line art (guidelines checked 2026-10-31).

The report uses 3.33 in compact figures or 7 in larger figures, readable type of at
least 8 pt, and explicit quantity/unit labels. PDF and SVG preserve vector
elements; the default color PNG is 300 dpi, with `--dpi 600` available for denser
exports. These settings provide a reproducible starting point. Authors still need
to inspect the final panel size and the destination journal's requirements.

## Reproducibility and input protection

The report saves the files actually read, SHA-256 digests and software versions,
along with plot data and configuration. Reference data need a source label; no
experimental values are bundled. The example CSV is deliberately synthetic.

Reporting does not execute saved Python inputs. It rejects a pre-existing output
directory unless `--overwrite` is explicitly requested and the directory carries
a valid marker identifying an owned PyRPMD report. Inputs and unrelated output
directories are not overwrite targets. The report itself does not replace the
original result files, trajectories or run manifest needed for reproducibility.
