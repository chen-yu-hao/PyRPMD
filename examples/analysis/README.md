# Analysis report examples

These examples show how to generate figures from native PyRPMD runs and how to
provide reference-data provenance. They are format and pipeline demonstrations,
not experimental benchmarks or converged reaction calculations.

## Install and report a harmonic FES

From the repository root:

```bash
python -m pip install -e '.[analysis]'
pyrpmd run examples/fes/harmonic2d.inp --output runs/report-harmonic
pyrpmd report runs/report-harmonic --output reports/harmonic \
  --formats png pdf svg --dpi 600
```

Open `reports/harmonic/index.html`. Inspect the FES mask and sampling diagnostics
along with the figure. The short harmonic input is a pilot; masked cells are not
filled by interpolation.

The combined shortcut is `pyrpmd run INPUT --report`, with default report output
under the new run's `analysis-report` directory. Existing results can be reported
with `pyrpmd analyze RUN --report --report-output DIR`.

## Report the teaching rate example

```bash
pyrpmd run examples/rate/pilot.inp --output runs/report-rate-pilot
pyrpmd report runs/report-rate-pilot --output reports/rate-pilot \
  --include-estimates
```

The teaching PES is not a fitted H3 surface. The workflow can withhold its final
rate when its diagnostics fail. `--include-estimates` displays saved finite-time
estimates as diagnostics; it does not promote them to published results or fit
them. No experiment should be inferred from these plots.

## Synthetic reference CSV

`reference-rates.csv` contains three invented rates and invented standard errors.
Every row is explicitly labeled synthetic. Its only purpose is to demonstrate
the required CSV columns:

```bash
pyrpmd report runs/report-rate-pilot --output reports/reference-format \
  --reference examples/analysis/reference-rates.csv
```

For your own reference data, keep `series`, `temperature_K`, `rate`, `rate_unit`
and `source`; `standard_error` is optional. Cite the actual experiment or dataset
in `source`, and use `standard_error` only for a documented standard error in the
same unit as the rate. Supported canonical rate units are
`cm3 molecule^-1 s^-1` and `s^-1`.

The synthetic reference is not chosen to agree with the teaching run. It does
not test the PES, rate normalization, sampling convergence or physical accuracy.

## Comparing your own temperature series

The following directories are placeholders for separately completed calculations:

```bash
pyrpmd report runs/rate-300 runs/rate-500 runs/rate-700 \
  --output reports/rate-series --labels '300 K' '500 K' '700 K' \
  --series RPMD RPMD RPMD --fit-arrhenius
```

`--fit-arrhenius` is optional. A fit requires three distinct temperatures with
positive published rates; it is an unweighted description, not a convergence
certificate. Use the same reaction, potential, rate definition and compatible
simulation settings throughout the series.

The report contains `index.html`, `report.md`, `report.json`, `summary.csv`,
figures and their CSV data. `report.json` indexes the files and records source
hashes. Existing report directories need explicit `--overwrite` and a valid
PyRPMD report marker; unrelated directories cannot be overwritten.

See the [English manual](../../docs/manual/analysis-reports.md),
[中文手册](../../docs/zh/manual/analysis-reports.md) and
[scientific design notes](../../docs/analysis-reports.md) for KIEs, parameter
comparisons, units, uncertainty and literature references.
