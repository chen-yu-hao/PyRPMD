# Dynamics-to-report validation

This benchmark runs deliberately short artificial-potential jobs, then compares
the numerical CSV exports against the actual saved archives. It checks the
postprocessing pipeline; it does not establish chemical accuracy or physical
convergence.

```bash
python -m pip install -e '.[test,plots]'
python -m benchmarks.analysis_reports.validate \
  --output runs/analysis-validation --style nature --formats png pdf svg
```

The output directory must be new or empty. It contains `validation.json`, the
resolved inputs and raw workflow results in `runs/`, and an offline HTML report,
figures and plotted CSV data in `report/`.

| Case | Checks |
| --- | --- |
| Adaptive 1D harmonic FES | Actual midpoint insertion, frozen production centers, pilot/production separation, numerical FES/overlap and saved pilot windows/overlap/ESS/step-count exports |
| Non-square 2D harmonic FES | 5 × 7 grid orientation, real supported-cell mask, energy/length conversion and overlap matrix orientation |
| Two-replica well-tempered metadynamics | Saved HILLS times, heights, bias values and replica labels, masked bias FES, absence of fabricated equilibrium standard errors |
| Free-potential rate mechanics | Saved flux-PMF gauge, values, masks and unavailable uncertainty; four actual recrossing children; transmission/time export; retained diagnostic estimate with thermal rate withheld |
| Exhausted adaptive pilot | Available pilot progress figures, no production FES fabricated |

Every file in each run is hashed before and after reporting, including files
that the report does not read. Tests also check completed adaptive resume,
recomputed analysis on a different non-square grid without new samples, strict
JSON and report generation with potential construction/dynamics blocked.
Numerical comparisons use relative tolerance `1e-12` and absolute tolerance
`1e-14` in the displayed units; masks and replica labels must agree exactly.
Tests deliberately alter a CSV value, precision, missing uncertainty or replica
label and require validation to fail. CSV agreement verifies preservation of
the producer's quantities and units; it is not an independent validation of
the PMF estimator or flux normalization theory.

The examples use atomic-unit masses and high temperatures to finish quickly.
The free potential and harmonic model are not reactive chemical surfaces. The
recorded convergence state remains the producer's state, independently of
whether rendering and numerical data checks pass.

The `Analysis reports` workflow runs the regression tests and this three-format
validation on Python 3.11 and 3.12, and uploads JUnit results, source data,
`validation.json` and report figures as workflow artifacts.
