# Automatic analysis and figure validation

Base: `13bc43ddc03a775812023ecaf90f7bf3975ba2f2` (main). The reporting
layer consumes saved numerical data and never constructs a PES or executes a
run's Python input. The dynamics kernels and acceptance criteria are unchanged.

## Verification

```bash
python -m pip install -e '.[test,plots]'
python -m pytest -q tests/test_report*.py
python -m sphinx -W --keep-going -b html docs /tmp/manual-en
python -m sphinx -W --keep-going -b html docs/zh /tmp/manual-zh
python setup.py bdist_wheel
```

- Final focused suite: **70 passed**. Cases exercise analytic Arrhenius recovery,
  independent KIE error propagation, dimensional conversions, masked 2D array
  orientation, withheld rates, signed transmission/electronic contributions,
  unresolved outcomes, distance traces, actual surface CSV schemas, optional
  dependencies, checkpoint integrity, provenance races, frozen adaptive plans,
  input-preserving replacement, CLI shortcuts and a real harmonic FES workflow.
- Both language manuals built with `sphinx -W` and no warnings. The wheel
  contains all five report modules and the optional `plots` dependency.
- An earlier full-repository run completed **1,649 passed, 23 optional skips,
  153 subtests passed**, with two existing socket tests blocked by this local
  sandbox's `PermissionError` on socket creation. This was before the final
  reporting-only regressions/distance panels; it is not represented as a final
  complete-suite pass. The normal GitHub tests cover the socket protocols.
- The new **Analysis reports** workflow runs all report tests and produces
  downloadable 1D/2D example reports in PNG/PDF/SVG. Existing tests, docs and
  package workflows continue to apply.

## Saved H + H2 / BKMP2 demonstration

Two existing datasets were read without modifying them. No fresh chemistry
simulation or converged chemical-rate claim is made here.

| Dataset | Temperature | Beads | Umbrella samples | Saved status |
| --- | ---: | ---: | ---: | --- |
| H3 rate pilot | 300 K | 128 | 44,400 | `pilot_requires_convergence` |
| Bond-coordinate H3 FES | 1500 K | 16 | 30,000 | `sampled` |

The combined report contains **29 figures**, each with a CSV table, plus
`index.html`, `report.md`, `report.json` and `summary.csv`. Every figure displays
its saved temperature/bead/timestep context. These two temperatures are separate
examples and are not used as a quantitative 1D/2D free-energy comparison.

- The 300 K source reports kappa = 0.6172200789, with SE = 0.1027314130.
  Its maximum child endpoint energy error is 0.1104913384 kBT per bead,
  above the producer's 0.1 threshold. The report therefore preserves
  `rate=null`; it does not promote `rate_estimate` into an accepted rate.
- The 1500 K FES uses the actual H1-H2 and H2-H3 centroid distances, converted
  from bohr to angstrom. 373 of 625 bins have support; all 252 unsupported
  bins remain grey. No interpolation fills that region.
- PMF/recrossing, uncertainty, overlap, ESS, sampling distributions,
  center offsets, variances, stiffness and selected saved coordinate traces
  were rendered and visually inspected. The numeric arrays keep their source
  gauge, masks and uncertainty interpretation.

Additional actual saved-output adapter checks covered NRPMD scattering, PIMD,
diffusion and PCET. These checks establish format/rendering compatibility,
not new validation of the underlying physical models.

The generated report records SHA-256 for every consumed source file and checks
the hashes again before publication. Source status/uncertainty are authoritative;
legacy accepted fields are not recertified by this visualization layer.
