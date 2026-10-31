# Analysis reporting: validation follow-up

This follow-up starts from PR #40 (`ffa2826162b952fecebd16184ca96a7fea9bd7a6`,
tree `003b2b6509ae83e626c7baae970519811a162bf0`). It strengthens saved-result
adapters and rendering without changing propagation or rate acceptance.

## Fixes exercised by the tests

- Adaptive progress remains available before production, when the FES archive
  is damaged, and in partial archives without a summary or resolved input.
  Nested umbrella/PMF plans receive distinct identifiers. Verified schema 1/2
  plans supply frozen production centers; pilot samples remain excluded.
- Vector CV input units and explicit quantity pairs are decoded once. Selected
  length units reach distances, radii, MSD/VACF and cross-section intervals.
- HDF5 CV units, measures, object types and integer stream identities are
  checked; a bad HILLS replica cannot leave a partially accepted curve.
- Explicit uncertainty masks override finite placeholders. Bad optional counts,
  overlap or ESS do not hide valid FES data and other diagnostics.
- Sparse available uncertainties on long curves are drawn as error bars.
  One-shot plot iterables work, headless exports do not initialize a GUI, and
  external Matplotlib state does not change the intended size/background.
- Optional Nature-style output uses 89/183 mm widths, 6–7 pt text, accessible
  colors, separate legends and editable PDF/SVG text. Full scientific labels,
  data, masks and uncertainty meanings are retained.

## Local verification

| Check | Observed result |
| --- | --- |
| Final `tests/test_report*.py` | 130 passed; no skips or failures |
| Repository regression, excluding the two socket cases | 1,721 passed; 23 optional-dependency skips; 153 subtests passed |
| Two i-PI local-socket cases | 2 passed with loopback socket permission |
| English and Chinese Sphinx builds | Both passed with `-n -W --keep-going` |
| Native extension, source archive and platform wheel | Built successfully |
| Installed-wheel report smoke, outside the source checkout | Nature SVG output and native capabilities passed |

The repository regression began before the final isolated nested-plan fallback
refinement. The final 130-test reporting run includes that refinement and its
regression; the complete repository run is not represented as a second final-head
run. The unchanged native and workflow code is covered by the repository run.
Optional skips require an independently built BKMP2/EAM model, ASE, PyTorch or
PyMBAR. No new chemical trajectory validation is claimed from those skips.

## Actual dynamics-to-report checks

`python -m benchmarks.analysis_reports.validate --output runs/analysis-validation
--style nature --formats png pdf svg` runs short artificial-potential jobs and
compares plot CSV values directly with their saved NPZ/HILLS arrays.

| Case | Observed evidence |
| --- | --- |
| Adaptive 1D | 3 pilot rounds; 4 frozen windows; 160 production samples; zero pilot samples reused |
| 2D FES | Non-square 5 × 7 grid; 5 supported cells; 30 cells remain masked |
| Two-replica well-tempered metadynamics | 8 hills per replica; original clocks/heights retained; 34 masked cells |
| Free-potential rate mechanics | Four paired recrossing children; exported transmission and time match the archive |
| Exhausted adaptive pilot | Production never started; progress figures exist; no final FES is invented |

The combined report contains **54 figures in PNG, PDF and SVG**. All **85 source
files** were unchanged after reporting. The recorded three-format experiment
used 120 dpi PNG previews; PDF/SVG are vectors. The reproducible script defaults
to 300 dpi. These intentionally short artificial-potential runs validate
software contracts, not chemical accuracy or physical convergence.

The final tests also reanalyze a saved FES onto a 7 × 4 grid without altering
checkpoints, verify completed adaptive resume, and generate reports while
potential construction and dynamics stepping are explicitly blocked.

## Saved real-BKMP2 report replay

Two existing H + H2 archives produced **29 figures × three formats** with the
Nature profile and 300 dpi PNGs. The report checked all 14 source artifacts it
read before publication:

- 300 K, P=128 rate pilot: 44,400 PMF production samples, 32 parent
  configurations and 128 child trajectories. The saved rate remains withheld
  after its child-energy diagnostic. A detected recrossing plateau is not
  promoted to complete rate acceptance.
- 1500 K, P=16 bond-distance FES: 30,000 samples, 373/625 supported cells.
  All 252 unsupported cells stay masked. Axes remain the sampled H1–H2 and
  H2–H3 distances.

Both source archives retain `converged=false`. This replay checks parsing,
units, source protection and plotting; it is not a new or converged rate
calculation. Numerical evidence is recorded in `testing-followup.json`.
