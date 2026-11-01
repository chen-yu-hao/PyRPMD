# Transition-state search validation

The complete study is recorded in
[the dated validation report](../../docs/validation/transition-state-search-2026-11-01/README.md).
It separates deterministic saddle checks, real-PES handoff checks and
finite-sampling rate comparisons. Failed searches and unconverged physical
estimates remain in the evidence.

Run the reproducible analytic benchmark from the repository root:

```bash
PYTHONPATH=src python benchmarks/transition_state_search/analytic.py \
  --output docs/validation/transition-state-search-2026-11-01/analytic.json
PYTHONPATH=src python -m pytest -q tests/test_ts_search_literature.py
```

NumPy and SciPy are required. If ASE is installed, the script also performs four
independent CI-NEB/FIRE searches using ASE's `improvedtangent` implementation:
both barriers with nine images and the lower barrier with five and seven images.
The benchmark records failed cases as well as successful searches.
ASE uses a maximum vector force norm for its convergence criterion; PyRPMD uses
the largest force component. Both reported metrics retain those definitions.

## What is compared

The Cerjan–Miller function uses `a=b=c=1` from the Applications section,
Eq. (21), of [Banerjee et al. (1985)](https://doi.org/10.1021/j100247a015).
Its minimum and two saddle coordinates are stated in that paper. Saddle energy
`exp(-1)` is an analytical consequence of the formula, not a quoted paper value.

For the Müller–Brown surface, the parameters and all five stationary points
come from Eqs. (1)–(2) and Table I of
[Doye and Wales (2002)](https://doi.org/10.1063/1.1436470), available in their
[primary preprint](https://arxiv.org/pdf/cond-mat/0108310), PDF page 3. The paper prints
coordinates and energies to three decimal places. Agreement is checked at that
precision. Differences between rounded paper energies are labelled as derived
barriers, rather than separately published values.

An additional Adams surface uses the explicit formula reproduced in Eq. (37)
of [Mauro et al. (2005)](https://doi.org/10.1021/jp053581t). Its reported
coordinate text is inconsistent with the retrieved formula, so no inferred
correction is used as gold data. Its results are comparisons to independently
solved stationary points only.

Each surface has separately implemented analytical gradients and Hessians.
`scipy.optimize.root` solves the stationary equations with those derivatives;
tests also check them against central differences of the energy and gradient.
PyRPMD's P-RFO implementation independently estimates its Hessian from forces.

## Units, coverage, and limits

These are dimensionless model landscapes, embedded as one artificial atom with
coordinates `[x,y,z]` and a stable harmonic `z` direction. The exact positive
energy scale and `z` curvature are recorded in JSON. Coordinates are passed
numerically as bohr and scaled energies as hartree; all paper comparisons use
the original, unscaled model units. These tests do not predict molecular rates.

The matrix covers six saddles, three perturbed local guesses per saddle, three
force tolerances, a Hessian-step sweep, remote initial guesses, and stationary
points that must be rejected. CI-NEB covers both Müller–Brown barriers, a reversed
path, a path across both barriers, four image counts, two force tolerances, and
bowed initial paths. Exact gradients and Hessian indices independently check
every accepted stationary point.

The saved report retains the four five-image lower-barrier runs that do not
converge within the 1,000-step budget. The other 32 band searches succeed.
Seven, thirteen, and twenty-one images succeed on these tested paths. The
independent ASE five-image search also exhausts its iteration limit, while its
seven-image search converges; this does not establish a unique cause for the
coarse-band difficulty or guarantee convergence on other surfaces. Use at least
seven images for these examples and check image-count convergence for new paths.
Remote local
refinement can reach a different saddle or fail; the method does not promise a
global search. The paper comparison concerns stationary coordinates, energies,
and indices. Iteration counts are measured for this implementation, not claimed
to reproduce either paper's optimizer settings or trajectory.

## Replaying failure evidence

Alongside the JSON report, the script saves
`analytic-neb-failure-trajectories.npz`. This contains all 1,001 evaluated bands
for each failed search, including the initial band, fixed endpoints, unscaled
energies, and physical forces. The JSON gives each case's array keys, shapes,
units, and the archive's SHA-256 hash. Load it without pickle:

```python
import numpy as np

with np.load("docs/validation/transition-state-search-2026-11-01/"
             "analytic-neb-failure-trajectories.npz", allow_pickle=False) as trace:
    positions = trace["neb_009_positions"]  # [iteration, image, atom, xyz]
```

P-RFO force/energy histories, numerical controls, primary-source provenance,
runtime versions, and hashes of the benchmark and optimizer source files are
recorded directly in the JSON.

The CLI exits nonzero if a core check fails: published precision, all local
refinement and Hessian-step cases, rejection of nontransition stationary points,
correct indices for every accepted refinement, and every tested path with at
least seven images. `core_checks_passed` and the individual checks are recorded
in JSON. Remote-start behavior, five-image searches, and optional ASE results
remain diagnostic; their exact success/failure counts are not CI requirements.

## Covalent-radius cross-check

The separate optional-ASE check compares all 96 H–Cm values against ASE's
Cordero dataset and exercises H/D/T aliases, isotope overrides, all 22 elements
beyond Cm, explicit length units, and unknown symbols:

```bash
PYTHONPATH=src python benchmarks/transition_state_search/covalent_radii.py \
  --output docs/validation/transition-state-search-2026-11-01/covalent-radii.json
```

It records every difference, the reference package version and source hash,
and the exact error messages for unsupported elements. This is a software-data
cross-check against ASE, whose table cites Cordero et al.; it does not claim a
new experimental fit or an independent digitization of the paper.

## Real BKMP2 surface and rate comparison

With the native backend installed and a Fortran compiler available:

```bash
python benchmarks/h3/build_pes.py
python benchmarks/transition_state_search/analytic.py --output validation/analytic.json
python benchmarks/transition_state_search/h3.py --output validation/h3.json
python benchmarks/transition_state_search/h3_rate_sanity.py --output validation/h3-rate-sanity.json
python benchmarks/transition_state_search/rates.py \
  --cases classical-a classical-b classical-c classical-halfdt quantum-128 \
  --workers 4 --output runs/endpoint-literature --evidence validation/rates.json
```

`h3.py` compares the electronic saddle with Boothroyd et al. (1996), checks
endpoint reversal, rotations/translations and approximate TS guesses, and
checks the geometry passed into rate sampling. It does not run rate dynamics.
`h3_rate_sanity.py` independently checks the classical harmonic-TST scale,
SI/atomic units, equivalent channels and equality of the endpoint-derived rate
coordinate with the pre-existing supplied-TS input.

`rates.py` runs actual endpoint-to-rate dynamics against the 300 K classical
and 128-bead entries of Suleimanov, Allen and Green (2013), Table 2. Its shorter
budgets are explicitly numerical experiments. It records the PMF/QTST/kappa
decomposition, independent seeds, a halved time step, replica and time-half
diagnostics, same-sample UI/MBAR comparisons and uncertainty limitations. A
comparison ratio near one is not a statistical-convergence criterion. Raw
resolved inputs, samples and checkpoints stay under `--output`; use a fresh
directory for a changed physical or sampling protocol.

To isolate child integration error, `recrossing_timestep.py` reuses a completed
quantum case's saved parents and reconstructs its exact child momentum streams.
It halves only the child time step at fixed physical duration, checks actual
initial-state/flux matching, and preserves the original checkpoints:

```bash
python benchmarks/transition_state_search/recrossing_timestep.py \
  --source runs/endpoint-literature/quantum-128 --output runs/child-halfdt \
  --evidence validation/h3-recrossing-halfdt.json --workers 4
```

This controlled replay does not rerun the PMF or constrained-parent sampling
and therefore cannot establish their convergence.

## Render saved evidence

```bash
python benchmarks/transition_state_search/render.py validation
```

The renderer reads saved JSON only and writes PNG/PDF scientific figures.
It requires the optional `plots` dependencies. Set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for the recorded execution
protocol. The rate driver parallelizes trajectories with `--workers`; this
does not increase the number of independent random seeds.
