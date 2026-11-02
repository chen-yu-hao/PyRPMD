# Published-model validation drivers

These opt-in experiments accompany the
[2026-11-01 validation report](../../docs/validation/tier2-literature-2026-11-01/README.md).
They run actual trajectories and independently computed references; the full
campaign is intentionally outside ordinary unit-test collection.

| Driver | Purpose |
| --- | --- |
| spectroscopy.py / spectroscopy_static.py | Cao–Voth oscillator, independent quantum references, PA-CMD and quadratic force matching |
| path_integral.py / path_integral_analysis.py | SC, canonical ideal bosons, PIGS, actual sampling and exact linear stationary maps |
| nonadiabatic.py / nonadiabatic_analysis.py | Direct-data Tully I trajectories, stationary quantum scattering and uncertainty |
| sampling.py / sampling_sensitivity.py | OPES/PLUMED, Müller sampling, exchange, shared MetaD, biased clocks and trained ML lifecycle |
| plot_sampling.py / report_sampling.py | Figures and tables from the saved sampling evidence |

Use each driver's `--help` and the corresponding detailed report for exact
commands, optional dependencies, units, seeds and declared budgets. Set
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when using concurrent
workers. Native trajectory campaigns require the compiled package backend;
the direct PLUMED comparison additionally needs the pinned kernel and Python
wrapper. Source versions and runtime details are recorded beside the data.

Use a fresh `--output` directory for numerical recomputation. The committed
reference data retain their original source identities; later compatibility
or reporting edits do not relabel old trajectories. PI cache preflight
rejects incompatible parameters, seeds and source hashes before reuse.

Selected NPZ files in the sampling report contain unique raw time series,
not resumable q/p/RNG state. They are included in its explicit curated list.
Local restart caches from other campaigns are excluded. CSV/JSON numerical
results, sources, reports and figures remain available for inspection even
when a long calculation is not rerun.

The test files named `test_*_literature_reference.py` and
`test_spectroscopy_literature.py` independently verify reference solvers and
processing conventions. Passing them does not certify molecular convergence
or reproduce every paper figure; the reports preserve unresolved sampling,
finite-step and method limitations.
