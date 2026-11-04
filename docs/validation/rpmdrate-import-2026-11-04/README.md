# T52 RPMDrate input conversion validation

Four of four pinned upstream inputs pass the real conversion, static CLI
validation and preparation paths with `--allow-unmapped`. All four refuse
conversion without that option, return exit code 2, and create zero output
files. The independently written small example converts without that option.
These are input/structure checks; `physical_converged=false`. The upstream conversion checks did not load/evaluate a PES or run
RPMDrate dynamics. External RPMDrate/PES execution: **未实测**.

## Source and acceptance agreement

- PyRPMD base: `cb28829ecb1cabe471136b74e585ba3e4b9728fb`, the latest main
  fetched before creating `agent/t52-rpmdrate-import`.
- RPMDrate: `710670c6f753b1159c0147c817e82998c2cc8486`,
  <https://github.com/chen-yu-hao/RPMDrate/tree/710670c6f753b1159c0147c817e82998c2cc8486>.
- Rules/cards read from `claude/round-5-task-cards@787f0491b3b21faa5709c1d1a5406d2859f55a8e`:
  G1–G19 and T52. No T28/T50 implementation is included.
- The task owner explicitly amended the H3 comparison: use a handwritten
  configuration faithful to the upstream input for field equality, and use
  existing BKMP2 configurations only for reaction topology. The original
  demand for identical geometry/Rinf was inconsistent: upstream has
  approximately 30.000158 bohr while both BKMP2 references have 10 bohr.

The compact [summary.json](summary.json) records each source input SHA256,
reaction fields, actual budgets, geometry/spring errors, every unmapped item,
and software/hardware information. The source scripts were executed only with
recording functions and PES import references. No external source or PES code
is redistributed.

## Commands and observed results

Environment: local host `cluster`, Intel Xeon Platinum 8176 @ 2.10 GHz,
2 sockets, 56 physical cores / 112 logical CPUs. CPU only;
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Python 3.11.5, NumPy 2.4.6,
SciPy 1.17.1, h5py 3.16.0, pytest 9.2.0, Sphinx 8.2.3.
The package and its native extensions were built with an editable install in
an isolated virtual environment. No existing environment was modified.

```bash
python -m pip install -e '.[test,docs]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_rpmdrate_import.py
python -m pytest -q -m "not campaign"
python -m sphinx -W --keep-going -b html docs docs/_build/html
git clone https://github.com/chen-yu-hao/RPMDrate.git /tmp/t52-rpmdrate
git -C /tmp/t52-rpmdrate checkout 710670c6f753b1159c0147c817e82998c2cc8486
python examples/rpmdrate_import/validate_upstream.py \
  --upstream /tmp/t52-rpmdrate --output /tmp/t52-validation
```

The actual install used separate dependency installation followed by
`python -m pip install --no-deps --no-build-isolation -e .`; the condensed
command above installs the same declared extras. Actual raw logs and generated
files are retained under `/tmp/pyrpmd-t52-*`; the checks do not depend on those
paths. Use a new output directory for each reproduction.

Raw focused-test summary:

```text
47 passed in 6.82s
```

Raw full non-campaign summary:

```text
9374 passed, 235 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1860.86s (0:31:00)
```

The strict Sphinx build returned zero with `build succeeded.`. The upstream
check returned zero with:

```text
H+H2: 110 windows, strict refusal, validate/prepare passed
H+CH4: 110 windows, strict refusal, validate/prepare passed
H+C2H6: 106 windows, strict refusal, validate/prepare passed
OH+CH4: 110 windows, strict refusal, validate/prepare passed
H3 handwritten identity and both BKMP2 topology comparisons passed
```

## Per-input comparison

Each row uses 1000 K, 4 beads, and the upstream timestep of 0.0001 ps
(0.1 fs). Every window keeps its center, 20 ps burn-in (200,000 steps),
100 ps production (1,000,000 steps), and one observation per production step.

| Input | Atoms | Windows | Trajectories/window | Rinf, bohr | Equivalent paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| H+H2 | 3 | 110 | 200 | 30.00015813923641 | 1 |
| H+CH4 | 6 | 110 | 200 | 30.00015813923641 | 3 |
| H+C2H6 | 9 | 106 | 150 | 15.00007906961820 | 2 |
| OH+CH4 | 7 | 110 | 200 | 15.00007906961820 | 3 |

All resolved springs are 100 hartree, with zero error against the independent
formula below. Geometry was checked independently by reading each upstream
literal TS with `ast.literal_eval` and converting its explicitly declared unit;
the maximum coordinate error is recorded per case in `summary.json`.
The reaction partitions, paired bonds and each atom permutation were compared
to handwritten expected lists. No geometry optimization or atom reordering
was performed by the importer.

All four recrossing requests retain 20 ps parent equilibration, 2 ps between
draws, 0.05 ps children, 1 parent chain, 1,000 configurations, 50 child pairs
per configuration, and 100,000 individual child trajectories. Omitted
`xi_current` selects the PMF maximum. None of these trajectories was executed.

Each input has four explicitly reported unmapped settings:

| Source setting | Why equivalence is unavailable | Explicit replacement |
| --- | --- | --- |
| `Window.trajectories > 1` | RPMDrate continues each previous final geometry; PyRPMD chains are independent | Preserve count and per-trajectory times as independent chains |
| `generateUmbrellaConfigurations` | No sequential, single-bead seed-generation input stage | Start windows from supplied TS, with their original equilibration budgets |
| H3 GLE / other examples' omitted Andersen period | Rate sampling lacks GLE; omitted periods depend separately on each upstream stage length | Fixed global Andersen refresh every 10 steps for umbrella and parent |
| `xi_min`, `xi_max` | PyRPMD UI grid starts at zero and ends at the largest center | Keep the requested histogram range and explicitly report the actual UI range |

The complete arguments of those calls remain in generated `import_report.json`.
No missing capability was added to another card's rate/sampling modules.

## H3 independent identity and topology

After real `JobSpec.resolve()`, the converted H3 input matches
`examples/rpmdrate_import/h3-reference.inp` in:

- fragments `[[0,1],[2]]`;
- forming bond `[[1,2]]`, breaking bond `[[0,1]]`;
- equivalent permutation `[[1,0,2]]`, symmetry factor 1;
- unchanged TS positions `[[0,0,-1.757],[0,0,0],[0,0,1.757]]` bohr;
- Rinf 15.8754 angstrom, compared in bohr with floating-point conversion tolerance.

The verifier calls `benchmarks/h3/rpmd.py`'s actual
`configuration(stage="rate")` and resolves
`benchmarks/h3/convergence_inputs/base.inp`. Mapping upstream atom labels to
benchmark labels with `[2,1,0]` makes the fragment partition and bond topology
equal. Conjugating the equivalent-path permutation gives `[0,2,1]`, as in
both benchmark inputs. Their geometry orientation and Rinf are deliberately
excluded from this topology comparison and are never substituted into output.

## Independent spring derivation

At the pinned commit, `rpmdrate/_main.f90:974` defines
`add_umbrella_potential`. For every bead k it adds

```text
V(k) += 0.5 * kforce * (xi - xi_current)**2
dVdq(i,j,k) += kforce * (xi - xi_current) * dxi(i,j)
```

Thus the stiffness in `W = K*(xi-xi_i)^2/2` is exactly the supplied `kforce`,
in hartree for dimensionless xi. The total ring Hamiltonian contains P copies
of W. PyRPMD applies that P internally. Its temperature-normalized input
therefore needs `force_constant = kforce / T`, giving
`K = force_constant * T = 0.1 * 1000 = 100 hartree` in these four inputs.
There is no extra `k_B`, temperature division in dynamics, or input bead factor.

The unit test independently evaluates the per-bead energy and gradient,
including heterogeneous springs, and compares them to actual resolved window
stiffnesses. The upstream verifier checks every one of the 436 windows.

## Isolation, remaining limits and artifact inventory

Tests compare all `sys.modules` keys and object identities before/after
successful and failed conversion, including explicit PES-call refusal,
attempted `rpmdrate` import, newly loaded helpers and deleted/replaced module
bindings. A real local PES file with a top-level side effect remains unexecuted,
including when an existing module binding has the same name. Global import
hooks are not installed. Trusted Python can still perform unrelated arbitrary
side effects; the command is not a sandbox and requires no concurrent imports.

CLI tests use actual parsing and `runner.prepare` workflow dispatch. They do
not mock a PyRPMD parser, resolver or workflow. A static preparation correctly
reports `ready=false`: the saddle and PES remain unchecked.

Remaining differences include the RNG, isotope-mass constants, estimator
discretization and the first parent draw's offset. The report documents
defaults; this conversion does not claim exact trajectory replay. Output/checkpoint
migration, dRPMD, dynamics performance, GPU speedup and physical convergence
were not tested or claimed.

Before this change this validation directory did not exist. It contains only
this record and `summary.json`, totaling less than 1 MB. Generated converted
inputs, full per-call reports, raw CLI/test/build logs and native binaries are
excluded from Git; the supplied command reproduces the input evidence. The
normal test/build products remain ignored. AI assistance: Codex implemented
and tested this change; human scientific review has not been claimed.
