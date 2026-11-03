# Development, validation and maintenance

## Source layout

| Path | Responsibility |
| --- | --- |
| `src/pyrpmd/input.py`, `units.py`, `system.py` | Input validation, units, molecular and periodic identity |
| `src/pyrpmd/runner.py`, `cli.py` | Python and CLI entry points, workflow dispatch |
| `src/pyrpmd/dynamics.py`, `native/` | Ring integrators, thermostats, constraints, C++17/OpenMP |
| `src/pyrpmd/potentials/` | Provider lifecycle, units and the batch contract |
| `src/pyrpmd/cv.py`, `bias.py` | CVs and Jacobians, biases and measures |
| `src/pyrpmd/workflows/` | rate, FES, materials, surface, PCET and the other workflows |
| `src/pyrpmd/analysis/` | Free-energy, diffusion and statistical analysis |
| `src/pyrpmd/pcet.py`, `pcet_input.py` | PCET kernels and model input |
| `src/pyrpmd/nonadiabatic.py`, `potentials/diabatic.py` | NRPMD mapping variables, matrix potentials and MInt propagation |
| `src/pyrpmd/checkpoint.py`, `io.py` | Atomic commits, verification and scientific identity |
| `src/pyrpmd/parallel/` | Task processes, bead pools, MPI and GPU placement |
| `examples/`, `tests/`, `benchmarks/` | Runnable examples, regression tests, source-pinned benchmarks |

### Shared utility contracts

Use `validation.py` for scalar integer, finiteness and sign/bounds checks. Integer
checks reject booleans and fractional values; callers can retain their domain
exception type and error message. Unit conversion and task-specific constraints
remain in their resolvers.

`io.jsonable` and `io.write_json` use strict metadata conversion by default.
`io.report_jsonable` and `io.write_report_json` encode unavailable numerical
estimates as JSON `null`. Callable, dataclass and `to_dict` policies preserve
existing representations. `write_json` delegates publication to
`atomic_write_text`; `ensure_ascii` and `trailing_newline` retain each workflow's
text format. Use `checkpoint.atomic_savez` for the common atomic
compressed NPZ container and integrity envelope; `save_metadata_npz` only
packages dRPMD metadata before delegating to it.

Fingerprints share an encoder while preserving their existing schemas:
`io.fingerprint(config, system)` keeps the manifest identity, including analysis
settings; supplying `extras` selects the stream identity, excluding analysis-only
and execution settings while retaining the sampling measure, flux axis and CV
geometry. FES and OPES use the stream form. dRPMD builds its versioned physics
payload, parent-chain policy and local dependency digests in
`drpmd_persistence.py`, then uses `io.fingerprint_payload`. Changing an encoding or
schema requires a restart compatibility decision and regression evidence.

### Focused implementation modules

The following paths are relative to `src/pyrpmd/`:

| Family | Responsibility |
| --- | --- |
| `input.py`, `input_*.py` | `JobSpec` loading, overrides and resolution order; separate schema, scientific-option, path, runtime, system/provider and task-contract modules. |
| `workflows/drpmd.py`, `drpmd_*.py` | Dispatch, sampling, trajectory propagation, outcome classification, analysis and restart conventions. |
| `workflows/rate.py`, `rate_*.py` | Coordination, constrained geometry, options, PMF preparation, parent/child trajectories and statistical aggregation. |
| `analysis/fes.py` | FES reconstruction and established analysis imports. |
| `analysis/mbar.py`, `umbrella.py`, `fes_statistics.py`, `rate_pmf.py` | MBAR/reusable weights, umbrella energies/histograms/UI, correlation/bootstrap statistics, and rate-barrier estimation respectively. |

Compatibility entry points use explicit re-exports or small adapters so existing
imports and serialized global references remain usable. Process/MPI workers stay
importable, and dRPMD passes its factory and persistence callbacks into the worker
kernels. FES adapters resolve the solver, resampler and UI callbacks used by
existing diagnostics at call time. New modules import their dependencies from the
owning implementation module, avoiding a dependency back to the compatibility
entry point.

A change to a public interface updates the documentation, the examples and tests that check the
numerical and state behavior in the same commit. Changes to sampling measures, forces, constraints,
random streams or restart identity come with their scientific reasoning and regression tests.

The [implementation and guide audit](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/implementation-guide-audit-2026-10-31.md)
records the review scope, prioritized work, implemented fixes and remaining validation boundaries.

The [2026-10-31 user-experience review](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/ux-review-2026-10-31.md)
tracks the audit scope, prioritized to-do list, implemented fixes and remaining work.

## Running the tests

```bash
python -m pip install -e '.[test,ase]'
python -m pytest -q
git diff --check
```

Native tests need the compiled extension, and socket tests need loopback access. Skipped tests did
not run; install the optional dependency to cover them. With MPI available, also run:

```bash
mpiexec -n 4 python tests/mpi_integration.py
mpiexec -n 4 python tests/mpi_workflow_smoke.py runs/mpi-workflow-check
```

The materials and surface MPI scripts and the real-model benchmarks are described in their
validation records. These scripts run on one machine; multi-node and CUDA runs need their own checks.

## Building packages

```bash
python -m pip install build
python -m build
```

The sdist contains the sources, native headers, manual sources, examples and provenance notes; the
wheel contains the compiled extension and the public ABI header. Install the wheel into a separate
environment and test it outside the source tree, so an editable install cannot mask a missing
module. Wheel names depend on Python and the platform; install the file that was built.

Portable release wheels now use the separate
[distribution workflow](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/distribution.md):
cibuildwheel builds CPython 3.11–3.14 wheels and tests each installed artifact. The same
guide covers opt-in PyPI/TestPyPI trusted publishing and the conda-forge submission
procedure. The local conda recipe builds and tests packages without uploading them.

## The manual

The manual has two editions, each its own Sphinx project:

| Edition | Entry and chapters | Configuration |
| --- | --- | --- |
| English (default) | `docs/index.md`, `docs/manual/*.md` | `docs/conf.py` |
| Chinese | `docs/zh/index.md`, `docs/zh/manual/*.md` | `docs/zh/conf.py` (reuses the English settings, sets language and title) |

Both use `docs/_static/manual.css` and `docs/_templates/language-switcher.html`, and are built with
[Sphinx](https://www.sphinx-doc.org/), [MyST](https://myst-parser.readthedocs.io/) and the
[PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/). The chapter structure follows the
Quickstart / User Guide / API / Contributor Guide layout of [PySCF](https://pyscf.org/user/index.html).

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
python -m sphinx -W --keep-going -b singlehtml docs/zh docs/_build/singlehtml/zh
```

The HTML build has navigation, search and one page per chapter; singlehtml puts everything on one
page. Build the Chinese edition into `zh/` under the English output so the language switcher finds
each page's counterpart. Single-page search/index controls target the sibling
`html/` edition, so publish both output layouts together. Build output is ignored by Git. The GitHub **Documentation** workflow builds
both editions strictly, runs the getting-started examples and uploads the `pyrpmd-manual-html`
artifact.

Both editions keep the same core chapter files, headings, commands and tables; update both in the
same commit. Extension guides and validation reports can be published in English, with a concise
entry in the Chinese API chapter. The language switcher sends those guides to the Chinese contents
until a translated counterpart exists. Links between the editions use full GitHub URLs, because
each Sphinx project contains only its own language.

### Checking navigation and links

Run the same local checks as Documentation CI after building both editions and
both HTML layouts:

```bash
python -m unittest discover -s docs/_tools -p 'test_*.py' -v
python docs/_tools/check_links.py --html docs/_build/html --html docs/_build/singlehtml --external-project docs/_build/external-links
python -m sphinx -W --keep-going -b linkcheck docs/_build/external-links docs/_build/linkcheck
```

The first check enforces the README byte budget and matching stable section IDs,
matching core chapter filenames, real local targets and rendered fragments. It
resolves `main` repository links against the checkout so new pages can be checked
before a PR merges. Links pinned to a full commit SHA are checked against that
exact Git tree/blob; missing local history is checked online instead. No object
fetch is performed by the offline checker. External URLs from both READMEs and both manuals are deduplicated
with their source locations and checked in a separate CI job. DOI URLs use
standard citation-metadata content negotiation to verify the registered DOI
without depending on a publisher's browser challenge; other HTTP targets are
checked normally. HTTP failures remain failures rather than being treated as
proof that a scientific reference is invalid. Inspect the saved linkcheck artifact
before changing a citation or adding an exception.

Add new current guides to `docs/conf.py` and a reachable toctree; do not use an
`orphan` flag to hide a missing navigation entry. Keep new release notes and
core chapters paired in English and Chinese. English-only extension guides need
an explicit, accurately labelled entry in the Chinese index. Preserve existing
paths when possible; a rename must update inbound links and old anchors.

## Writing conventions

Each tutorial gives the problem it solves, a runnable input, the required environment, the output
paths and units, and the key parameters. Include real code with `literalinclude` and generate API
signatures with autodoc so they follow the source.

Write direct, task-first prose: say what a command does and what it produces. State each limitation
once, where the reader acts on it, as a plain description of behavior.

Public examples pass the input parser, and the getting-started examples run, read their results and
resume. Links inside the manual are relative `.md` paths; links to validation reports, example
sources and specifications are full GitHub URLs. Placeholders in examples are marked as such.

When periodic geometry, fixed-atom support or model identity changes, check how old checkpoints are
interpreted: migrate states whose old identity can be verified, and reject the others explicitly.

## Provenance and licenses

The project license and third-party notices are in
[LICENSE](https://github.com/chen-yu-hao/PyRPMD/blob/main/LICENSE) and
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md).
Code adapted from sources such as RPMDrate keeps its source commit, authors and license. External
electronic-structure programs, machine-learning models and EAM/PES data carry their own licenses.

Commit descriptions report the tests run, skips, hardware and known limitations.
