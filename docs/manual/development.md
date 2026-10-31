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

A change to a public interface updates the documentation, the examples and tests that check the
numerical and state behavior in the same commit. Changes to sampling measures, forces, constraints,
random streams or restart identity come with their scientific reasoning and regression tests.

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
each page's counterpart. Build output is ignored by Git. The GitHub **Documentation** workflow builds
both editions strictly, runs the getting-started examples and uploads the `pyrpmd-manual-html`
artifact.

Both editions keep the same core chapter files, headings, commands and tables; update both in the
same commit. Extension guides and validation reports can be published in English, with a concise
entry in the Chinese API chapter. The language switcher sends those guides to the Chinese contents
until a translated counterpart exists. Links between the editions use full GitHub URLs, because
each Sphinx project contains only its own language.

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
