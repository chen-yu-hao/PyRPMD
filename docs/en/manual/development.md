# Development, validation and maintenance

## Source layout

| Path | Responsibility |
| --- | --- |
| `src/pyrpmd/input.py`, `units.py`, `system.py` | Strict input, units, molecular/periodic identity |
| `src/pyrpmd/runner.py`, `cli.py` | Python and CLI public entry points, workflow dispatch |
| `src/pyrpmd/dynamics.py`, `native/` | Ring integrators, thermalization, constraints, C++17/OpenMP |
| `src/pyrpmd/potentials/` | Provider lifecycle, units and batch contract |
| `src/pyrpmd/cv.py`, `bias.py` | CVs/Jacobians, biases and measures |
| `src/pyrpmd/workflows/` | rate, FES, materials, surface, PCET and other workflows |
| `src/pyrpmd/analysis/` | Free-energy, diffusion and statistical analysis |
| `src/pyrpmd/pcet.py`, `pcet_input.py` | Local PCET numerical kernels and model input |
| `src/pyrpmd/nonadiabatic.py`, `potentials/diabatic.py` | Experimental NRPMD mapping variables, matrix potentials and MInt propagation |
| `src/pyrpmd/checkpoint.py`, `io.py` | Atomic commits, verification and scientific identity |
| `src/pyrpmd/parallel/` | Task processes, bead pools, MPI and GPU placement |
| `examples/`, `tests/`, `benchmarks/` | Runnable examples, regressions, source-pinned system validations |

When changing a public interface, update the documentation, the related examples and tests that
actually check numerical/state behavior together. Changes to sampling measures, forces, constraints,
random streams or restart identity need a scientific justification and matching regressions.

## Local regression

```bash
python -m pip install -e '.[test,ase]'
python -m pytest -q
git diff --check
```

Native tests need the extension to be built; tests skipped because of missing optional dependencies
do not count as passed. Socket tests need local loopback permission. In an MPI environment also run:

```bash
mpiexec -n 4 python tests/mpi_integration.py
mpiexec -n 4 python tests/mpi_workflow_smoke.py runs/mpi-workflow-check
```

The materials- and surface-specific MPI scripts, real-model benchmarks and their environment settings
are described in their validation records. Passing tests with several ranks on one machine does not
prove cross-node performance or the real CUDA path.

## Building packages

```bash
python -m pip install build
python -m build
```

Check that the sdist contains the sources, native headers, manual sources, examples and provenance
notes, and that the wheel contains the compiled extension for the current platform and the public ABI
header. Install the built wheel into a separate environment and verify it outside the source tree so
that an editable install cannot hide a missing module. Wheel file names depend on Python and the
platform; install the file that was actually produced.

## Manual structure and build

The manual has Chinese and English editions, each its own Sphinx project:

| Edition | Entry and chapters | Configuration |
| --- | --- | --- |
| Chinese | `docs/index.md`, `docs/manual/*.md` | `docs/conf.py` |
| English | `docs/en/index.md`, `docs/en/manual/*.md` | `docs/en/conf.py` (reuses the Chinese settings, changes only language and title) |

Both share `docs/_static/manual.css` and `docs/_templates/language-switcher.html`. The manual uses
[Sphinx](https://www.sphinx-doc.org/), [MyST](https://myst-parser.readthedocs.io/) and the
[PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/). Its organization follows the
Quickstart / User Guide / API / Contributor Guide layout of [PySCF](https://pyscf.org/user/index.html);
the content is based on this project's implementation and keeps its own commands and object
conventions.

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/en docs/_build/html/en
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
python -m sphinx -W --keep-going -b singlehtml docs/en docs/_build/singlehtml/en
```

The regular HTML build provides navigation, search and separate chapters; singlehtml puts the text on
one page. The English edition must be written to `en/` under the Chinese output so that the language
switcher in the navigation bar finds the matching page. Generated files live in an ignored directory
and are not committed. The GitHub **Documentation** workflow runs strict builds of both languages and
the getting-started smoke tests, and uploads the `pyrpmd-manual-html` artifact.

Both languages keep the same chapter file names, heading levels, example commands and table contents.
When you change a chapter, update the other language in the same commit. Cross-language links use full
GitHub URLs, because each Sphinx project contains only the files of its own language.

## Writing and validation conventions

Every tutorial should state the problem it solves, a directly runnable input, the required
environment, output paths and units, the key parameters, and the physical conclusions the example
cannot prove. Include real code with `literalinclude`; generate new API signatures from the source
with autodoc so hand-written signatures cannot drift between versions.

Public examples must at least pass the current input parser. Key getting-started examples should run
completely, read their results and resume after completion. Documentation links come in two kinds:
links inside the current manual use relative `.md` paths; historical reports, example sources and
existing specifications use full GitHub paths. Placeholders in examples must be marked as such.

When periodic geometry, fixed-atom support or model identity changes, compare the semantics of old
checkpoints first; states that can be migrated should verify their old identity first, and states that
cannot be interpreted unambiguously should be rejected explicitly. Do not accept a restart whose
physical meaning changed just because the array shapes match.

## Provenance, licenses and scientific evidence

The project license and third-party notices are in
[LICENSE](https://github.com/chen-yu-hao/PyRPMD/blob/main/LICENSE) and
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md).
When adapting or porting sources such as RPMDrate, keep the source commit, authors and license.
External electronic-structure programs, machine-learning models and EAM/PES data have their own
licenses and domains of validity.

Report actual test results, skips, hardware and known limitations in the commit description. Code
regressions, independent numerical cross-checks, finite-sample system validations and final physical
predictions are different levels of evidence. Do not substitute statistics from a historical branch
for results of a new commit, and do not present a model pilot as a converged materials constant.
