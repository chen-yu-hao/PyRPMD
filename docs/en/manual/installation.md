# Installation and environment

## Install from source

The reproducible starting point for PyRPMD is the Git repository. The commands below install the
checked-out source into a fresh virtual environment and report the computational capabilities
that were actually loaded. Do not assume that a PyPI project with the same name corresponds to
this source tree.

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pyrpmd --version
pyrpmd info
```

PyRPMD needs **Python 3.11 or newer**, NumPy ≥ 1.26, SciPy ≥ 1.11 and h5py ≥ 3.10. A C++17
extension is built by default, with OpenMP enabled on Linux. The authoritative versions and
dependencies are in the repository's
[`pyproject.toml`](https://github.com/chen-yu-hao/PyRPMD/blob/main/pyproject.toml).
The activation command above is for Linux/macOS; in Windows PowerShell use
`.venv\Scripts\Activate.ps1`.

`pyrpmd info` prints JSON. The key fields are:

| Field | Meaning |
| --- | --- |
| `native.available`, `native.openmp` | Whether the C++ extension and OpenMP were loaded |
| `native.integrators` | Compiled integrators; normally `cayley-bcb`, `normal-mode-bab`, `velocity-verlet` |
| `workflows` | The 10 available `task` values |
| `input_schema_version` | Current input schema version (1) |

A finished installation does not mean that every external potential library, MPI, CUDA or
optional interface is ready; install the dependencies for the features you actually use.

## Compilers and the reference implementation

The build looks for usable C/C++ compilers and honors explicit `CC` and `CXX`. On Linux, for
example:

```bash
CC=gcc CXX=g++ python -m pip install -e .
```

Without a C++ compiler, install the NumPy reference path in a clean environment:

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

If only OpenMP is unavailable, keep the C++ kernels and disable OpenMP:

```bash
PYRPMD_NO_OPENMP=1 python -m pip install -e .
```

These two environment variables control the **build**. Setting them does not remove an old `.so`
that is already in the tree; check `pyrpmd info`, or use a fresh checkout and environment. After
updating the source, especially native interfaces, reinstall so that Python and an old extension
are never mixed.

In an input, `dynamics.native="required"` demands compiled integrator kernels, while
`dynamics.loop_backend="native"` demands that the whole propagation loop qualifies for the fused
native path. They are different switches. A custom Python potential or a periodic/fixed-atom
system can use compiled integrator substeps driven by a Python outer loop; see
[Execution and parallelism](execution.md).

## Optional dependencies

| Install option | Purpose | Also required |
| --- | --- | --- |
| `python -m pip install -e '.[ase]'` | ASE calculator potentials | The calculator's own model or program |
| `python -m pip install -e '.[mpi]'` | MPI trajectory groups and bead splitting | An MPI library and launcher matching mpi4py |
| `python -m pip install -e '.[gpu]'` | Torch potentials | A PyTorch/CUDA build matching the node driver, and a model |
| `python -m pip install -e '.[analysis]'` | Independent PyMBAR cross-checks | Routine FES uses the built-in MBAR; PyMBAR is not required |
| `python -m pip install -e '.[test]'` | pytest regression suite | Tests for missing optional interfaces are skipped |
| `python -m pip install -e '.[docs]'` | Building this manual | Sphinx, MyST, PyData theme |

The MPI launcher and the MPI library loaded by Python must come from the same environment. On a
cluster, follow the site's module and job conventions first, then install mpi4py. Choose the
PyTorch CPU/CUDA wheel for the actual driver; installing the extra does not prove that a GPU is
usable. The BKMP2 benchmark additionally needs GNU Fortran and downloads and checksums the
third-party potential source through a repository script.

## Verify the installation

From the root of the source repository:

```bash
python -m pytest -q
pyrpmd validate examples/manual/quickstart.inp --check-pes
python examples/manual/low_level.py
```

Record passed, skipped and failed results separately. With only `.[test]` installed, tests that
need ASE, BKMP2, an EAM model, PyMBAR, Torch or matplotlib are skipped, which means that path was
not verified in this environment; any failure should be reported. Running the full
[Quickstart](quickstart.md) additionally checks output, analysis and restart.

## Build this manual

The manual has Chinese and English editions with matching chapters. Build the English edition
into the `en/` directory of the Chinese output so the language button in the navigation bar links
the two:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/en docs/_build/html/en
```

Open `docs/_build/html/en/index.html` (English) or `docs/_build/html/index.html` (Chinese) in a
browser. The build includes the table of contents, cross-references and the public API imported
from the current source; `-W` turns documentation warnings into errors. Building the manual does
not require external potential programs or long trajectories. See the
[Developer guide](development.md) for development and release notes.
