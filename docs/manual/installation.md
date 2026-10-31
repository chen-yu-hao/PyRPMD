# Installation and environment

## Install from source

Install PyRPMD from the Git repository into a fresh virtual environment, then check which
capabilities were built:

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

PyRPMD requires **Python 3.11 or newer**, NumPy ≥ 1.26, SciPy ≥ 1.11 and h5py ≥ 3.10; the full list
is in [`pyproject.toml`](https://github.com/chen-yu-hao/PyRPMD/blob/main/pyproject.toml). The
installer builds a C++17 extension, with OpenMP on Linux. On Windows PowerShell, activate the
environment with `.venv\Scripts\Activate.ps1`.

`pyrpmd info` prints JSON. The key fields are:

| Field | Meaning |
| --- | --- |
| `native.available`, `native.openmp` | The C++ extension and OpenMP are loaded |
| `native.integrators` | Compiled integrators: `cayley-bcb`, `normal-mode-bab`, `velocity-verlet` |
| `workflows` | The 11 available `task` values, including `opes` |
| `input_schema_version` | Current input schema version (1) |

Optional features such as ASE, MPI, Torch and external potential libraries need their own
dependencies; see [Optional dependencies](#optional-dependencies).

## Compilers and the reference implementation

The build uses the C/C++ compilers it finds, or the ones named in `CC` and `CXX`:

```bash
CC=gcc CXX=g++ python -m pip install -e .
```

Without a C++ compiler, install the NumPy reference implementation:

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

To keep the C++ kernels but build without OpenMP:

```bash
PYRPMD_NO_OPENMP=1 python -m pip install -e .
```

Both variables act at build time. Reinstall after updating the source, especially the native
interfaces, so Python never loads an extension built from older code; `pyrpmd info` shows what is
loaded.

Two input settings control the compiled code at run time. `dynamics.native="required"` demands
compiled integrator kernels. `dynamics.loop_backend="native"` demands that the whole propagation
loop runs in the fused native path. With the default workflow engine, a custom Python potential or
a periodic/fixed-atom system uses compiled integrator substeps inside a Python outer loop.
The separately selected native PIMD engine has its own PES and execution requirements;
see [Execution and parallelism](execution.md).

## Optional dependencies

| Install option | Enables | Also needs |
| --- | --- | --- |
| `python -m pip install -e '.[ase]'` | ASE calculator potentials | The calculator's model or program |
| `python -m pip install -e '.[mpi]'` | MPI trajectory groups and bead splitting | An MPI library and launcher matching mpi4py |
| `python -m pip install -e '.[gpu]'` | Torch potentials | A PyTorch/CUDA build matching the node driver, and a model |
| `python -m pip install -e '.[plots]'` | Matplotlib figures and `pyrpmd report` | Saved PyRPMD results |
| `python -m pip install -e '.[analysis]'` | PyMBAR cross-checks and the plotting dependencies | Routine FES uses the built-in MBAR |
| `python -m pip install -e '.[test]'` | pytest suite | Tests for optional interfaces skip when those are missing |
| `python -m pip install -e '.[docs]'` | Building this manual | Sphinx, MyST, PyData theme |

Use the MPI launcher and MPI library from the same environment; on a cluster, load the site
modules first, then install mpi4py. Pick the PyTorch CPU or CUDA wheel that matches the driver. The
BKMP2 benchmark also needs GNU Fortran; its build script downloads and checksums the third-party
potential source.

## Verify the installation

From the repository root:

```bash
python -m pytest -q
pyrpmd validate examples/manual/quickstart.inp --check-pes
python examples/manual/low_level.py
```

With only `.[test]` installed, tests that need ASE, BKMP2, an EAM model, PyMBAR, Torch or
matplotlib are skipped; everything else should pass. The [Quickstart](quickstart.md) then
exercises output, analysis and restart.

## Build this manual

The manual has English and Chinese editions with matching chapters. Build the Chinese edition into
the `zh/` directory of the English output so the language button links the two:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

Open `docs/_build/html/index.html` (English) or `docs/_build/html/zh/index.html` (Chinese). The build
includes the API reference generated from the current source, and `-W` turns warnings into errors.
No external potentials or long trajectories are needed. Release and development notes are in the
[Developer guide](development.md).
