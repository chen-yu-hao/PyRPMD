# Cited Ni/H EAM4: isolated static reference

This directory makes the author-attributed **EAM4** implementation inspectable
and testable. It does not reproduce a diffusion coefficient, supply a production
RPMD provider, or replace the separate Angelo–Moody–Baskes NiAlH integration
pilot in `benchmarks/materials`.

## Source identity

The [Cambridge OPTIM page](https://www-wales.ch.cam.ac.uk/OPTIM/) links to the
public [software archive](https://www-wales.ch.cam.ac.uk/svn.tar.bz2) and describes
its distribution as GPL. The inspected archive contains `wales/COPYING`, the
GPL version 3 text. Its member `wales/OPTIM/source/nimetpot.f90` credits Yury
Suleimanov, identifies EAM4 Ni(100)/H, and explicitly cites his
[2012 RPMD paper](https://doi.org/10.1021/jp302453z). The original header and
entire source file are preserved outside the repository in the recorded build.

| Item | Bytes | SHA256 |
| --- | ---: | --- |
| Public archive fetched 2026-11-01 | 122758544 | `8463fbbbab3f613c1daa021df99cf89791be3ca61c7f3b9c2f5307e484bf8029` |
| Original `nimetpot.f90` | 43812 | `b294b106cac20038eb941037b63f35dcce374ed40559d68217f5751b90d514a6` |
| Exact suffix, original byte offset 6150 onward | 37662 | `07abd148d739f60a065791f1459bd20ac6434264ffd69af39334937011036384` |

This establishes an available, author-attributed implementation associated with
the cited paper. It does not establish byte identity with the authors' 2012
executable or supply the missing original trajectory protocol. The archive URL
is mutable; the recorded hashes identify the particular inspected release.

## Build and replay

Obtain the public archive separately. No test or builder makes network requests.
Choose a new output directory outside this repository:

```bash
python -m benchmarks.materials_reference.build_eam4 \
  --source-archive /absolute/path/svn.tar.bz2 \
  --output /absolute/path/new-eam4-build \
  --fc /absolute/path/gfortran

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.materials_reference.verify_eam4 \
  --library /absolute/path/new-eam4-build/libeam4_static.so \
  --output /absolute/path/new-static-report.json

PYRPMD_EAM4_STATIC_LIBRARY=/absolute/path/new-eam4-build/libeam4_static.so \
  python -m pytest -q tests/test_materials_reference_eam4.py
```

The builder also accepts `--source-file` for an already extracted file with the
exact pinned length and hash. It copies the suffix starting with
`eamnih_forces_umb` unchanged and compiles it together with the small C bridge.
Only the OPTIM-dependent front wrapper is excluded. Coefficients, cutoff,
smoothing functions, energy convention and gradient calculations are untouched.
The original source, derived suffix, compiler logs, library and build receipt
remain outside Git. The build records compiler/version/flags and all relevant
hashes. A missing or changed source, changed library, populated output directory
or output directory inside the repository is rejected.

The adapter uses **angstrom, eV and eV/angstrom** without conversion. A stale
upstream comment in the energy-only routine mentions atomic units; its conversion
calls are commented out. The file header, force routine and implemented constants
use the units above. The two energy routes agree in the recorded checks.

## Narrow interface contract

The interface fixes `nb=1` and atom order to H, mobile Ni, fixed Ni. The caller
supplies the geometry, partition and box lengths. Nonperiodic or rectangular
XY-periodic boundaries are available; periodic lateral lengths must exceed
20 angstrom for the unchanged 10 angstrom minimum-image cutoff.

The original routine omits the fixed–fixed Ni pair-energy constant and returns
zero fixed-Ni derivatives. Consequently, absolute energies from different fixed
partitions must not be compared as though they used the same zero, and zero fixed
rows must not be interpreted as physical reaction forces. Finite differences
cover H and mobile Ni only. The matching active gradients across partitions are
also checked. The unmodified energy-only routine prints diagnostic components;
the saved stdout preserves them.

This interface does not certify the upstream mixed-bead branch, bath dynamics,
virial, arbitrary cells, multiple H atoms, or an all-quantum metal lattice. The
OPTIM wrapper's own default lattice dimensions and the manual's slab example are
not promoted to the 2012 paper protocol.

## Recorded bounded validation

The [predeclared protocol](eam4_static_protocol.json) uses an expressly excluded
nine-atom static fragment, three finite-difference spacings, four mobility/boundary
cases and four smoothing/cutoff probes. Arbitrary rotations apply only to
nonperiodic cases. The periodic case uses whole-system translations and integer
lateral lattice shifts. Every case is retained.

All static checks and all **10** tests passed. The largest gradient discrepancy
over the three difference spacings was **3.83e-8 eV/angstrom**; the finest-spacing
main-case error was at most **2.42e-9 eV/angstrom**. Symmetry energy and gradient
errors were at most **1.07e-14 eV** and **6.67e-15 eV/angstrom**. Compiler plus
verification/test subprocesses consumed about **2.02 CPU seconds**. These values
are implementation consistency checks, with no external diffusion target.

The complete [source audit and validation receipts](../../docs/validation/literature-convergence-2026-11-01/other-reactions/materials-source-index/README.md)
retain the unresolved original paper tables, lattice, masses, bead choices and
sampling protocol. No diffusion or literature-reproduction flag is set.
