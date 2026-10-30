# Native numerical kernel and PES ABI

The normal installation builds `pyrpmd._native` through setuptools. It uses the
Python buffer API, C++17, and OpenMP on Linux; it does not depend on the NumPy C
ABI or require pybind11. Build from a checkout with:

```bash
python setup.py build_ext --inplace
```

`CC` and `CXX` select compilers. `PYRPMD_NO_NATIVE=1` installs the NumPy reference
implementation only. `PYRPMD_NO_OPENMP=1` builds the native single-thread kernel.
Requesting more than one dynamics thread without an OpenMP build is an error.
The optional CMake build is equivalent and is not required by pip.

## Cayley map

The native map uses `M = I + dt^2*(P/beta)^2*L/4`, solves for midpoint positions,
and updates the complete periodic ring. P=1 and P=2 have explicit special cases;
P>=3 uses Thomas factorization plus a rank-two periodic correction. Parallel
work is over complete Cartesian rings, never independent broken bead segments.
Each Python object owns an immutable factorization. Per-call workspace avoids
shared mutable state; the GIL is released during native propagation and kicks.

The map is adapted from the user's RPMDrate cyclic-Cayley implementation at
commit `19ba2602bad5b2145b0d0ba0716d7ba3d90a1003` and its thermal-dRPMD successor
`9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78`. Original RPMDrate MIT attribution is
retained in `THIRD_PARTY_NOTICES.md`. The NumPy path evaluates the same Cayley
map in Fourier space and is checked against an independent dense midpoint solve.

BCOCB sampling uses the mathematically specified square-root Cayley half-step
and an OU update in real orthonormal modes. It is currently a NumPy sampling
path; the production full-ring Cayley map has native OpenMP acceleration.

## Potential plugin ABI

`include/pyrpmd/pes_api_v1.h` is the stable v1 physical PES ABI. It accepts host
memory `float64[B,N,3]` in bohr and returns hartree and physical forces in
hartree/bohr. It has separate context lifecycle, version/shape checks, metadata,
global and per-frame statuses, and explicit buffer capacities. The library
must not add ring springs, biases, or divide its forces by the bead count.

The v1 ABI describes nonperiodic fixed-composition single-surface molecular
potentials. It has no MPI communicator/device-pointer/cell parameters. Fortran
wrappers use ISO_C_BINDING, signed int32/int64 and double with BIND(C).

`examples/harmonic_pes.c` is a working ABI example with one Cartesian harmonic
test atom. It is only an interface/numerical reference model, not a chemical
PES. It can be built using:

```bash
cc -std=c11 -fPIC -shared -I native/include native/examples/harmonic_pes.c -o libharmonic.so
```

The tests also compile this example to an object file and exercise the real
manifest-driven `.o` to `.so` linking path. Arbitrary legacy Fortran signatures
are not inferred: supply an explicit wrapper or select the RPMDrate/f2py adapter.

## Fused production blocks

`fused.hpp`, included by `cayley.cpp`, adds `bind_pes` and `run_block` without
replacing the validated Cayley class. The Python public entrypoint is
`Dynamics.propagate`; see [native propagation](../docs/native-propagation.md).
It supports direct C ABI/Fortran calls and native centroid CV/umbrella/flux-metric
observations, with a GIL-free step loop and per-block private workspaces.
