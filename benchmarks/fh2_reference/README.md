# SW1 and HSW: unchanged author-source static checks

This directory checks the two distinct distributions linked by the
[Werner group](https://www.itheoc.uni-stuttgart.de/research/werner/research/h2f/potential/):
SW1 without spin–orbit correction and HSW with that correction. It builds
and evaluates the original Fortran source in isolated processes. It does not
register a production PyRPMD potential, calculate a thermal rate, supply
analytic HSW forces or identify which variant belongs to a still-unretrieved
corrected literature table.

The exact source and parameter identities are in [sources.json](sources.json).
The original archive readmes request attribution; no explicit redistribution
license file was present in either archive. Source, parameter files, compiled
libraries and the original verification executable stay outside this
repository. The associated publications named by the original source are
Stark and Werner, JCP 104, 6515 (1996),
[doi:10.1063/1.471372](https://doi.org/10.1063/1.471372), and Hartke and Werner,
Chem. Phys. Lett. 280, 430 (1997), for the spin–orbit correction.

## Reproduce the bounded static audit

Obtain the two exact author archives, `sw1.tar` and `hsw.tar`, under the
terms supplied by their authors. Supply a GNU Fortran compiler and a new
external output directory:

```sh
python benchmarks/fh2_reference/run_static_check.py \
  --archives-dir PATH_TO_AUTHOR_ARCHIVES \
  --compiler PATH_TO_GFORTRAN \
  --output NEW_EXTERNAL_STATIC_OUTPUT
```

The runner checks every archive member and SHA256 before materializing a
flat, regular-file-only model directory. It compiles unchanged source with
the flags frozen in [static_protocol.json](static_protocol.json).
`-fno-automatic` retains the legacy `PREPOT`/`POTE` ENTRY parameter state
across calls. Default Fortran literal precision is preserved; no parameter,
constant, energy shift or unit factor is edited.

Each model runs in its own single-threaded process and fixed parameter-file
directory. This avoids sharing the original COMMON blocks, ENTRY state or
relative file initialization between models. The tested interfaces are the
documented original GNU Fortran symbols and COMMON storage, not a portable
general-purpose provider ABI.

The fixed ten geometries contain both author verification points, additional
bent triangles and two distant-channel diagnostics. All coordinates are the
three internuclear distances `(F–H1, H1–H2, F–H2)` in bohr. The audit checks:

- SW1's independent energy and energy/derivative routines.
- Its three analytic distance derivatives against centered differences at
  all three fixed spacings, `1e-4`, `1e-5` and `1e-6` bohr.
- Hydrogen exchange in energies and the SW1 derivative components.
- The common SW1 component in the two separate distributions.
- Both HSW author decimal targets and the unchanged `verify.f` executable.
- Unchanged source, parameter and compiled-library identities.

These are finite static checks. They do not establish global PES accuracy,
Cartesian force integration, RPMD convergence or the electronic population
factor used by a thermal-rate reference. HSW's spline implementation has
its original domain projections and branch conditions; no derivative
continuity claim is made for them.

## Recorded result and legacy precision

All eight fixed implementation gates pass in the
[2026-11-01 static record](../../docs/validation/literature-convergence-2026-11-01/other-reactions/fh2-author-static/README.md).
The largest SW1 finite-difference discrepancy is 4.67 × 10⁻⁹ hartree/bohr.
The two independent SW1 energy routines differ by at most 2.78 × 10⁻¹⁷
hartree; the SW1 component is identical between archives at all ten points.

The HSW calculation differs from each author-written decimal target by
2.849847 × 10⁻¹¹ hartree. This equals the rounding of the original default-real
`0.0006` energy offset when promoted to double precision, within rounding
of the final sum. The original `verify.f` target assignments also use
default-real literals. Both its actual printed targets and the written
decimal targets are retained. The predeclared static tolerance is
5 × 10⁻¹⁰ hartree; the source was not changed to erase these differences.

The full build and check consumed 1.96 child CPU seconds and 2.33 seconds
of wall time. Every child has one CPU affinity and a 30-second CPU/wall
limit; the parent admits no new child after 90 seconds. There are no
trajectory steps, statistical samples or converted rate constants.
