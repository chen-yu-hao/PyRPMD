# H + H₂ on the BKMP2 H₃ surface

This benchmark uses the published ground electronic state Born–Oppenheimer
potential of A. I. Boothroyd, W. J. Keogh, P. G. Martin and M. R. Peterson,
“A refined H₃ potential energy surface,” *Journal of Chemical Physics* **104**,
7139–7152 (1996), [DOI: 10.1063/1.471430](https://doi.org/10.1063/1.471430).
It is a fitted molecular PES for hydrogen exchange, with analytic derivatives.
It is not the teaching exchange model in `examples/rate`.

The [authors' distribution page](https://www.cita.utoronto.ca/~boothroy/bkmp2.html)
describes the surface, original Fortran and verification program. The PyRPMD
integration follows the same physical surface used in the RPMDrate H + H₂
example. Distances are in bohr, energies in hartree, and the energy zero is three
separated hydrogen atoms. Isotope changes affect nuclear masses, not this
electronic potential.

## Exact downloaded source

`source.json` records the immutable source identity:

- Repository: `chen-yu-hao/RPMDrate`.
- Commit: `19ba2602bad5b2145b0d0ba0716d7ba3d90a1003`.
- File: `examples/H+H2/h3_bkmp2.f`.
- [Pinned source](https://github.com/chen-yu-hao/RPMDrate/blob/19ba2602bad5b2145b0d0ba0716d7ba3d90a1003/examples/H%2BH2/h3_bkmp2.f).
- Bytes: `61792`, preserving upstream CRLF line endings.
- SHA-256: `6fa37803d0d1db4f43e5bc4bfccd5e26585f01116dae9dc4f6d0d28d5d71d237`.

The builder downloads this file only into the ignored `build/` directory.
A size or checksum mismatch is a hard failure, including a server returning an
HTML error page. `--source FILE` permits offline use of the exact same bytes.

No explicit redistribution license was identified in this PES source. It is
therefore fetched from its public upstream location and is **not included in
the repository or relicensed under PyRPMD's MIT license**. The new Python code
and ISO C binding wrapper belong to PyRPMD. Preserve the original scientific
attribution and check upstream terms before redistributing the fetched source
or compiled PES. Benchmark result files and checksums identify what was used.

## Build and use

```sh
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3/pes-verification.json
```

GNU Fortran is required. `FC` or `--fc` selects its executable. Additional
compiler flags can be passed with `FFLAGS` or `--fflags`; no shell is invoked.
The default library is `benchmarks/h3/build/libbkmp2.so`. `--output PATH`
changes this location. The corresponding `.build.json` records the compiler,
flags, upstream source hash, wrapper hash, binary hash and the first independent
numerical check. The factory verifies the recorded binary/source identity.

The legacy RPMDrate copy contains some unsuffixed real constants. The required
GNU flags `-fdefault-real-8 -fdefault-double-8` retain binary64 constants and
binary64 `DOUBLE PRECISION`. This implements the precision concern documented
by the authors for older distributions. Do not remove these flags: the
independent numerical reference checks are specifically intended to detect
such precision or ABI changes. No fast-math option is used by default.

The importable factory is `benchmarks/h3/pes.py:build`, accepting optional
`library=...`. Its batch method consumes `q[B,3,3]` in atomic units and returns
energies and **forces**, with the negative gradient sign. Each process owns its
provider; calls are serialized within that process. MPI/process workers can
evaluate independent bead batches. The upstream routine contains `STOP` on
invalid geometry, so the wrapper rejects pair distances below 0.2 bohr and
invalid triangles before entering it, returning per-frame failure status.

## Independent verification

The numerical oracles below come from the authors'
[test program](https://www.cita.utoronto.ca/~boothroy/data/bkmp2/bkmp2test.f)
and [reference output](https://www.cita.utoronto.ca/~boothroy/data/bkmp2/bkmp2test.out).
They are independent of PyRPMD's finite-difference calculations.

| Distances, bohr | Published reference |
|---|---|
| `(1.757, 1.757, 3.514)` | Energy `-0.159176000` hartree |
| `(1.401, 99, 99)` | Energy `-0.174495730` hartree |
| `(0.9, 1.1, 1.5)` | Radial derivatives `(-0.524470263, -0.271446000, -0.108650526)` hartree/bohr |

The check command additionally optimizes the isolated H₂ bond and symmetric
collinear saddle, differentiates Cartesian energies independently, checks
translation/rotation/atom permutation invariance, compares batch and scalar
evaluation, and verifies that the TS Hessian has exactly one negative mode.

A GNU Fortran 13.3 build gave H₂ equilibrium distance `1.4014719755` bohr,
saddle adjacent distances `1.7570000251` bohr, and an electronic barrier
`0.0153197707762` hartree (`9.6133013026` kcal/mol) relative to the optimized
H + H₂ asymptote. This barrier differs slightly from the authors' nominal test
barrier because their H₂ reference uses the rounded bond length 1.401 bohr.
The generated JSON contains the actual results, tolerances and machine details.

These checks verify the electronic surface and adapter. They do not establish
convergence of RPMD with bead count, time step, sampling time, separation,
impact cutoff, umbrella overlap or recrossing plateau.
