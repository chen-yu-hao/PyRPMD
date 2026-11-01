# CBE/PES-2008: actual build and electronic validation

**Status: electronic model and adapter checked; thermal rate not run.**
This report is restricted to the exact upstream source, public geometries and
force/interface checks. It establishes a usable potential candidate for the
requested H + CH₄ benchmark; it does not establish RPMD sampling convergence.

## What was built

The official GreenGroup/RPMDrate source is pinned to commit
`8c1828d2302aacf09a15faf5c7fc50a18d4c113e`.
The [source manifest](../../../../../benchmarks/hch4/source.json) contains exact
byte lengths, SHA-256 and Git blob SHA-1 values. Files were read through the
GitHub connector as base64 and their Git blob hashes independently verified
from the decoded bytes. The two compiled `.f` files remain outside the Git
repository and are unmodified. No PES source or shared library is redistributed
in this evidence directory.

The actual compiler was GNU Fortran 16.2.0 from an isolated conda-forge
environment. The successful flags were `-O2 -fPIC -shared -std=legacy
-ffixed-line-length-72`. The first attempted unlimited fixed-line setting
failed because `utility.f` has historical sequence labels after column 72;
[initial-failure.json](initial-failure.json) preserves the compiler error and
command from one exact repeat used to expose stderr. Standard 72-column
parsing fixed the build without changing upstream bytes. The two subsequent
builds produced the same binary SHA-256:
`aa33e4f55246d5b99839cd3b08702f1d091a6b3dddac1d6c97d34bb542d95aaf`.

The [build receipt](build.json) records all inputs, commands and hashes;
[verification.json](verification.json) records replay coordinates, energies,
forces, exact current source hashes and every gate. The observed host was Linux,
Python 3.12.14 and NumPy 2.5.4. The verification itself took under one second.
No portability claim is made from that one environment.

## Independent paper check and numerical results

The primary potential paper is Corchado, Bravo and Espinosa-García,
[DOI 10.1063/1.3132223](https://doi.org/10.1063/1.3132223).
Its [author-uploaded text](https://www.researchgate.net/publication/24434786_The_hydrogen_abstraction_reaction_OP-3_CH4_A_new_analytical_potential_energy_surface_based_on_fit_to_ab_initio_calculations)
has the correct H + CH₄ title and DOI in the paper; the host's landing-page title
incorrectly says O(³P) + CH₄. The PES column of Table III supplies the rounded
saddle and methane geometries, and the 15.01 kcal/mol barrier.

| Check | Actual result | Interpretation |
|---|---:|---|
| Reconstructed Table III electronic barrier | 15.0142843084 kcal/mol | +0.0042843 from 15.01; passes the predeclared 0.03 tolerance |
| CH₄ + H reference, 50 to 80 bohr | 0 observed energy change | Asymptotic check for this small oracle |
| Largest raw force/central-difference difference, ε = 10⁻⁵ bohr | 1.01669 × 10⁻⁷ Ha/bohr | Known upstream gradient rounding retained |
| Largest residual after analytically accounting for that constant ratio | 3.38172 × 10⁻¹¹ Ha/bohr | Diagnostic only; no scaled force used |
| Translation/rotation force error | 8.77770 × 10⁻¹⁶ Ha/bohr | Covariance passes |
| Equivalent methane-H permutation force error | 5.20417 × 10⁻¹⁷ Ha/bohr | Four abstraction labels agree |
| Batch versus repeated scalar evaluation | Exact equality | No bead or batch scaling |
| New object loading the same library | Exact equality | Destructive initialization is not repeated |

The 19 geometries comprise the official RPMDrate input, the independently
reconstructed saddle and asymptote, and 16 fixed perturbations selected before
evaluation. Four finite-difference steps (10⁻⁴, 3 × 10⁻⁵, 10⁻⁵, 3 × 10⁻⁶ bohr)
all satisfy the predeclared raw-force tolerance. Full data, force and torque
conservation errors, and invalid-frame isolation are retained in the JSON.
The [four focused tests](pytest.xml) passed, including same-size checksum
tampering, reconstruction of paper bonds/angles, real-source finite differences
and rejection of a false library receipt.

## Constants are preserved and disclosed

The source's internal energy unit is **10⁵ J/mol**, confirmed by its comments
and the `.041840` conversion from kcal/mol. Its `.03812` conversion into the
upstream Hartree scale is **0.0840463% larger** than the modern CODATA conversion
`.03808798847134393`. Its internal coordinate factor is `.52918` Å/bohr.
The energy/coordinate product `.0201723416` differs from the separately rounded
gradient factor `.0201723`; the latter produces a **−2.0622296 ppm** gradient
scale discrepancy. The finite differences measure precisely that effect.

Neither coefficient has been changed. This is an *as-distributed* historical
PES adapter, with the paper comparison at its stated precision. A future
modern-constant variant must have its own model identity, reference values and
validation; silently changing constants would break historical reproducibility.
The generic POTLIB printed zero-energy/1999 text is stale for this file: the
actual separated-reactant energy is −0.7098507306 in the upstream Hartree scale.
Only differences are used for the barrier.

## Reproduction

From the repository root, with the exact source pair in the external asset
directory and the compiler path adjusted to the local machine:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.hch4.build_pes \
  --source-dir /path/to/exact/cbe-sources \
  --output /path/to/hch4/libcbe2008.so --fc /path/to/gfortran
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.hch4.verify \
  --library /path/to/hch4/libcbe2008.so --output verification.json
PYRPMD_CBE2008_LIBRARY=/path/to/hch4/libcbe2008.so \
  python -m pytest tests/test_hch4_pes.py -q
```

Omitting `--source-dir` fetches only checksum-pinned sources. The original
source/reference URLs and usage statements are in the manifest. A full rate
comparison still requires canonical six-atom ensembles, product-channel
accounting, independently converged PMFs and transmission dynamics, along with
bead, timestep, time-horizon and independent-seed checks.
