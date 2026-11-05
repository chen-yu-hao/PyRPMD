# H + CH₄: pinned CBE/PES-2008 adapter

This directory makes the **electronic potential** in the published RPMDrate
example reproducible. It does not contain a thermal-rate reproduction or a
converged classical/RPMD ensemble.

The model is Corchado, Bravo and Espinosa-García,
[JCP 130, 184314 (2009), DOI 10.1063/1.3132223](https://doi.org/10.1063/1.3132223),
usually called CBE or PES-2008. It is reference 49 of the
[RPMDrate CPC paper](https://doi.org/10.1016/j.cpc.2012.10.017).
`source.json` pins the official GreenGroup/RPMDrate commit and the byte length,
SHA-256 and Git blob SHA-1 of each relevant source/interface file. No external
PES source or binary is included in this repository.

## Build and check

Run from the PyRPMD repository root with its Python dependencies and GNU Fortran
available. Use a build destination outside the repository for external assets:

```bash
python -m benchmarks.hch4.build_pes --output /path/to/hch4/libcbe2008.so
python -m benchmarks.hch4.verify \
  --library /path/to/hch4/libcbe2008.so \
  --output /path/to/hch4/verification.json
```

The builder fetches only the two byte-pinned `.f` compile inputs. For an offline
build, add `--source-dir /path/to/exact/sources`; `--fc /path/to/gfortran` selects
the compiler. `utility.f` contains sequence labels after column 72, so standard
fixed-form 72-column parsing is required. The builder retains the Fortran files
unchanged and writes compiler flags, source hashes, compiler version, wrapper
hash and library hash to `libcbe2008.build.json`. A build failure writes a
separate failure receipt.

```python
from benchmarks.hch4.pes import CBE2008Potential

potential = CBE2008Potential("/path/to/hch4/libcbe2008.so")
result = potential.evaluate(q)  # q has shape [batch, 6, 3], in bohr
```

The atom order is **H, C, H, H, H, H**. The last atom is the incoming H; the
other four H atoms are equivalent methane hydrogens. The adapter returns the
unmodified upstream energy and **forces = −dV/dq**, with no bead scaling.
The batch entry point admits all frames in one pass and passes the whole
ring to upstream `get_potential(q, 6, P, ...)` in a single call, writing
energies and gradients directly into the output arrays; only a batch that
contains rejected frames is compacted first. Its results are bit-identical to
per-frame calls. Its canonical-path lock serializes the legacy `COMMON` blocks. Initialization
runs once per loaded library: calling upstream `PREPOT` twice would rescale
parameters twice. Separate Python objects therefore share the initialized
library without reinitializing it. A 0.2 bohr pair-distance rejection prevents
obvious numerical singularities; it is not a validated physical-domain bound.

## Known source conventions

The upstream code converts its internal energy unit, **10⁵ J/mol**, using
`0.03812`; this is not an eV conversion. The modern conversion is approximately
`0.03808798847`, making the legacy scale 0.0840463% larger. Internal coordinates
use `0.52918` Å/bohr. The gradient uses a separately rounded `0.0201723`, whereas
the product of the energy and coordinate factors is `0.0201723416`. Thus the
reported gradient is approximately 0.9999979377703975 times the exact derivative
of the reported energy. **Neither coefficient is changed here.**

The generic upstream console text also contains stale 1999 reference and
zero-reactant-energy wording. The file header explicitly identifies the 2009
PES, and the returned energy is unshifted (`ANUZERO=0`); the separated reactants
do not have zero numerical energy. Model identity is checked against the
published barrier, using energy differences, rather than that generic text.

The verification uses the official RPMDrate input geometry, a saddle and
tetrahedral methane reconstructed from the primary potential paper's Table III,
and 16 fixed random perturbations. Before execution the checks specify a
0.03 kcal/mol barrier tolerance and a force tolerance of
`3e-8 + 5e-6 * abs(finite_difference_force)` in atomic units. The barrier
tolerance allows rounded published coordinates/energies and the unchanged
legacy constants. The force tolerance allows the known upstream rounding;
the residual after the analytically known scale is also reported, without
using that scaled force for simulation.

The saved [validation report](../../docs/validation/literature-convergence-2026-11-01/other-reactions/hch4/README.md)
records the actual Linux build, 19 geometries, four finite-difference step sizes,
symmetries, repeated-instance initialization, invalid-frame isolation and all
replay coordinates/energies/forces. Optional real-source tests run as:

```bash
PYRPMD_CBE2008_LIBRARY=/path/to/hch4/libcbe2008.so \
  python -m pytest tests/test_hch4_pes.py -q
```

The upstream RPMDrate README declares an MIT license; the legacy PES files
contain scientific references without an additional file-specific license
statement. This wrapper records those statements and preserves the sources
externally. It does not assert a new license over third-party PES files.
