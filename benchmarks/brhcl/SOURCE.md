# BrHCl_LEPS1 source and atomic-unit adapter

This benchmark uses the published **BrHCl_LEPS1** London–Eyring–Polanyi–Sato
(LEPS) potential distributed by ChemPotPy. It is an actual analytic reaction
potential with three diatomic arrangements. It is a historical semiempirical
model, and the benchmark does not establish modern quantitative accuracy for
H + BrCl product branching.

## Exact source

- Repository: <https://github.com/shuyinan/chempotpy>
- Revision: `4716d22e87ae01770cad9ec995c4a02a8e479296`
- File: [`chempotpy/BrHCl/BrHCl_LEPS1.f`](https://github.com/shuyinan/chempotpy/blob/4716d22e87ae01770cad9ec995c4a02a8e479296/chempotpy/BrHCl/BrHCl_LEPS1.f)
- SHA-256: `55df493416823e16ea26309b474353381e36b982aea95dcc4aaf39097ba02f5e`
- Length: 58,274 bytes
- Distribution license: [ChemPotPy repository Apache-2.0](https://github.com/shuyinan/chempotpy/blob/4716d22e87ae01770cad9ec995c4a02a8e479296/LICENSE).

`source.json` contains the machine-readable pin. The builder verifies both byte
count and checksum before compilation. The unchanged third-party source is
downloaded into ignored `build/`; it is not included in the PyRPMD repository
or relicensed under PyRPMD's MIT license. The adapter sources are PyRPMD code.

The source header cites these parameter studies:

1. D. J. Douglas, J. C. Polanyi and J. J. Sloan, *Chemical Physics* **13**, 15–31
   (1976), [doi:10.1016/0301-0104(76)80121-8](https://doi.org/10.1016/0301-0104(76)80121-8),
   for the Morse parameters.
2. J. A. Kaye and A. Kuppermann, *Chemical Physics Letters* **92**, 574–580
   (1982), [doi:10.1016/0009-2614(82)83653-1](https://doi.org/10.1016/0009-2614(82)83653-1),
   for the Sato parameters. This paper concerns collinear Br + HCl in vibrational
   states 2, 3 and 4. Using the same analytic surface for H + BrCl is useful for
   testing multiple product arrangements, but is not an independent validation
   of that entrance channel against modern electronic-structure data.

## Coordinates, units and energy zero

The fixed Cartesian atom order is **Br, H, Cl**, atomic numbers **35, 1, 17**.
The neutral model is represented as one doublet electronic surface. Internal
distances are ordered **Br–H, H–Cl, Br–Cl**. The parameters stored in the source
are:

| Pair | Dissociation energy, kcal/mol | Equilibrium distance, Å | Morse beta, Å⁻¹ | Sato parameter |
|---|---:|---:|---:|---:|
| Br–H | 90.24 | 1.414 | 1.851 | 0.154 |
| H–Cl | 106.40 | 1.275 | 1.867 | 0.154 |
| Br–Cl | 52.09 | 2.136 | 1.923 | 0.154 |

The source's nominal energy zero is separated **Br + HCl** at the HCl equilibrium
distance. Relative to it, HBr + Cl is 16.16 kcal/mol and H + BrCl is 54.31 kcal/mol
in the asymptotic Morse limit. The upstream source retains a rounded internal
constant `sqrt(2) = 1.41421356`; its actual zero differs from the nominal zero by
approximately −2.99 × 10⁻¹⁰ hartree. This tiny source-level rounding is preserved
and explicitly allowed by the build oracle.

The public adapter takes bohr and returns hartree and **forces = −dE/dq** in
hartree/bohr. A frame `[3,3]` or batch `[B,3,3]` is accepted; results always have
energy `[B]`, forces `[B,3,3]`, and per-frame status `[B]`. The auxiliary `radial`
method returns the bare energy and **positive** derivatives `dE/dr`.

The wrapper calls the source's `PREPOT` and `POT` directly in atomic units.
It avoids the ChemPotPy `pes` wrapper's rounded Å/eV interface factors.
The original `627.5095` kcal/mol/hartree and `0.529177106` Å/bohr parameter
conversions are retained, so this is the pinned source model, not a refit.

## Reproducible build and checks

From the repository root, with GNU Fortran, a C++17 compiler and PyRPMD installed:

```bash
python benchmarks/brhcl/build_pes.py
python -m benchmarks.brhcl.pes --check --output runs/brhcl-validation/pes-verification.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_brhcl_pes.py
```

An offline build accepts `--source /path/to/exact/BrHCl_LEPS1.f`. The same pin is
required. The default binary is `benchmarks/brhcl/build/libbrhcl.so` and its
`.build.json` records the source, wrapper and binary hashes, compilers, commands
and three asymptotic energy checks. The loader verifies these hashes before use.
Tests requiring the real library skip when it has not been explicitly built;
pytest never downloads or substitutes a mock model.

Ordinary Fortran `DOUBLE PRECISION` is sufficient. The builder does not inherit
`FFLAGS` or introduce default-real/default-integer promotion, which could change
the legacy COMMON-block ABI. Separate GNU Fortran and C++ compilation preserves
the declared types. A GNU linker export map hides the upstream `POT`, `PREPOT`
and COMMON symbols, exposing only the intended PyRPMD entry points.

Checks compare 40 noncollinear geometries with an independent NumPy evaluation
of the LEPS formula, Cartesian and radial analytic derivatives with central
finite differences, rigid translation and rotation, net force and torque,
batch and scalar calls, bad-geometry handling, and common-wall forces.
Native propagation is tested against the Python propagation loop with four
beads, a two-distance umbrella and an active common wall.

## Common confinement and native execution

`BrHClPotential()` is **bare by default**. A scan explicitly requests:

```python
potential = BrHClPotential(
    wall_angstrom=6.0,
    wall_k_kcal_mol_angstrom2=20.0,
)
```

This adds the same physical wall on all three pairs, on each bead:

\[
V_\mathrm{wall}(q)=\tfrac12 k\sum_{a<b}\max(r_{ab}-R,0)^2.
\]

The wall is evaluated inside the native C ABI context, so native dynamics has
the same energy and forces as Python calls. It is part of the target confined
system and remains in the PMF after umbrella removal. It does not represent
an unconstrained gas-phase equilibrium or a standard-state rate constant.
The wall uses PyRPMD's current atomic-unit conversions; it does not alter the
legacy model parameters. Setting either radius or force constant to zero
disables confinement. `bare_evaluate()` and `radial()` always omit the wall;
`wall_energy_force()` provides an independent NumPy check of the wall alone.

The source uses process-global COMMON blocks. `PREPOT` converts parameters in
place and must run exactly once per loaded library. The binding serializes all
calls with a C++ mutex, and PyRPMD also shares a lock by canonical library path.
Multiple contexts retain independent wall parameters while sharing the safely
initialized bare source. Native bindings retain their context owner and reject
use after closure. Parallel sampling should use separate worker processes.

Pair distances below 0.2 bohr are rejected before entering the source. The LEPS
radical has a singular derivative when all three exchange integrals coincide;
all-separated configurations can also underflow this radical. Nonfinite source
outputs are reported as failed frames with NaN energy/forces, never replaced
by an invented regularization. These guards are not evidence of quantitative
PES validity throughout every finite-distance configuration.
