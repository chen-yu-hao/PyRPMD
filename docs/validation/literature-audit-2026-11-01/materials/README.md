# Ni–H EAM: independent engine and derivative audit

This audit compares **ASE 3.29.0 and LAMMPS 20250722** using the same
SHA-256-verified Angelo–Moody–Baskes Ni–Al–H table. Only Ni and H species are
used. It checks energy, force, periodic images, and the physical virial; it does
not report a literature diffusion coefficient.

## Results

[The complete numerical record](cross-engine-eam.json) contains all 12 atomic
configurations, exact model provenance, source hashes, dependency versions,
tolerances, and derivative estimates. The geometry is a 27-Ni-atom
3 × 3 × 3 Ni(100) slab with one H atom, a 3.52 Å lattice constant, periodic
in-plane boundaries and a nonperiodic vacuum direction. All atoms are mobile.
Eleven configurations add seeded displacements to the reference geometry.

| Comparison | Largest absolute error |
| --- | ---: |
| ASE versus LAMMPS total energy | 5.93378 × 10⁻⁷ eV |
| ASE versus LAMMPS Cartesian force component | 2.86044 × 10⁻⁵ eV/Å |
| ASE versus LAMMPS virial component | 3.62910 × 10⁻⁴ eV |
| LAMMPS force versus central energy derivative | 1.06854 × 10⁻⁸ eV/Å |
| LAMMPS virial versus homogeneous strain derivative | 5.02800 × 10⁻⁷ eV |

The force derivative uses six components, two atoms including H, and two
displacements (10⁻⁴ and 5 × 10⁻⁵ Å). The strain derivative checks three normal
strains and one shear at ±2 × 10⁻⁵. The virial is `−stress × volume`; it includes
the periodic many-body contribution instead of substituting a Cartesian
`sum(r*f)`. Every strained partial-periodic slab uses a fresh fixed-cell engine,
respecting the provider's full-periodicity requirement for variable-cell
dynamics. Translating H by whole in-plane lattice vectors preserves forces,
and provider evaluation leaves the supplied unwrapped coordinates unchanged.

All declared engineering checks pass. ASE and LAMMPS use different interpolation
implementations for the same tabulated functions, so exact floating-point
agreement is not assumed. These tolerances do not express the uncertainty of a
physical diffusion coefficient.

## Reproduce

Install the `ase` and `lammps` optional dependencies with a compatible LAMMPS
shared-library/MPI runtime. Obtain and verify the pinned table using
`benchmarks/materials/ni_h_eam.py`; its source URL, byte count, and SHA-256 are
recorded in the numerical record. Then run from the checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/materials/cross_engine_eam.py \
  --model /absolute/path/NiAlH_jea.eam.alloy \
  --output /new/path/cross-engine-eam.json
```

The table is fetched only with explicit opt-in by the existing model helper;
this audit never downloads a potential implicitly and does not redistribute
the table.

## What is still missing for a paper comparison

The reference [Suleimanov, J. Phys. Chem. C 116 (2012),
10.1021/jp302453z](https://doi.org/10.1021/jp302453z) concerns H diffusion on
Ni(100). This audit has **not established that its PES, surface size, lattice
constraints, coverage, and thermodynamic state match that paper**. It therefore
cannot establish either the reported tunnelling crossover or a diffusion
coefficient. The model tested here comes from [Angelo, Moody and Baskes,
Modelling Simul. Mater. Sci. Eng. 3 (1995),
10.1088/0965-0393/3/3/001](https://doi.org/10.1088/0965-0393/3/3/001); table
provenance is separately documented by
[NIST](https://www.ctcms.nist.gov/potentials/entry/1995--Angelo-J-E-Moody-N-R-Baskes-M-I--Ni-Al-H/).

The separate Pd bulk/cluster paper [Calvo and Costa, JCTC 6 (2010),
10.1021/ct900554r](https://doi.org/10.1021/ct900554r) also requires its particular
many-body model, loading, and geometry. A Ni table or an unrelated Pd surface
model cannot replace it. Both literature reproductions remain unvalidated.
