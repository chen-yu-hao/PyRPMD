# Published NaH₂ multistate molecular PES

This example uses the full-dimensional NaH₂ 5G diabatic matrix of Halvick and
Truhlar, not a Tully or generic vibronic toy potential. The two electronic states
describe the lowest two doublet A′ surfaces relevant to Na(3p)+H₂ electronic
quenching and reaction to NaH+H. All three internal distances enter the fit;
the provider accepts all nine Cartesian components and returns the complete
2×2 matrix and every positive Cartesian matrix gradient.

The exact upstream source, its license, immutable commit and hashes are in
[vendor/chempotpy/NOTICE.md](vendor/chempotpy/NOTICE.md). The portable NumPy
implementation preserves the original fit and obtains its derivatives by
forward chain-rule differentiation. No compiler or external PES installation
is needed for normal use.

## Run the molecular API example

From an installed checkout, or with `PYTHONPATH=src`:

```bash
python benchmarks/molecular_nonadiabatic/run_molecular.py \
  --method both --steps 100 --trajectories 8 --beads 4 \
  --output results/molecular_nonadiabatic
```

This writes JSON provenance and diagnostics plus integrity-protected NPZ
trajectory observations. The recorded local executions are
[results/nrpmd.json](results/nrpmd.json) and [results/rpsh.json](results/rpsh.json).
They establish that the real molecular matrix and nuclear forces run through
the two propagators; they do not establish physical convergence.

The selected initial geometry lies in the interaction region. Nuclear positions
and momenta are a specified collapsed-ring phase point, with real Na/H atomic
masses and zero total initial linear momentum. It is not a thermal bath or an
asymptotic rovibrational collision ensemble. The electronic initial conditions
also differ: fixed diabatic state 1 for NRPMD, centroid adiabatic state 1 for
RPSH. Consequently, their populations are not a controlled method comparison.

## Use the restartable task and CLI

```bash
pyrpmd validate benchmarks/molecular_nonadiabatic/nah2_nrpmd.inp --check-pes
pyrpmd run benchmarks/molecular_nonadiabatic/nah2_nrpmd.inp --output runs/nah2-nrpmd
pyrpmd resume runs/nah2-nrpmd
pyrpmd analyze runs/nah2-nrpmd --recompute

pyrpmd run benchmarks/molecular_nonadiabatic/nah2_rpsh.inp --output runs/nah2-rpsh
```

The direct API script and the task have separately specified random stream
layouts. Their equal integer seeds do not promise identical finite-sample
populations. Each route records its own initialization and reproducibility
contract.

## Compare to the unmodified original Fortran kernel

Install a Fortran compiler, then run:

```bash
python benchmarks/molecular_nonadiabatic/compare_fortran.py \
  --output results/nah2_fortran_comparison.json
```

The script authenticates the original 15,875-byte source. It excludes the
unused ChemPotPy angstrom/eV and LAPACK wrapper, compiles the original
`PREPOT`/`POT` routines without editing their equations or constants, and calls
them directly using ctypes. No f2py, ChemPotPy installation, BLAS or LAPACK is
needed. The explicit fixed-form 72-column setting respects the source's
historical sequence-number columns.

For 64 reproducible rotated and translated molecular geometries, it compares
all matrix elements and adiabatic eigenvalues to the Python port. It also
compares the port's derivatives to five-point differences of the independently
compiled original matrix. Thresholds are 3e-12 hartree for energies and 3e-8
hartree/bohr for gradients. Failure exits nonzero; the JSON records the actual
compiler, source pin, errors and thresholds. This is an implementation
equivalence test, not a validation of the fit against new electronic-structure
data or of RPMD's approximation.

See [VALIDATION.md](VALIDATION.md) for executed checks and limits, and the
[molecular provider guide](../../docs/molecular-nonadiabatic.md) for units,
electronic phases and external ChemPotPy use.
