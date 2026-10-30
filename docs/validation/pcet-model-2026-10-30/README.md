# Independent local PCET model comparison

PyRPMD's new local golden-rule PCET calculation was compared with the official
[Hammes-Schiffer group pyPCET implementation](https://github.com/shsgroup/pyPCET/tree/031a679eab50e210aa7f76698ef5f302de31a460),
pinned to **`031a679eab50e210aa7f76698ef5f302de31a460`**. The reference performs its
own Fourier Grid Hamiltonian (FGH) eigensolves, wavefunction integration,
Boltzmann weighting, and vibronic rate calculation. The harness imports that
external checkout; no reference implementation is copied into PyRPMD.

The largest raw difference across 12 forward/reverse rates is **0.06208%**;
the largest difference across six H/D isotope effects is **0.003583%**.
These results validate a fixed-geometry teaching model. They do not reproduce
a BIP, CoTPP, explicit-solvent, or constant-potential electrode calculation.

## Model and conventions

Both implementations receive the same harmonic curves, in eV and angstrom:

\[
V_O(x)=\tfrac12(4.0)(x+0.2)^2,\qquad
V_R(x)=\tfrac12(5.0)(x-0.2)^2.
\]

The domain is \([-2,2]\) angstrom, \(T=300\) K, \(\lambda=0.7\) eV,
\(V_{el}=0.001\) eV, and the constant electrode DOS is
\(1\,\mathrm{eV}^{-1}\). The well-minimum free-energy offset is zero at the
reference potential; its reduction-side shift is \(+e(U-U_{ref})\).
The scan uses \(U-U_{ref}=-0.2,0,+0.2\) V. This specifies potential differences,
without assigning an absolute reference-electrode scale.

H and D use the identical explicit masses on both sides of the comparison:
1837.1526474396187 and 3671.4829415365825 electron masses, respectively.
The reference's default proton mass is therefore not used.

### Reference input mapping

At the pinned revision, `calc_reaction_free_energy_matrix()` uses excitation
energies relative to each surface's vibrational ground state. Consequently,
to match PyRPMD's **well-minimum** driving-force convention, the reference
receives

\[
\Delta G_{forward}=e(U-U_{ref})+E_{R0}-E_{O0}-\epsilon,
\]

and a second reference object with swapped surfaces receives

\[
\Delta G_{reverse}=-e(U-U_{ref})+E_{O0}-E_{R0}+\epsilon.
\]

The harness integrates the reference rates against \(\rho f(\epsilon)\)
and \(\rho[1-f(\epsilon)]\), respectively. Each ground-state correction comes
from the reference's own FGH result and is archived separately for H and D.
This explicit mapping is needed despite the reference constructor's
description of `DeltaG`; it follows the actual pinned implementation.

One compatibility shim is applied in memory:
`simps(y, x)` calls `scipy.integrate.simpson(y, x=x)` because SciPy 1.17
requires a keyword for `x`. This changes call syntax only. No source file,
physical constant, eigensolver, or rate formula in the reference is changed.

## Results

The final comparison uses **1601 finite-difference points / 24 states** in
PyRPMD and **512 FGH points / 24 states** in pyPCET. Rates are local first-order
rates in \(\mathrm{s}^{-1}\), not bimolecular rate constants or current densities.

| Isotope | \(U-U_{ref}\), V | PyRPMD reduction | pyPCET reduction | PyRPMD oxidation | pyPCET oxidation |
|---|---:|---:|---:|---:|---:|
| H | −0.2 | 4,535,487.186 | 4,534,584.405 | 2,665.649 | 2,664.142 |
| H | 0.0 | 143,575.348 | 143,522.589 | 193,246.194 | 193,179.301 |
| H | +0.2 | 1,976.672 | 1,975.496 | 6,092,813.940 | 6,091,677.810 |
| D | −0.2 | 2,947,004.664 | 2,946,330.485 | 1,600.754 | 1,599.793 |
| D | 0.0 | 92,655.926 | 92,619.074 | 115,257.497 | 115,213.473 |
| D | +0.2 | 1,353.692 | 1,352.852 | 3,856,274.080 | 3,855,435.743 |

| \(U-U_{ref}\), V | PyRPMD reduction KIE | pyPCET reduction KIE | PyRPMD oxidation KIE | pyPCET oxidation KIE |
|---|---:|---:|---:|---:|
| −0.2 | 1.53901595 | 1.53906170 | 1.66524528 | 1.66530402 |
| 0.0 | 1.54955386 | 1.54960077 | 1.67664750 | 1.67670756 |
| +0.2 | 1.46020871 | 1.46024577 | 1.57997430 | 1.58002317 |

KIE means \(k_H/k_D\), at the same potential, temperature, and mass-independent
potential curves. All unrounded values are in `reference_results.json`.

## Numerical checks and remaining difference

The figures below are maximum absolute **fractional** changes, not percentages.

| Comparison | Rate change | KIE change |
|---|---:|---:|
| PyRPMD vs official pyPCET, final settings | 6.2080 × 10⁻⁴ | 3.5824 × 10⁻⁵ |
| PyRPMD grid: 801 → 1601, 16 states | 1.2065 × 10⁻⁴ | 1.5459 × 10⁻⁵ |
| PyRPMD states: 16 → 24, 1601 points | 3.5226 × 10⁻¹² | 8.3312 × 10⁻¹³ |
| Reference grid: 256 → 512, 16 states | 3.2808 × 10⁻¹³ | 5.7499 × 10⁻¹³ |
| Reference states: 16 → 24, 512 points | 4.4409 × 10⁻¹⁶ | 3.3307 × 10⁻¹⁶ |

The reference integration's reported absolute error, divided by the smaller
rate in each forward/reverse pair, is at most 2.0 × 10⁻⁸. Adding the energy
tails between 4 and 6 eV on either side changes a final rate by less than
4.8 × 10⁻⁷¹ fractionally. The largest probability in a retained PyRPMD state's
outer 5% boundary bands is 5.94 × 10⁻¹⁰.

The reference deliberately retains rounded constants: its Boltzmann constant
is 8.616901760079889 × 10⁻⁵ eV/K, compared with PyRPMD's
8.617333262145177 × 10⁻⁵ eV/K. Its \(\hbar\), hartree/eV and bohr/angstrom
conversions also differ slightly. Their values are archived. The raw
cross-code difference includes these conventions and finite-difference error;
no constant substitution or fitted correction is applied. The separate
refinements above do not uniquely apportion the remaining difference.

## Reproduce

Install PyRPMD and the **validation-only** `numba` dependency in a suitable
environment, then fetch the external reference checkout and run from the
PyRPMD repository root:

```bash
git clone https://github.com/shsgroup/pyPCET /path/to/pyPCET-reference
git -C /path/to/pyPCET-reference checkout 031a679eab50e210aa7f76698ef5f302de31a460
python -m pip install numba
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python docs/validation/pcet-model-2026-10-30/compare_pypcet.py \
  --reference-checkout /path/to/pyPCET-reference
```

The archived run used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Numba
0.68.1. The JSON records reference commit, source SHA-256 hashes, working-tree
status, environment, inputs, per-isotope ground energies, quadrature estimates,
refinement results and raw rates. The harness rejects a modified reference
checkout or changes to either implementation's source files during a run.
