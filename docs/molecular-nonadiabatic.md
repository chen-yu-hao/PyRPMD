# Nonadiabatic dynamics on a published molecular PES

`NaH2LEPS5G` supplies the published NaH₂ surface set 5G as a complete two-state
diabatic matrix with Cartesian derivatives. This is a fitted molecular PES for
a chemically specified three-atom system. It is available directly in Python
and through `task="nonadiabatic"`, `model.name="nah2_leps_5g_1992"`.

## Chemical system and source

Halvick and Truhlar constructed the coupled lowest two doublet A′ surfaces for
Na(3p)+H₂ → Na(3s)+H₂ electronic quenching and Na(3p)+H₂ → NaH+H reaction.
The representation covers the reaction asymptotes and the three-body crossing
region described in their paper. The 5G implementation retained here is the
historical POTLIB/ChemPotPy surface, including its long-range term and energy
zero shift. It has three internal degrees of freedom and accepts unrestricted
three-dimensional Cartesian geometries; it is not a one-coordinate cut.

- P. Halvick and D. G. Truhlar, *J. Chem. Phys.* **96**, 2895–2909 (1992),
  [doi:10.1063/1.461986](https://doi.org/10.1063/1.461986), with
  [1994 erratum](https://doi.org/10.1063/1.467282).
- Y. Shu, Z. Varga, D. Zhang and D. G. Truhlar, ChemPotPy,
  *J. Phys. Chem. A* **127**, 9635–9640 (2023),
  [doi:10.1021/acs.jpca.3c05899](https://doi.org/10.1021/acs.jpca.3c05899).
- [POTLIB](https://comp.chem.umn.edu/potlib/) and the
  [pinned upstream Fortran](https://github.com/shuyinan/chempotpy/blob/4716d22e87ae01770cad9ec995c4a02a8e479296/chempotpy/NaH2/NaH2_LEPS_5G_1992_DPEM.f).

The source commit is `4716d22e87ae01770cad9ec995c4a02a8e479296`; the Fortran
SHA-256 is `b98e2cb6a7355d868346b2a901441998705f9dead1dd233146285fd618d08f61`.
The upstream Apache-2.0 license is retained with the source and distributed with
the Python implementation. The translation changes the implementation language,
batch evaluation and derivative calculation; it does not refit the PES.

## Python contract and atom order

```python
import numpy as np
from pyrpmd.potentials.multistate_external import NaH2LEPS5G

pes = NaH2LEPS5G(symbols=("Na", "H", "H"))
q = np.array([[[3.2, 0.8, 0.0], [-0.85, 0.0, 0.0], [0.85, 0.0, 0.0]]])
value = pes.evaluate(q)
assert value.matrix.shape == (1, 2, 2)
assert value.gradients.shape == (1, 2, 2, 3, 3)
```

| Quantity | Convention |
| --- | --- |
| Input geometry | Real finite `[batch, atom, xyz]`, bohr |
| Atom order | Any fixed order containing one `Na` and two `H`; declare it in `symbols` |
| Electronic matrix | Real symmetric `[batch, 2, 2]`, hartree, original fixed diabatic convention |
| Derivatives | Positive `dV/dq`, `[batch, 2, 2, 3, 3]`, hartree/bohr, including off-diagonal elements |
| Masses | Supplied separately to dynamics, in electron masses; the PES is mass-independent |
| Batch semantics | Independent frames; concatenating beads or trajectories does not alter an evaluation |
| Electronic provenance | Neutral system, charge 0, spin multiplicity 2 for both represented doublet states |
| Energy convention | Original native POTLIB hartree expression with its 5G common zero shift |

The provider canonicalizes the atom order once and maps all gradients back to
the declared order. It never sorts atoms by distance during propagation.
Identical-hydrogen exchange, translations and rotations obey the molecular
symmetries. Both matrix elements and their gradients retain the original
electronic phase convention; the coupling is not replaced by its magnitude.

Forward differentiation of the fitted energy expression gives its analytic
chain-rule derivative with floating-point arithmetic. This does not use a
finite-difference force step. The native fit contains different historical
eV conversion factors in the diagonal and coupling formulas; these are retained
as part of the published parameterization. Replacing them with a modern common
constant would change the original numerical PES. The unused ChemPotPy
angstrom/eV wrapper is therefore not part of the portable provider's numerical
definition.

## Nuclear and electronic preparation

Use actual Na/H masses and a geometry appropriate to the desired calculation.
The ready-to-run
[NRPMD input](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/molecular_nonadiabatic/nah2_nrpmd.inp)
and [RPSH input](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/molecular_nonadiabatic/nah2_rpsh.inp)
specify isotope atomic masses and an interaction-region phase point with zero
total initial linear momentum. The nuclear ring is initially collapsed.
These examples demonstrate propagation and restart; they do not prepare a
rovibrationally selected asymptotic collision or an equilibrium nuclear ensemble.

For NRPMD, `initial_state=1` selects the fixed diabatic state. For RPSH,
`active_surface=1` selects the upper centroid adiabatic eigenstate. At the
example's coupled geometry these are different electronic states. Population
differences between those example runs cannot be interpreted as a comparison
of the two dynamics approximations. A controlled physical comparison must match
the full initial nuclear and electronic density and the observation basis.

The [nonadiabatic workflow](nonadiabatic-workflow.md) retains independent
trajectory histories, signed MMST observations, separate RPSH amplitudes and
active-state occupations, and state/RNG checkpoint consistency. The complete
molecular identity includes atom order, source commit, source hash and
translation version; restoring a different source under the same model name is
rejected.

## Optional ChemPotPy installation

`ChemPotPyDPEM` bridges the same pinned 5G entry through the installed
ChemPotPy `u(system, surface, labeled_geometry)` interface:

```python
from pyrpmd.potentials.multistate_external import ChemPotPyDPEM

# Requires the matching upstream source and a separately compiled ChemPotPy PES.
pes = ChemPotPyDPEM(symbols=("Na", "H", "H"), difference_step_bohr=1e-4)
result = pes.evaluate(q)
```

ChemPotPy's public geometry and matrix units are angstrom and eV. The bridge
converts them to PyRPMD atomic units with the appropriate coordinate chain rule.
This particular upstream entry supplies energies only: the bridge explicitly
uses five-point Cartesian numerical derivatives, records the step, and requires
its convergence to be checked. The portable `NaH2LEPS5G` implementation above
provides the directly differentiated path. Because the upstream wrapper's
historical conversion constants are rounded, its converted results can differ
slightly from a direct call to the native POTLIB kernel.

The adapter verifies the installed Fortran source hash. A matching source file
does not prove that an independently supplied binary was built from it; use a
controlled build. Calls are serialized across instances because upstream
Fortran routines hold saved state. Every geometry in a numerical-difference
stencil is checked before calling Fortran, including its singularity that would
otherwise invoke `STOP`. Nothing is downloaded or compiled on import.

The complete-matrix contract preserves real or complex Hermitian entries and
their phases. It does not infer a global diabatic basis from independently
phased adiabatic eigenvectors. A scalar ASE energy/force callback, an array of
adiabatic energies, or energies without all required coupling derivatives does
not meet the contract. Other molecular surfaces require their own verified
multistate provider; this adapter does not advertise the entire ChemPotPy
catalog as uniformly usable.

## Validation and physical limits

The [molecular benchmark](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/molecular_nonadiabatic)
contains an executable comparison against the independently compiled original
Fortran, molecular trajectory examples and recorded local results. Unit tests
check Cartesian gradients, atom permutations, rotation/translation covariance,
net forces and torques, the asymptotic excitation gap, source/license hashes,
strict restoration, unit conversion and actual propagation with both methods.

The independently compiled reference compares complete matrices and adiabatic
energies at reproducible molecular geometries, plus Cartesian derivatives from
five-point differences of the original energies. It **passed in GitHub Actions**
on 2026-11-01 using GNU Fortran 15.2.0 and 64 molecular geometries. Maximum
absolute differences were `1.51094e-15` hartree for the complete diabatic matrix,
`1.44329e-15` hartree for the adiabatic eigenvalues, and `3.43297e-12`
hartree/bohr for the Cartesian matrix gradients. The gradient reference used a
five-point step of `2e-4` bohr.

The [completed molecular job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41290725504/job/124801235314)
tested PR head `c53fd1dff83917d4b3137abaebf03135880d9b68` through merge checkout
`a04d478f0b52d938fafe45d427d4906703bc12c6`. The subsequent head `4cc1049` changed
only workflow concurrency and API documentation; the physical implementation
was identical. The [retained result](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/molecular_nonadiabatic/results/original_fortran_ci.json)
contains the measured errors, source hash, compiler flags and complete CI
provenance. This establishes implementation equivalence to the published
kernel, without establishing new accuracy against electronic-structure data.

The physical scope is the historical two-state NaH₂ fit. It omits spin-orbit
coupling, higher electronic states and all other chemical compositions. It is
not an on-the-fly electronic-structure interface or a modern refit. No universal
error bound is asserted for compressed, high-energy or otherwise unexplored
configurations. Noncoincident nuclei and a nonzero Na-to-H₂-midpoint distance
are mathematical requirements; they are not an empirical applicability test.
RPSH also retains its method-specific degeneracy/gap restrictions at crossings.

Before reporting quenching probabilities or reaction observables, specify and
converge initial rovibrational preparation, collision energy, angular momentum,
asymptotic separation, stopping rules, beads, timesteps and trajectory sampling.
A short run with small energy drift establishes numerical operation; the
software continues to report `physical_converged=false`.
