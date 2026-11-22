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

## Optional ChemPotPy DPEMs

`ChemPotPyDPEM` bridges the diabatic potential energy matrices of
[ChemPotPy](https://github.com/shuyinan/chempotpy) (its `multi_state_list`: for example
`OH3_PIP_FFW1_2019_DPEM` and `OH3_PIP_FFW2_2022_DPEM` for OH + H$_2$,
`NH3_GEN_FFW1_2006_DPEM` and `NH3_GEN_FFW2_2007_DPEM` for NH$_3$, `PHOH_APRP_DPEM`,
`C6H5SH_APRP_DPEM`, `CH3NH2_APRP_DPEM`, the LiFH, NaFH, NaH$_2$, BrH$_2$ and O$_3$ entries, and
the MCH/MXH/YRH model surfaces) through the upstream `u` and `ug` calls:

```python
from pyrpmd.potentials.multistate_external import ChemPotPyDPEM

pes = ChemPotPyDPEM(system="OH3", surface="OH3_PIP_FFW1_2019_DPEM",
                    symbols=["O", "H", "H", "H"])   # caller's atom order
result = pes.evaluate(q)        # q[B, 4, 3] bohr -> matrix [B,3,3] hartree, gradients [B,3,3,4,3]
matrices = pes.matrices(q)      # matrices only, one upstream call per frame
pes.metadata["gradient_mode"]   # "analytic" or "difference"
```

ChemPotPy works in angstrom and eV (gradients in eV/angstrom). The bridge converts to bohr
and hartree with the coordinate chain rule, and `nstates` is read from the Fortran source.
`gradient` selects how the derivatives are obtained:

| `gradient` | Derivatives |
| --- | --- |
| `"auto"` (default) | upstream analytic `ug` if a one-time check passes, otherwise five-point differences of `u`; the outcome is recorded in `metadata["gradient_check"]` |
| `"analytic"` | upstream `ug`, checked once; a failed check raises |
| `"difference"` | five-point Cartesian differences of `u` with `difference_step_bohr` (12N+1 calls per frame) |

The check matters for two reasons. First, some entries are energy-only: called through `ug`,
such an entry prints `Only DPEM is available` and returns an **all-zero matrix and gradient**
without raising. Second, the coupling gradients of some entries do not match their own matrix.
The check compares `ug` with `u` and with a five-point difference at the first evaluated frame
and at three deterministic random displacements of it (`check_points`, `check_displacement_bohr`).
A symmetric frame, where a coupling and its gradient error may both vanish, therefore cannot
pass on its own. In ChemPotPy 1.0.7, compiled here for Python 3.13:

| Entry | `ug` versus differences of `u` | `auto` uses |
| --- | --- | --- |
| `NH3_GEN_FFW1_2006_DPEM` | agrees within $10^{-7}$ hartree/bohr | analytic |
| `OH3_PIP_FFW2_2022_DPEM` | agrees within about $10^{-5}$ hartree/bohr | analytic |
| `OH3_PIP_FFW1_2019_DPEM` | the gradients of couplings (0,1) and (1,2) are wrong at nonplanar geometries (up to several hartree/bohr; planar frames can pass) | difference |
| `NaH2_LEPS_5G_1992_DPEM` | energy-only (`ug` returns zeros) | difference |

ChemPotPy returns total energies, for example about $-76.8$ hartree for OH$_3$. Five-point
differences of such energies with `difference_step_bohr = 1e-4` leave round-off of order
$10^{-9}$ hartree/bohr in the forces. A Hessian built by differencing those forces can then
show spurious negative rigid-body eigenvalues. For these polynomial surfaces, a step of
`1e-3` bohr and a transition-state `hessian_step` of `1e-3` bohr keep that noise below the
saddle-index tolerance without changing the physical frequencies.

The same DPEM is available from input tables. Symbols default to the atom order of the system:

```toml
[potential]                       # ground-state rate on the lowest adiabat
provider = "diabatic_adiabat"
diabatic = { provider = "chempotpy", system = "OH3", surface = "OH3_PIP_FFW1_2019_DPEM" }

[electronic]                      # subspace-correct: eta_NA from the same matrix
source = "diabatic"
potential = { provider = "chempotpy", system = "OH3", surface = "OH3_PIP_FFW1_2019_DPEM" }
nstates = 3
```

Along the subspace path only `matrices()` is called.

**Fitted domain.** Check that a surface was fitted for the configurations you need. The OH$_3$
`FFW1` and `FFW2` DPEMs were fitted for reactive quenching of OH(A$^2\Sigma^+$) by H$_2$. Their
asymptotes are sensible: the OH + H$_2$ van der Waals well is about $-0.6$ kcal/mol, and
H$_2$O + H lies $-15.0$ (FFW1) or $-15.8$ (FFW2) kcal/mol below OH + H$_2$. On the lowest adiabat,
however, steepest descent from the ground-state saddle ends in spurious regions:

| Surface | Saddle | Descent towards reactants | Descent towards products |
| --- | --- | --- | --- |
| FFW1 | 7.2 kcal/mol, 3900i cm$^{-1}$ | $-22$ kcal/mol | collapses (below $-10^3$ kcal/mol) |
| FFW2 | 5.1 kcal/mol, 4637i cm$^{-1}$ | well at $-40$ kcal/mol | well at $-31$ kcal/mol |

These regions lie within about 1 angstrom of the saddle. Neither surface is therefore suitable
for thermal ground-state OH + H$_2$ $\to$ H$_2$O + H rates. Note also that the analytic-gradient
check is local: FFW1 can pass it near its saddle and fail elsewhere, so prefer
`gradient = "difference"` for an entry known to fail at generic geometries.

**Installation.** PyPI ChemPotPy ships extensions compiled for Python 3.11 only. For another
Python, compile the entries you use with f2py, linking LAPACK, inside the installed package:

```bash
cd "$(python -c 'import chempotpy, os; print(os.path.dirname(chempotpy.__file__))')/OH3"
python -m numpy.f2py -c OH3_PIP_FFW1_2019_DPEM.f90 -m OH3_PIP_FFW1_2019_DPEM -llapack
```

Nothing is downloaded or compiled by PyRPMD. The Fortran source of each entry is hashed into
`metadata["source_identity"]`. The default `NaH2_LEPS_5G_1992_DPEM` entry must match its
pinned hash, which is the same in ChemPotPy 1.0.7, and any entry can be pinned with
`source_sha256`. A matching source does not authenticate an independently supplied binary.
Calls are serialized across instances because upstream Fortran routines hold saved state.
Upstream routines print banners to standard output. An invalid geometry can make their
Fortran `STOP` the interpreter; only the NaH$_2$ 5G entry is screened for its known singularity
before every call, so keep other surfaces inside their fitted domain. Cite ChemPotPy (Y. Shu,
Z. Varga, D. G. Truhlar) and the original paper of each surface.

The complete-matrix contract preserves real or complex Hermitian entries and their phases.
It does not infer a global diabatic basis from independently phased adiabatic eigenvectors.
A scalar ASE energy/force callback, an array of adiabatic energies, or energies without all
required coupling derivatives does not meet the contract, and the single-state ChemPotPy
entries are refused. The portable `NaH2LEPS5G` implementation above provides the directly
differentiated NaH$_2$ path; because the upstream wrapper's historical conversion constants are
rounded, its converted results differ from it by about $10^{-8}$ hartree.

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
