# RPSH-BA, RPSH-DM, DBOC and complex surface hopping

`pyrpmd.advanced_surface_hopping` adds explicitly selected variants alongside the
existing real centroid driver. All formulas below use atomic units, including
`hbar = 1`. All beads occupy **one shared active adiabatic surface**. These are
approximate dynamical methods; implementation tests establish numerical and
representation consistency, not agreement with exact quantum dynamics for an
arbitrary molecular system.

## API and provider contract

```python
from pyrpmd.advanced_surface_hopping import AdvancedRPSHDynamics

dynamics = AdvancedRPSHDynamics(
    potential, masses, beta, nbeads, dt, nstates,
    method="dm",                   # "ca", "ba", or "dm"
    electronic_substeps=8,
    rescaling="bead",             # or "centroid"; both conserve ring energy
    frustrated_hops="keep",       # or "reverse"
    dboc=False,
    dboc_step=1e-4,                # bohr; used only when dboc=True
    geometric_force=True,
    seed=1729, native=False,
)
state = dynamics.initialize(nuclear_state, active_surface=0)
dynamics.step(state)
rho_diabatic = dynamics.density_matrix(state)
adiabatic_population = dynamics.populations(state)
energy = dynamics.energy(state)
```

`RPSHBADynamics`, `RPSHDMDynamics`, and `ComplexRPSHDynamics` are convenience
constructors for `method="ba"`, `"dm"`, and `"ca"`, respectively. Complex matrices
are accepted by CA and DM. BA requires real symmetric matrices and gradients.
The original `RPSHDynamics` remains available with its previous conventions.

The provider must return `DiabaticResult(matrix, gradients)` from `evaluate(q)`:

| Quantity | Shape | Convention |
| --- | --- | --- |
| `q` | `[P,N,3]` | Cartesian positions in bohr |
| `matrix` | `[P,S,S]` | Hermitian potential matrices, hartree |
| `gradients` | `[P,S,S,N,3]` | Positive Cartesian derivatives, hartree/bohr |

Each bead's matrix must depend only on that bead's Cartesian geometry. The
electronic basis must be a **position-independent orthonormal diabatic basis**.
A provider in a moving orbital basis must first include the required basis
connection in a suitable adapter; returning only the derivative of matrix
elements in that moving basis does not meet this contract. For position-dependent
SOC, its derivatives must be included in `gradients`.

The implementation diagonalizes the Hermitian matrix as `V = U E U†` and uses

\[
d_{ij,I} = \frac{(U^\dagger\partial_I V U)_{ij}}{E_j-E_i},\qquad i\ne j.
\]

Off-diagonal couplings use the **bare electronic gaps**, including when DBOC is
enabled. Labels follow the ordered bare eigenvalues. Degeneracies and gaps at or
below `gap_tolerance` are rejected; no gap floor silently changes the equations.

## Electronic states, observables and restarts

| Method | State class | Stored amplitudes | Extra restart data |
| --- | --- | --- | --- |
| CA | `RPSHState` | `[S]`, fixed diabatic basis | None |
| BA | `RPSHBAState` | `[S]`, common adiabatic state labels | `basis_vectors[P,S,S]` |
| DM | `RPSHDMState` | `[P,S]`, fixed diabatic basis | None |

Each state also stores `q`, `p`, `step`, `time`, and scalar `active_surface`, and
implements detached `copy()`, `to_dict()`, and `from_dict()`. A stochastic restart
must additionally retain `dynamics.rng.bit_generator.state`. The BA frames cannot
be discarded or reconstructed with unrelated eigenvector signs on restart.

`initialize(..., adiabatic_coefficients=c)` accepts a normalized `[S]` vector.
For DM it also accepts normalized `[P,S]` vectors; a supplied `[S]` is interpreted
as the same **local bead adiabatic amplitudes**, not as one identical fixed
diabatic wavefunction. A fixed-diabatic preparation can instead be supplied by
constructing `RPSHDMState` directly. Initialization fixes each centroid eigenvector
phase by making its largest component positive real and aligns bead frames with
that reference. For a coherent superposition, the input coefficients refer to
this stated gauge. This initialization requires nonzero overlaps with the
centroid reference.

`density_matrix(state)` always returns an electronic amplitude density in the
**fixed diabatic basis**. DM returns the average of bead wavefunction outer
products. BA maps the common adiabatic vector through each bead frame and
averages those outer products. CA returns one pure-state outer product.
`populations(state)` returns centroid adiabatic populations for CA/DM and the
common adiabatic coefficient populations for BA. These are distinct from the
active-surface occupation counts and from a mixed quantum/classical estimator
that substitutes active populations on a density matrix's diagonal.

This distinction matters when reproducing the donor populations in
[Limbu et al. (2025)](https://doi.org/10.1063/5.0301249). Their mixed estimator
uses active-state indicators for the diagonal adiabatic terms and electronic
amplitude coherences for the off-diagonal terms before transformation to the
donor basis. Neither `density_matrix(state)` nor an active-surface fraction alone
is that observable. A reproduction must construct the paper's estimator with
its specified basis and bead averaging, as well as its Debye-bath parameters
and initial distribution. See the [reproduction plan](manual/reproduction.md).

## The bead approximation

The BA used here is Eq. (17) of Shakya et al. and Eq. (4) of Limbu et al.:

\[
i\dot c = \left[\operatorname{diag}\!\left(\frac1P\sum_b E_b\right)
 - i\frac1P\sum_b v_b\cdot d_b\right]c.
\]

It propagates one electronic vector with bead-averaged **adiabatic energies and
velocity-contracted derivative couplings**. It neither diagonalizes a
bead-averaged diabatic Hamiltonian nor assigns an independent active state to each
bead. The latter interpretation is explicitly excluded by Section II.2 of the
2025 paper, even though its abstract describes transitions at the bead level.

Real adiabatic frames are initially aligned to the centroid and subsequently
transported by overlap with the preceding frame. This preserves relative signs
when averaging NACs. A vanishing transport overlap fails with an explicit error;
it requires refining the path/time discretization or choosing a different
method. Complex BA would require an additional common gauge prescription and is
not inferred from the real BA equations.

## The density-matrix construction

RPSH-DM follows the defining construction of Shakya et al., Eqs. (18)–(28): each
bead has its own electronic wavefunction, while hopping is determined by their
density averaged in a common centroid adiabatic basis. In the fixed diabatic
representation used by the code,

\[
i\dot z_b=V(R_b)z_b,\qquad
\tilde\rho_b=U_c^\dagger z_bz_b^\dagger U_c,\qquad
\rho=\frac1P\sum_b\tilde\rho_b.
\]

The generator in that moving common basis is

\[
\tilde H_b=U_c^\dagger V(R_b)U_c-i\dot R_c\cdot d(R_c).
\]

This is an algebraic simplification of the paper's bead-to-centroid basis
transformation, retaining its centroid basis-motion term. Omitting either the
off-diagonal part of `Uc† V(Rb) Uc` or the centroid connection gives an incorrect
current. The probability of hopping from active state `a` to `j` is calculated
from the integrated outgoing current,

\[
g_{a\to j}=\max\!\left(0,
 \frac{\int_t^{t+\Delta t}\frac{2}{P}\sum_b
 \operatorname{Im}[\tilde\rho_{b,aj}\tilde H_{b,ja}]\,dt'}{\rho_{aa}(t)}\right).
\]

For `P=1` this reduces to the usual FSSH current. Different bead phases can
reduce the purity of the ring's electronic density even while every bead
wavefunction remains normalized. This is the specified RPSH-DM mechanism;
arbitrary density-matrix damping is not used. Finite-bead recoherence is possible.
DM requires `decoherence="none"`. Optional `decoherence="edc"` on CA/BA uses
centroid gaps and centroid kinetic energy, after propagation and hopping, with
the same amplitude-damping convention as the existing CA driver.

## Nuclear motion and hop energy accounting

The propagation energy is the **sum-bead** Hamiltonian,

\[
H_P=\sum_b\left[\sum_I\frac{p_{bI}^2}{2M_I}
 +\frac12\sum_I M_I\omega_P^2(R_{bI}-R_{b+1,I})^2
 +\mathcal E_a(R_b)\right],\qquad\omega_P=P/\beta.
\]

Here `mathcal E = E` normally and `mathcal E = E + D` with DBOC. Every bead feels
the gradient of its own active-surface energy. The existing ring free-step
integrators are reused between force kicks.

For `rescaling="bead"`, the full momentum direction contains the NAC at each
bead, using consistent electronic phases. For `rescaling="centroid"`, a centroid
NAC direction is broadcast to all beads, preserving internal bead momenta.
Both choices solve the mass-weighted quadratic for a common shift along that
direction using the **sum of corrected bead energy gaps**. The smaller momentum
change is selected. This differs intentionally from the old CA driver's
`rescaling="centroid"`, which conserves centroid electronic plus kinetic energy.

An energetically forbidden hop either retains momenta or reflects their component
along the selected direction, according to `frustrated_hops`. A zero direction
frustrates the hop. `HopResult.ring_energy_change` reports the realized full-ring
energy change, including any reflection roundoff. Deterministic `attempt_hop`
and stochastic `step` commit state and diagnostics only after validation.

Electronic propagation uses unitary midpoint exponentials. Hopping currents are
integrated at electronic substep midpoints along the linear interpolation between
the old and new bead geometries, with chord velocities `(q_new-q_old)/dt`.
A zero active electronic population
makes the conditional hopping probability undefined and is rejected. A total
outgoing probability above one beyond roundoff fails transactionally with a
request to reduce the timestep, rather than rescaling the probabilities. Failed
steps restore the state, random-number generator and hop diagnostics.

## Diagonal Born–Oppenheimer correction

The finite-state correction is

\[
D_a(R_b)=\sum_{j\ne a,I}\frac{|d_{aj,I}(R_b)|^2}{2M_I}.
\]

`dboc=True` includes this term in active-surface energies, nuclear forces and
hop gaps. The gradient is computed by a central difference of the **scalar**
`D_a` with displacement `dboc_step`, so nuclear force is consistently
`-grad(E_a + D_a)`. Differentiating the scalar norm also avoids spurious derivatives
from independent eigenvector phase changes. This requires additional provider
evaluations and a convergence study in the finite-difference displacement.

This implementation also uses corrected diagonal energies in electronic phase
propagation: in the fixed diabatic basis its effective matrix is
`U diag(E + D) U†`. The NACs and adiabatic projectors remain those of the supplied
electronic matrix. This is an explicit consistent-potential convention for
multistate and DM combinations; it is not a claim that all those combinations
were benchmarked in the cited two-state DBOC study. In a two-state model the
correction is equal on both surfaces and adds only a common electronic phase.

Only states supplied by the provider contribute to the sum. A truncated electronic
model therefore supplies a truncated DBOC, not the complete molecular correction
from omitted electronic states. DBOC can become very large near small gaps and
does not universally improve FSSH or RPSH. Its scalar finite-difference step and
the dynamical timestep both require convergence; no divergence is hidden by a
coupling cap. The implementation follows the energy definition and the force
identity of Eqs. (7)–(9) in Limbu et al.

## Complex SOC and geometric forces

Complex CA/DM use Hermitian conjugation throughout. For momentum rescaling the
implementation selects the real direction that maximizes

\[
\|\operatorname{Re}(e^{i\eta}d_{aj})\|^2,
\]

which is Method #2 of Miao, Bellonzi and Subotnik (2019). For bead rescaling, the
bead NACs are first placed in a common centroid phase convention and concatenated
before maximizing. The resulting ray is invariant under adiabatic eigenvector
rephasing; the largest Cartesian component fixes its overall sign. If the two
largest singular values tie, the maximizing ray is not unique. Such a hop fails
explicitly rather than using an arbitrary gauge-dependent direction.

With `geometric_force=True`, each bead also feels the nondegenerate adiabatic
Berry-force ansatz from Eq. (6) of that paper:

\[
F^{\mathrm B}_{a,I}=2\operatorname{Im}\sum_{j\ne a,J}
d_{aj,I}d_{ja,J}v_J=G_{a,IJ}v_J.
\]

`G` is real antisymmetric and this force does no kinetic work. The code advances
the force at fixed geometry by a Cayley rotation of mass-weighted momenta,
preserving kinetic energy to roundoff, and splits it symmetrically with the
ordinary force kicks. `geometric_force=False` explicitly omits this effect.
For a real Hamiltonian, the curvature vanishes.

This is a local, nondegenerate surface-hopping extension. The Berry force has an
adiabatic justification; combining it beadwise with CA/DM is an explicitly
specified approximation. It does not recover all nonlocal geometric phase
interference, implement the later phase-space surface-hopping algorithms, or
resolve Kramers/other degenerate subspaces and their non-Abelian connections.
Those degeneracies are rejected, including when encountered in a molecular SOC
model. Passing a complex matrix alone is not evidence of a complete spin-dynamics
or intersystem-crossing treatment.

## Validation and references

The dedicated tests use analytic rotating eigenvectors, an independently
exponentiated Hamiltonian, mass-dependent DBOC formulas, Berry curvature of a
gapped Pauli Hamiltonian, and arbitrary electronic basis changes. They cover
`P=1`, coincident beads, internal bead motion, DM purity loss, moving centroid
currents, three-state DBOC hop gaps, norm and energy accounting, random
eigenvector phases, and transactional failures. Run with:

```bash
PYTHONPATH=src python -m pytest tests/test_advanced_surface_hopping.py \
  tests/test_surface_hopping.py tests/test_rpsh_decoherence.py
```

For predictive calculations, numerical convergence, trajectory sampling error
and disagreement between approximate methods remain separate checks.

The [method-to-paper catalog](manual/references.md) distinguishes each original
method from the extra combinations exposed by this API.

1. Y. Shakya, R. Welsch, L. Inhester and R. Santra, *Capturing electronic
   decoherence in quantum-classical dynamics using the ring-polymer-surface-
   hopping–density-matrix approach*, **Phys. Rev. A 107**, 062810 (2023).
   [DOI](https://doi.org/10.1103/PhysRevA.107.062810).
2. D. K. Limbu, S. Bhusal, D. M. Castañeda-Bagatella and F. A. Shakib,
   *Diagonal Born–Oppenheimer corrections in condensed-phase ring polymer
   surface hopping*, **J. Chem. Phys. 163**, 234102 (2025).
   [DOI](https://doi.org/10.1063/5.0301249),
   [author manuscript](https://arxiv.org/html/2509.10629v2).
3. G. Miao, N. Bellonzi and J. E. Subotnik, *An extension of the fewest switches
   surface hopping algorithm to complex Hamiltonians and photophysics in
   magnetic fields: Berry curvature and “magnetic” forces*, **J. Chem. Phys.
   150**, 124101 (2019).
   [DOI](https://doi.org/10.1063/1.5088770),
   [author manuscript](https://arxiv.org/pdf/1902.05115).
