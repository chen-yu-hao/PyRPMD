# Complex couplings and ring-polymer surface hopping

The matrix-PES contract accepts **complex Hermitian** electronic energies and
their nuclear derivatives. The existing state-selected NRPMD collision driver
can therefore propagate a supplied spin-orbit-coupled diabatic Hamiltonian.
A separate Python API implements centroid-approximation ring-polymer surface
hopping (RPSH-CA) for **real symmetric** matrices.

Run both APIs on a small synthetic model:

```bash
python examples/nonadiabatic/soc_and_rpsh.py
```

The example prints deterministic JSON containing mapping norm/energy checks and
RPSH amplitude, active-surface, and hop-energy diagnostics. It starts from a
specified unthermalized ring and explicitly retains `physical_converged=false`.

## Complex spin-orbit couplings in NRPMD

Return `DiabaticResult(matrix, gradients)` with shapes `[B,S,S]` and
`[B,S,S,N,3]`. Energies are in hartree, positions in bohr, and gradients are
positive derivatives, not forces. Both arrays must satisfy Hermiticity in their
two electronic indices. For example, if `V[0,1] = a + i*b`, then
`V[1,0] = a - i*b`; the gradients obey the same conjugation rule. All diagonal
elements are real. The Python adapter applies the declared energy/length unit
conversion to both real and imaginary parts.

The mapping Hamiltonian is

\[
H_{\mathrm{el}}(q,x,y)
=\frac12\left[(x+i y)^\dagger V(q)(x+i y)-\operatorname{Tr}V(q)\right].
\]

The result is real for a Hermitian matrix. In particular, writing `V=A+iB`
with real symmetric `A` and real antisymmetric `B` gives
`H_el = (x.T A x + y.T A y - 2 x.T B y - Tr A)/2`.
Dropping the imaginary matrix entries would remove the cross term and its
nuclear force.

The exact fixed-coordinate MInt substep diagonalizes `V=U diag(E) U†`, rotates
`z=x+i*y` by `exp(-i*V*h)`, and integrates the full complex force bilinear before
taking its real part. This preserves the mapping norm, constant-unitary basis
covariance, and the canonical symplectic structure. Degenerate energies use a
continuous sinc integral. The real-valued scalar diagonal adapter explicitly
extracts the real diagonal only after the complete Hermitian contract is checked.

The state-selected initialization, signed populations, collision stopping tests,
and physical restrictions of the original NRPMD workflow remain applicable.
SOC values and derivatives must come from a physically justified fixed diabatic
basis. This feature does not infer spin-orbit couplings from uncoupled singlet
and triplet surfaces. Reversing momenta and conjugating mapping amplitudes at
fixed complex `V` is generally not the same operation as reversing physical time;
the inverse numerical map uses negative time steps.

## Centroid RPSH API

```python
from pyrpmd.dynamics import State
from pyrpmd.surface_hopping import RPSHDynamics

driver = RPSHDynamics(
    matrix_potential, masses, beta, nbeads, dt, nstates,
    seed=42, electronic_substeps=8,
    rescaling="centroid", frustrated_hops="keep",
)
state = driver.initialize(State(q, p), active_surface=0)
driver.step(state)
populations = driver.populations(state)
last_hop = driver.last_hop
```

Every bead follows the same active adiabatic surface, using its own position to
evaluate that surface's force. The existing free-ring integrator handles the
springs and real nuclear masses. The electronic Hamiltonian, derivative
couplings, and fewest-switches currents are evaluated along the centroid path.
Electronic amplitudes are stored in the **fixed diabatic basis**, so arbitrary
sign changes of adiabatic eigenvectors do not require persistent gauge tracking.
This is the centroid electronic equation expressed in an equivalent basis.

The centroid path between nuclear endpoints is linearly interpolated for the
electronic substeps. A midpoint unitary electronic propagator and midpoint
current quadrature provide second-order convergence on a prescribed smooth
centroid path. Hops are selected at the end of the nuclear step; nuclear timestep
convergence is needed in addition to electronic-substep convergence.

For active surface `a`, the implementation integrates the outgoing current to
each `j`, divides by the initial active-state amplitude population, and sets
negative integrated probabilities to zero. It rejects a step if their total is
greater than one beyond roundoff. Increase electronic resolution and reduce the
nuclear timestep when necessary. A zero active-state population also fails
explicitly instead of dividing by zero.

### Energy rescaling and frustrated hops

At a hop, every bead receives the same momentum shift along the centroid
derivative-coupling vector. Internal ring-mode momenta remain unchanged.

| Setting | Conserved at an accepted hop | Diagnostic |
| --- | --- | --- |
| `rescaling="centroid"` (default) | Centroid kinetic energy plus centroid adiabatic energy | Full-ring energy can jump; `HopResult.ring_energy_change` records the jump |
| `rescaling="ring"` | Full sum-bead Hamiltonian, using the mean bead energy gap in the centroid rescaling equation | `ring_energy_change` is zero to floating-point accuracy |

For one bead the conventions agree. The second setting is a distinct energy
rescaling choice, not a claim that RPSH is a rigorously derived thermal theory.
An upward hop without enough kinetic energy is frustrated. With
`frustrated_hops="keep"` the momenta remain unchanged; with `"reverse"` their
centroid component along the coupling direction reverses. Neither choice adds
a universal detailed-balance or decoherence correction.

### Restarting and supported scope

`RPSHState.to_dict()` and `RPSHState.from_dict()` provide detached JSON-compatible
state snapshots. Save and restore `driver.rng.bit_generator.state` alongside the
state, using the same physical and numerical parameters. Failed steps leave the
nuclear coordinates, momenta, coefficients, active surface, clocks, RNG, and
last-step diagnostics unchanged. Provider-internal side effects cannot be
rolled back by the dynamics driver.

This API supports nonperiodic, all-mobile nuclei and nondegenerate real
adiabatic states in a fixed diabatic basis. A gap below `gap_tolerance` fails
explicitly. Complex SOC hopping, Berry-force corrections, DBOC, decoherence
schemes, periodic/fixed-atom workflows, MPI/GPU propagation, and a thermal RPSH
rate driver are outside this interface. Real `V` does not imply that electronic
amplitudes remain real; the coefficients are complex throughout propagation.

RPSH amplitude populations and the fraction of trajectories on each active
surface are different observables. Frustrated hops can make them differ; the
API reports them separately without silently equating them.

## Numerical checks

The tests compare SOC MInt against independent matrix-exponential force
quadrature and complete Hamiltonian ODE integration. They check constant-unitary
basis covariance, imaginary-gradient forces, degenerate gaps, negative-time
inversion, a finite-difference symplectic Jacobian, and unit-aware gradients.
RPSH tests cover the one-surface RPMD limit at one and four beads, a known
rotating electronic frame, integrated probability currents, both energy
conventions, frustrated-hop reversal, state/RNG restarts, and failed-step
rollback. These are numerical implementation checks, not a validation of real
chemical scattering probabilities.

## References

- F. A. Shakib and P. Huo, *J. Phys. Chem. Lett.* **8**, 3073–3080 (2017),
  [doi:10.1021/acs.jpclett.7b01343](https://doi.org/10.1021/acs.jpclett.7b01343).
- D. K. Limbu and F. A. Shakib, *J. Phys. Chem. Lett.* **14**, 8658–8666 (2023),
  [doi:10.1021/acs.jpclett.3c02085](https://doi.org/10.1021/acs.jpclett.3c02085),
  [author manuscript](https://arxiv.org/abs/2305.13320). Equations 1–4 specify the
  centroid RPSH equations; frustrated-hop choices are discussed separately.
- Y. Shakya, R. Welsch, L. Inhester, and R. Santra, *Phys. Rev. A* **107**, 062810
  (2023), [doi:10.1103/PhysRevA.107.062810](https://doi.org/10.1103/PhysRevA.107.062810),
  [accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevA.107.062810).
  Section II D distinguishes the centroid and bead approximations. The new
  density-matrix RPSH method in that paper is not implemented here.
