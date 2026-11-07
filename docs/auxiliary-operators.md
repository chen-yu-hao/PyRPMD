---
orphan: true
---

# Auxiliary operators in the propagation stack

PyRPMD represents electronic mapping variables and other non-nuclear degrees of freedom as **auxiliary operators**. The protocol is implemented in `pyrpmd.auxiliary` and re-exported by `pyrpmd.engine`; `NRPMDDynamics` is the first adapter. The design separates three responsibilities:

1. the nuclear integrator owns nuclear coordinates, momenta, the ring-polymer spring map, and the trajectory clock;
2. an auxiliary operator owns its state arrays and the fixed-nuclear-coordinate auxiliary flow;
3. the adapter declares which physical combinations are supported and commits a complete step only after every sub-operator succeeds.

All quantities use atomic units. Auxiliary arrays are writable, C-contiguous `float64` arrays. A layout is either `per-ring`, whose leading dimension is one, or `per-bead`, whose leading dimension is the number of beads. The electronic-state dimension is unrestricted apart from being positive.

## Protocol

An `AuxiliaryOperator` supplies:

- `state_arrays`: named arrays that form the auxiliary state;
- `propagate(dt, q)`: the fixed-`q` auxiliary flow, including any integrated nuclear impulse required by that flow;
- `nuclear_force(q)`: the instantaneous auxiliary contribution to the nuclear force;
- `energy(q)`: the auxiliary Hamiltonian contribution;
- `state_dict()` and `restore_state_dict(...)`: versioned JSON persistence;
- `identity()`: a finite JSON scientific identity;
- `layout`, `nstates`, and an explicit set of permitted split insertion points;
- an `AuxiliaryCompatibility` whitelist.

`AuxiliaryPropagationCore` operates on a copied trial state. It validates the nuclear and auxiliary arrays, applies the requested symmetric composition, validates the result, and only then copies all fields and the clock back to the public state. A failed PES evaluation or auxiliary operation therefore leaves the caller's state unchanged.

## NRPMD operator order

For the existing NVE API, one step is exactly the historical map

\[
  \Phi_{\Delta t} = M_{\Delta t/2}\,C_{\Delta t}\,M_{\Delta t/2},
\]

where `M` is the exact fixed-nuclear-coordinate MMST mapping flow (MInt), including its time-integrated nuclear momentum impulse, and `C` is the selected free-ring map. The adapter invokes the same `_mint` implementation and the same `CayleyIntegrator`; no extra force kick is introduced. Existing inputs therefore retain their numerical map and scientific fingerprint.

With an explicitly supplied scalar nuclear potential, the reversible composition is

\[
 B_{\Delta t/2}\,M_{\Delta t/2}\,C_{\Delta t}\,M_{\Delta t/2}\,B_{\Delta t/2}.
\]

For `SplitPotential` plus `ForceSplitting(inner_steps=M)`, the slow scalar correction is applied at the outer boundary and the fast scalar, MMST, and free-ring maps are repeated with \(\delta t=\Delta t/M\):

\[
 B_{\mathrm{s},\Delta t/2}
 \left[B_{\mathrm{f},\delta t/2}M_{\delta t/2}C_{\delta t}
 M_{\delta t/2}B_{\mathrm{f},\delta t/2}\right]^M
 B_{\mathrm{s},\Delta t/2}.
\]

RPC changes the scalar Hamiltonian through `SplitPotential`; its lifted forces are not reinterpreted as independent bead forces.

## Reversibility and symplecticity

Each deterministic composition is palindromic. For a deterministic, time-independent PES, reversing all momenta and applying the map with the opposite timestep gives the inverse map up to floating-point roundoff. The free-ring map, scalar kicks, and exact fixed-`q` MMST Hamiltonian flow are symplectic; compositions of these maps remain symplectic. Symplecticity does **not** imply exact energy conservation at finite timestep, so validation reports bounded energy error rather than physical convergence.

Periodic propagation stores unwrapped nuclear coordinates. Only coordinates passed to fixed-cell PES evaluations are wrapped. The ring-polymer spring therefore remains continuous across the boundary. NPT is intentionally unsupported: changing the cell would require a separately derived mapping/cell Hamiltonian and virial.

## Compatibility whitelist

| Coupling | NRPMD status | Physical scope |
| --- | --- | --- |
| NVE, fixed cell | Supported | Historical MInt/free-ring/MInt map. |
| Fixed-cell PBC | Supported | PES coordinates are wrapped; propagated coordinates remain unwrapped. |
| Scalar nuclear potential | Supported | Symmetric scalar kicks around the MMST/free-ring map. |
| `SplitPotential` / RPC | Supported | Scalar RPC Hamiltonian and exact chain-rule lifted forces. |
| RESPA with scalar `SplitPotential` | Supported | Explicit outer slow and inner fast/MMST/free-ring composition. |
| Andersen nuclear preparation | Supported | Nuclear momenta/coordinates sampled with mapping variables frozen. |
| PILE-L nuclear preparation | Supported | PILE acts on the frozen-mapping scalar Hamiltonian. |
| Real-time thermostat | Rejected | No thermostatted NRPMD correlation-function interpretation is claimed. |
| Barostat / NPT | Rejected | No mapping/cell Hamiltonian or nonadiabatic physical virial is implemented. |
| Bond constraints | Rejected | A constrained MMST splitting has not been validated. |
| Bias potential | Rejected | Bias placement and estimator semantics require a method-specific derivation. |

The whitelist is checked before PES evaluation, RNG consumption, provider cell updates, or workflow output creation. Code reachability is not treated as physical validation.

## Nuclear-coordinate preparation

`NRPMDDynamics.thermalize(...)` supports Andersen and PILE-L preparation. The mapping variables are copied and held fixed while a scalar potential evaluates the instantaneous MMST Hamiltonian and its nuclear gradient. The routine advances only nuclear `q`, `p`, and the shared clock; it verifies that both mapping arrays are bitwise unchanged before committing.

This operation prepares a nuclear distribution conditional on one fixed mapping realization. It is not an electronic thermal trace, not thermostatted real-time NRPMD, and not evidence of physical convergence. PILE-L follows the normal-mode path-integral thermostat of Ceriotti, Parrinello, Markland, and Manolopoulos, *J. Chem. Phys.* **133**, 124104 (2010), DOI `10.1063/1.3489925`. The NRPMD mapping Hamiltonian and MInt context follow Richardson and Thoss, *J. Chem. Phys.* **139**, 031102 (2013), DOI `10.1063/1.4816124`.

## Identity and checkpoints

New NRPMD options are represented in `NRPMDDynamics.identity()`: cell and PBC flags, scalar/RPC potential parameters, force-splitting depth, electronic backend/device, and auxiliary layout/state count. Known built-in and composite scalar potentials receive complete parameter identities. An unknown external scalar provider must supply `nuclear_potential_identity`; PyRPMD does not silently fingerprint only its class name.

`operator_state_dict()` uses version 2 and contains the immutable scientific identity, complete RNG state, and a versioned auxiliary state. `load_operator_state_dict(...)` accepts version 2 and the legacy RNG-only forms. Nuclear and mapping trajectory arrays remain in the trajectory checkpoint; restore is transactional. The existing nonadiabatic trajectory container remains version 1 and readable, so pre-T26 workflow checkpoints resume through the same loader.

## Deliberate limits

This change does not migrate spin mapping, MASH, surface hopping, IESH, KC-RPMD, MV-RPMD, or isomorphic Ehrenfest dynamics. It does not add an auxiliary native kernel. The Python composition may still use the existing native free-ring kernel, but auxiliary ordering and commits remain owned by Python.
