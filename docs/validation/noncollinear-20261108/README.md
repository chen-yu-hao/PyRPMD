# Noncollinear Cl + HBr validation checkpoint

**No selectively quantum-corrected 3D gas-phase rate is reported in this record.**
This is a measured software/physics validation checkpoint, not a universal-method success claim.

Reproduction and scope: [benchmark README](../../../benchmarks/noncollinear_quantum/README.md).
Machine-readable measurements are in `summary.json`. Inputs, sampler, propagator,
quantum audit and self-contained report generator are in
`benchmarks/noncollinear_quantum/`.

## Measured work

- 19,456 actual collision propagations across 10 noncollinear ensembles, plus
  a small smoke calculation. The paired integrator runs share initial samples.
- 32 new local tests passed; compileall passed. This was a repository overlay
  test, not an assertion that the entire PyRPMD suite ran locally.
- Three independent restrained-PIMD pilots, 256 samples each, at 150/300/600 K.
- Two finite-channel J=0 quantum scattering calculations in three internal
  coordinates, with both radial arrangements retained.

Representative baseline rates (cm^3 molecule^-1 s^-1; one sampling SE):

| T / K | P | Integrator | RPMD rate |
|---|---:|---|---|
|150|32|Verlet, dt=4 au|3.242e-13 +/- 9.523e-14|
|300|32|Cayley, dt=2 au|1.801e-12 +/- 3.403e-13|
|600|16|Verlet, dt=4 au|9.444e-12 +/- 1.196e-12|

The different bead counts and steps do not define a fully converged temperature
curve. Electronic/symmetry factors are explicitly 1 on the historical LEPS
surface. Statistical uncertainty excludes all systematic/model errors.
The wider-launch run retains three unfinished trajectories; their weight is
reported as a rate interval, not discarded.

## Failed or unestablished physical gates

1. The conservative local selector chooses all 3 internal coordinates at each
   temperature, including all 64 bootstrap resamples. A budget of 2 is refused
   or leaves excessive residual loading. This is heuristic, not proof that no
   better reduced coordinate system exists.
2. J=0 CRPs change by 49.086% in relative L2 on a common TOTAL-energy window
   under combined numerical refinement. No threshold shifts were fitted.
   The fine discrete S-matrix unitarity defect is 3.86e-12, which does not fix
   the unconverged channel/basis boundary problem.
3. No J>0 scattering blocks or complete partial-wave thermal integral were run.
   J=0 is not compared directly against an all-impact thermal rate.
4. No noncollinear selective-quantum transfer is accepted. The workflow rejects
   inconsistent units, state/normalization/observable identities, absent quantum
   gates, and non-reduced selector status before producing a corrected rate.
5. Arbitrary-atom collision APIs have 5/7-atom software controls, not a real
   polyatomic high-accuracy reaction benchmark.

A successful universal method still needs a transferable reduced quantum
Hamiltonian/kinetic metric, a justified environment approximation, and
independent all-J quantum or other reliable physical references.
