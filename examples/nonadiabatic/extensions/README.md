# Explicit nonadiabatic method examples

These seven inputs are small, executable workflow examples. All use four
independent trajectories and 6–12 steps so that validation and restart can be
checked quickly. They are not converged physical predictions.

| Input | Method and initialization | Population output |
| --- | --- | --- |
| `sm_nrpmd.inp` | Two-state W-spin SM-NRPMD; harmonic reference rings and focused diabatic state 0 | Signed W-spin estimator in the fixed diabatic basis, with complex coherence |
| `rp_mash.inp` | CA-RP-MASH; harmonic reference rings and upper adiabatic hemisphere sampled with `abs(Sz)` importance | Centroid adiabatic active-surface populations; no generic density/coherence output |
| `rpsh_ba.inp` | RPSH-BA; one adiabatic amplitude vector and transported bead frames | Fixed diabatic amplitude density; common bead-adiabatic amplitudes and active counts stored separately |
| `rpsh_dm.inp` | RPSH-DM; one wavefunction per bead, common-centroid density/hop current | Fixed diabatic amplitude density; centroid adiabatic amplitudes and active counts stored separately |
| `rpsh_dboc.inp` | Advanced CA RPSH with DBOC on Tully model 1 | Fixed diabatic amplitude density plus separate adiabatic amplitudes/active counts |
| `soc_rpsh.inp` | Complex LVC model, CA hopping and local Berry-force ansatz | Complex fixed diabatic amplitude density plus separate adiabatic amplitudes/active counts |
| `nah2_sm_nrpmd.inp` | Published molecular NaH₂ 5G PES; W-spin SM-NRPMD, focused diabatic state 1 | Signed fixed diabatic W-spin density |

## Run from the repository root

After installing PyRPMD, select any input above:

```bash
pyrpmd validate examples/nonadiabatic/extensions/sm_nrpmd.inp --check-pes
pyrpmd run examples/nonadiabatic/extensions/sm_nrpmd.inp --output runs/example-sm
pyrpmd resume runs/example-sm
pyrpmd analyze runs/example-sm --recompute
```

`python -m pyrpmd` is equivalent to `pyrpmd`. For a source checkout, set
`PYTHONPATH=src`. Use a different output directory for each input. The examples
select portable CPU paths; SM can additionally select the explicit native or
Torch electronic backend when available. MASH uses adaptive DOP853 event
integration and requires `native=false`.

The NPZ output contains per-trajectory contributions and means; the CSV gives
population time series. SEM uses independent trajectories, not beads or saved
frames. The checkpoints preserve nuclear/electronic state, phases or transported
frames as required, random streams, and observation history. The examples use
block sizes distinct from sampling strides to exercise this separation.

## Interpret initialization and measurement together

The harmonic ring distribution is the finite-bead Boltzmann distribution of
the specified **nuclear harmonic reference**. It does not prepare the coupled
electronic-nuclear thermal density. Point initialization repeats one specified
nuclear phase point on all beads. `beta` determines the ring frequencies in
either case; it does not turn a point phase point into a thermal ensemble.

`initial_state` denotes a fixed diabatic state for SM. `active_surface` denotes
an energy-ordered adiabatic surface, with 0 lower and 1 upper for two states.
These preparations differ at coupled geometries. Different rows of this table
therefore do not constitute a controlled method-comparison benchmark.

The BA/DM amplitude density is generally mixed after bead averaging. Its
diagonal and active-surface fractions answer different questions and should
not be interchanged. The MASH workflow deliberately reports active populations
only. General MASH diabatic/coherence correlations require the separate weighted
operator-correlation API and full-sphere initial sampling.

## Molecular and method scope

NaH₂ is a **neutral doublet molecular system**: the portable 5G model covers its
coupled lowest two doublet A′ surfaces with all three atoms and Cartesian
derivatives. This input specifies Na/H isotope masses and an interaction-region
geometry with zero total initial linear momentum. It demonstrates molecular
propagation; asymptotic collision, rovibrational and reaction-probability
preparation require additional physical choices and convergence. The historical
fit contains no SOC and is not an on-the-fly electronic-structure calculation.

`soc_rpsh.inp` is an explicitly complex **model** LVC Hamiltonian, not a fitted
SOC PES for NaH₂ or another molecule. Its `harmonic` values are force constants,
not frequencies. SOC hopping uses the documented nondegenerate local
Berry-force/rescaling prescription. DBOC uses a finite-difference derivative of
the correction; converge `dboc_step` separately from the dynamics timestep.

SM and MASH currently implement two electronic states. MASH and BA hopping
require real matrices; degeneracies require care and are rejected by the
nondegenerate hopping implementations. Full method definitions, original
references and limits are in
[spin methods](../../../docs/nonadiabatic-spin-methods.md),
[hopping extensions](../../../docs/nonadiabatic-hopping-extensions.md), and
[molecular PES documentation](../../../docs/molecular-nonadiabatic.md).
