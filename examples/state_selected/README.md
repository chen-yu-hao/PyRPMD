# Specified vibrational initial states

These two small examples prepare H + a Morse diatom with the diatom's harmonic
stretch set to `v=1`, zero centroid angular momentum, and collision energy
0.1 eV. The atom has no interaction with the molecule in this toy PES. These
inputs exercise initialization, propagation, diagnostics and restart; they do
not predict an H + H2 reaction cross section.

```bash
pyrpmd validate examples/state_selected/ne_rpmd.inp
pyrpmd state-check examples/state_selected/ne_rpmd.inp --time '10 fs' --samples 8 --stride 10 --output runs/ne-state-check.json
pyrpmd run examples/state_selected/ne_rpmd.inp --output runs/ne-rpmd-toy
pyrpmd extend runs/ne-rpmd-toy --max-time '4 fs'
pyrpmd resume runs/ne-rpmd-toy
pyrpmd run examples/state_selected/qct.inp --output runs/qct-toy
```

`ne_rpmd.inp` uses 32 beads and an explicitly chosen preparation beta. This is
an auxiliary ring parameter and does not assign a gas temperature to a selected
initial vibrational state. `qct.inp` uses one bead, random harmonic phase and
the explicit `j0_energy_policy = "preserve_actions"` option to retain the
initial energy `(v+1/2)*omega` in the construction mode basis while enforcing
zero fragment angular momentum. The example's single radial diatom mode
already has zero angular momentum. For a polyatomic fragment, the solver may
adjust the drawn phases and introduce correlations between them. It raises
an error on nonconvergence, without redrawing or using another policy.

This constrained-phase option is an experimental preparation convention, not
a validated sampler of the conditional microcanonical distribution. Inputs
that omit `j0_energy_policy` retain the original `"project"` policy, whose
angular-momentum projection can change the assigned harmonic mode energies.
A P=1 thermal ring is not a QCT control; select the `qct` method explicitly.
Neither selected-state example samples a thermal rotational distribution or
assigns final rovibrational quantum states.

The input Hessian is the analytic Cartesian Hessian of the toy Morse minimum.
PyRPMD projects out rigid translation and rotation and diagonalizes it using
the actual isotope masses. Mode labels and the full reference are saved in
`states/initial-ensemble.json`. No Andersen parent trajectory is run. Independent
initial-state and collision RNG streams are derived from the global seed and
trajectory ID, so worker allocation does not change a launch.

The short 2 fs budget deliberately leaves incoming trajectories unresolved.
`extend --max-time '4 fs'` raises the **total** budget to 4 fs, retaining their
saved q/p and classification persistence state. It does not add another 4 fs,
resample the state, reset the phase, or relaunch completed trajectories.
Unresolved trajectories and numerical failures stay in the statistical
denominator and appear as separate counts.

## Native production PES

The built-in toy Morse model evaluates in Python, so these portable examples
exercise the compatibility backend. For production, configure an existing
native C ABI or Fortran batch PES and require the C++ block driver:

```toml
[potential]
provider = "native"
library = "/absolute/path/to/libpes.so"

[dynamics]
integrator = "cayley-bcb"
timestep = "0.02 fs"
loop_backend = "native"
block_steps = 256
```

Use that PES's equilibrium geometries and Hessians, correct atom ordering and
isotope masses. See [native potential plugins](../../docs/manual/potentials.md)
for the provider's complete configuration and ABI checks. Nuclear state
preparation is outside propagation; the physical collision and its energy/ring
observations use the existing continuous C++ block. A strict native request
errors when the PES cannot supply the required native binding.

## Interpreting diagnostics

For QCT, `launch.preparation_diagnostics` records the requested and observed
mode energies in the unrotated construction basis, their relative errors,
the complete Cartesian harmonic-energy error, and residual fragment momenta.
`state-check` saves the same preparation report for each QCT sample as
`samples[i].preparation_diagnostics`. These values quantify initialization
constraints before the collision orientation and kick; they do not include
the difference between the harmonic reference and the full Morse PES.

The `energy_parts` columns in each trajectory NPZ sum to the conserved classical
ring Hamiltonian `H_P/P`; they are not vibrational quantum energies. Molecular
mode actions are approximate harmonic diagnostics from `state-check`, computed
after Eckart alignment at sparse observation times. The check evolves the full
PES at the separated reference geometry without a collision kick, so residual
interfragment forces can still matter. Check separation as well as timestep,
beads, beta and harmonic/full-PES consistency before using a realistic system.

The `preserve_actions` constraint applies to the reference basis used to
construct a QCT initial state. Eckart/Kabsch alignment at a later observation
can change the mode projections, particularly for large deformations or
degenerate modes. For a linear molecule, alignment does not uniquely fix
rotation about its axis. Thus an aligned mode-action estimate is not an exact
state population or a guarantee of preserved action on the full PES.

See the [method and statistical conventions](../../docs/state-selected.md) for
equations, literature, assumptions and the boundary between initial-state
selection and final quantum state-to-state scattering.
