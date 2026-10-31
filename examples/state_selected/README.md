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
energy `(v+1/2)*omega`. A P=1 thermal ring is not a QCT control; select the
`qct` method explicitly.

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

The `energy_parts` columns in each trajectory NPZ sum to the conserved classical
ring Hamiltonian `H_P/P`; they are not vibrational quantum energies. Molecular
mode actions are approximate harmonic diagnostics from `state-check`, computed
after Eckart alignment at sparse observation times. The check evolves the full
PES at the separated reference geometry without a collision kick, so residual
interfragment forces can still matter. Check separation as well as timestep,
beads, beta and harmonic/full-PES consistency before using a realistic system.

See the [method and statistical conventions](../../docs/state-selected.md) for
equations, literature, assumptions and the boundary between initial-state
selection and final quantum state-to-state scattering.
