# Constant-pressure PIMD

The materials `pimd` workflow accepts an isotropic Bussi–Zykova–Parrinello
(BZP) barostat. The cell changes volume while preserving its shape and
orientation. All three directions must be periodic and every atom must be
mobile. Orthogonal and tilted reference cells use the same algorithm.

## Input and execution

Add a `[barostat]` table to a PIMD input:

```toml
[barostat]
method = "bzp"
pressure = "1 bar"
time_constant = "1000 fs"
# Optional; the default is the inverse of time_constant.
friction = "0.001 fs^-1"
```

`pressure` is required. Its canonical unit is hartree/bohr³; explicit pressure
units include Pa, kPa, MPa, GPa, bar and atm. `time_constant` sets the piston
mass and must be positive. The piston friction has inverse-time units.
These are scientific settings and belong to the checkpoint fingerprint.
The initial `system.cell` remains the run reference; saved frames and the
barostat checkpoint carry the changing cell.

The particle thermostat remains `sampling.thermostat = "pile-l"` or
`"andersen"`. PILE-L uses the chosen Cayley BCOCB or exact-normal-mode BAOAB
particle sampler. Andersen also supports velocity-Verlet. Use
`dynamics.loop_backend = "auto"` or `"python"`; the volume coupling is
orchestrated in Python and retains the existing compiled particle substeps.
Independent replicas can use trajectory process workers. The initial NPT
implementation rejects bead-worker pools and MPI execution, whose provider
cell updates require a separate collective protocol.

`examples/materials/npt_free.inp` is an executable ideal-gas check with a known
volume distribution. It is not a fitted material model. Run it with:

```bash
pyrpmd validate examples/materials/npt_free.inp --check-pes
pyrpmd run examples/materials/npt_free.inp --output runs/materials-npt-free
pyrpmd resume runs/materials-npt-free
```

A 1-bar gas cell can be much larger than a dense material cell. Choose the
initial volume and barostat time constant for the intended system; a strongly
compressed initial structure still needs a stable PES and timestep.

## Pressure and ensemble convention

The implementation follows the centroid version of BZP described by
[Ceriotti, More and Manolopoulos, Section 4](https://arxiv.org/abs/1402.1045),
with the sum-beads Hamiltonian sampled at inverse temperature `beta/P`.
It couples only the bead centroid to the volume. Internal ring offsets are
unchanged by the dilation operation.

For `P` beads, volume `V`, bead forces `F`, and physical per-bead virials `W`,
the instantaneous pressure used by the piston is

\[
p_{\mathrm{int}} =
\frac{P\sum_i |\bar{p}_i|^2/m_i +
\sum_j\operatorname{Tr}W_j -
\sum_{ji}(q_{ji}-\bar q_i)\cdot F_{ji}}{3PV}.
\]

The piston momentum `alpha` has mass `mu = 3*N*time_constant**2/beta`,
OU variance `P*mu/beta`, and pressure force

\[
\dot\alpha = 3P\left[V(p_{\mathrm{int}}-p_{\mathrm{ext}})
                 +\frac{1}{\beta}\right].
\]

The `1/beta` correction uses freely moving center-of-mass coordinates and
momenta. It is essential for the NPT volume measure. No COM removal or
fixed-atom constraint is applied.

The reversible deterministic part has the extended invariant

\[
E_{\mathrm{ext}} = H_P + Pp_{\mathrm{ext}}V
                    +\frac{\alpha^2}{2\mu}
                    -\frac{P}{\beta}\log V.
\]

Thermostats exchange heat, so this expression alone is not conserved in a
thermostatted run. `BZPBarostat.extended_energy` exposes it for deterministic
integration tests. It is distinct from physical internal energy.

The integrator composes pressure half-kicks and centroid/cell half-dilations
around the existing fixed-volume particle step, with half OU steps on the
piston at each end. This symmetric splitting reuses the particle sampler;
it has its own finite-timestep error and is not a bitwise reproduction of
i-PI's combined force/piston update.

## Potential interface

A variable-cell potential must implement `set_cell(cell_bohr)` and provide
`PESResult.virial` with shape `[P,3,3]` in hartree. The convention is
`W = -V * stress` for tensile-positive stress. Include every explicit cell
derivative, including any volume-dependent tail correction. Exclude particle
kinetic energy and ring-spring terms. A missing virial raises an error;
`q dot F` is insufficient for a general periodic potential.

Supported adapters are:

- **ASE:** updates the calculator cell without scaling its stored positions;
  the calculator must implement stress.
- **Periodic Python with `evaluate(q, *, cell, pbc)`:** receives the current
  cell in the provider's declared length unit on every evaluation.
- **Bound Python with `metadata['supports_pbc'] = True`:** additionally needs
  an explicit `set_cell(cell_bohr)` callback in canonical atomic units.
- **i-PI:** sends the new cell and inverse in subsequent POSDATA requests.
- **Free particles:** returns zero physical virial for ideal-gas validation.

The fixed-lattice `periodic_cosine` builtin has no variable-cell contract.
Other adapters are accepted only when they meet the same explicit interface.
The evaluation cache is invalidated on every cell update, including rollback,
even if Cartesian coordinates happen to be unchanged.

## Checkpoints and numerical checks

`BZPBarostat.snapshot()` returns its current cell and piston momentum.
`restore()` verifies that the cell is a positive isotropic scaling of the
reference cell. The workflow saves this state alongside `q`, `p`, RNG,
sampling clocks and the committed trajectory prefix. A failed propagation or
observation restores both particle and piston state before recording a
checkpoint.

The tests independently check the pressure against energy differences under
centroid/cell dilation, the `1/P` normalization, reversible maps for all three
particle integrators, second-order deterministic invariant error, and exact
checkpoint continuation. Force-failure injection checks state/RNG/cell
rollback. Ideal-gas runs at one and four beads check

\[
p(V)\propto V^N e^{-\beta p_{\mathrm{ext}}V},\qquad
\langle V\rangle=\frac{N+1}{\beta p_{\mathrm{ext}}},\qquad
\operatorname{var} V=\frac{N+1}{(\beta p_{\mathrm{ext}})^2}.
\]

These checks verify the implementation. A material result still requires
timestep, bead, cell-size and sampling convergence with a suitable potential.
The workflow keeps its physical-convergence status false after a pilot run.

## Scope

This feature samples equilibrium NPT PIMD. `diffusion`, rate calculations,
metadynamics, fixed layers, partial PBC, anisotropic stress and cell-shape
fluctuations remain outside this barostat's scope. NPT PIMD configurations can
inform an equilibrium volume choice; a physical RPMD transport trajectory
still uses its fixed-cell, unthermostatted workflow.

References:

- M. Ceriotti, J. More and D. E. Manolopoulos, *Computer Physics Communications*
  **185**, 1019–1026 (2014),
  [doi:10.1016/j.cpc.2013.10.027](https://doi.org/10.1016/j.cpc.2013.10.027).
- G. Bussi, T. Zykova-Timan and M. Parrinello, *Journal of Chemical Physics*
  **130**, 074101 (2009),
  [doi:10.1063/1.3073889](https://doi.org/10.1063/1.3073889).
