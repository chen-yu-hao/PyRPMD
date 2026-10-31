# Periodic materials, diffusion and surface reactions

PyRPMD handles periodic systems and slabs with frozen layers, plus constant-pressure PIMD. This chapter
covers equilibrium PIMD, thermostat-free RPMD transport and basin-normalized local hop rates. The
default materials engine and the opt-in native PIMD engine have different potential, barostat and
checkpoint contracts; select the engine explicitly when using the native route.

## Tasks and boundary conditions

| Task | Purpose | Periodic cell | `fixed_atoms` | Main result |
| --- | --- | --- | --- | --- |
| `pimd` | Equilibrium quantum statistics; optional BZP NPT | Yes | Yes at fixed cell | Energy estimators, radius of gyration |
| `diffusion` | Continuous centroid trajectories and transport | Yes | Yes | MSD, VACF, D and diagnostics |
| `site_rate` | A directed exit in a fully mobile material | Yes | No | Basin rate along one supported smooth scalar centroid CV |
| `fes` | 1D/2D geometric free energy | Yes | Yes | Position-density FES, overlap, mask |
| `surface_rate` | A directed exit on a surface, optionally with fixed layers | Yes, or a finite cluster | Yes | First-order rate with the mass metric of the mobile atoms |

Fixed atoms are supported by `pimd`, `diffusion`, `fes`, `metadynamics` and `surface_rate`; they keep
reference positions on every bead and zero momenta. Materials kinetic-energy estimators count only
mobile atoms. `site_rate` still requires all atoms mobile. Gas-phase `rate`, `drpmd` and `network`
take molecular systems without PBC or fixed atoms.
Metadynamics accepts one to three CVs, with independent CV and atomic periodicity.

All mobile atoms use the same bead number. Cells are fixed unless `pimd` explicitly uses `[barostat]`.

## Cells, coordinates and isotopes

A slab with a skewed cell:

```toml
[system]
geometry = "slab.xyz"
geometry_unit = "angstrom"
cell = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
```

Each cell row is a lattice vector and Cartesian positions satisfy `r = s @ cell`. The matrix must be
finite, invertible and well conditioned; the vacuum direction of a slab also needs a non-zero
vector. PBC flags refer to lattice vectors. A cell without `pbc` is periodic in all three directions,
so declare `pbc` explicitly.

`cell_unit` defaults to the geometry unit: bohr for inline arrays, angstrom for XYZ files. Setting
`geometry_unit` and `cell_unit` explicitly keeps an input unambiguous when its geometry source
changes.

Propagated coordinates `q[P,N,3]` and exported centroid trajectories stay unwrapped, which keeps
ring springs, centroids and MSD continuous. Periodic potentials and periodic CVs apply the minimum
image themselves; in a skewed cell that is not the same as rounding each fractional coordinate.
Nuclei are distinguishable closed rings; exchange and winding are not sampled.

H, D and T map to hydrogen on the electronic PES and differ in nuclear mass. Use `symbols=["D"]`,
`isotopes` or `masses_amu`. An isotope keeps the same Born–Oppenheimer PES; converge the bead number
and equilibration time for each isotope.

### Periodic CVs

| CV kind | Geometry | Use for |
| --- | --- | --- |
| `distance`, `bond_difference`, `angle` | Unwrapped coordinates, no minimum image | Molecular definitions, explicitly unwrapped geometry |
| `periodic_distance`, `periodic_bond_difference`, `periodic_angle` | Nearest image using the system cell and PBC | Periodic bond lengths, angles and FES |
| `projection`, `fractional` | Continuous centroid projection or fractional coordinate | Directed site hops that distinguish neighboring cells |
| `surface_height` | Adsorbate-group COM along the surface normal | Height free energies, planar dividing surfaces |
| `surface_lateral` | Unwrapped COM along an in-plane direction | Surface migration free energies and directed exits |

`periodic_bond_difference` also accepts the alias `periodic_difference`. A nearest-image CV has a
derivative branch where the nearest image switches; keep the biased range away from it.
`site_rate` accepts its supported linear and smooth nonlinear scalar centroid CVs, with the
constrained-parent mass-metric weighting described in [general-coordinate rates](../general-coordinate-rates.md).
`surface_rate` also accepts unwrapped
`distance` with its constant endpoint mass metric and full curved constraint. For
`periodic_distance`, the dividing surface must be below half the smallest singular value
of the periodic lattice basis, keeping it away from image switches. Bond differences and
variable-metric coordinates are outside this surface-rate formula.

## Periodic potentials

ASE receives the true cell, PBC and per-bead coordinates and converts units through the calculator.
If the calculator provides stress, the adapter returns the configurational virial `-volume * stress`
in hartree as `PESResult.virial[B,3,3]` (no kinetic or spring terms); chunked execution keeps it.

Python periodic plugins use one of two contracts:

- Declare `metadata["periodic"] = True` and accept the current cell in
  `evaluate(q, *, cell, pbc)`; coordinates and cell use the provider's declared length unit.
- Declare `metadata["supports_pbc"] = True` and implement `bind_system(system)`, called before
  `evaluate(q)`. The bound System is in atomic units, so a plugin working in angstrom/eV converts the
  stored cell itself.

Run `validate --check-pes` on a new calculator to check input/provider setup and finite PES results.
For ordinary adiabatic potentials this is an interface/evaluation check, not an automatic
finite-difference force, lattice-translation or finite-strain virial validation. Perform those
checks independently before a research calculation.

i-PI sends the true cell and its inverse in POSDATA, using the protocol's column-vector convention.
POSDATA carries no PBC flags, so a slab sets `potential.params.init.pbc` to match the system and the
server must honor this INIT field. Periodic models can use C ABI v2, ASE and its named ML adapters,
explicitly periodic Python or Torch, tblite, or i-PI. C ABI v1 and RPMDrate/f2py remain molecular
interfaces. The generic Torch model must explicitly select `periodic=true` and implement
`forward(q, cell, pbc)`; merely loading a Torch model does not make it periodic. DeePMD currently
accepts full PBC or no PBC, while the PySCF adapter is molecular. See
[potential contracts and restrictions](potentials.md#c-abi-v2-and-periodic-torch).

Match the electronic-structure solver's boundary convention: a plane-wave slab may use full 3D
periodicity with vacuum. PyRPMD's PBC flags do not configure a calculator's electrostatic solver
or dipole correction. Check the chosen treatment of repeated slabs, net charge and asymmetric
surface dipoles, together with vacuum, k-point, cutoff and electronic-smearing convergence.
The [GPAW dipole-layer tutorial](https://gpaw.readthedocs.io/tutorialsexercises/electrostatics/dipole_correction/dipole.html)
illustrates distinct real-space and plane-wave boundary treatments. For a fully periodic slab,
declare the surface normal explicitly and supply its area when requesting second-order normalization.

With the default workflow engine, periodic and fixed-atom systems use a Python time-step loop
with compiled deterministic substeps; use `loop_backend="auto"` or `"python"`. An explicit
`task="pimd"`, `sampling.engine="native"` selects a separate C++ trajectory engine with supported
builtins or a compiled `native_periodic` evaluator. C ABI v2 batch evaluation does not by itself
supply that evaluator. See the
[native PIMD workflow](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md)
for its potential ABI, thermostats and restart format.

## PIMD: equilibrium properties

The controls below describe the default (`sampling.engine="legacy"`) workflow. The explicit
`sampling.engine="native"` route uses a separate native thermostat/barostat engine and checkpoint
format; its compatible potentials, inputs and examples are documented in the
[native PIMD guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md).

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/manual-pimd
pyrpmd analyze runs/manual-pimd
pyrpmd resume runs/manual-pimd
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/pimd.inp) uses the
default (`sampling.engine="legacy"`) materials engine and a
periodic cosine potential. `sampling.equilibration` is the initial equilibration time, `production`
the sampling time, `interval` the save interval and `replicas` the number of independent streams.

| `dynamics.integrator` | `sampling.integrator` with PILE-L | Andersen name |
| --- | --- | --- |
| `cayley-bcb` | `bcocb` | `cayley-bcb-andersen` |
| `normal-mode-bab` | `baoab` | `normal-mode-bab-andersen` |
| `velocity-verlet` | (PILE-L unavailable) | `velocity-verlet-andersen` |

With `sampling.thermostat="pile-l"`, `gamma_centroid="0.01 fs^-1"` sets the centroid friction and the
internal modes are thermalized per normal mode. Sampling and production use the same time step.
With `loop_backend="auto"`, PILE-L uses a C++ free-ring half-step–OU–half-step kernel for `P<=8` and
power-of-two bead numbers, with potential calls and NumPy random numbers in Python; other bead
numbers use the NumPy implementation. PILE-L requires `loop_backend="auto"` or `"python"`.

The summary's `estimators` are `potential_energy`, `kinetic_primitive`, `kinetic_centroid_virial`,
`internal_energy_centroid_virial` and `ring_polymer_hamiltonian`. The first four are physical energy
estimators; the last is a propagation diagnostic. Each has a `mean` and a
`standard_error_between_replicas`, which is `null` with a single replica.
`radius_of_gyration_squared_bohr2` gives each atom's mean squared ring radius of gyration.

`samples/materials_XXXX/manifest.json` describes immutable NPZ chunks containing centroid positions,
velocities, sample steps, energy estimators and radii of gyration. `checkpoints/materials_XXXX.npz`
holds the state and the authoritative committed prefix. `[output] trajectory_storage="stream"`
is the default; `chunk_frames=256` bounds the staging buffer. `trajectory_storage="memory"` retains
the earlier single-NPZ layout. Completed runs whose scientific identity remains compatible can
migrate storage without loading the PES. Historical ASE-backed runs need a new output directory
under the [force-consistent energy convention](potentials.md); saved summaries remain readable.
Correlation analysis still loads scalar or selected-atom histories; budget that memory separately. See
[streaming and recovery](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/streaming-trajectories.md).
For transport properties use the `diffusion` task.

### Native PIMD engine

`sampling.engine="native"` selects the separate C++ ensemble engine. It has explicit built-in
and compiled PES requirements, additional thermostats, NVT support for fixed atoms, and BZP,
MTTK, RGB or SCR barostats subject to their documented restrictions. It uses transactional
`samples/native_pimd_NNNN/journal.json` files and immutable chunks instead of the default
materials checkpoint layout. Engine selection changes the run identity. See the
[native PIMD guide and examples](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md)
for the full input and output contract; the BZP table below belongs to the default engine.

### Constant-pressure PIMD

For the default materials engine:

```toml
[barostat]
method = "bzp"
pressure = "1 bar"
time_constant = "1000 fs"
```

The isotropic Bussi–Zykova–Parrinello barostat scales centroids and the cell while preserving the
internal ring modes. It requires a fully periodic 3D cell, all atoms mobile, and a variable-cell
PES returning physical configurational virials. ASE, i-PI and compatible Python models are
supported. Independent trajectory processes are supported; bead pools, MPI and GPU execution are
not supported for NPT. Pressure accepts Pa, kPa, MPa, GPa, bar, atm and atomic pressure.
The optional `friction` defaults to the inverse time constant. Samples include cells, pressures
and volumes; checkpoint state includes the cell piston. See the
[BZP method and example](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials-npt.md).
The opt-in native PIMD engine additionally exposes `mode="mttk"` and `mode="rgb"` flexible-cell
barostats, along with native `bzp` and one-bead `scr`. All these NPT modes require compiled physical
virials, 3D PBC and all atoms mobile. Native fixed-cell NVT does support `fixed_atoms`.
The native `mode`/`tau`/`gamma` controls do not replace the legacy
`method`/`time_constant`/`friction` table above. Flexible-cell inputs have a specified triangular
cell convention. A cell mask does not establish a slab or frozen-substrate NPT ensemble. The
[native ensemble guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-ensembles.md#barostats)
defines these restrictions and the tested numerical controls.

RPMD diffusion production remains fixed-cell and thermostat-free by default.

## Diffusion: MSD and D from continuous trajectories

The optional `[electronic_friction]` table selects a physical Markovian centroid EF-RPMD
production bath, including fixed substrates. These trajectories exchange energy and do not
receive an NVE conservation verdict. This approximation has no non-Markovian memory or bath
renormalization of the imaginary-time distribution; see the
[electronic-friction guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md).

`task="diffusion"` saves t=0 after thermalization and by default propagates RPMD without thermostat,
umbrella or metadynamics. It saves the centroid positions and velocities of every atom, so the
measured atoms, reference frame and fit can be changed afterwards.

```bash
pyrpmd run examples/materials/diffusion.inp --output runs/manual-diffusion
pyrpmd analyze runs/manual-diffusion --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40 --bootstrap 100
pyrpmd resume runs/manual-diffusion
```

The [example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/diffusion.inp) is a
free particle, a ballistic test case: the analysis reports `analysis_status="ballistic_or_drifting"`
and `diffusion_m2_s=null`.

| `diffusion` field | Meaning and unit |
| --- | --- |
| `atom_indices` | 0-based indices of the measured atoms; H/D/T if omitted |
| `axes` | Measured Cartesian components, e.g. `[0,1]` for xy |
| `reference_atoms` | Atoms whose mass-weighted COM is subtracted every frame; disjoint from the measured atoms |
| `max_lag` | Maximum lag in saved frames |
| `fit_range` | Lag-time range of the fit, within the saved range |
| `bootstrap` | Resamplings over independent replicas; 0 disables it |
| `block_length` | Saved frames per block for a single replica; at least `4*max_lag` |

For d analyzed components, the displacement and diffusion tensors are

$$
M_{\alpha\beta}(t)=\langle\Delta\bar r_\alpha(t)\Delta\bar r_\beta(t)\rangle,
\qquad D_{\alpha\beta}=\frac12\frac{dM_{\alpha\beta}}{dt},
\qquad D=\frac{\operatorname{tr}D_{\alpha\beta}}d,
$$

so the in-plane MSD slope is divided by 4 and the 3D slope by 6. PyRPMD also stores the symmetric
VACF tensor and its running integral. Use `reference_atoms` to remove lattice drift.

`analysis_status` and `results/diffusion.json` report the outcome of the fit:

| Status | Meaning |
| --- | --- |
| `insufficient_data` | Too few frames, fit points or time origins |
| `no_observed_motion` | No motion in the recorded window |
| `ballistic_or_drifting` | Ballistic motion or drift |
| `non_diffusive_window` | The fit window is not linear |
| `indefinite_diffusion_tensor` | The fitted tensor is not positive semidefinite |
| `unstable_fit_window` | The two halves of the window give different slopes |
| `diffusive_regime_candidate` | Linear diffusive regime; D and the tensor are reported |

For every status, `raw_fit`, MSD and VACF are saved; D and the tensor are filled in only for
`diffusive_regime_candidate`. Errors come from bootstrapping independent replicas or blocks of one
replica. Confirm D against the fit window, production length, number of replicas and system size.

## Site and surface rates

For a linear centroid coordinate s with constant mass metric, both local-rate workflows take
the dividing-surface value and basin probability from the same position density. Nonlinear
`site_rate` additionally averages the constrained-parent mass metric; see
[general-coordinate rates](../general-coordinate-rates.md). `surface_rate` also accepts
the nonlinear distance coordinates described above. For the constant-metric case:

$$
k_{\mathrm{QTST}}=
\sqrt{\frac{g^TM^{-1}g}{2\pi\beta}}
\frac{\rho(s^\ddagger)}{\int_A\rho(s)\,ds},
\qquad k(t)=k_{\mathrm{QTST}}\kappa(t).
$$

g is the gradient of the coordinate with respect to the atomic centroids and A the explicit reactant
basin. `surface_rate` removes fixed atoms from the mass metric. The basin and the dividing surface
need connected, unmasked sampling, and each bootstrap draw computes the density and the basin
integral together to keep their covariance.

Both give the first-order rate in `s^-1` for one directed exit. Total escape rates and macroscopic
diffusion coefficients combine several exits with hop lengths, site counts and hop correlations.

### `site_rate`: material sites

```bash
pyrpmd prepare examples/materials/site_rate.inp --pilot
pyrpmd run examples/materials/site_rate.inp --output runs/manual-site-rate
pyrpmd analyze runs/manual-site-rate
```

The [example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/site_rate.inp)
uses an unwrapped `fractional` centroid coordinate. The constant-metric route supports `cartesian`, `projection` and
`fractional`, the last two optionally relative to a group of lattice atoms. Smooth distances, bond
differences, angles and compatible Python centroid CVs use the nonlinear metric route described in
[general-coordinate rates](../general-coordinate-rates.md). `site_rate.cv` names the
umbrella CV, `basin=[lo,hi]` defines a finite basin, and `direction=1` places the dividing surface
at hi, `direction=-1` at lo.

Windows and the analysis range cover the whole basin and extend past the exit. Windows and
constrained parents use Andersen thermostats; ±p children run without one. Finite-sample κ keeps
its sign. Outputs are `site_qtst.json` / `.npz`, `pmf/fes.csv` and `recrossing.csv`; the summary
fields are `k_qtst_s`, `kappa`, `rate_s`, `rate_estimate_s`, `rate_standard_error_s`,
`rate_percentile_95_ci_s` and `rate_uncertainty`.

### `surface_rate`: surfaces with fixed layers

```bash
pyrpmd validate examples/surface/diffusion.inp --check-pes
pyrpmd run examples/surface/diffusion.inp --output runs/manual-surface-rate
pyrpmd resume runs/manual-surface-rate

pyrpmd run examples/surface/fes2d.inp --output runs/manual-surface-fes
pyrpmd analyze runs/manual-surface-fes --recompute --bins 24 12 --bootstrap 40
```

The first example, `diffusion.inp`, runs **`surface_rate`** (not the MSD `diffusion` task): the rate
of leaving one local well to the right on a separable analytic surface. It takes about 1–2 minutes
on one core; the 2D surface FES takes about 1 minute.

`system.fixed_atoms` freezes every bead of the listed atoms with zero momentum; they still enter the
PES and act on the mobile atoms. The rate coordinate is one centroid `cartesian`, `surface_height`,
`surface_lateral`, `distance` or smooth `periodic_distance` CV, subject to the image-switch restriction
above. `surface.reactant_basin`, `surface.dividing_surface` and
`surface.product_side="higher"/"lower"` define the basin and the product side; the whole basin lies
on the reactant side. Give these lengths with units.

Outputs are `surface_pmf.csv` / `.npz` / `.json`, `recrossing.csv`, `umbrella/` and the summary,
with the rate in `rate_s_inv`, `rate_estimate_s_inv`, `rate_standard_error_s_inv` and
`rate_percentile_95_ci_s_inv` (same unit as `site_rate`, different field names). If basin support
fails in some bootstrap draws, the joint interval is left empty.

For an explicitly defined second-order surface normalization, set
`surface.normalization.reaction_order=2` and `reference_adsorbates` to the number of co-reactants
in the reference cell. The conversion is `k2 = k_cell * area / reference_adsorbates`, in `cm²/s`.
Area is inferred only when exactly two lattice vectors are periodic, using their cross product;
their indices need not be 0 and 1. For a fully periodic cell, a wire or a finite cluster, supply
`surface.normalization.area` with area units to identify the intended surface. This conversion
does not account for coverage-dependent interactions or turn a local rate into a total turnover
frequency. See the
[surface normalization definition](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/surface-catalysis.md).

Surface `fes` supports adaptive umbrellas; `surface_rate` uses fixed windows. To run a real ASE
calculator, install ASE:

```bash
python -m pip install -e '.[ase]'
pyrpmd validate examples/surface/ase_emt.py --check-pes
pyrpmd run examples/surface/ase_emt.py --output runs/manual-surface-emt
```

This input runs a Cu adatom on Cu(111) with ASE EMT, exercising the calculator, a skewed cell and
fixed layers. Models and sources are described in the
[materials notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials.md) *(Chinese)*,
the [surface notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/surface-catalysis.md) *(Chinese)*
and the [Ni–H materials pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md).

## From a runnable slab to a research calculation

The Cu/EMT, Ni–H/EAM and CO/O/Pt(111) examples exercise different models and workflows; their short
pilots do not establish a converged catalytic rate or diffusion coefficient. Use the following
sequence for a new material:

1. Relax and identify the slab, adsorption sites, fixed layers, coverage and electronic state with
   the intended PES. Declare all three lattice vectors and each PBC flag. Start a molecule or
   adsorbate group in one consistent unwrapped image before using a COM coordinate; a nearest-image
   pair distance does not reconstruct a whole group.
2. Check the adapter's energy/force agreement with the underlying calculator, finite-difference
   forces and lattice-translation invariance. For variable-cell sampling also compare the virial
   with finite strains. For a metal with electronic smearing, verify that the reported energy is
   the thermodynamic potential differentiated by its forces.
3. Converge equilibrium preparation, bead count and timestep at each temperature and isotope.
   Check the number of mobile degrees of freedom, ring radii and independent-replica statistics.
   Equilibrium PIMD sampling time is not a physical transport trajectory.
4. For diffusion, retain unwrapped centroids, choose a disjoint lattice reference group and test
   the fit window and production length. `axes` denotes Cartesian axes; rotate a tilted slab into
   the desired analysis frame or analyze its full tensor. For a local rate, establish connected
   basin/transition-state sampling and an independently supported recrossing plateau.
5. Vary lateral supercell, slab thickness, vacuum, coverage and frozen-layer choice. Report model
   accuracy, finite-size effects and statistical/numerical convergence separately. A single directed
   first-order rate does not determine total escape, sticking, turnover frequency or conductivity.

The [periodic-materials audit and TODO list](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/periodic-materials-audit.md)
records implementation findings, open-source comparisons and remaining validation work.
