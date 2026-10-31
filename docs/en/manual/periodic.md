# Periodic materials, diffusion and surface reactions

A periodic structure describes the geometric boundaries, the potential model determines the
material interactions, and the workflow defines the physical quantity to estimate. A fixed cell
does not by itself make diffusion measurable, and a surface free-energy barrier does not directly
give a diffusion coefficient or a catalytic rate. This chapter covers equilibrium PIMD,
thermostat-free RPMD transport and basin-normalized local rates in turn.

## Choose the task and the boundaries first

| Task | Purpose | Periodic cell | `fixed_atoms` | Core result |
| --- | --- | --- | --- | --- |
| `pimd` | Equilibrium quantum statistics | Supported | Currently rejected | Energy estimators, radius of gyration |
| `diffusion` | Continuous centroid trajectories and transport analysis | Supported | Currently rejected | MSD, VACF, D and diagnostics |
| `site_rate` | A selected local exit in a fully mobile material | Supported | Currently rejected | Basin rate along one constant-gradient CV |
| `fes` | 1D/2D geometric free energy | Supported | Supported | Position-density FES, overlap, mask |
| `surface_rate` | A local surface exit, possibly with fixed layers | Supported, finite clusters too | Supported | First-order rate with the mass metric of the mobile degrees of freedom |

The materials energy and transport estimators currently assume that all atoms are mobile, so the
integrated program explicitly rejects fixed atoms in `pimd`, `diffusion` and `site_rate`. To fix
bottom layers, use surface `fes` / `surface_rate`. Gas-phase `rate`, `drpmd`, `network` and the
current `metadynamics` reject periodic or fixed-atom systems.

All mobile atoms use the same bead number; there is no mixed-resolution entry such as "many beads
for the adsorbate, one bead for the mobile substrate". NPT/variable cells, electronic friction,
nonadiabatic hopping, collective conductivity and automatic kMC construction are not available.

## Cells, continuous coordinates and isotopes

A structure table looks like this; replace `slab.xyz` with the actual file of your system:

```toml
[system]
geometry = "slab.xyz"
geometry_unit = "angstrom"
cell = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
```

Each cell row is a lattice vector and Cartesian positions satisfy `r = s @ cell`. The matrix must be
finite, invertible and well conditioned; the non-periodic vacuum direction of a slab still needs a
non-zero vector. The PBC flags refer to lattice vectors, not necessarily to the Cartesian x/y/z
axes. If a cell is given without pbc, all three directions default to periodic; always declare pbc
explicitly.

`cell_unit` defaults to the unit of the geometry input: inline arrays default to bohr and XYZ files
to angstrom. Stating `geometry_unit` and `cell_unit` keeps the input meaning stable when the
geometry source changes.

The propagated `q[P,N,3]` and the exported centroid trajectories stay continuous and unwrapped.
Do not wrap each bead into the cell after every step: that changes the ring springs, the centroid
and the MSD. Periodic potentials and explicit periodic CVs handle images; the nearest image of a
skewed cell is not simply per-fractional-coordinate rounding. This path-integral representation
uses closed rings of distinguishable nuclei and samples neither nuclear exchange nor non-zero
winding topologies.

H/D/T all map to hydrogen on the electronic PES; they differ only in nuclear mass. Use
`symbols=["D"]` or explicit `isotopes` / `masses_amu`. Replacing the mass does not require
redefining the same Born–Oppenheimer PES, but the bead number, equilibration time and the
convergence of the results must be checked separately.

### Choose periodic CVs explicitly

| CV kind | Geometric meaning | Suitable use |
| --- | --- | --- |
| `distance`, `bond_difference`, `angle` | Ordinary geometry on continuous unwrapped coordinates, no automatic minimum image | Existing molecular definitions, explicitly unwrapped geometry |
| `periodic_distance`, `periodic_bond_difference`, `periodic_angle` | Nearest-image geometry using the system cell/PBC | Periodic bond lengths, angles and FES |
| `projection`, `fractional` | Continuous centroid projection or fractional coordinate, no modulo | Directed site hops that distinguish neighboring cells |
| `surface_height` | Projection of an adsorbate-group COM onto the surface normal | Height free energies, linear surface dividing planes |
| `surface_lateral` | Continuous unwrapped COM projection on a given direction | In-plane migration free energy and selected exits |

`periodic_bond_difference` also accepts the alias `periodic_difference`. An ordinary geometric CV
never silently changes its definition because of the task type or fixed layers. A nearest-image CV
can have derivative branches where the image switches; check that the chosen range suits smooth
biasing and sampling. The constant mass-metric constraints of `site_rate` / `surface_rate` accept
only their explicitly supported linear centroid CVs; an arbitrary periodic distance or proton
transfer coordinate is not an implemented constrained rate.

## Periodic potential interfaces

ASE receives the true cell, PBC and per-bead coordinates and converts units through the calculator.
If the calculator supports stress, the adapter can return the configurational virial
`-volume * stress` in hartree. `PESResult.virial[B,3,3]` excludes momenta and ring springs and is
not automatically a quantum pressure estimator. Chunked execution keeps this optional output.

A Python periodic plugin supports two explicit contracts:

- Declare `metadata["periodic"] = True` and receive the current cell via
  `evaluate(q, *, cell, pbc)`; coordinates and cell use the length unit declared by the provider.
- Declare `metadata["supports_pbc"] = True` and implement `bind_system(system)` to bind the
  geometry before `evaluate(q)`. The bound System uses atomic units; if evaluate works in
  angstrom/eV, the plugin must convert the stored cell itself.

A bound periodic model must still agree with the current System. Writing a capability flag while
ignoring the cell/PBC does not satisfy the physical contract. For any new calculator, run
`validate --check-pes` first, then do independent force and periodic-translation checks.

i-PI sends the true cell and its inverse in POSDATA, using the protocol's column-vector convention.
POSDATA does not carry partial PBC separately; a slab needs `potential.params.init.pbc` matching the
system, and the server must implement this INIT convention. A large vacuum gap alone does not
replace partial-PBC semantics. The periodic C ABI v1, RPMDrate/f2py and Torch entries currently do
not support the cell contract; connect such models through a suitable ASE/Python/i-PI interface.

Periodic/fixed-atom systems use a compatible Python time-step loop and can use compiled
deterministic integrator substeps; use `loop_backend="auto"` or `"python"`. This does not mean that
the whole periodic propagation runs in the fused native loop or resides on a GPU.

## PIMD: equilibrium properties

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/manual-pimd
pyrpmd analyze runs/manual-pimd
pyrpmd resume runs/manual-pimd
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/pimd.inp) uses a
periodic cosine analytic potential. `sampling.equilibration` is the initial equilibration time,
`production` the subsequent equilibrium sampling time, `interval` the save interval and `replicas`
the number of independent random streams.

| `dynamics.integrator` | `sampling.integrator` for PILE-L | Explicit Andersen name |
| --- | --- | --- |
| `cayley-bcb` | `bcocb` | `cayley-bcb-andersen` |
| `normal-mode-bab` | `baoab` | `normal-mode-bab-andersen` |
| `velocity-verlet` | Not supported | `velocity-verlet-andersen` |

With `sampling.thermostat="pile-l"`, `gamma_centroid="0.01 fs^-1"` sets the centroid friction and
the internal modes are thermalized per normal mode. The sampling and deterministic production time
steps of materials tasks must agree. PILE-L `auto` uses a C++ free-ring half-step–OU–half-step
kernel for `P<=8` or power-of-two bead numbers, while Python keeps the potential calls and NumPy
random numbers; larger non-power-of-two bead numbers keep the NumPy reference path. It does not
accept a forced `loop_backend="native"` request for a fully native loop.

The summary's `estimators` contain `potential_energy`, `kinetic_primitive`,
`kinetic_centroid_virial`, `internal_energy_centroid_virial` and `ring_polymer_hamiltonian`. The
first four describe physical energy estimators; the last one is a ring-propagation diagnostic.
Each has `mean` and `standard_error_between_replicas`; with a single replica the between-replica
error is `null`. `radius_of_gyration_squared_bohr2` gives the mean squared ring radius of gyration
of each atom.

`samples/materials_XXXX.npz` exports `positions`, `velocities`, `times`, `observables` and
`radii_gyration_squared`, where positions and velocities are continuous centroid quantities.
`checkpoints/materials_XXXX.npz` is the authoritative restart state. PIMD sampling time must not be
used directly to compute diffusion or reaction lifetimes.

## Diffusion: from continuous trajectories to MSD and D

`task="diffusion"` stores t=0 after thermalization, then runs RPMD production without a thermostat,
umbrella or metadynamics. It saves the centroid positions and velocities of all atoms so that the
measured atoms, reference frame and fitting settings can be changed offline.

```bash
pyrpmd run examples/materials/diffusion.inp --output runs/manual-diffusion
pyrpmd analyze runs/manual-diffusion --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40 --bootstrap 100
pyrpmd resume runs/manual-diffusion
```

This [free-particle negative control](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/diffusion.inp)
should be recognized as ballistic and keep `diffusion_m2_s=null`. Finishing successfully without
publishing D is exactly the expected behavior of this example.

| `diffusion` field | Meaning and unit |
| --- | --- |
| `atom_indices` | 0-based indices of the measured atoms; H/D/T are selected if omitted |
| `axes` | Measured Cartesian components, e.g. `[0,1]` for xy; not fractional lattice axes |
| `reference_atoms` | Their mass-weighted COM position/velocity is subtracted every frame; must not overlap the measured atoms |
| `max_lag` | Maximum lag in saved-frame intervals |
| `fit_range` | Physical lag-time range of the fit; must lie within the saved range |
| `bootstrap` | Number of resamplings over independent replicas; 0 means no such error estimate |
| `block_length` | Saved frames per block for single-replica blocking; at least `4*max_lag`, with enough independent blocks |

For d analyzed components the displacement and diffusion tensors are

$$
M_{\alpha\beta}(t)=\langle\Delta\bar r_\alpha(t)\Delta\bar r_\beta(t)\rangle,
\qquad D_{\alpha\beta}=\frac12\frac{dM_{\alpha\beta}}{dt},
\qquad D=\frac{\operatorname{tr}D_{\alpha\beta}}d.
$$

The in-plane 2D MSD slope is therefore divided by 4 and the 3D slope by 6. The program also stores
the symmetric VACF tensor and its finite-time integral for comparison; it does not certify a VACF
integral plateau automatically. A fixed material reference frame removes overall lattice drift,
but subtracting a fitted drift from each diffusing atom does not manufacture linear diffusion.

Look at `analysis_status` and `results/diffusion.json` first:

| Status | How to read it |
| --- | --- |
| `insufficient_data` | Too few frames, fit points or time origins |
| `no_observed_motion` | No motion seen in this record; does not prove the true D is zero |
| `ballistic_or_drifting` | Still ballistic, or drifting |
| `non_diffusive_window` | The fitting window is not an approximately linear diffusive regime |
| `indefinite_diffusion_tensor` | The fitted tensor is not positive semidefinite |
| `unstable_fit_window` | The slopes of the two halves of the window disagree |
| `diffusive_regime_candidate` | Passed heuristic screening of a finite record; still needs a physical convergence study |

Non-candidate statuses still save `raw_fit`, MSD and VACF, while the accepted top-level D/tensor
stays `null`. Many overlapping time origins of one replica are not many independent experiments,
and the OLS residual of the fit does not replace a sampling error. Compare the stability of the
result after changing the fitting time, production length, number of independent replicas and
system size.

## The common physical definition of site and surface rates

Both local-rate workflows use, for a linear centroid coordinate s with a constant mass metric, the
dividing-surface value and basin probability of the same position density:

$$
k_{\mathrm{QTST}}=
\sqrt{\frac{g^TM^{-1}g}{2\pi\beta}}
\frac{\rho(s^\ddagger)}{\int_A\rho(s)\,ds},
\qquad k(t)=k_{\mathrm{QTST}}\kappa(t).
$$

g is the gradient of the coordinate with respect to the atomic centroids and A the explicit
reactant basin. `surface_rate` removes the fixed-atom components from the mass metric. The whole
basin and the dividing surface need connected, unmasked sampling support; the density and the basin
integral are computed together within each bootstrap draw to keep their covariance.

Both outputs are first-order rates of the chosen directed exit in `s^-1`, without gas volume,
surface area, pressure, coverage or site-degeneracy factors. One exit rate is not an automatically
summed total escape rate and does not convert to a macroscopic D by itself; that also needs the hop
length, number of neighbors and correlations between successive hops.

### `site_rate`: material sites

```bash
pyrpmd prepare examples/materials/site_rate.inp --pilot
pyrpmd run examples/materials/site_rate.inp --output runs/manual-site-rate
pyrpmd analyze runs/manual-site-rate
```

The [example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/site_rate.inp)
uses an unwrapped `fractional` centroid coordinate. Supported CVs are `cartesian`, `projection` and
`fractional`; the latter two may refer to a group of lattice atoms. `site_rate.cv` names the same
umbrella CV, `basin=[lo,hi]` defines a finite basin, `direction=1` requires the dividing surface at
hi and `direction=-1` at lo.

Windows and the analysis range must cover the whole basin and provide the needed support on the far
side of the exit. Windows and constrained parents use Andersen, ±p children are unthermostatted,
and a negative finite-sample κ is not clipped to zero. Results include `site_qtst.json` / `.npz`,
`pmf/fes.csv` and `recrossing.csv`; the summary fields are `k_qtst_s`, `kappa`, `rate_s`,
`rate_estimate_s`, `rate_standard_error_s`, `rate_percentile_95_ci_s` and `rate_uncertainty`.

### `surface_rate`: surfaces with fixed layers

```bash
pyrpmd validate examples/surface/diffusion.inp --check-pes
pyrpmd run examples/surface/diffusion.inp --output runs/manual-surface-rate
pyrpmd resume runs/manual-surface-rate

pyrpmd run examples/surface/fes2d.inp --output runs/manual-surface-fes
pyrpmd analyze runs/manual-surface-fes --recompute --bins 24 12 --bootstrap 40
```

Although the example is named `diffusion.inp`, its task is **`surface_rate`**: it computes the
right-hand exit of one local well. Do not confuse it with the MSD `diffusion` task. The analytic
potential is a separable teaching model and cannot validate real thermal site hopping or a material
diffusion coefficient. These two surface examples are more expensive than the Quickstart: on a
single core `surface_rate` takes about 1–2 minutes and the 2D surface FES about 1 minute.

`system.fixed_atoms` freezes all beads of the bottom layers with zero momenta; these atoms still
enter the real PES and act on the mobile atoms. One centroid `cartesian`, `surface_height` or
`surface_lateral` coordinate is supported. `surface.reactant_basin`, `surface.dividing_surface` and
`surface.product_side="higher"/"lower"` define the normalization region and the product side; the
whole basin must be on the reactant side of the dividing surface. Strings with units are
recommended for these lengths.

The surface outputs are `surface_pmf.csv` / `.npz` / `.json`, `recrossing.csv`, `umbrella/` and
the summary. Surface rate fields are `rate_s_inv`, `rate_estimate_s_inv`,
`rate_standard_error_s_inv` and `rate_percentile_95_ci_s_inv`; the names differ from `site_rate`
but the physical unit is the same. If basin support fails in a bootstrap draw, the main joint
interval may be left empty; conditional statistics over the successful draws cannot pose as the
full error.

Surface `fes` supports adaptive umbrellas; `surface_rate` currently requires fixed declared
windows. To test a real ASE connection, install ASE and run:

```bash
python -m pip install -e '.[ase]'
pyrpmd validate examples/surface/ase_emt.py --check-pes
pyrpmd run examples/surface/ase_emt.py --output runs/manual-surface-emt
```

This input uses Cu(111) / a Cu adatom with ASE EMT to validate the calculator, skewed cell and
fixed-layer connection; it does not show that EMT is suitable for arbitrary catalytic reactions.
More model details and sources are in the
[materials notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials.md) *(Chinese)*,
the [surface notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/surface-catalysis.md) *(Chinese)*
and the [Ni–H materials pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md).
