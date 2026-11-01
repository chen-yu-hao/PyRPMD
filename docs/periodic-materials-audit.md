# Periodic materials and surface catalysis: implementation audit and improvements

**English** · [中文](zh/periodic-materials-audit.md)

Audit date: **2026-10-31**. Code baseline:
[`c4fa743f552c1d71db35fab791a2e63409bc1c3e`](https://github.com/chen-yu-hao/PyRPMD/tree/c4fa743f552c1d71db35fab791a2e63409bc1c3e).
This record is based on that baseline's implementation, tests and English/Chinese
guides. It distinguishes implemented capabilities, reproducible defects and extensions
requiring research validation. Numerical unit tests, short pilots with real potentials,
and successful program completion are not evidence of converged materials physics.
This English translation preserves the historical audit's dates, test counts and scope.

## Conclusions and audit scope

PyRPMD already provides a foundation for periodic-materials research: skewed cells,
partial PBC, continuously unwrapped ring coordinates, frozen substrates, periodic
PES interfaces, PIMD, RPMD diffusion, free-energy sampling and basin-normalized local
rates. The audit prioritized specific problems affecting surface geometry, energy
consistency and rate units, and brought outdated guides into agreement with the code.
Real catalytic mechanisms, finite-size effects and sampling convergence still require
material-specific evidence.

| Layer | Code and validation entry points | Existing capabilities and main boundaries |
| --- | --- | --- |
| Input and system | `input.py`, `system.py`, `units.py`; `test_materials_input_cv.py`, `test_periodic_materials.py` | Atomic units, row lattice vectors, strict PBC/fixed-atom checks; gas-phase `rate`/`drpmd` reject periodic systems |
| Periodic geometry and CVs | `geometry.py`, `cv.py`, `cv_periodic.py`; `test_periodic_geometry_cache.py`, `test_surface_geometry.py` | Exact MIC for general skewed cells, gradients, fixed-lattice caching; ordinary molecular CVs do not automatically become periodic |
| Propagation and ensembles | `dynamics.py`, `barostat.py`, `native_ensemble.py`, `native/`; `test_surface_dynamics.py`, `test_barostat.py`, `test_native_ensembles.py` | Fixed layers retain reference positions and zero momenta on every bead; default BZP and separate native flexible-cell routes exist; slab/fixed-layer NPT is unsupported |
| Potentials and execution | `potentials/`, `parallel/execution.py`; `test_cabi_v2.py`, `test_torch_periodic.py`, `test_potential_adapters.py`, `test_ipi_clients.py` | Periodic C ABI v2, ASE/ML, Python/Torch, tblite and i-PI; cell-aware evaluation does not automatically supply the `set_cell` needed for NPT |
| Workflows and restart | `workflows/materials.py`, `native_pimd.py`, `fes.py`, `metadynamics.py`, `site_rate.py`, `surface.py`; stream/native/workflow tests | Streamed trajectories, atomic commits and integrity checks; native and legacy checkpoints are not interchangeable |
| Observables and rates | `analysis/diffusion.py`, `surface.py`, `projection.py`, `recrossing.py`; diffusion/surface/statistics tests | Self-diffusion tensors, rejection of nondiffusive fit windows, basin normalization and correlated-block errors; complete rate uncertainty from projected source FES remains unresolved |
| Related extensions | `electronic_friction.py`, `electrochemistry/`, `potentials/contraction.py`, `multitime.py`, `transition_interface.py` | EF-RPMD, local electrochemistry, RPC/MTS and TIS interfaces have explicit scopes; they are not all connected to fixed-layer surface rates |
| Guides and examples | English/Chinese READMEs, `docs/manual/`, `docs/zh/manual/`, materials/surface guides, `examples/materials/`, `examples/surface/`, `benchmarks/materials/` | Extensive examples and mathematical conventions, with documentation drift around periodic potentials, native engines and nonlinear rates |

This was an implementation and usage-path audit with targeted regression validation.
It did not rerun every optional electronic-structure, GPU, multi-node or long-production
benchmark, or rederive all nonadiabatic/quantum-statistical extensions.

## Improvements implemented in the audited change

### 1. Minimum images must not depend on the representation of an equivalent lattice

At the baseline, nearest-image search for
`cell=[[1000,1,0],[1,0,0],[0,0,15]]`, `pbc=[true,true,false]` could hit the candidate
limit and report an excessively skewed cell, while swapping the two periodic basis
vectors succeeded. Both represent the same two-dimensional lattice. The problem was
the sphere-search range in an unreduced basis; an equivalent valid representation
must not be treated as a physical error.

`geometry.py` now performs integer size/Lovasz reduction in the **periodic subspace**
before the existing exact QR sphere search. Original cell/PBC, coordinates and the
physical lattice are unchanged. Beads are not wrapped, and exact search is not replaced
by simple fractional-coordinate rounding. Tests cover all eight PBC combinations,
equivalent integer basis changes, exhaustive skew-cell references, periodic-CV
gradients and independent ASE `find_mic` comparisons. ASE's general MIC implementation
also uses lattice reduction. This change uses independently implemented preconditioning,
without copying ASE source. [S1]

### 2. Electronic-smearing calculations in metals need force-consistent energies

The baseline ASE adapter requested ordinary `energy` by default. Some electronic-
structure calculators return a zero-temperature extrapolated energy for that property,
while nuclear forces and stress differentiate the finite-electronic-temperature free
energy. Combining them affects energy-conservation diagnostics and free-energy weights.
ASE explicitly distinguishes these two energy interfaces. [S2]

The adapter now prefers `get_potential_energy(force_consistent=True)`. It falls back
to ordinary energy only if the calculator does not declare `free_energy` and raises
`PropertyNotImplementedError`. Failure while evaluating a declared free energy still
propagates as an error. This electronic free energy is not a nuclear PMF, and electronic
temperature does not automatically equal the nuclear thermostat temperature.
Old smearing calculations using extrapolated energy should be rerun in new output
directories; do not append to their energy histories or reuse their weights. Existing
energy conventions for classical calculators remain applicable.

### 3. A failed periodic Python-model bind must preserve wrapper state

When rebinding a system, the Python adapter first validates candidate cell/PBC and
calls the external model's binding interface. Only after every operation succeeds
does it commit the wrapper's cell/PBC and call mode. Invalid cells, invalid boundaries
and callback exceptions no longer leave partially updated wrapper state. This guarantees
the PyRPMD wrapper's commit boundary; it does not undo arbitrary side effects already
performed inside an external Python object.

### 4. Surface area must follow the actual periodic plane

The baseline two-dimensional rate normalization assumed the in-plane vectors were
always `cell[0]` and `cell[1]`. It therefore rejected valid `[true,false,true]` and
`[false,true,true]` slabs and could assign an area to a fully periodic three-dimensional
cell without an explicitly chosen surface.

Area is now inferred from a cross product only when exactly two lattice vectors are
periodic. Other cases require explicit `area`. Outputs record the area source and the
periodic-vector indices used. Conversion remains `k2=k_cell*A/n`, in `cm²/s`; this
introduces neither coverage interactions nor a new rate-statistics model. Previously
ambiguous fully periodic inputs must explicitly state the surface area of interest.

### 5. Include the ASE energy convention in run identity

Changing energy evaluation while allowing unconditional restart would mix two PES
energy conventions in one output directory. The new ASE contract therefore enters
run-identity validation. Restart and reanalysis of old ASE-backed checkpoints are
rejected, including classical-calculator jobs whose energy equivalence cannot be
proved offline. Existing saved summaries remain readable, and new-convention runs
restart normally. This conservative compatibility boundary protects PMF weights,
energy records and checkpoints; it does not mean every old classical trajectory is
physically wrong.

The static fingerprint marks the contract as
`potential_energy_convention="ase-force-consistent-v1"` for explicit `ase`, `mace`,
`nequip` and `fairchem` providers. This is not a new user input parameter. An ASE
calculator hidden inside an arbitrary Python factory cannot be detected from the
static provider name; record its source and energy convention separately.

### 6. Public preparation and validation for native PIMD

The baseline native `resolve_materials` returned the native engine's options, but
`runner.prepare` still directly read legacy-only `barostat`/`storage` fields. Valid
native PIMD inputs therefore raised `KeyError` during `prepare` or
`validate --check-pes`, making the documented validation commands unusable.

Native and legacy preparation results are now handled separately, with an actual
lightweight native-engine preflight. Native barostat parameters are not sent to the
legacy BZP path. Regressions cover public native NVT/NPT examples and invalid inputs.
For ordinary adiabatic PES providers, `validate --check-pes` still checks interface,
evaluation and finite values. It does not replace finite-difference force, lattice-
translation or finite-strain scientific validation.

### 7. Align user guides with existing implementation

The English/Chinese periodic guides and materials/surface notes now make clear that:

- C ABI v2, explicitly periodic Torch and named ML interfaces already exist. C ABI v1,
  f2py and similar interfaces retain molecular boundaries.
- The default Python materials workflow and explicit native PIMD are separate engines.
  Native MTTK/RGB is implemented, but that does not establish fixed-layer slab NPT.
- Nonlinear `site_rate` already has a constrained-parent ensemble metric correction,
  and `surface_rate` supports a continuous-distance subset. It is inaccurate to call
  all nonlinear rates unimplemented.
- EF-RPMD, constant-potential effective PES models and local HER postprocessing have
  modules with defined physical limitations; older descriptions of them as entirely
  absent obscure those boundaries.
- A practical sequence now connects structure preparation, PES contracts,
  thermalization, rates/diffusion and finite-size convergence.

## Comparisons with open-source implementations

| Reference | Useful practice | PyRPMD status and direction |
| --- | --- | --- |
| ASE geometry / Atoms / socketio [S1–S3] | Reduce the lattice for general MIC; distinguish force-consistent energy; exchange cell and physical virial explicitly | This change improves MIC and the energy contract; expand real-calculator force/strain checks rather than infer model applicability from an adapter name |
| i-PI [S4] | Decouple nuclear propagation from forces; constant-pressure and flexible cells; explicit RPC/MTS force splitting | These core capabilities exist in PyRPMD; connect interfaces and validate materials instead of adding duplicate algorithm names |
| LAMMPS PIMD / i-PI client [S5] | Distinguish equilibrium PIMD from dynamics; retain image flags; define coordinates/normal-mode data and variable-cell client responsibilities | Preserve unwrapped q/centroid conventions; validate real force-engine interoperability and matched ensembles rather than demand trajectory-by-trajectory RNG identity |
| PLUMED 2.10 [S6] | Distinguish nearest-image distances, continuous components and molecular/COM image reconstruction | Periodic pair CVs exist, but users must ensure initially consistent group images; topology helpers can assist without changing propagation coordinates |
| GPAW [S8] | Specify real-space/plane-wave slab electrostatic boundaries and dipole-layer corrections | PyRPMD passes cell/PBC; electronic boundaries and long-range corrections remain PES settings; fully periodic slabs require explicit normals and normalization areas |
| OpenPathSampling [S7] | Multiply TIS crossing probabilities by an independent effective flux; preserve full paths and independent-chain statistics | PyRPMD TIS already separates these roles; validate surface channels/long-time dynamics through the existing API, rather than convert one MetaD crossing time directly into a rate |

These projects provide algorithm and interface comparisons, not evidence of numerical
equivalence for arbitrary PES models, temperatures, bead counts or constraints.

## TODO: priorities and acceptance criteria

`P0` affects numerical/unit correctness directly. `P1` concerns research reliability
and integration of periodic capabilities. `P2` concerns cost, usability and extensions.
Checked entries belong to the historical audited change. Unchecked entries remain
future work and must not be counted as delivered scientific results.

- [x] **P0 — MIC reliability under equivalent lattice bases.** Reduce the periodic
  subspace; pass independent nearest-image/gradient comparisons while preserving
  general skew-cell and partial-PBC behavior.
- [x] **P0 — ASE energy/force consistency.** Strictly evaluate declared free energy;
  cover electronic smearing, classical fallback and failure propagation.
- [x] **P0 — ASE energy-convention identity.** Prevent restart/recomputation from
  mixing old energies; retain new-convention restart and read-only saved summaries.
- [x] **P0 — Python binding commit boundary.** Preserve original cell, PBC and call
  mode after invalid input or callback failure.
- [x] **P0 — Surface areas and units.** Cover all three two-periodic-axis choices,
  row permutations, explicit areas and rejection of fully periodic ambiguity,
  preserving the default first-order rate behavior.
- [x] **P0 — Public native-PIMD preflight.** Resolve and validate according to the
  actual engine in `prepare` and `validate --check-pes`; cover native NVT/NPT without
  reading nonexistent legacy fields.
- [x] **P1 — Synchronize English/Chinese guides.** Distinguish legacy/native engines,
  existing features and unvalidated extensions; explain compatibility changes and
  the research preparation sequence.
- [ ] **P1 — Physical validation matrix for real periodic materials.** Scan H/D,
  beads, timestep, equilibration, independent seeds, supercell size, layer count,
  vacuum and coverage on source-pinned PES models. Electronic-structure references
  additionally require converged k points, cutoff, smearing and SCF force accuracy,
  with explicit electrostatic/dipole treatment of charged or asymmetric slabs; PBC
  flags do not replace those choices. [S8] Acceptance: report statistical and
  systematic errors separately, stable long-time diffusion windows, connected PMFs
  and recrossing plateaus, and failed/unconverged samples.
- [ ] **P1 — Joint source-FES → surface-rate uncertainty.** `surface.free_energy`
  remains `projected_pmf_pilot`, with formal `rate_s_inv` unset. Acceptance: jointly
  resample correlated source blocks, conditional integration domains and parent/child
  pairs; propagate support failures and covariances, and verify coverage on known models.
- [ ] **P1 — General nonlinear rates with fixed layers.** Reuse the all-mobile
  `site_rate` metric framework; setting frozen velocities to zero is insufficient.
  Acceptance: consistent mobile-subspace mass metrics, constrained-surface measures,
  flux weights and joint uncertainty, with coordinate-reparameterization and analytic
  conditional-distribution tests.
- [ ] **P1 — Complete variable-cell contracts for periodic providers.** Cell-aware
  Torch/C ABI v2/tblite/DeePMD evaluation does not automatically provide the dynamic
  `set_cell` required by legacy NPT; retain current guards. Acceptance: explicit
  update interfaces, finite-strain virial checks, failure rollback, checkpoint
  restoration and correct provider neighbor-list updates.
- [ ] **P1 — Explicit molecular and bead-image preparation.** Use PLUMED's ordered-
  topology reconstruction ideas to add optional group-image initialization and ring-
  winding diagnostics. Acceptance: consistent cross-boundary COMs/bonds, invariance
  under atom reordering and lattice translations, and continuously unwrapped q during
  dynamics without bead-by-bead wrapping.
- [ ] **P2 — Connect existing RPC/MTS to materials workflows.** Python APIs already
  support force splitting and contraction, but complete workflow, pressure/virial,
  bias/constraint and dedicated checkpoint integration is missing. Acceptance: define
  `Vfast + Vslow`, converge contraction beads and outer timestep independently, report
  target-PES call counts and total runtime, and compare full-bead observables. [S4]
- [ ] **P2 — More efficient real periodic PES execution.** The C ABI v2 batch interface
  and `native_periodic` whole-trajectory ABI differ; ASE models evaluate frame by
  frame. Acceptance: model-specific cell-aware native batches/device-resident paths,
  preserving physical virials, state errors and memory ownership, with performance
  across N, P and PES costs.
- [ ] **P2 — Transport and structure analysis for arbitrary surface directions.**
  `axes` currently selects Cartesian components; long-trajectory offline analysis
  still needs selected atom or scalar histories. Acceptance: diffusion-tensor
  covariance in explicit orthonormal analysis bases, correct in-plane Einstein factors,
  optional structural statistics and bounded-memory correlations. Self-D must not be
  presented as collective conductivity.
- [ ] **P2 — Equivalent-lattice tests for periodic-distance dividing surfaces.**
  `surface.py` currently uses half the smallest singular value of the original periodic
  basis as its safety radius. This avoids image switching but may reject a valid
  surface after an equivalent integer shear. The MIC improvement did not change this
  conservative rate limit. Acceptance: a validated shortest-lattice-vector or reduced-
  basis bound giving equivalent representations consistent usable ranges while
  preserving image-seam protection.
- [ ] **P2 — Surface networks, coverage and incidence.** Existing local HER
  postprocessing does not replace general adsorption/desorption/reaction kMC. Gas
  incidence and sticking require new initial states and normalization. Acceptance:
  explicit standard states, site/channel counting, detailed balance and coverage
  interactions, without counting equivalent paths already included in a FES twice.

Do not add an ad hoc mixed model with many beads only for H and one bead for metals
as part of these fixes. All mobile atoms currently share P. Arbitrarily assigning one
bead to selected atoms changes the Hamiltonian and statistical measure. Validate
existing controlled RPC/MTS approximations first, then derive and validate regional
contraction or a classical substrate separately.

## Validation record

The audited change reran the full suite and four real Ni–H EAM pilots after freezing
the source. Commands, environment, source hashes and raw outputs are in the
[validation evidence](validation/periodic-materials-2026-10-31/README.md). Calculations
ran in the modified pre-commit worktree. The archive records the baseline and per-file
SHA256 hashes explicitly; it does not present the baseline as containing the fixes.

| Historical audit check | Result |
| --- | --- |
| New regressions | 80: MIC/CV 14, ASE energy/strain 7, energy identity/restart 28, failed Python binding 3, surface area 13, native preflight 15 |
| Final full pytest | **3,885 passed, 114 skipped, 264 subtests passed**, zero failures; 471.94 s |
| Real Ni–H EAM | Four H/D × P=1/4 interface and finite-timestep checks passed; the 16 recorded source files did not change during calculation |
| English/Chinese Sphinx | Four strict HTML/singlehtml builds with `-W --keep-going` passed without warnings |

The full suite enabled compiled native extensions, ASE and source-pinned EAM.
Optional electronic-structure, PyTorch, PLUMED, external PES and MPI checks whose
environment requirements were unmet skipped under their own conditions. Exact reasons
remain in the raw logs. GPU and multi-node validation was not performed.

The archived [raw EAM report](validation/periodic-materials-2026-10-31/eam-validation.json)
uses 27 Ni + H/D, 600 K, a 0.1 fs timestep, 64 equilibration steps and 160 unthermostatted
production steps, saving 81 frames per case:

| Isotope | Beads | Maximum energy drift per bead (hartree) | Mean squared ring radius of gyration (bohr²) | Diffusion diagnostic |
| --- | ---: | ---: | ---: | --- |
| H | 1 | 1.6704e-7 | 0 | `ballistic_or_drifting` |
| D | 1 | 1.1438e-7 | 0 | `ballistic_or_drifting` |
| H | 4 | 1.8282e-7 | 0.039323 | `ballistic_or_drifting` |
| D | 4 | 1.6073e-7 | 0.019857 | `ballistic_or_drifting` |

The largest absolute error in the selected finite-difference force checks was
`4.4348e-10 eV/angstrom`. This validates interface and finite-timestep implementation.
None of the four cases has an acceptable diffusion coefficient:
`physical_converged=false` and `equilibration_converged=false`. Ring sizes from these
short trajectories cannot be interpreted as equilibrium isotope effects. See the
[Ni–H EAM pilot](../benchmarks/materials/README.md) for fixed source and SHA256.
Potential files were not redistributed in this change.

Other existing materials evidence has its own limits. The
[CO/O/Pt(111) example](../examples/surface/pt111/README.md) uses real EANN and full
active-atom propagation, but its short pilot does not cover the complete FES or
reproduce a published rate. Its production calculation was not rerun in this audit.
Long calculations, PES physical accuracy and material conclusions remain P1 work above.

## Reference implementations and primary sources

Sources were consulted on 2026-10-31. Access dates for mutable documentation and pinned
source revisions are listed separately. No upstream implementation code was copied.

- **[S1] ASE periodic geometry.** [General MIC source](https://docs.ase-lib.org/_modules/ase/geometry/geometry.html)
  and [geometry API](https://docs.ase-lib.org/ase/geometry.html): lattice reduction,
  partial PBC and geometric-derivative conventions.
- **[S2] ASE energy contract.** [Atoms.get_potential_energy](https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_potential_energy)
  and [Atoms source](https://docs.ase-lib.org/_modules/ase/atoms.html): `force_consistent`
  distinguishes force-consistent electronic free energy from extrapolated energy.
- **[S3] ASE i-PI protocol.** [socketio source](https://docs.ase-lib.org/_modules/ase/calculators/socketio.html):
  cell/inverse/force/virial conventions. The protocol has no per-direction PBC flags;
  the external driver must implement matching boundary conditions.
- **[S4] i-PI.** [Features and theoretical references](https://docs.ipi-code.org/features.html),
  [authors' RPC/MTS tutorial](https://atomistic-cookbook.org/examples/pi-mts-rpc/mts-rpc.html),
  and [pinned source `93162d280d15fd954da6dc3580a5ed3503e207ac`](https://github.com/i-pi/i-pi/tree/93162d280d15fd954da6dc3580a5ed3503e207ac).
- **[S5] LAMMPS.** [`fix pimd/langevin`](https://docs.lammps.org/fix_pimd.html) and
  [`fix ipi`](https://docs.lammps.org/fix_ipi.html), documentation labeled release
  `30Sep2026`; [develop source at access, `7dbed49448a17dbc9d4bc3eee5512e9ea2ad3a04`](https://github.com/lammps/lammps/tree/7dbed49448a17dbc9d4bc3eee5512e9ea2ad3a04).
- **[S6] PLUMED 2.10.** [DISTANCE](https://www.plumed.org/doc-v2.10/user-doc/html/_d_i_s_t_a_n_c_e.html),
  [COM](https://www.plumed.org/doc-v2.10/user-doc/html/_c_o_m.html) and
  [WHOLEMOLECULES](https://www.plumed.org/doc-v2.10/user-doc/html/_w_h_o_l_e_m_o_l_e_c_u_l_e_s.html).
- **[S7] OpenPathSampling authors' paper.** [Python framework, part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6374749/):
  full TIS path ensembles, conditional crossing probabilities and effective-flux normalization.
- **[S8] GPAW.** [Dipole-layer corrections](https://gpaw.readthedocs.io/tutorialsexercises/electrostatics/dipole_correction/dipole.html):
  real-space/plane-wave boundaries, vacuum potential and dipole corrections for asymmetric
  slabs. No GPAW calculation was run as part of the audited change.
