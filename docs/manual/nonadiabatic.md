# Nonadiabatic dynamics and electronic friction

PyRPMD provides the following distinct interfaces. Model-level Python methods
require their own initialization and analysis; they are not implicit substitutes
for the gas-phase `rate` workflow.

| Method | Public entry point | Current scope |
| --- | --- | --- |
| Focused NRPMD with complex SOC | `task="drpmd"`, `NRPMDDynamics` | Fixed diabatic basis, complex Hermitian matrix and gradients |
| Population ensembles on models or the supported molecular PES | `task="nonadiabatic"` | Eight explicit selectors listed below; independent point/harmonic-reference parents, method-specific observables and restart |
| NRPMD/RPSH Python ensemble API | `run_nonadiabatic_ensemble` | Original focused NRPMD and real centroid RPSH; custom complete matrix providers |
| W-spin SM-NRPMD | `SMNRPMDDynamics`, `method="sm-nrpmd"` | Two-state complex Hermitian matrices, signed W estimator, focused nonequilibrium preparation |
| CA-RP-MASH | `RPMASHDynamics`, `method="rp-mash"` | Two real states, deterministic centroid-equator events and population importance sampling |
| Classical-nuclei MASH API | `MASHDynamics` | Original two-state setting at `P=1`; experimental centroid extension at `P>1`, distinct from `RPMASHDynamics` |
| Quasi-diabatic/adiabatic API | `QDNRPMDDynamics`, `AdiabaticRPSHDynamics` | Provider-supplied energies, gradients, derivative couplings and electronic overlaps |
| Thermal NRPMD | `ThermalNRPMDSampler`, `ThermalNRPMDDynamics` | Signed SEO thermal preparation, finite-time flux/population rate curves; real matrices |
| Electronic friction | `[electronic_friction]` on `diffusion`, `ElectronicFrictionDynamics` | Markovian centroid bath, tensor friction, periodic/fixed slabs |
| Original centroid RPSH | `RPSHDynamics`, `method="rpsh"` | Real matrices, original hopping/rescaling policy and optional EDC |
| Extended RPSH | `AdvancedRPSHDynamics` | CA, real BA, common-centroid-density DM, optional DBOC and nondegenerate complex CA/DM hopping |
| MV-RPMD | `MVRPMDDynamics` | SEO mapping trace Hamiltonian and its own Hamiltonian flow |
| KC-RPMD | `KCRPMDDynamics` | Two real states, kinetic constraint and explicit calibration |
| Isomorphic RPMD | `IsomorphicPotential`, `IsomorphicEhrenfestDynamics` | Two-state trace-matched isomorphic Hamiltonian with Ehrenfest propagation |

Detailed contracts and executable examples are in the
[thermal guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/thermal-nrpmd.md),
[friction guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md),
[SOC/RPSH guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-soc-rpsh.md),
[spin-method guide](../nonadiabatic-spin-methods.md),
[hopping-extension guide](../nonadiabatic-hopping-extensions.md),
and [method guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-methods.md).

## Original papers and reproduction scope

The [method-to-paper catalog](references.md) gives full citations and code
locations; the [reproduction plan](reproduction.md) specifies the original
systems, observables, runnable entries and remaining work. Cite the method,
initial-state preparation and physical model used in a calculation. A generic
LVC or discrete spin-boson provider still needs a citation for its parameters
and bath discretization.

| Implemented component | Original sources | Scope of the attribution |
| --- | --- | --- |
| Focused NRPMD and its MInt subflow | [Richardson–Thoss 2013](https://doi.org/10.1063/1.4816124); [Chowdhury–Huo 2019](https://doi.org/10.1063/1.5096276); [Church et al. 2018](https://doi.org/10.1063/1.5005557) | The 2019 paper supplies the focused preparation; the Church citation is for the integrator component, not a complete MQC-IVR implementation. |
| Signed thermal NRPMD | [Richardson–Thoss 2013](https://doi.org/10.1063/1.4816124) | SEO thermal sampling and correlations; the dedicated paper benchmark uses the original lowest-adiabatic split, while the library defaults to a trace split. |
| Two-state SM-NRPMD | [Bossion et al. 2021](https://doi.org/10.1063/5.0051456) | Focused spin propagation is implemented; the paper's equilibrium correlations additionally require its thermal trace and phase-weighted estimators. |
| MASH and CA-RP-MASH | [Mannouch–Richardson 2023](https://doi.org/10.1063/5.0139734); [Cao et al., ChemRxiv v1, 2025](https://doi.org/10.26434/chemrxiv-2025-xps2g) | `MASHDynamics(P=1)` follows the first; `RPMASHDynamics`/`rp-mash` follows the CA variant of the second. |
| FSSH and centroid RPSH | [Tully 1990](https://doi.org/10.1063/1.459170); [Shushkov et al. 2012](https://doi.org/10.1063/1.4766449); [Shakib–Huo 2017](https://doi.org/10.1021/acs.jpclett.7b01343) | Match the chosen paper's preparation, momentum-rescaling policy and amplitude/active-state observables. |
| RPSH-BA and RPSH-DM | [Y. Shakya et al. 2023](https://doi.org/10.1103/PhysRevA.107.062810) | BA has one common amplitude vector; DM averages bead wavefunctions in a common centroid basis. Both keep one active surface. |
| EDC and DBOC | [Granucci–Persico 2007](https://doi.org/10.1063/1.2715585); [Limbu et al. 2025](https://doi.org/10.1063/5.0301249) | Separate extensions with stated ring conventions; reproducing the DBOC donor population requires its mixed active-state/coherence estimator. |
| Complex hopping and geometric force | [Miao et al. 2019](https://doi.org/10.1063/1.5088770) | Local nondegenerate NAC/rescaling and Berry-force prescription, adapted to the ring. |
| MV-RPMD and KC-RPMD | [Ananth 2013](https://doi.org/10.1063/1.4821590); [Menzeleev et al. 2014](https://doi.org/10.1063/1.4863919) | Separate model APIs with their own thermal weights; KC also requires crossing calibration and auxiliary mass. |
| Isomorphic potential | [Tao et al. 2018](https://doi.org/10.1063/1.5005544) | The two-state trace construction is paired with Ehrenfest here; the paper's QCLE/iso-surface-hopping calculations need different propagators. |
| Quasi-diabatic transport | [Mandal et al. 2018](https://doi.org/10.1021/acs.jctc.7b01178); [Zhou et al. 2019](https://doi.org/10.1021/acs.jpclett.9b02747) | QD-NRPMD reuses the local-basis strategy; the original PLDM/SQC and ethylene calculations are not provided by the analytic interface demo. |
| Tully models and NaH₂ PES | [Tully 1990](https://doi.org/10.1063/1.459170); [Halvick–Truhlar 1992](https://doi.org/10.1063/1.461986), [1994 erratum](https://doi.org/10.1063/1.467282) | Cite the physical potential independently of the dynamics used with it. |
| Independent 1D quantum reference | [Feit et al. 1982](https://doi.org/10.1016/0021-9991(82)90091-2) | Fourier spectral propagation on a finite periodic grid; converge domain, grid and timestep. |

For a first quantitative paper comparison, use
`examples/nonadiabatic/published_nrpmd_benchmark.py` for Richardson–Thoss
Fig. 1(c), with `--potential-split lowest_adiabatic`. Other short population and
molecular examples establish workflow operation; their synthetic parameters
and initial distributions must be replaced where the reproduction plan calls
for a published system. The CA-RP-MASH citation records the implemented v1
source; a journal successor was not verified in the 2026-11-02 review.

## Population ensembles

For time-dependent populations, use the [population workflow](../nonadiabatic-workflow.md):

```bash
pyrpmd validate examples/nonadiabatic/population.inp --check-pes
pyrpmd run examples/nonadiabatic/population.inp --workers 2 --output runs/population
pyrpmd analyze runs/population --recompute
pyrpmd resume runs/population --block-steps 64
```

The exact `method` selectors are:

| Selector | Propagation and important restrictions |
| --- | --- |
| `nrpmd` | Focused MMST NRPMD; any supported state count, complex Hermitian matrix |
| `sm-nrpmd` | Two-state W-spin SM-NRPMD; one unit spin per bead, complex Hermitian matrix |
| `rp-mash` | Two-state CA-RP-MASH; real matrix, `integrator="adaptive-dop853"`, `native=false` |
| `rpsh` | Original real centroid RPSH; preserves its existing `centroid`/`ring` rescaling semantics |
| `rpsh-ca` | Extended centroid RPSH; nondegenerate Hermitian matrix, optional DBOC and geometric force |
| `rpsh-ba` | Real RPSH-BA; one shared adiabatic coefficient vector, bead-averaged energies and velocity-contracted NACs |
| `rpsh-dm` | Per-bead wavefunctions with a common centroid density and hopping current; nondegenerate Hermitian matrix |
| `soc-rpsh` | Explicit complex-SOC selector for the same extended CA implementation as `rpsh-ca` |

The [model library](../nonadiabatic-models.md) supplies Tully I–III, general LVC
and finite discrete spin-boson Hamiltonians. The registry also includes the
published molecular PES described below. The original
[ensemble API](../nonadiabatic-ensembles.md) accepts custom matrix providers and
prepared NRPMD/RPSH states; the additional methods have their own drivers and
restartable CLI workflow.

For `nrpmd` and `sm-nrpmd`, `initial.initial_state` is a fixed-diabatic state
index. Hopping methods use `initial.active_surface`; BA shares a bead adiabatic
label, while CA/DM use the centroid basis for their population decision. MASH
draws spins from the chosen centroid adiabatic hemisphere with its required
population importance distribution. Equal integer labels across methods do
not establish equal physical electronic preparations.

NRPMD and SM save signed fixed-diabatic density estimates without clipping.
RPSH saves its amplitude density, adiabatic amplitude populations and active
fractions separately. **MASH CLI populations are adiabatic active-surface
indicators** under the specified importance sampling. Its propagation-spin
outer product is not a MASH density estimator; the CLI therefore does not
publish it as diabatic population or coherence. General MASH operator
correlations use the full-sphere, weighted Python API in the spin-method guide.

The four extended RPSH selectors accept `dboc=true` and `geometric_force`.
DBOC enters energies, nuclear gradients and hop energy gaps consistently.
For complex hopping, the documented maximum-real-NAC direction and local
Abelian Berry-force ansatz apply only to nondegenerate states. This does not
implement Kramers/other degenerate subspaces or phase-space surface hopping.
`decoherence="edc"` is available for the original/extended CA and real BA;
RPSH-DM requires `decoherence="none"` and uses its bead-density construction.
Extended `rescaling="bead"|"centroid"` always uses full-ring hop energy
accounting; it does not reuse the original CA centroid-energy convention.

Initial nuclear parents may repeat a point or sample an independent finite-bead
harmonic reference. Per-trajectory histories and pointwise SEM use independent
trajectories, not beads or frames, as samples; one trajectory has no SEM.
Checkpoints retain the complete electronic state, including BA transported
frames, MASH sampling labels, RNG states and observation histories. These
nonequilibrium preparations do not supply the coupled electronic-nuclear
thermal distribution or a thermal rate. The SM thermal trace/phase reweighting,
BA-RP-MASH, multistate MASH and SU(N) SM-NRPMD remain outside these interfaces.

## Published molecular PES

`model.name="nah2_leps_5g_1992"` selects the historical coupled two-doublet NaH₂
5G surface set. `NaH2LEPS5G` accepts three-dimensional Na/H/H geometries and
returns the full real diabatic matrix and Cartesian gradients, including the
coupling derivative. It is a fitted molecular PES for Na + H₂ electronic
quenching/reaction, with pinned source identity and a comparison against the
independently compiled original Fortran. It does not include SOC, higher
electronic states, arbitrary molecules or on-the-fly electronic structure.

See the [molecular guide](../molecular-nonadiabatic.md) for source, atom-order and
unit conventions, the optional pinned ChemPotPy bridge, and
[molecular examples](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/molecular_nonadiabatic).
The examples start from explicit interaction-region phase points; they are not
converged rovibrational scattering or thermal reaction calculations. Source
agreement and small energy drift establish numerical operation, not a new
accuracy assessment of the fit or `physical_converged=true`.

## Electronic backends and NRPMD batching

The population workflow permits `dynamics.electronic_backend="numpy"`,
`"native"` or `"torch"` for **NRPMD and SM-NRPMD**. The native backend combines
Hermitian diagonalization with a C++/OpenMP kernel for exact fixed-coordinate
electronic propagation and its integrated nuclear impulse. The separate
`native` flag controls the nuclear free-ring integrator.

Torch supports `electronic_device="cpu"` or `"cuda[:index]"` when available.
The accelerated operation is the mapping/impulse kernel: nuclear propagation
and the current molecular PES remain on CPU, and the host driver transfers
arrays as needed. Missing requested dependencies/devices raise an error rather
than silently selecting a different backend. The current validation makes no
CUDA hardware or GPU speedup claim. Hopping and MASH do not use this GPU kernel.

`BatchedNRPMDDynamics` is a separate Python API that batches trajectories and
beads for compatible independent-frame providers and the selected electronic
backend. It is not an automatic batching mode for every CLI method. Details,
measured CPU comparisons and limitations are in the
[performance guide](../nonadiabatic-performance.md).

## State-selected collisions

NRPMD is an experimental mode of `task="drpmd"`. It adds electronic mapping variables to every
nuclear bead, propagates electrons and nuclei together on the full diabatic potential-energy matrix,
and combines the electronic populations with the chemical product classifiers to give
electronic-state-resolved contributions.

Collisions start from a chosen incoming diabatic electronic state that is decoupled at long range.
Initial conditions, impact parameters and classification follow ordinary dRPMD; see the
[gas-phase workflows](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/manual/workflows.md).

## Run the example

From the repository root:

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)
runs four beads, two electronic states, two parent chains and eight trajectories at a fixed
collision energy. The [matrix potential](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)
is an artificial two-state model with a local coupling; channels are labeled as H + HD → H₂ + D.

With `collision.sampling="fixed_energy"`, the result is a cross section in `bohr^2`. Switching to
`thermal_flux` and removing `energy` gives a thermal collision distribution from the selected
incoming state.

## Configuration and the matrix PES

Add the following to an ordinary dRPMD input:

```toml
[potential]
provider = "diabatic_python"
factory = "matrix_pes.py:make"
derivative = "gradient"

[nonadiabatic]
method = "nrpmd"
nstates = 2
initial_state = 0
state_labels = ["entrance", "excited"]
initialization = "focused"
asymptotic_coupling_tolerance = "1e-8 hartree"
```

`initial_state` is 0-based, and labels name states of the fixed diabatic basis. The factory returns
an object with `evaluate(q)`, or a callable, that returns
`pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`:

| Data | Shape and unit |
| --- | --- |
| Nuclear coordinates q | `[B,N,3]`, bohr |
| Energy matrix | `[B,K,K]`, hartree |
| Gradient of the matrix | `[B,K,K,N,3]`, hartree/bohr |

B is the batch size and K the number of electronic states. Both arrays are finite and
Hermitian in the electronic indices, with real or complex entries and gradients for every diagonal element and coupling. The
matrix interface takes **positive gradients** (scalar potentials return forces). If the provider
declares coordinate or energy units, the adapter converts them.

Put any state-independent energy on every diagonal element; the matrix is the complete potential.
`validate --check-pes` checks the matrix and finite-differences each element's gradient.

## Mapping variables and signed estimators

Nuclei carry `q,p[P,N,3]` and electrons `mapping_q,mapping_p[P,K]`. Parents are sampled on the
diagonal potential of the incoming state. Each launch draws independent phases for every bead and
state; focused initialization gives the incoming state population 1 and the others 0. Collisions
run without a thermostat.

With electronic variables x and y, the population estimator of a trajectory is

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

The populations sum to 1 within rounding; individual values can fall outside $[0,1]$. Averages over
trajectories give the state-resolved contributions.

The joint contribution of chemical channel α and electronic state s is `w * I_alpha * n_s`,
normalized by the total number of launched trajectories; summing over states recovers the chemical
contribution. All channels share the parent bootstrap draws, which gives their joint covariance.

## Stopping, output and restart

At launch, the off-diagonal couplings of every bead must be below `asymptotic_coupling_tolerance`. A
product or returning reactant counts as resolved when it is both geometrically separated and
electronically decoupled; otherwise the trajectory runs to the time budget. Converge the coupling
threshold, separation distance and propagation time.

| Output | Contents |
| --- | --- |
| `nonadiabatic` in `results/summary.json` | `final_estimates_available`, state labels, population-sum error, channel results |
| `results/electronic_channels.csv` | Signed observed contributions and thermal rates or cross sections |
| `nonadiabatic.channels` | `signed_observed_contributions`, `rate_cm3_molecule_s` or `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | Joint error information ordered by `covariance_columns` |

If any trajectory failed or remains unresolved, final electronic rates or cross sections are left
empty and the observed contributions are still reported. Summaries carry `experimental=true` and
`physically_validated=false`.

Checkpoints store the nuclear and mapping arrays, clocks, classification persistence and RNG state,
and a failed step rolls back electrons and nuclei together. `resume` finishes interrupted work within
the original time budget; a longer maximum collision time needs a new run.

## Supported configurations

The focused **`task="drpmd"` scattering workflow** runs NRPMD on CPU with NumPy,
serially or with local trajectory processes; the nuclear free ring can use the
compiled kernels. Use `loop_backend="auto"` or `"python"`. This collision
interface does not expose the population task's electronic backend selectors or
the separate NRPMD batch API. MPI, bead-worker splitting, GPU, a fused native
electronic trajectory loop, and periodic or fixed-atom systems are not available
in this scattering workflow. Complex Hermitian SOC is supported in its matrix
propagation. The signed thermal sampler and the original centroid `RPSHDynamics`
require real matrices; extended CA/DM complex hopping has the distinct
restrictions stated above.

For a real system, provide a diabatic basis with the full matrix and its gradients, including the
couplings between states, and converge the asymptotic states, the energy and mapping norm, the bead
number, the time step, parent correlation, collision parameters and product commitment. Method and
sources are in the [NRPMD topic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md)
and the [example notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md).
