---
orphan: true
---

# Molecular quasi-centroids and experimental f-QCMD

PyRPMD provides an experimental **f-QCMD** production route for nonperiodic,
fixed-topology, bent triatomic molecules. Its training tools construct
bead-averaged internal-coordinate distributions, update separable correction
potentials by iterative Boltzmann inversion (IBI), and save a portable,
temperature-specific model. The spectroscopy workflow runs classical dynamics
on the original potential plus that correction with `method = "f-qcmd"` and
`beads = 1`.

A separate constrained PIMD API samples ring polymers at fixed mean internal
coordinates. It is an equilibrium research tool; its Markov-chain trajectory
is not a real-time QCMD trajectory. Existing PA-CMD and quadratic force-matched
f-CMD retain their behavior and input identities.

This implementation does not establish converged liquid-water IR spectra.
Real-time adiabatic QCMD and mixed molecular partitions for h-CMD remain outside
this first implementation. Requests for spectroscopy methods `qcmd` or `h-cmd`
continue to fail explicitly.

## Method and supported molecular geometry

For a triatomic molecule with atom order `(center, end1, end2)`, let the bond
lengths and included angle in bead $b$ be $r_{1b}$, $r_{2b}$ and $\theta_b$.
Its quasi-centroid internal coordinates are

$$
R_1=\frac{1}{P}\sum_{b=1}^{P}r_{1b},\qquad
R_2=\frac{1}{P}\sum_{b=1}^{P}r_{2b},\qquad
\Theta=\frac{1}{P}\sum_{b=1}^{P}\theta_b.
$$

Averaging the Cartesian positions first and measuring their bonds and angle
generally gives different values. This distinction is the geometric basis of
QCMD's treatment of the CMD curvature problem. The original QCMD construction
and its approximations are described by
[Trenins, Willatt and Althorpe, JCP 151, 054109 (2019)](https://doi.org/10.1063/1.5100587).

`TriatomicQuasiCentroid` reconstructs Cartesian quasi-centroid positions with
these internal coordinates. Each reconstructed molecule has the center of
mass of its Cartesian centroid. A proper mass-weighted Kabsch rotation aligns
its orientation to the Cartesian centroid, satisfying the corresponding
Eckart stationarity condition. Internal coordinates are invariant under
translation and rotation; reconstructed positions transform with the molecule.

The current implementation requires all of the following:

- Explicit disjoint molecular triples covering every atom exactly once.
  Atom indices are zero-based. The center atom is first; no bond discovery or
  terminal-atom permutation is inferred.
- Bent molecules with positive bond lengths. Zero bonds, nearly linear
  molecules and a Cartesian centroid with degenerate orientation are errors.
- Nonperiodic, unwrapped positions. The production workflow rejects any cell,
  including a cell with all PBC flags false. There is no minimum-image or
  molecular-unwrapping operation in the quasi-centroid geometry routines.
- Mobile atoms without fixed atoms or additional bond constraints.
- An explicit CPU potential for the f-QCMD correction wrapper, and serial
  spectroscopy using the NumPy reference engine. Native-required execution is
  rejected.

Multiple triatomics can share coordinate types. General polyatomic topology,
dihedrals, dissociation, changing molecular connectivity, and intermolecular
IBI corrections are not implemented. The original potential retains its
couplings and interactions; only the added quantum correction is restricted
to the supplied intramolecular terms. The validated application scope is
isolated molecules and gas-phase model checks.

## Units and array shapes

Python APIs use atomic units throughout. Explicit units in input files are
resolved through the usual PyRPMD unit parser before model compatibility is
checked.

| Quantity | Unit or shape |
| --- | --- |
| Cartesian position and bond length | bohr |
| Angle | radians |
| Mass | electron mass |
| Energy and correction table values | hartree |
| Force | hartree/bohr, equal to $-\nabla V$ without bead scaling |
| Time and timestep | atomic time |
| Inverse temperature `beta` | inverse hartree, $1/(k_B T)$ |
| `internal_coordinates(q)` | input `[...,atom,3]`, output `[...,molecule,3]` ordered `(r1,r2,theta)` |
| `internal_gradients(q)` | output `[...,molecule,coordinate,local_atom,xyz]` |
| `mean_coordinates(q)` | input `[bead,atom,3]` or `[frame,bead,atom,3]`; average the bead axis |
| `quasi_centroid(q)` | the same ring input; output `[atom,3]` or `[frame,atom,3]` |

For example, given a ring `q` with shape `[P,3,3]` and a resolved molecular
`system`:

```python
from pyrpmd.qcmd import TriatomicQuasiCentroid

geometry = TriatomicQuasiCentroid([[0, 1, 2]], system.masses)
mean_internal = geometry.mean_coordinates(q)  # [1,3], bohr/bohr/radian
qc = geometry.quasi_centroid(q)               # [3,3], bohr
reconstructed_internal = geometry.internal_coordinates(qc)
# reconstructed_internal agrees with mean_internal to numerical precision.
```

The molecular definition and numerical singularity tolerance are immutable
parts of the saved model identity.

## The f-QCMD training procedure

The f-QCMD procedure of
[Fletcher et al., JCP 155, 231101 (2021)](https://doi.org/10.1063/5.0076704)
uses **ordinary unconstrained PIMD** to obtain target distributions. It does not
require a set of fixed-quasi-centroid sampling windows. Training and production
are distinct stages:

1. Sample the original physical potential with unconstrained equilibrium PIMD
   at the intended temperature and isotope masses. Save representative ring
   configurations after equilibration.
2. For each saved ring, calculate the mean bond lengths and mean angle defined
   above. Accumulate their target distributions on explicit grids.
3. Start with zero correction. Sample classical equilibrium configurations
   under the original potential plus the current correction, using one bead
   and the same temperature and masses.
4. Compare classical and target distributions, update the correction by IBI,
   re-equilibrate, and sample again. An update does not certify that the newly
   fitted model already reproduces the target.
5. Validate the final correction on independent data. Run one-bead classical
   production trajectories on the corrected potential and analyze the saved
   property traces.

For coordinate group $g$, the update is

$$
\Delta V_g^{(n+1)}(s)=\Delta V_g^{(n)}(s)
 +\frac{\lambda}{\beta}
 \log\left[\frac{p_g^{(n)}(s)}{p_g^{\rm target}(s)}\right],
\qquad 0<\lambda\leq 1.
$$

The effective potential is

$$
V_{\rm eff}(\mathbf q)=V_{\rm physical}(\mathbf q)
 +\sum_{\text{molecular coordinates }c}\Delta V_{g(c)}(s_c(\mathbf q)).
$$

Coordinates sharing a group use the same correction. For water,
`groups = [[0, 0, 1]]` pools the two OH bonds and gives the HOH angle its own
correction. A group cannot mix bonds and angles. Group indices must be
contiguous starting from zero.

`coordinate_histograms` returns counts in the raw coordinate measure $dR$ or
$d\Theta$. The IBI routine separately normalizes target and sampled counts,
including each bin's width. Radial and angular measure factors cancel between
the two distributions on the same grid. Taking `-log(target)/beta` alone and
using it as a Cartesian molecular potential would omit those measure factors
and the original potential; that is not this algorithm.

### Python training API

Training is a Python API and research example, not an additional CLI training
task. In this snippet, `config` and `system` are the resolved input and system,
`pimd_frames` has shape `[frame,bead,atom,3]`, and `classical_frames` contains
samples from the current corrected model with shape `[frame,atom,3]`:

```python
import numpy as np
from pyrpmd.qcmd import (
    FQCMDModel, FQCMDPotential, TriatomicQuasiCentroid,
    coordinate_histograms, fqcmd_base_identity, update_fqcmd_model,
)

geometry = TriatomicQuasiCentroid([[0, 1, 2]], system.masses)
groups = [[0, 0, 1]]
edges = (
    np.linspace(0.5, 3.5, 101),              # OH distances, bohr
    np.linspace(0.15, np.pi - 0.15, 101),    # HOH angle, radians
)
model = FQCMDModel(
    geometry, groups, edges,
    [np.zeros(len(edge) - 1) for edge in edges],
    beta=config["beta"],
    base_identity=fqcmd_base_identity(config["potential"], system),
    provenance={
        "training": "unconstrained PIMD followed by separable IBI",
        "pimd_beads": pimd_frames.shape[1],
        "physical_converged": False,
    },
)

target = coordinate_histograms(
    pimd_frames, geometry, groups, edges, bead_average=True,
)
# Obtain fresh classical_frames with FQCMDPotential(base_pes, model)
# at this model's beta and masses before each IBI update.
sampled = coordinate_histograms(
    classical_frames, geometry, groups, edges, bead_average=False,
)
updated, diagnostics = update_fqcmd_model(
    model, target, sampled,
    damping=0.5, regularization=0.01, polynomial_degree=4,
)
updated.save("fqcmd-model.npz")
restored = FQCMDModel.load("fqcmd-model.npz")
corrected_pes = FQCMDPotential(base_pes, restored)
```

The grids and fit settings above illustrate the API; they are not universal
water parameters. Choose domains wide enough to contain all sampled and
production coordinates, and establish convergence for the chosen grids.

The correction wrapper borrows `base_pes`; its owner remains responsible for
closing the original potential. Its `evaluate(x)` follows the ordinary
`PESResult(energy[B], forces[B,N,3], ...)` contract, so batch length is a batch
of configurations, not an instruction to re-quantize the effective potential.
A pre-existing centroid or quasi-centroid PMF cannot serve as the original
physical PES for another f-QCMD correction.

### Interpolation, empty bins and smoothing

Each correction table stores energies at bin centers. A not-a-knot cubic
spline interpolates the energy, and its analytic derivative supplies the force.
The supported domain extends to the first and last bin edges; the two outer
half-bins use the same terminal cubic polynomials. Coordinates outside that
closed domain raise an error. No coordinate is silently dropped, clipped, or
extrapolated beyond the explicit model domain.

By default, zero-density bins make an IBI update fail. A positive explicit
`regularization` adds a common density floor, scaled by the larger maximum
of the target and sampled densities, before the logarithm. This changes the
fit and must be converged. It does not create independent observations in an
unsampled tail.

With `polynomial_degree=None`, the update uses the tabulated IBI increments.
An explicit degree from zero through eight instead fits the increment using
weighted least squares on commonly occupied bins. Weights are proportional
to $n_t n_s/(n_t+n_s)$, where $n_t$ and $n_s$ are target and sampled counts.
The fitted polynomial is evaluated over the whole specified grid. This is an
additional finite-basis approximation, especially in sparsely sampled tails;
rank-deficient fits fail. The final table still uses the same conservative
energy-spline interpolation.

Each update records count hashes, parent-model identity, damping,
regularization and polynomial degree. Reported marginal errors refer to the
supplied **pre-update** classical samples. The routine keeps `converged` and
`physical_converged` false.

## Production through the spectroscopy workflow

The production input names the **original** physical PES and a previously
trained model. The workflow loads the model and applies the correction.
The following is a template for compatible model and property files in the
same directory; the numerical settings are illustrative:

```toml
task = "spectroscopy"
temperature = "150 K"
beads = 1
seed = 1729

[system]
symbols = ["O", "H", "H"]
geometry_unit = "bohr"
geometry = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.74, 0.0]]

[potential]
provider = "python"
factory = "water_model.py:build_potential"

[dynamics]
integrator = "normal-mode-bab"
timestep = "0.1 fs"

[sampling]
thermostat = "pile-l"
integrator = "baoab"
equilibration_steps = 2000
production_steps = 20000
sample_interval = 2
replicas = 4
gamma_centroid = "0.05 fs^-1"

[spectroscopy]
method = "f-qcmd"
fqcmd_model_file = "fqcmd-model.npz"
spectra = ["ir"]
window = "hann"
normalize = false

[properties]
provider = "python"
factory = "water_properties.py:build"

[output]
directory = "runs/fqcmd"
checkpoint_steps = 500
```

The model must have been trained with the resolved masses, temperature and
physical PES specification used here. Supplying a different isotope, different
PES parameters, or a different model-source file fails compatibility checking.
Explicit isotope masses in training and production must match exactly.

```bash
pyrpmd validate fqcmd.inp --check-pes
pyrpmd run fqcmd.inp --output runs/fqcmd
pyrpmd resume runs/fqcmd
pyrpmd analyze runs/fqcmd --recompute
```

Structural validation checks the input without loading the fit or property
provider. Numerical validation checks the saved model, original-PES identity,
initial fitted-coordinate support, and property contract before propagation.

Production uses physical masses with no centroid thermostat. The production
stepper and existing spectral estimators are reused. PA-CMD-specific
`target_frequency` and `friction_scale` are not valid f-QCMD options, and a
multi-bead production input is rejected to prevent quantizing the learned
potential a second time.

### Observable approximation

Properties are evaluated on the propagated quasi-centroid geometry. A general
property surface at that geometry is not the same as a conditional average of
per-bead property values. This is an additional observable approximation,
recorded in preparation and results metadata. Nonlinear dipole or
polarizability surfaces require their own validation or an appropriate
conditional property model. Matching coordinate marginals alone does not
validate IR/Raman intensities or correlation functions.

The common spectroscopy outputs remain relative, approximate dynamical
spectra; physical convergence and absolute electromagnetic intensity
prefactors are not inferred from a successful run.

## Model identity, restart and offline analysis

`FQCMDModel` is immutable and its atomic NPZ model file contains its schema,
temperature, masses, ordered molecular triples, coordinate groups, grids,
energy corrections, interpolation rule, singularity tolerance, original-PES
identity and training provenance. The model's scientific fingerprint hashes
that complete definition. `fqcmd_base_identity` includes the potential
specification, referenced local source/model content, atom ordering, masses,
charge, spin and boundary conditions. Starting coordinates are omitted so
thermal starting geometries can share a trained model. Indirect imports and
remote services still require an independently pinned software environment.

Record training timesteps, thermostat settings, equilibration and collection
lengths, seeds, bead count, file digests and sampling uncertainty in
`provenance`; the API cannot infer an external dataset's history. Changing
provenance changes model identity as well as changing a correction table.

The spectroscopy trajectory identity additionally includes the resolved model
file path and its complete file SHA256. An old checkpoint is rejected if that
file changes, including after a trajectory has finished. Atomic spectroscopy
checkpoints retain phase clocks, positions, momenta, RNG state and the exact
saved property-trace prefix. Restart resumes from that state without repeating
equilibration or duplicating a frame. Existing methods do not acquire these
new identity fields unless f-QCMD is explicitly selected.

Offline reanalysis uses authenticated saved traces and their recorded model
hash and provenance. It can change spectral processing options without loading
the potential, property provider or model file again, and it does not replace
the raw trace history. Changing the recorded model path or trajectory physics
is not a reanalysis operation and is rejected. Offline reanalysis does not
validate a new model against old trajectories.

## Fixed-quasi-centroid conditional PIMD API

`ConstrainedQuasiCentroidSampler` is a separate equilibrium sampler for
research on rings conditioned on fixed $R_1,R_2,\Theta$. It is useful for
checking the geometry and studying conditional ring distributions. A single
fixed window is not the unconstrained target ensemble required by the f-QCMD
IBI procedure above.

For constraints $C(q)=\bar s(q)-s_{\rm target}$, let
$J=\partial C/\partial q$ and $G=J M^{-1}J^T$, using the physical bead masses.
The desired positional measure is

$$
\rho(q)\,dq\ \propto
 \exp[-\beta_P U_P(q)]\,\delta(C(q))\,dq,
\qquad \beta_P=\beta/P,
$$

where $U_P$ is the sum of physical bead potentials and primitive ring springs.
Rigid constrained Hamiltonian dynamics instead carries the factor
$\sqrt{\det G}$ in that Cartesian delta-function representation. The sampler
corrects it in the Metropolis ratio:

$$
\log A=-\beta_P\bigl[H_P(q',p')-H_P(q,p)\bigr]
       -\tfrac12\bigl[\log\det G(q')-\log\det G(q)\bigr].
$$

At each HMC proposal, momenta are fully redrawn from the projected canonical
Gaussian satisfying $J M^{-1}p=0$. A primitive velocity-Verlet RATTLE trajectory
uses the unmodified ring Hamiltonian. A final momentum reversal makes the
proposal an involution, and an explicitly checked reverse trajectory guards
the nonlinear projection. The Fixman factor is applied in acceptance; a
Fixman Hessian or Hessian force is not constructed. Disjoint molecular triples
permit independent `3 x 3` constraint-metric solves.

Position projection, momentum projection, model evaluation or reversibility
failure raises an error and leaves state, RNG and counters unchanged. The
primitive free-spring timestep stability limit is enforced; molecular
stiffness, acceptance and statistical mixing still need checking. This sampler
fixes internal mean coordinates only. Overall translation and rotation remain
unconstrained.

The following initializes a window from `initial_ring`, with the original
`base_pes` and previously resolved `config` and `system`:

```python
import numpy as np
from pyrpmd.dynamics import State
from pyrpmd.qcmd import (
    ConstrainedQuasiCentroidSampler, TriatomicQuasiCentroid,
    fqcmd_base_identity,
)

geometry = TriatomicQuasiCentroid([[0, 1, 2]], system.masses)
target = geometry.mean_coordinates(initial_ring)
sampler = ConstrainedQuasiCentroidSampler(
    base_pes, geometry, target,
    beta=config["beta"], nbeads=len(initial_ring), dt=0.5,
    trajectory_steps=4, seed=27,
    tolerance=1e-10, max_iterations=50,
    reversibility_tolerance=1e-8,
)
q = sampler.constraint.project_positions(initial_ring)
state = State(q, np.zeros_like(q))
for _ in range(100):
    sampler.step(state)

identity = fqcmd_base_identity(config["potential"], system)
sampler.save_checkpoint("conditional-ring.npz", state,
                        potential_identity=identity)
# This call also works on a new sampler constructed with identical settings.
state = sampler.load_checkpoint("conditional-ring.npz",
                                potential_identity=identity)
```

The numeric `dt=0.5` above is in atomic time, not femtoseconds; it is an API
example, not a recommended universal sampling timestep. `State.step` counts
Markov proposals. `State.time` accumulates their proposal integration time.
Neither axis can be used as a real-time QCMD correlation trajectory.

Checkpoint save/load requires a nonempty explicit `potential_identity`.
Settings, target, masses, geometry, complete RNG state, counters and the state
are saved atomically with array integrity checks. Loading verifies all these
invariants and rolls back sampler changes if validation fails.

## Runnable gas-phase research example and validation

The repository includes `examples/spectroscopy/fqcmd_water.py`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/spectroscopy/fqcmd_water.py --output runs/fqcmd-water
```

It runs unconstrained PIMD, f-QCMD IBI iterations and classical production,
with the existing PA-CMD as its default comparator. Independent,
noninteracting water-inspired molecules share vectorized evaluations, and
their molecular spectra are averaged without intermolecular dipole cross
terms. The analytic surface contains a quartic expansion of two Morse
stretches and a harmonic bend. It is neither q-TIP4P/F nor the
Partridge--Schwenke potential used in the published gas-phase study.

The example saves fitted models, training distributions, spectra and a JSON
report containing scientific settings, seeds, identity, histogram errors,
frequency estimates, timings and unresolved convergence parameters. The
optional `--comparison centroid-ibi` fits Cartesian-centroid internal-coordinate
distributions with the same separable IBI model; it is a separate comparator,
not the existing quadratic force-matched f-CMD implementation.

Implementation tests cover translation/rotation behavior, exact reconstructed
internal coordinates, finite-difference gradients and conservative forces,
constraint and reversibility residuals, analytic tangent-momentum covariance,
Fixman acceptance ratios, identity rejection and exact RNG/state restart.
Zero correction is checked against the ordinary classical trajectory. These
checks establish implementation properties, not molecular spectral accuracy.

Numerical pilot results and their interpretation should be reported with the
actual generated JSON report, including both the largest grid-resolved OH
peak and the stretch-band first moment. Finite record length, window choice,
rotation and unresolved stretch components can move an individual band maximum.
A direction consistent with removal of the CMD red shift is necessary evidence,
but does not establish quantitative agreement with a different water potential.
No fixed gas-phase frequency shift is asserted from unrelated condensed-phase
results or from a QCMD-versus-exact-quantum residual.

### Parameters requiring independent convergence

- **Quantum target ensemble:** bead number, PIMD timestep, thermostat and
  equilibration, independent seeds, sampling length and correlation times.
- **IBI model:** histogram domain and resolution, pooling of equivalent
  coordinates, overlap counts, density floor, damping, polynomial degree,
  iteration count, and equilibrium sampling under the final corrected model.
  Assess held-out marginals and joint-coordinate correlations; accurate
  separate marginals do not uniquely determine a joint PMF.
- **Dynamics and spectroscopy:** production timestep, thermal launch ensemble,
  trajectory length and replica count, observable model, sampling cadence,
  spectral window and frequency resolution. Check one-bead NVE energy behavior
  independently of spectral appearance.
- **CMD comparison:** PA-CMD adiabatic target frequency, internal friction and
  its smaller timestep. A difference from an unconverged PA-CMD comparator
  cannot be assigned entirely to the curvature correction.
- **Conditional sampling, when used:** RATTLE tolerances, reverse-check
  tolerance, proposal timestep and length, acceptance, mixing and window
  equilibration. These do not replace the unconstrained f-QCMD target sampling.

The separable intramolecular correction and effective-property approximation
remain method limitations after numerical parameters have been converged.

## Further methods and primary references

Original real-time QCMD requires additional adiabatic dynamics and mean-force
machinery beyond this fixed-window equilibrium sampler. h-CMD requires explicit
molecular partitions and consistent combinations of f-QCMD and f-CMD reference
distributions and corrections. Neither method is selected implicitly here.

- [Trenins, Willatt and Althorpe, QCMD, JCP 151, 054109 (2019)](https://doi.org/10.1063/1.5100587).
- [Fletcher et al., f-QCMD, JCP 155, 231101 (2021)](https://doi.org/10.1063/5.0076704).
- [h-CMD primary paper, DOI 10.1063/5.0248115](https://doi.org/10.1063/5.0248115).
- [DL_POLY Quantum 2.1 software paper, DOI 10.1021/acs.jpca.4c08644](https://doi.org/10.1021/acs.jpca.4c08644),
  also available as [arXiv:2412.17216](https://arxiv.org/abs/2412.17216).
