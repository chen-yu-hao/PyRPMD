# PA-CMD, property models and vibrational spectra

The `spectroscopy` workflow propagates independent **PA-CMD**, RPMD, or TRPMD
replicas, evaluates a dipole/polarizability model, and saves relative IR, Raman
and causal complex SFG spectra. A temperature-tagged fitted centroid PMF can
also be propagated with one bead as **f-CMD**. The current workflow is a serial
NumPy reference implementation with atomic checkpoints, separate equilibration
and production clocks, and offline reanalysis.

These implementations add working spectroscopy tools. They do **not** establish
physical convergence on a molecular material or complete QCMD/h-CMD coverage.
Constrained adiabatic QCMD, quasi-centroid reconstruction and the complete
iterative h-CMD/f-QCMD fitting loop are not implemented. An explicit `qcmd` or
`h-cmd` method request fails instead of selecting Cartesian CMD.

## Run from an input file

```bash
pyrpmd validate examples/spectroscopy/pacmd.inp --check-pes
pyrpmd run examples/spectroscopy/pacmd.inp --output runs/pacmd
pyrpmd resume runs/pacmd
pyrpmd analyze runs/pacmd --recompute
```

The example uses an analytic harmonic PES and a Python toy property surface. It
checks the workflow; it is not a spectrum of a real chemical system. Replace
both surfaces with appropriate models and converge the calculation before
interpreting peaks or intensities.

`validate --check-pes` evaluates the initial complete ring through both the PES
and the requested property model, including shape, units-contract and symmetry
checks. It also checks the temperature tag for f-CMD. It does not propagate a
trajectory; validation without that flag only checks the input structure and
does not import the property model.

```toml
task = "spectroscopy"
temperature = "300 K"
beads = 16

[system]
geometry = "molecule.xyz"

[potential]
provider = "python"
factory = "my_pes.py:build"

[dynamics]
integrator = "normal-mode-bab"
timestep = "0.02 fs"

[sampling]
thermostat = "pile-l"
equilibration = "500 fs"
production = "2000 fs"
interval = "0.1 fs"
gamma_centroid = "0.01 fs^-1"
replicas = 4

[spectroscopy]
method = "pa-cmd"
target_frequency = "14000 cm^-1"
friction_scale = 0.5
spectra = ["ir", "raman", "sfg"]
sfg_component = "xxz"
window = "hann"
sfg_window = "cosine"
normalize = false

[properties]
provider = "python"
factory = "my_properties.py:build"
```

`target_frequency` is an **angular** frequency. A number is inverse atomic
time; a wavenumber such as `14000 cm^-1` is converted using
\(\hbar\Omega=hc\tilde\nu\). Inverse-time quantities such as `fs^-1` are also
accepted and mean angular frequency. It is not a prescribed universal
adiabaticity setting. `friction_scale` sets the internal friction to
\(2\lambda\Omega\) for PA-CMD or \(2\lambda\omega_k\) for TRPMD. The centroid
thermostat is applied only during equilibration.

The workflow requires all atoms mobile, a fixed cell, and serial execution. It
rejects biased/constant-pressure production, native-required execution and an
incompatible integration scheme. It does not silently use a real-mass native
RPMD kernel for fictitious-mass CMD. Equilibration can have its own timestep;
saved production time always uses the production timestep.

## PA-CMD Hamiltonian and checks

In the repository's sum-bead convention, the distribution is sampled at
\(\beta_P=\beta/P\). The physical free-ring frequencies are
\(\omega_k=2P\sin(k\pi/P)/\beta\). PA-CMD assigns

\[
\mu_{0,i}=m_i,\qquad
\mu_{k,i}=m_i\omega_k^2/\Omega^2\quad(k>0),
\]

to the kinetic terms while retaining the ordinary physical spring and PES
terms. All internal *free* modes therefore oscillate at \(\Omega\), and the
centroid retains its physical inertia. `PACMDDynamics` combines exact free-mode
rotations, physical-force kicks, and a normal-mode Langevin update. Its
`State.p` contains canonical momenta represented in the bead basis; internal
velocities must be obtained with `dynamics.velocities(state.p)`, not beadwise
`p/m`. The centroid velocity still equals mean bead momentum divided by the
physical mass. Its `energy` is the fictitious Hamiltonian, not a quantum energy
estimator.

```python
from pyrpmd.cmd import PACMDDynamics
from pyrpmd.dynamics import State

dyn = PACMDDynamics(potential, masses, beta, beads, dt,
                   target_frequency=omega, friction_scale=0.5, seed=12)
state = State(initial_ring, dyn.sample_momenta())
for _ in range(equilibration_steps):
    dyn.sample_step(state, gamma_centroid=gamma)
for _ in range(production_steps):
    dyn.step(state)  # internal thermostat only
```

Tests compare free propagation to an independently constructed dense
Hamiltonian matrix exponential, check free-ring Gibbs covariances, and verify
that the harmonic centroid motion is independent of internal frequency and
friction. They also check the one-bead limit, force-failure/RNG rollback, and
restart equivalence. These are implementation checks. PA-CMD calculations
still require timestep and adiabatic-separation studies, appropriate initial
equilibration, and statistical convergence. Cartesian CMD can retain the
low-temperature curvature problem.

## Common dipole and polarizability interface

`pyrpmd.properties.PropertyModel` accepts real coordinates `[batch,atom,3]` in
bohr. The result is `PropertyResult` or a dictionary containing:

| Key | Shape | Unit and meaning |
| --- | --- | --- |
| `dipole` | `[batch,3]` | Total-system dipole in e·bohr, laboratory axes |
| `polarizability` | `[batch,3,3]` | Real symmetric total-system nonresonant polarizability in atomic units |

A model can return either property or both. A requested spectrum fails if its
required property is missing. Complex resonant polarizabilities are outside
this interface. The validator owns output buffers, so a calculator's reusable
buffers cannot alter already recorded values.

A Python factory returns a callable or an object with `evaluate(q)`; an
optional `bind_system(system)` receives composition and fixed-cell geometry.
Outputs must already follow the atomic-unit convention. Local factory files
and explicitly named model/file parameters resolve relative to the input and
are included in source fingerprints.

The optional `TorchScriptPropertyModel` uses this standard signature:

```python
def forward(inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # inputs["positions"]:       [B,N,3], floating point
    # inputs["atomic_numbers"]:  [N], int64
    # inputs["cell"]:            [3,3], rows are cell vectors
    # inputs["pbc"]:             [3], bool
    return {"dipole": dipole, "polarizability": polarizability}
```

Nonperiodic systems receive a zero cell and false PBC. A model-specific wrapper
can adapt a package's own signature to this contract. Example input:

```toml
[properties]
provider = "torchscript"
model = "models/properties.pt"
device = "cpu"
dtype = "float64"
position_unit = "angstrom"
dipole_unit = "debye"
polarizability_unit = "angstrom3"
```

Supported alternate units are explicit; returned properties always use atomic
units. The default input position unit is `bohr`, and both output-unit options
default to `au`. PyTorch is imported only when this adapter is constructed.
The offline `evaluate_property_trajectory` helper batches model calls and
retains the bead axis. No property is evaluated on a centroid unless the user
specifically supplies centroid configurations: for nonlinear surfaces,
\(\langle A(q_j)\rangle_j\) and \(A(\langle q_j\rangle_j)\) differ.

## Spectral conventions and output

IR retains the existing \(\beta\omega^2\) Kubo dipole-spectrum convention.
Raman decomposes the polarizability into the mean trace \(a\) and the traceless
tensor. With \(I_{\rm iso}\) the spectrum of \(a\) and \(I_{\rm aniso}\) the
Frobenius spectrum of \(\sqrt{3/2}(\alpha-aI)\), it reports
\(VV=45I_{\rm iso}+4I_{\rm aniso}\), \(VH=3I_{\rm aniso}\), their sum and the
depolarization ratio. These are rotational averages for the stated
polarization convention, not oriented single-crystal tensor intensities.

The Python Raman helper returns a reduced Kubo spectrum when `beta=None`.
Supplying beta, as the workflow does, applies the detailed-balance conversion
\(\beta\omega/[1-\exp(-\beta\omega)]\) for Stokes scattering. The optional
`laser_wavenumber_cm1` adds the fourth power of the scattered frequency;
frequencies at/above the laser are excluded. This conversion is distinct from
applying an extra empirical correction to an already quantum Kubo spectrum.

SFG uses the stated convention
\(C_{ijk}(t)=\langle\delta\mu_k(0)\delta\alpha_{ij}(t)\rangle\),
\(R_{ijk}(t)=-\beta\,dC_{ijk}/dt\), and
\(\chi_{ijk}(\omega)=\int_0^{t_{\max}}R_{ijk}(t)e^{+i\omega t}dt\).
Its saved NPZ includes complex susceptibility, the signed imaginary part and
\(|\langle\chi\rangle_{\rm replicas}|^2\). Responses are averaged *before*
squaring, preserving cancellation between inversion-related contributions.
The default lag taper is unity at zero lag and vanishes at the largest lag.
Derivatives use finite differences; sampling cadence and lag cutoff must be
converged. The caller chooses lab axes and selects a physical interface. The
workflow does not automatically separate the two sides of a slab.

All spectra are **relative**. Electromagnetic constants, optical geometry,
volume/area normalization, refractive-index/Fresnel factors, instrumental
effects and an electronic nonresonant SFG background are absent. Replicas are
kept separate; time origins and Cartesian components are not counted as
independent replicas. Peak-normalized errors condition on a common fitted
scale. The output always reports `physical_converged=false`.

`results/spectroscopy-traces.npz` stores `[replica,time,...]` averaged property
traces. `ir-spectrum.npz`, `raman-spectrum.npz` and `sfg-spectrum.npz` contain
frequency axes, spectral arrays and conventions. Frequency axes in Hz/cm⁻¹ do
not silently change the underlying spectral density measure. Per-replica
checkpoints include state, RNG and the complete recorded trace prefix. Offline
reanalysis uses the saved traces without loading a PES or property model.

## f-CMD fitting and QCMD/h-CMD boundary

`centroid_force_training_data(q, physical_forces)` exports Cartesian centroids
and mean physical bead forces from canonical PIMD arrays `[frame,bead,atom,3]`.
It must receive **unbiased physical forces**, excluding springs and external
biases. These are noisy training targets for the conditional centroid mean
force. Fit a conservative model at the intended temperature, test held-out
trajectories and then propagate it with **one bead and physical masses**.

`fit_quadratic_centroid_pmf` provides a fully working, conservative local
harmonic reference fitter. It jointly fits the independent coefficients of a
symmetric Hessian and reports training RMSE and design rank. Its Cartesian
quadratic basis is limited to a local basin; it is not invariant to global
rotation/translation/permutation and cannot represent a general molecular PMF.
For molecular production, use an appropriate invariant ML model with the same
training-data interface. A provider used by `method='f-cmd'` must explicitly
declare `metadata['centroid_pmf_beta_au']` matching the job temperature. The
reference fitted potential supplies that tag.

For f-CMD, the one-bead geometry represents the centroid. A nonlinear CMD
observable is the conditional mean of its bead-averaged surface at that
centroid, at the same temperature as the PMF. Supply an independently validated
conditional property model when this distinction matters. Evaluating an ordinary
geometry-dependent dipole or polarizability directly on the centroid introduces
an additional observable approximation (a linear dipole has no such difference).
The workflow does not automatically train these conditional property models.

`ibi_update` implements a checked Boltzmann-inversion update on supplied,
consistently normalized coordinate distributions, including damping and
explicit zero-bin regularization. It is a building block for f-QCMD/h-CMD
development. It does not construct quasi-centroids, implement Eckart
constraints or complete an IBI sampling iteration; these methods remain on the
coverage backlog.

## Method references

- Hone, Rossky and Voth, *J. Chem. Phys.* **124**, 154103 (2006),
  [doi:10.1063/1.2186636](https://doi.org/10.1063/1.2186636).
- Hone, Izvekov and Voth, *J. Chem. Phys.* **122**, 054105 (2005),
  [doi:10.1063/1.1836731](https://doi.org/10.1063/1.1836731).
- [DL_POLY Quantum method definitions](https://dlpolydocs.readthedocs.io/en/latest/intro.html),
  including the PA-CMD mass convention and the f-CMD/f-QCMD/h-CMD distinctions.
- Fletcher et al., *J. Chem. Phys.* **155**, 231101 (2021),
  [doi:10.1063/5.0076704](https://doi.org/10.1063/5.0076704).
- [h-CMD method paper](https://doi.org/10.1063/5.0248115).
- [Nonresonant Raman tensor invariants](https://doi.org/10.1021/acs.jctc.0c01323).
- [SFG response-function and interface conventions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7181271/).
