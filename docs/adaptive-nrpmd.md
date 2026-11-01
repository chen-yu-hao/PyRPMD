# Adaptive reaction paths at constant potential with thermal NRPMD

This experimental Python workflow connects the existing signed thermal NRPMD
sampler and propagator to a separable constant-potential matrix model. It learns
a restrained path in collective variables, freezes its umbrella windows, and
reweights fresh production samples before launching unbiased NRPMD trajectories.

[中文说明](zh/manual/adaptive-nrpmd.md) · [Thermal NRPMD](thermal-nrpmd.md) ·
[Electrode models](electrochemistry.md)

## Run the complete example

Install the checkout as usual, then run from the repository root:

```bash
python examples/electrochemistry/adaptive_constant_potential.py \
  --beads 2 --potential-V -0.3 --output runs/adaptive-electrode.json
```

`--quick --beads 1` provides an inexpensive classical-limit smoke test; the
short `--quick --beads 2` budget can correctly return unavailable sign/ESS
diagnostics. `--beads`, `--potential-V` and `--seed` are explicit model parameters.
The output records the initial and
final path, sign and overlap diagnostics, electron-number solver residuals,
and, when reweighting is available, short unbiased trajectories and their
propagation Hamiltonian drift. Every output retains `physical_converged=false`.
The analytic two-state curved valley is an interface/science test model, not a
calibrated electrode material or a prediction of a chemical reaction rate.

This is a Python API workflow. It does not introduce an `.inp` task or a new
`pyrpmd run` mode, and does not claim checkpoint/restart for the entire adaptive
outer loop. The underlying individual thermal samplers retain their existing
snapshot/restore support, now including the frozen umbrella identity.

## 1. The constant-potential matrix model

```python
from pyrpmd.electrochemistry import (
    ElectronReservoir, AdiabaticElectrodePotential,
    ConstantPotentialDiabaticPotential,
)

reservoir = ElectronReservoir(
    potential_V=-0.3,
    reference_potential_V=0.0,
    chemical_potential_at_reference_au=0.1,
    potential_reference="explicit model calibration",
)
electrode = AdiabaticElectrodePotential(
    charging_model, reservoir, electron_bounds=(-5.0, 5.0),
)
potential = ConstantPotentialDiabaticPotential(bare_matrix_model, electrode)
```

`bare_matrix_model` returns `DiabaticResult` with the existing full matrix
contract: `matrix[B,S,S]` and **positive** gradients `[B,S,S,N,3]` in atomic
units. `charging_model.evaluate(q, electron_numbers)` returns the existing
`ElectronicPESResult`: energy, physical nuclear forces, and positive `dE/dN`.

The explicitly separable Hamiltonian is

\[
\mathbf V(R,N)=\mathbf V_{\rm bare}(R)+E_{\rm charge}(R,N)\mathbf I,
\qquad
\mathbf V_\mu(R)=\mathbf V_{\rm bare}(R)+\Omega_{\rm charge}(R;\mu)\mathbf I,
\]

\[
\Omega_{\rm charge}(R;\mu)=\min_N\{E_{\rm charge}(R,N)-\mu N\},
\qquad
\mu(U)=\mu_{\rm ref}-(U-U_{\rm ref})\,\mathrm{EV\_TO\_AU}.
\]

The charging term must be absent from the bare matrix. All states share the
same charge response. The existing solver requires a smooth, strictly convex
charging model with an interior root of `dE/dN=mu`, and rejects unconverged or
unbracketed roots. Envelope-theorem nuclear forces are converted to positive
matrix gradients with the correct sign. The adapter retains electronic gaps
and couplings at fixed geometry, including complex Hermitian couplings.

`potential.solve(q)` additionally exposes electron numbers, chemical potentials
and residuals. Both providers are borrowed; the caller closes their contexts.
No native binding for either separate model is advertised for the composite.

This is **BO constant-potential polarization**, evaluated independently at each
nuclear configuration/bead. It changes nuclear forces and can therefore alter
a path and its subsequent electronic dynamics indirectly. It does not directly
shift an electron-transfer energy gap at fixed geometry. It is not a full
Faradaic reservoir model, state-dependent charging, electron-number jump
dynamics, or a shared-electron-number grand-canonical ensemble. The latter
remains available through the separate `GrandCanonicalSampler` interface.
Their initial ensembles must not be silently interchanged. [4,5]

The path runner also accepts an existing provider for a physically consistent
full constant-potential matrix, including state-dependent gaps/couplings if the
model defines them. Such a provider can replace the separable wrapper directly;
this extension does not derive that matrix from unspecified charged surfaces.
Although the matrix wrapper preserves complex Hermitian couplings, the current
SEO thermal sampler/path workflow requires **real symmetric** matrices; complex
phase reweighting is not implemented.

## 2. Signed thermal umbrellas

```python
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.thermal_nrpmd import ThermalNRPMDSampler

cv = CVStack([Cartesian(0, axis="x"), Cartesian(0, axis="y")])
bias = UmbrellaBias(cv, center=[0.0, 0.3], stiffness=[5.0, 5.0])
sampler = ThermalNRPMDSampler(
    potential, masses, beta, initial_ring, nstates=2,
    seed=17, bias=bias,
)
window = sampler.sample(500, burn_in=200, stride=2)
```

The sampler continues to sample the absolute SEO-projected thermal trace [1].
It does not replace that trace with a Boltzmann factor of the real-time mapping
Hamiltonian. Initial SEO signs and electronic projector estimators are retained.
With `P` beads, the existing bias contract returns `B=P*W`; the new log target is

\[
\log|\rho_k|=\log|\rho_0|-\frac{\beta}{P}B_k
             =\log|\rho_0|-\beta W_k.
\]

This factor is independent of bead count for a physical centroid restraint.
Nuclear proposals include it; mapping-only proposals retain the unchanged bias
and the original independence-Hastings correction. A failing sweep still rolls
back coordinates, mapping variables, random generator and counters together.

Only an unmodified `UmbrellaBias` with supported built-in molecular CVs is
accepted. The sampler copies the CV and bias, freezes its internal arrays, and
binds snapshots to the semantic CV/center/stiffness identity. History-dependent
biases, arbitrary mutable Python CV callbacks, and unsupported periodic/surface
CV identities are rejected. Both thermal correlation entry points reject
unreweighted biased ensembles.

## 3. Adaptive restrained path

```python
from pyrpmd import AdaptivePathOptions, run_adaptive_nrpmd_path

options = AdaptivePathOptions(
    max_iterations=5,
    pilot_samples=200, pilot_burn_in=200,
    production_samples=1000, production_burn_in=500,
    production_stride=3,
    relaxation=0.25, max_step=0.15, path_tolerance=0.02,
    min_average_sign=0.1, min_effective_samples=20,
)
result = run_adaptive_nrpmd_path(
    potential, masses, beta, initial_images, 2, cv,
    stiffness=[5.0, 5.0], cv_scales=[1.0, 1.0],
    options=options, seed=42,
)
print(result.summary())
```

`initial_images` has shape `[K,P,N,3]`, with at least three images. Endpoint CV
centers remain fixed while endpoint nuclei still sample their restrained
windows. The bead count, masses, inverse temperature and potential remain fixed.
The initial images are a proposed pathway: this algorithm does not discover
unknown basins or guarantee finding every competing channel.

At each iteration, independent random streams run fixed umbrellas. Signed
restrained means estimate the force on each center. In scaled coordinates
`y=xi/cv_scales`, the update applies the stiffness matrix transformed into this
metric, removes the tangent component, damps/caps the force step, and
redistributes images at equal arclength. `max_step` caps the perpendicular force
step before arclength redistribution. Mixed-unit CVs require explicit scales;
periodic path topology is not inferred.

This follows a **finite-restraint, smoothed free-energy path in the selected
Euclidean metric**. It is inspired by string methods [2], without implementing
all metric/conditional-expectation terms needed to identify a general CV MFEP or
committor. Finite stiffness, coordinate scales, initialization, sampling noise
and step size affect the learned path. A small last path displacement is a
geometric stopping criterion, not evidence of physical convergence.

An unresolved signed mean stops adaptation and is recorded as
`unresolved_pilot_sign`. It is not replaced by an absolute-weight mean. The last
defined path can still receive a larger production budget. After adaptation,
centers are frozen and new random streams plus a fresh burn-in generate
production chains; pilot configurations only initialize these chains. **No
pilot sample is included in MBAR.** Monte Carlo steps and adaptation rounds do
not measure physical time.

## 4. MBAR and physical averages

`reweight_thermal_umbrellas` can also combine separately sampled windows. It
checks their original bias/CV identities, identical provider object, beta,
masses, bead count, electronic state count and source distribution. All windows
must share an unchanged base potential and mapping convention.

With that common base density, MBAR uses only
`u_kn=beta*W_k(x_n)` and targets the **unbiased absolute SEO measure**. The
existing positive-density MBAR solver [3] is reused. The signed physical average
is then

\[
\langle A\rangle=\frac{\sum_n w_n\,\sigma_n A_n}
                         {\sum_n w_n\,\sigma_n},
\qquad w_n\geq0,\quad\sigma_n\in\{-1,+1\}.
\]

```python
weighted = result.reweighted
print(weighted.summary())
mean_cv = weighted.expectation(weighted.cv_values)
electronic_population = weighted.expectation(weighted.ensemble.electronic_populations)
```

Physical estimates and parent resampling are unavailable if MBAR has not
converged, the overlap graph is disconnected, the weighted average sign is too
small/unresolved by the diagnostic, or the target effective-sample diagnostic
is below the requested threshold. The measured signs are never clipped.
`absolute_measure_window_free_energies` are explicitly **not** physical signed
PMF values.

`importance_ess=1/sum(w**2)` measures weight concentration. The reported
correlation-adjusted ESS and sign error use source-window CV/sign/projector
autocorrelations as heuristic diagnostics. They ignore uncertainty in fitted
MBAR weights and do not constitute source-chain bootstrap error bars.
`min_effective_samples` checks the positive absolute-measure ESS; the separate
`signed_ess_diagnostic` can be smaller and is not an additional threshold. All
source window/sample IDs remain available. Independent chains, adequate
equilibration and target-region coverage still require separate checks.

## 5. Unbiased real-time nuclear/electronic propagation

```python
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics, thermal_population_rate

parents = weighted.resample_parents(200, seed=23)
dynamics = ThermalNRPMDDynamics(
    potential, masses, beta, len(parents.states[0].q),
    dt=0.01, nstates=2, native=False,
)
correlation = thermal_population_rate(
    parents, dynamics, 0, 1, steps=100, min_average_sign=0.1,
)
```

Positive MBAR weights select entire parent states; every bead, nuclear momentum,
mapping coordinate/momentum, initial SEO sign and projector estimator is copied.
The same constant-potential matrix is used for propagation. No umbrella or
sampling thermostat acts in these trajectories. There is no electron-number
Monte Carlo clock in the dynamics.

Use the existing **`ThermalNRPMDDynamics` scalar/traceless split**. Thermal
mapping norms fluctuate, so the scalar charging force must act once, without
being multiplied by that norm. The one-electronic-state limit is ordinary
RPMD, including nuclear springs and the selected free-ring integrator.

Resampling duplicates configurations; it does not create new independent
equilibrium information. The output retains source ancestry. Existing rate
summaries suppress naive parent-block uncertainty/plateau certification for
these ensembles and request a bootstrap of the original source chains and
MBAR fit. NRPMD remains an approximate real-time method whose propagation need
not preserve the full initial SEO density [1]. A finite-time curve is not a
validated rate. Bead number, time step, sign, source-chain mixing, CV choice,
and a physically meaningful time plateau need independent convergence studies.

## References

1. Richardson and Thoss, *Communication: Nonadiabatic ring-polymer molecular
   dynamics*, JCP 139, 031102 (2013), [doi:10.1063/1.4816124](https://doi.org/10.1063/1.4816124).
2. Maragliano et al., *String method in collective variables: minimum free
   energy paths and isocommittor surfaces*, JCP 125, 024106 (2006),
   [doi:10.1063/1.2212942](https://doi.org/10.1063/1.2212942).
3. Shirts and Chodera, *Statistically optimal analysis of samples from multiple
   equilibrium states*, JCP 129, 124105 (2008),
   [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177).
4. Scalfi et al., *Charge fluctuations from molecular simulations in the
   constant-potential ensemble*, PCCP 22, 10480–10489 (2020),
   [doi:10.1039/C9CP06285H](https://doi.org/10.1039/C9CP06285H).
5. Sun et al., *Probing nuclear quantum effects in electrocatalysis via a
   machine-learning enhanced grand canonical constant potential approach*,
   Nature Communications 16, 3600 (2025),
   [doi:10.1038/s41467-025-58871-7](https://doi.org/10.1038/s41467-025-58871-7).
