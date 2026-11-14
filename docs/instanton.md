# Ring-polymer instanton rates and convergence workflows

`pyrpmd.instanton` computes low-temperature **stationary imaginary-time paths**
using the existing batched potential interface. The molecular workflow in
`pyrpmd.workflows.instanton` combines these paths with harmonic reactants, the
full fluctuation prefactor, geometric zero modes and bead convergence scans.
The legacy one-dimensional rate API remains available. These calculations are
separate from real-time RPMD propagation.

The companion [low-temperature cross-checks](low-temperature.md) provide the
multidimensional Gaussian rate prefactor, gas translation/rotation factors,
cached-Hessian TST/Wigner/Eckart reports, and open-path ground-doublet splittings
for equivalent bound/constrained wells. Their scope and normalization are
separate from the path-only result documented here.

## Supported calculations and boundaries

| Calculation | Current support |
| --- | --- |
| Physical PES energy/force calls | Existing `Potential.evaluate([P,N,3])`, in atomic units |
| Cartesian instanton action, gradient, Hessian | Arbitrary active Cartesian subset; optional analytic Hessians, otherwise central force differences |
| Thermal periodic orbit | Folded half-ring root or Bofill trust solve; asymmetric barriers supported |
| Crossover temperature | Stationary TS with exactly one negative active mass-weighted Hessian mode |
| Full-ring mode diagnostics | Negative modes, near-zero modes, phase residual, stationary gradient |
| Legacy one-dimensional rate | Exactly one active Cartesian degree of freedom, explicit reactant partition function |
| Multidimensional rate prefactor | `pyrpmd.low_temperature.instanton_rate`; explicit harmonic reactants and gas/constrained convention |
| Ground-doublet splitting | `pyrpmd.tunnelling.tunnelling_splitting`; equivalent bound/constrained wells, open path |
| Bead convergence | `run_instanton(..., mode="rate")`; interpolation, optimization and rate diagnostics at every P |
| Splitting convergence | `run_instanton(..., mode="splitting")`; independent duration and time-step scans |
| Nonadiabatic golden-rule rate | `run_instanton(..., mode="golden_rule")`; two bound/constrained diabatic surfaces, constant weak coupling, normal regime |
| Command line | `pyrpmd instanton INPUT.py --output report.json` |

No transverse vibrational, rotational, translational, molecular symmetry, or
three-dimensional bimolecular standard-state factor is guessed. An optimized
action by itself is **not a multidimensional rate**. Numerical stationarity
does not establish bead, PES, partition-function or semiclassical accuracy. Periodic paths must be supplied
in a continuous, unwrapped Cartesian representation with a PES that implements
the required boundary conditions; the solver does not infer winding sectors.

## Action and stationary-point solve

For even bead count $P$, inverse temperature $\beta=1/(k_B T)$ and
$\beta_P=\beta/P$, the primitive Euclidean action in atomic units is

$$
S_P=\sum_{j=0}^{P-1}\left[
 \frac{\sum_a m_a|q_{j,a}-q_{j-1,a}|^2}{2\beta_P}
 +\beta_P V(q_j)\right],\qquad q_{-1}=q_{P-1}.
$$

The solver finds a stationary point of this action. An ordinary minimization
would usually find a collapsed reactant configuration. The default seed obeys
the link-reflection relation $q_j=q_{P-1-j}$, so only half the ring is optimized.
This fixes the imaginary-time phase without fixing the centroid, which would
distort an asymmetric-barrier orbit. Bead-reflected seeds $q_j=q_{P-j}$ are also
accepted, but can retain an extra unstable discretization phase on coarse rings.
Folded half-rings and full-ring mode checks are standard thermal-instanton
procedures; see the [Molpro instanton implementation](https://www.molpro.net/manual/doku.php?id=instantons)
and [Richardson and Althorpe (2009)](https://doi.org/10.1063/1.3267318).

Spring derivatives are analytic. The physical Hessian comes from an optional
provider or central force differences. Optimization evaluates the unique
folded beads in one batch, rather than evaluating both copies of the ring.
The full-ring Hessian is reconstructed and freshly evaluated after optimization.
Valid numerical results require a noncollapsed
orbit, a stationary active gradient, and exactly one negative mode. The phase
mode is only approximately zero at finite $P$; its spectral-derivative residual
is reported for convergence inspection. A finite-P phase curvature can be
slightly negative, as in the BKMP2 H + H2 benchmark. If the raw Hessian has
more than one negative eigenvalue and the phase residual is resolved within
`phase_tolerance`, the solver counts physical negative modes in the exact
subspace orthogonal to the Fourier phase tangent. The unmodified full spectrum,
raw count and whether phase projection was applied remain in diagnostics.
Another physical negative direction still causes failure. The rate prefactor
independently checks its geometric rigid and phase projections.
The local nonlinear solver needs a
reasonable path seed and does not prove uniqueness or locate every tunneling
path. Dense Hessians are limited to 4096 active bead coordinates; storage is
quadratic and diagonalization cubic in that dimension. The workflow supports
small real molecules within this explicit limit; it is not a large-system
sparse-determinant implementation.

If a stationary transition-state geometry is supplied, the code checks

$$ T<T_c=\frac{|\omega_b|}{2\pi k_B}, $$

where $\omega_b^2$ is the negative eigenvalue of the active mass-weighted PES
Hessian. This is the usual local crossover criterion. Above crossover the
ordinary thermal orbit collapses; this module does not provide a uniform
crossover or high-temperature instanton formula.

## Python usage

```python
import numpy as np
from pyrpmd.instanton import instanton_seed, optimize_instanton

# potential: any existing PyRPMD physical Potential
# ts: stationary transition-state Cartesian geometry [N,3], bohr
# masses: physical masses [N], in electron masses
# direction: a nonzero Cartesian unstable-mode displacement [N,3]
seed = instanton_seed(ts, direction, nbeads=64, amplitude_bohr=1.0)
orbit = optimize_instanton(
    potential, masses, temperature_kelvin=100.0, initial_path=seed,
    transition_state=ts,
    method="bofill", gradient_tolerance=1e-8,
    hessian_step_bohr=1e-4,
)
report = orbit.to_dict()  # includes the path, action, modes and scope
```

An optional `active_mask[N,3]` selects coordinates to optimize. Inactive
coordinates must be identical on every bead. This is a constrained model, not
an automatic removal of molecular rotations/translations. Numerical force
accuracy must be sufficient for the Hessian difference step; test that step
when using noisy electronic-structure or fitted potentials. Default failures
raise `InstantonError`. `raise_on_failure=False` is available for path diagnosis;
the rate function always requires successful paths.

Set `retain_hessian=True` when the orbit will be passed to the multidimensional
rate postprocessor. This caches the final action Hessian without additional PES
calls and leaves the path diagnostic JSON unchanged; see
[Hessian reuse](low-temperature.md#multidimensional-thermal-instanton-rate).

### Steepest-descent RPMD defect

Below the crossover temperature the RPMD rate and the instanton rate differ, to
steepest-descent order, only in the factor of the unstable mode
(Richardson and Althorpe, J. Chem. Phys. 131, 214106 (2009)):

```python
from pyrpmd.instanton import steepest_descent_rpmd_defect

defect = steepest_descent_rpmd_defect(orbit)
defect["log_defect"]             # ln[2 pi / (beta hbar eta_1)]
defect["unstable_frequency_au"]  # eta_1
```

`-eta_1^2` is the negative eigenvalue of the mass-weighted Hessian of the
ring-polymer potential at the orbit; it is read from `orbit.hessian_eigenvalues`,
so no further PES call is made. `eta_1` equals the barrier frequency at the
crossover, where the factor is one, and lies below the first Matsubara frequency
`2 pi / (beta hbar)` under it. The orbit must be converged with exactly one
negative mode; anything else raises `InstantonError`.

`log_defect` is the leading part of `ln(k_QM / k_RPMD)` and not the whole of it:
anharmonic fluctuations about the instanton, and the entire difference above the
crossover, are missing. It is a diagnostic of how far an RPMD rate lies from the
quantum rate and of what a [subspace correction](subspace-instanton-anchor.md)
has to carry; it is not applied to any rate by PyRPMD.

### Hessian providers and optimizer choice

A potential may additionally implement:

```python
def hessian(self, x):
    # x: float64 [batch, atoms, 3], bohr
    # Return unweighted physical d²V/dq², hartree/bohr².
    return h  # float64 [batch, 3*atoms, 3*atoms]
```

The equivalent shape `[batch, atoms, 3, atoms, 3]` is accepted. The full tensor
must be finite, symmetric and **float64**, even when an active mask selects a
smaller subspace. It has no bead scaling or mass weighting. The code falls back
to central force differences only when `hessian` is absent. A noncallable
attribute, wrong shape/precision, asymmetry, nonfinite result or provider error
fails explicitly. `HarmonicState.from_potential` and the open-path splitting
solver use this same contract. A supplied cached stationary-point Hessian still
takes precedence during reactant preparation.

`optimize_instanton` keeps `method="root"` as its existing default. The new
workflow selects `method="bofill"`. Bofill uses a mass-scaled symmetric Hessian,
Newton steps with a bead displacement trust radius, and a squared-gradient
merit function appropriate to an action saddle. Accepted steps update the
Hessian with the Bofill mixture of symmetric rank-one and Powell symmetric
Broyden updates. This method follows the approach of reusing Hessian information
between electronic-structure evaluations; see
[Bofill (1994)](https://doi.org/10.1002/jcc.540150102) and
[Beyer et al. (2016)](https://doi.org/10.1021/acs.jpclett.6b02115).

| Option | Meaning |
| --- | --- |
| `gradient_tolerance=1e-8` | Maximum absolute active **full-ring action gradient**, in atomic units |
| `max_iterations=200` | Residual evaluation budget; final diagnostics require additional PES work |
| `trust_radius_bohr=0.3` | Maximum Euclidean displacement of each unique bead in a Bofill step |
| `hessian_update_interval=0` | No scheduled physical-Hessian refresh; a positive integer refreshes after that many accepted steps |
| `hessian_step_bohr=1e-4` | Central-force displacement when no analytic provider exists |
| `phase_tolerance=0.05` | Maximum relative phase residual permitting its removal from a raw extra-negative-mode count |

Bofill computes a true Hessian initially, on trust-model stagnation, at an
explicit refresh interval when requested, and at the final path. It does not
silently switch to another optimizer. Invalid final mode counts and collapsed
paths are rejected even if a solver reports success. Hessian-update/trust
settings differing from their defaults are rejected with `method="root"`.

`orbit.optimizer_diagnostics` reports actual batch `force_calls`,
`force_configurations`, physical-Hessian builds, analytic-Hessian calls,
accepted/rejected Bofill steps, refreshes and residual history. Counted work
includes validation and final mode diagnostics. A batch call can contain many
configurations: both counts matter for electronic-structure cost. Dense linear
algebra can dominate for an inexpensive analytic PES, so fewer force calls
do not imply the same factor of elapsed-time speedup.

## Unified molecular workflow and command line

```python
from pyrpmd.workflows.instanton import run_instanton

report = run_instanton(
    potential, masses, seed,
    potential_identity={"model": "my-pes", "model_sha256": model_sha256},
    temperature_kelvin=temperature_kelvin,
    transition_state=ts,
    reactants=[prepared_reactant_1, prepared_reactant_2],
    bead_counts=[32, 64, 128, 256],
    optimizer_options={"method": "bofill", "gradient_tolerance": 1e-8},
    rate_options={"rigid_body": True, "symmetry_number": 1,
                  "electronic_degeneracy": 2, "reaction_degeneracy": 2},
    convergence_relative_tolerance=0.01,
    output="instanton-rate.json",
)
```

The symmetry and degeneracy values above are examples, not defaults appropriate
to every reaction. Prepare the isolated `HarmonicState` reactants with the same
PES energy convention, isotope masses and electronic/rotational factors; see
[reactant normalization](low-temperature.md#translation-rotation-and-reactant-normalization).
Use all Cartesian coordinates for an isolated gas molecule. An active mask
instead defines a constrained model and must agree with its reactant state.

At least three strictly increasing even bead counts are required. The workflow
interpolates each optimized path to seed the next P, caches the final Hessian,
and evaluates the full multidimensional rate. It checks a supplied TS and
rejects temperatures at or above its crossover before the scan. Missing TS data
produce an explicit `null` crossover diagnostic; the final negative and phase
modes remain mandatory. An invalid level aborts with its P and reason rather
than silently skipping a failed calculation.

| Result field | Interpretation |
| --- | --- |
| `numerical_converged` | Every requested orbit and prefactor passed their stationarity/mode checks |
| `bead_convergence` | Successive rate changes, the chosen tolerance and the requirement of two passing refinements |
| `discretization_converged`, `converged` | Both of the last two rate refinements passed the requested tolerance |
| `rows` | Every optimized path, Hessian spectrum, cost counters, partition components and rate |
| `physics_fingerprint`, `scientific_identity` | Versioned identity and all resolved scientific settings |

The relative change is `max(k_old,k_new)/min(k_old,k_new)-1`, computed from
logarithmic rates so underflow does not hide a failed convergence test.
The last two P refinements must each increase P by at least a factor of 1.5;
doubling is recommended. A sequence such as `[32,34,36]` is still evaluated,
but its small changes cannot certify convergence. The report records the
refinement ratios and `sufficient_refinement_span`. These checks describe the
requested numerical study and are not a rigorous bound on continuum error.
`potential_identity` is required: identify all PES parameters, external model
files/checksums and remote model versions. Provider source hashes and metadata
are also recorded. The identity includes the initial path, masses, reactants,
energy reference, numerical options, degeneracies and convergence thresholds.
It is separate from existing RPMD identities, which are unchanged.

The command accepts a **trusted Python file** defining `build()` that returns
the keyword arguments of `run_instanton`. This allows real molecular adapters
and analytic Hessians without another input schema or hidden unit conversions:

```bash
pyrpmd instanton examples/instanton/production.py --output instanton-rate.json
```

Inputs use atomic units, except the explicitly named kelvin temperatures. The
Python input source hash also enters the identity. Unknown options are rejected.
Results are written atomically after a successful scan; an unconverged bead
study is saved with its false convergence fields. The output is a deterministic
result, not an optimizer checkpoint. This workflow adds no RNG state and does
not alter existing checkpoint/resume behavior. A previous path can be supplied
explicitly as a new seed, with a new identity.

## One-dimensional thermal rate

For one active Cartesian coordinate, the canonical steepest-descent formula is

$$
kQ_r=\sqrt{-\frac{1}{2\pi}\frac{dE}{d\beta}}\exp(-S),
\qquad E=\frac{dS}{d\beta}.
$$

It follows by applying steepest descent to
$kQ_r=(2\pi)^{-1}\int dE\exp[-\beta E-W(E)]$, with
$W'(E)=-\beta$. This is the one-dimensional limit of the thermal instanton
theory derived by [Richardson (2016)](https://doi.org/10.1063/1.4943866).
The code obtains $E$ from the action derivative at fixed coordinates and
computes $dE/d\beta$ from separately optimized neighboring paths. Two difference
steps, $h$ and $h/2$, expose derivative sensitivity; the default $h=0.01$.
Nonnegative $dE/d\beta$ or an unresolved derivative is rejected.

```python
from pyrpmd.instanton import instanton_rate_1d

result = instanton_rate_1d(
    potential, masses, temperature_kelvin, seed,
    active_mask=np.array([[True, False, False]]),
    reactant_partition=Qr,
    normalization="dimensionless",  # returns rate in s^-1
    transition_state=ts,
)
```

For a one-dimensional scattering reactant, use its partition function **per
bohr** and `normalization="per_bohr"`; the result is a flux-normalized velocity
in m/s. It is not a three-dimensional bimolecular rate coefficient. The
partition function must use the same energy zero as the action. The optional
`energy_reference_hartree` subtracts an energy constant consistently from the
action and imaginary-time energy. Logarithmic rates are retained in output
even when an extremely small rate underflows floating-point exponentiation.

## Reproducible Eckart validation

```bash
PYTHONPATH=src python examples/instanton/eckart.py
python -m pytest -q tests/test_instanton.py
```

The example uses $V(x)=0.01\,\mathrm{sech}^2(0.5x)$ hartree, mass 1822 electron
masses and $T=T_c/2\simeq41.6273$ K. It compares the optimized discrete action
with the analytic continuum instanton and the rate with an independent thermal
integral of the **exact quantum Eckart transmission probability**. The exact
transmission and normalization are documented in Appendix C of
[Dušek, Lawrence and Richardson, arXiv:2509.01454](https://arxiv.org/html/2509.01454v1#A3).
The numerical run is deterministic and writes all parameters and comparisons
to JSON on standard output.

| Beads | Optimized action $S/\hbar$ | Absolute action discretization error |
| ---: | ---: | ---: |
| 32 | 56.812594 | 0.080600 |
| 64 | 56.873163 | 0.020030 |
| 128 | 56.888193 | 0.005000 |
| Analytic continuum instanton | 56.893193 | 0 |

The action error decreases by approximately a factor of four when beads double,
as expected for this primitive discretization. The continuum instanton velocity
is $7.09246\times10^{-22}$ m/s; the exact quantum result is
$7.56940\times10^{-22}$ m/s. Thus, even after bead convergence, the leading
instanton approximation is about **6.30% lower** in this example. This explicitly
separates discretization error from the remaining semiclassical approximation;
agreement of an instanton action with its analytic limit does not make the
result exact quantum dynamics.

## Molecular and cost validation

The production validation script evaluates the exact Eckart transmission
integral independently, compares its continuum semiclassical action and rate,
counts actual PES calls for both optimizers, and optionally builds the same
full Cartesian molecular rate used in the workflow:

```bash
python benchmarks/h3/build_pes.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_production.py \
  --h3 --temperatures 250 300 --h3-beads 32 64 128 256 384 \
  --baseline-ref 8cc402f --output validation-250-300.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_production.py \
  --h3 --temperatures 200 --h3-beads 128 192 288 432 \
  --baseline-ref 8cc402f --output validation-200.json
```

The optional baseline argument actually executes the selected repository
revision's instanton module; it is not an estimated baseline. It should only
name a trusted source revision. With an installed package but no repository
history, omit that argument to compare the two current solvers. Without
`--h3`, only the analytic validation and cost comparison run. Missing BKMP2
data fail explicitly when the molecular test is requested.

### H + H2 on BKMP2

The stored runs use all nine Cartesian coordinates, the checksum-pinned BKMP2
library, hydrogen masses of 1.00782503223 amu, separately prepared H and H2
reactants, and classical rigid rotation. Both reactant H2 and the instanton
use rotational symmetry number 1, while the two equivalent reaction paths
give `reaction_degeneracy=2`. The H-atom doublet degeneracy and the electronic
degeneracy of the reacting system are both 2 and cancel. This follows the
equivalent-transition-state convention described in
[Richardson's review](https://doi.org/10.1080/0144235X.2018.1472353).

The calculated crossover is 345.2593 K. The action uses the isolated H2
minimum as its energy reference. The stationary action-gradient threshold is
`1e-8`, the physical Hessian step is `1e-4 bohr`, and the phase tolerance is
0.05. Rates below are in **cm³/s per reacting pair**; the API returns m³/s and
the validation report applies the explicit factor of one million.

| T / K | Finest P | Calculated instanton rate / cm³ s⁻¹ | Published SCI / cm³ s⁻¹ | Last two rate changes |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 384 | 5.21397e-16 | 5.2e-16 | 0.703%, 0.130% |
| 250 | 384 | 4.34636e-17 | 4.3e-17 | 0.964%, 0.179% |
| 200 | 432 | 2.24618e-18 | 2.2e-18 | 0.452%, 0.201% |

The reference is Table III of
[Karandashev et al., Structural Dynamics 4, 061501 (2017)](https://doi.org/10.1063/1.4996339),
which reports SCI rates on BKMP2 to two significant figures. All three results
are within those rounding intervals. The largest relative difference from
the printed number is 2.10% at 200 K; this does not justify a claim of matching
more digits than the reference supplies. All three scans pass the requested
two-successive-refinement 1% criterion; semiclassical and model validation
remain separate.

The same table's exact quantum rates are 3.24e-16, 3.67e-17 and 2.01e-18 cm³/s
at 300, 250 and 200 K, respectively. The substantial difference near crossover
is a limitation of the leading semiclassical rate, not a bead convergence
failure. Coarse P=32 and 64 paths at 200 K fail the unchanged phase-resolution
guard; the 200 K scan explicitly starts at P=128 instead of weakening it.

Full numeric outputs, counters, spectra diagnostics, potential hashes and
source provenance are available in
[the 250/300 K report](validation/instanton-production-2026-11-02/validation-250-300.json),
[the 200 K report](validation/instanton-production-2026-11-02/validation-200.json),
and [the BKMP2 build record](validation/instanton-production-2026-11-02/bkmp2-build.json).

### Equal-tolerance optimizer cost

For a P=32 Eckart barrier with a separable harmonic bath (six atoms, 18 active
Cartesian components), all methods use the same seed and `1e-8` gradient
tolerance. Both the initial/final Hessian work and final diagnostics are counted.
The comparison against the original `8cc402f` implementation is:

| Method | Force batch calls | Force configurations | Analytic Hessian calls | Median elapsed time / s |
| --- | ---: | ---: | ---: | ---: |
| Original root, force-difference Hessians | 118 | 3776 | 0 | 0.12829 |
| Bofill, force-difference Hessians | 79 | 1264 | 0 | 0.13152 |
| Bofill, analytic provider | 7 | 112 | 2 | 0.12041 |

Bofill reduces force calls by **1.49x** and evaluated configurations by
**2.99x** versus the original implementation. The analytic-provider case
reduces force calls by 16.86x but also makes two Hessian calls; their cost must
be counted for a real electronic-structure provider. The original hybrid
solver already reuses its Jacobian and does not necessarily rebuild it every
iteration.

These runs used an AMD EPYC 9V74 guest, Python 3.12.14, NumPy 2.3.5 and
SciPy 1.17.0, with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
The guest exposed nine logical CPUs with an eight-core CPU quota and an 8 GiB
memory limit. Other jobs shared the runner. The table preserves medians of
three actual measurements: the finite-difference Bofill run was slightly
slower on this inexpensive model. Force-call savings are the demonstrated
benefit; a universal wall-clock speedup is not claimed. For cheap potentials,
the existing root solver can remain the faster choice.

The source-distribution manifest already includes the new Python workflow,
tests, examples and these small JSON/log validation records. The production
path keeps the 4096-coordinate dense-Hessian limit; P=512 for full Cartesian
H3 would exceed it. Sparse fluctuation determinants, uniform crossover theory,
and free-molecule rovibrational splitting networks remain outside this
implementation. The separate golden-rule extension is described below.

## Golden-rule nonadiabatic rates

The optional second stage implements the normal-regime golden-rule instanton
of [Richardson, Bauer and Thoss (2015)](https://doi.org/10.1063/1.4932361) and
[Richardson (2015)](https://doi.org/10.1063/1.4932362). It evaluates a weak-coupling,
leading semiclassical rate between two diabatic surfaces. The electronic
coupling is the real, constant off-diagonal Hamiltonian element `Delta`, in
hartree. Its squared magnitude multiplies the rate; a global sign change does
not change the result. Convergence does not establish that a chosen coupling
is physically in the golden-rule limit.

This implementation supports **bound or Cartesian-constrained coordinates**,
one explicitly normalized reactant, and positive imaginary times on both
surfaces. It does not supply free-molecule translation/rotation factors,
coordinate-dependent electronic couplings, or negative-time paths in the
Marcus inverted regime. Those extensions need different prefactors or saddle
searches; they are rejected rather than approximated by clipping a time split
or deleting small Hessian eigenvalues. The molecular rigid-mode extension is
discussed by [Ansari et al. (2022)](https://doi.org/10.1098/rsta.2020.0378), and
negative-time instantons by
[Heller and Richardson (2020)](https://doi.org/10.1063/1.5137823).

### Two paths and the time saddle

For `tau0 + tau1 = beta`, the action is

$$
S(q,\tau_1)=S_0(q_0,\beta-\tau_1)+S_1(q_1,\tau_1),
\qquad 0<\tau_1<\beta.
$$

Each segment has `Pi` intervals, `Pi+1` nodes, and step `dtaui=taui/Pi`.
Its spring action is analytic; each interval assigns half its potential
weight to each endpoint. The directed paths close as
`path0[0] == path1[-1]` and `path0[-1] == path1[0]`. Both hopping coordinates
remain independent integration variables in the fluctuation Hessian, even
when they coincide at the optimized solution.

At fixed time allocation, spring-preconditioned L-BFGS minimizes the nuclear
action. The preconditioner factors a matrix of size `P0+P1`, shared by all
mass-weighted coordinates. An outer root search matches the segment energies,
`E1-E0=0`. The search must establish a normal-regime sign bracket; merely
finding an energy mismatch smaller than an absolute tolerance does not prove
that such a root exists. Hessian-based Newton refinement is available when
gradient or energy roundoff prevents the requested accuracy. Every physical
Hessian and force evaluation is counted, including final diagnostics.

The final coordinate Hessian must be positive definite, and the reduced time
curvature must be resolved and negative:

$$
\kappa=S_{\tau\tau}-S_{\tau q}H_{qq}^{-1}S_{q\tau}<0.
$$

This Schur complement includes relaxation of all nuclear coordinates when the
time split changes. Using only `S_tautau` gives the wrong prefactor. In atomic
units, with `f` active coordinates and `K=M^(-1/2) Hqq M^(-1/2)`, the complete
coordinate Gaussian factor is

$$
\log A=-\frac12\left[f(P_0\log\Delta\tau_0+P_1\log\Delta\tau_1)
                         +\log\det K\right],
\qquad
k=\frac{g|\Delta|^2}{Q_R}\sqrt{\frac{2\pi}{-\kappa}}\,A e^{-S}.
$$

Here `M` repeats the physical active-coordinate masses on all independent
nodes. The free-particle normalization and Gaussian integrations cancel the
remaining mass determinants and coordinate powers of `2*pi`. There is
**no adiabatic time-translation zero-mode factor**: the two labelled hopping
points already fix the segment boundaries. The only unstable direction is
the time saddle. The algorithm does not apply the adiabatic crossover test or
require a noncollapsed orbit; the high-temperature limit can recover Marcus
theory.

### Python and existing diabatic providers

The low-level APIs are in `pyrpmd.golden_rule`:

```python
from pyrpmd.golden_rule import optimize_golden_rule_instanton, golden_rule_rate

orbit = optimize_golden_rule_instanton(
    surface0, surface1, masses, temperature_kelvin, path0, path1,
    active_mask=mask, energy_reference_hartree=energy_reference,
)
rate = golden_rule_rate(
    orbit, coupling_hartree=delta,
    log_reactant_partition=log_Qr, reaction_degeneracy=1,
)
```

Each surface implements the ordinary physical `Potential.evaluate` contract
and may supply the existing optional analytic `hessian(x)` interface.
`log_Qr` must be dimensionless and include the electronic minimum-energy
Boltzmann factor with the same reference as the action. Subtracting a common
reference from both surfaces shifts the total action by `-beta*reference`,
not by twice that amount. Reported rates are in s^-1, with logarithmic values
retained through floating-point underflow or overflow.

An existing two-state `DiabaticResult` provider can instead be adapted:

```python
from pyrpmd.golden_rule_adapters import diabatic_surfaces

surface0, surface1 = diabatic_surfaces(
    matrix_potential, coupling_hartree=delta,
)
```

The adapter checks the real constant off-diagonal value and its zero gradient
on every evaluation, including finite-difference Hessian displacements. The
rate coupling must have the same magnitude as the adapter declaration. Full
provider errors and failed matrix frames propagate; extra electronic states
and non-Condon couplings are not silently discarded. The adapters borrow the
matrix provider, whose owner remains responsible for closing it. They expose
finite-difference scalar Hessians through the common fallback rather than
inventing a matrix-Hessian protocol.

### Independent segment convergence and CLI

The same production entry supports this calculation:

```python
report = run_instanton(
    potential=(surface0, surface1), masses=masses,
    initial_path=(path0, path1), mode="golden_rule",
    temperature_kelvin=temperature_kelvin, reactants=[reactant],
    active_mask=mask, energy_reference_hartree=energy_reference,
    bead_counts=[(8, 8), (16, 16), (32, 32), (64, 64)],
    potential_identity={"surface0": identity0, "surface1": identity1},
    golden_rule_options={"coupling_hartree": delta,
                         "partition_discretization": "matched"},
    convergence_relative_tolerance=0.01,
    output="golden-rule.json",
)
```

With `m` refinement levels, this evaluates `3*m-2` grids: the joint refinement
and each segment refined independently with the other at its finest requested
resolution. **Time allocation is reoptimized at every grid.** All three scans
must pass two successive refinements, each increasing the relevant interval
count by at least 1.5. Four requested levels therefore require ten complete
optimizations. A converged joint scan alone cannot hide cancellation between
the two discretization errors.

Harmonic reactants default to a partition matched to both optimized segment
steps; see [nonuniform harmonic normalization](low-temperature.md#golden-rule-reactant-normalization).
`partition_discretization="continuous"` explicitly selects continuum harmonic
reactants. An externally supplied `log_reactant_partition` requires
`partition_discretization="external"` and no `reactants` argument. Each choice
is recorded, and none infers molecular standard states or rotational factors.

The golden-rule report has a separate versioned identity. Both PES identities,
coupling, all resolved controls, paths, active mask, masses, reactant data and
three convergence scans enter that identity. Original thermal-rate and splitting
identity contents are unchanged. Outputs are atomic result files, not new
optimizer checkpoints.

```bash
pyrpmd instanton examples/instanton/golden_rule.py --output golden-rule.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_golden_rule.py \
  --modes 4 8 --intervals 8 16 32 64 --output golden-rule-validation.json
```

The first example uses the existing `DiscreteSpinBoson` matrix potential and
the checked diabatic adapter. The validation script additionally compares
against independent analytic harmonic instantons, the classical Marcus limit,
and the full quantum golden-rule correlation integral for a continuous Ohmic
bath. It reports **bead discretization, bath discretization, and semiclassical
approximation errors separately**. A finite set of isolated oscillators has
an unbroadened delta-function spectrum; it is never labelled as having an
exact smooth irreversible rate. The companion
`examples/instanton/golden_rule_reference.py` evaluates its reference formulas
without importing the production optimizer or partition implementation.

### Golden-rule validation results

The reproducible continuous-bath test uses atomic units, `beta=2000`
(`T=157.887512 K`), exponential Ohmic cutoff `omega_c=0.001`, total
reorganization energy `lambda=0.01`, driving bias `epsilon=0.002`, and constant
electronic coupling `Delta=1e-6`. These are specified validation parameters,
not rates copied from a publication. Gauss-Laguerre quadrature supplies the
four- and eight-mode discretizations. The production path uses the existing
matrix provider, its checked Condon adapters, finite-difference Hessians,
`gradient_tolerance=1e-8`, and `energy_tolerance_hartree=1e-10`.

| Bath modes | Final intervals `(P0, P1)` | Production rate / atomic time^-1 | Bead error vs finite-bath analytic SCI | Bath error vs continuous SCI |
| ---: | --- | ---: | ---: | ---: |
| 4 | (64, 64) | 5.478111387e-11 | -0.017426% | +0.265920% |
| 8 | (64, 64) | 5.463387745e-11 | -0.017415% | -0.003578% |

For comparison, the independently evaluated continuum SCI rate is
`5.464534872e-11`, while quadrature of the full quantum golden-rule correlation
function gives `5.544580321e-11` in the same units. Their difference is
**-1.443670%**, which persists after bead and bath convergence. The final
eight-mode production result is `2.258639524e6 s^-1`, or 1.464359% below the
continuous-bath exact GR result. The exact reference is exact within the
golden-rule weak-coupling model; it is not arbitrary-coupling dynamics.

All three eight-mode scans pass the requested 1% criterion:

| Scan | Penultimate rate change | Final rate change |
| --- | ---: | ---: |
| Refine both segments | 0.208801% | 0.052241% |
| Refine surface 0 with `P1=64` | 0.160348% | 0.040252% |
| Refine surface 1 with `P0=64` | 0.047949% | 0.011967% |

Each reported change is `max(k_old,k_new)/min(k_old,k_new)-1`, not an error
estimate against the analytic reference. The finite-bath SCI comparison above
provides that independent estimate. A separate high-temperature one-mode test
at `beta=40`, with `(4,4)`, `(8,8)`, and `(16,16)` grids, is 0.003228% above the
classical Marcus rate. It checks the mathematical high-temperature limit;
it does not validate a molecular potential at that temperature.

| Complete study | Optimizations | Force batch calls | Evaluated configurations | Physical-Hessian batches | Measured time / s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-mode bath, four levels | 10 | 7834 | 286922 | 28 | 4.936 |
| Eight-mode bath, four levels | 10 | 28062 | 1101750 | 72 | 16.963 |
| One-mode Marcus limit, three levels | 7 | 2924 | 34764 | 22 | 1.340 |

Counts include time-bracketing searches, any accuracy refinement, all three
grid scans, and final fluctuation Hessians on both surfaces. All Hessians in
this matrix-adapter benchmark use force differences. Additional Hessian
batches appear only when a coordinate or coupled refinement requires them.
These are individual measured runs on the same AMD EPYC 9V74 guest described
above, with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and both thread controls
set to one. Other work shared the machine; the timings are reproducibility
records and do not establish a speedup over another golden-rule code.

The [complete compact validation report](validation/golden-rule-instanton-2026-11-02/validation.json)
preserves every grid result, partition and prefactor, residual, cost counter,
resolved control, model identity, and source hash. Large paths, full Hessian
spectra, and iteration histories can be regenerated with `run_instanton`;
the [CLI example result](validation/golden-rule-instanton-2026-11-02/workflow.json)
retains those arrays for its smaller example. The [run record](validation/golden-rule-instanton-2026-11-02/run.json)
links the raw test and documentation-build logs. Numerical and discretization
success are reported separately.
