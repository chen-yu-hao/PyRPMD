# Low-temperature rates, splittings and diagnostics

These APIs extend the [stationary instanton path solver](instanton.md). They are
separate from real-time RPMD propagation and do not modify an RPMD transmission
coefficient or mark an RPMD production calculation as converged.

| Quantity | API | Scope |
| --- | --- | --- |
| Classical and quantum harmonic TST | `harmonic_cross_checks` | Bound/constrained models or isolated gas molecules |
| Wigner and asymmetric Eckart estimates | `harmonic_cross_checks` | Explicitly labelled correction to a selected TST baseline |
| Multidimensional thermal instanton rate | `instanton_rate` | Full Gaussian fluctuation determinant, including transverse modes; harmonic reactants |
| Translation/rotation in the rate | `rigid_body=True` | Isolated gas, classical linear/nonlinear rigid rotors, explicit symmetry/electronic factors |
| Ground-state tunnelling doublet | `tunnelling_splitting` | One open kink between symmetry-equivalent, bound/constrained Cartesian wells |
| Nonadiabatic golden-rule instanton | `pyrpmd.golden_rule` | Normal regime; weak constant real coupling; bound/constrained reactants |
| Reproducible convergence studies | `pyrpmd.workflows.instanton.run_instanton` | Molecular thermal rates or independent duration/grid studies for splittings |

**Not covered:** uniform crossover theory, hindered or quantum rotors, gas
rovibrational splitting patterns, asymmetric-well or multiwell splitting
networks, and production validation of a particular molecular PES. The splitting
API deliberately rejects free translational/rotational modes. The thermal-rate
API does support these modes, through the external partition factors described
below. Every result retains `physical_converged=False`: stationarity and a finite
prefactor do not establish semiclassical, bead or PES accuracy.

## Reuse the stationary-point Hessians

`HarmonicState` contains a geometry `[atoms,3]` in bohr, physical masses in
**electron masses**, an electronic energy in hartree, and an **unweighted
Cartesian PES Hessian** in hartree/bohr². With an `active_mask`, the Hessian is
restricted to the selected Cartesian components in NumPy boolean-index order.
It is not a normal-mode frequency array, a mass-weighted Hessian, or a
ring-polymer action Hessian.

```python
from pyrpmd.low_temperature import HarmonicState, harmonic_cross_checks, print_cross_checks

# Reuse a Hessian already obtained during stationary-point preparation.
# from_potential makes one energy/force call to check stationarity. It only
# uses potential.hessian when available, otherwise central force differences.
ts_state = HarmonicState.from_potential(
    potential, ts, masses, hessian=ts_cartesian_hessian,
    rigid_body=True, symmetry_number=1, electronic_degeneracy=1,
)
reactant = HarmonicState.from_potential(
    potential, reactant_geometry, masses, hessian=reactant_cartesian_hessian,
    rigid_body=True,
)
report = harmonic_cross_checks(100.0, ts_state, [reactant])
print_cross_checks(report)
```

For a bimolecular entrance, supply **two isolated reactant states**, each with
its own geometry, masses, Hessian and electronic energy. Their combined atom
masses must match the reacting system, including isotope substitutions. The
isolated-fragment energies and the combined-system PES must use compatible
energy zeros. A frozen Cartesian model instead uses `rigid_body=False`, a
single reactant with the same atom order/mask/frozen coordinates, and no gas
translation or rotation factor.

The constructor `HarmonicState(...)` also accepts already saved energies and
Hessians without any PES calls; the caller must establish stationarity. Rigid
modes are constructed from geometry and mass, not identified by deleting a
fixed number of small eigenvalues. An inconsistent rigid-body Hessian, an extra
internal zero mode, a nonminimum reactant or a TS with other than one negative
mode is rejected. Soft torsions are not automatically converted into rotors.

The estimates themselves make **no PES calls**. However, a TS Hessian alone
cannot provide reactant partition functions, barrier heights or symmetry
numbers. Missing reactant/product data must be calculated or supplied; they are
not inferred from the local TS curvature.

`state.partition_diagnostics(beta, nbeads=P)` exposes the vibrational,
translation, rotation and electronic degeneracy contributions, positive
frequencies, zero-point energy, removed rigid-mode count and normalization.
The electronic energy Boltzmann factor is kept separate so that one shared
energy reference can be applied to the sum of reactant energies. The molecular
rate report includes this breakdown for every reactant and the combined
`log_reactant_electronic_boltzmann` term.

### Printed and JSON reports

A cached-Hessian input is included:

```bash
python -m pyrpmd.low_temperature examples/instanton/cross_checks.json
python -m pyrpmd.low_temperature examples/instanton/cross_checks.json --json
```

The JSON has `temperature_kelvin`, `transition_state`, `reactants`, and optional
`products`, `eckart_barriers_hartree`, `reaction_degeneracy` and
`tunnelling_baseline`. Each stationary-point object contains the arguments of
`HarmonicState`; boolean arrays use JSON `true`/`false`. This is a standalone
postprocessor, not a new `task` value in a simulation input. It prints classical
TST, harmonic TST, Wigner and Eckart estimates, their units, the chosen baseline,
and warnings. An unavailable Eckart estimate is printed as unavailable, not zero.

The supplied analytic two-dimensional quartic example at 100 K prints
approximately 0.84244 s⁻¹ (classical TST), 1407.7 s⁻¹ (quantum harmonic TST),
14248 s⁻¹ (Wigner), and 1.3823×10⁷ s⁻¹ (Eckart). These different approximations
are **not interchangeable low-temperature reference truths**.

## TST, Wigner and Eckart conventions

For inverse temperature `beta=1/(kB*T)`, the classical harmonic oscillator
partition is `1/(beta*omega)`. The quantum harmonic partition is
`1/(2*sinh(beta*omega/2))`, including its zero-point energy. With the unstable TS
mode omitted,

$$
k_{\rm TST}=\frac{g}{2\pi\beta}
 \frac{Q^\ddagger}{\prod_i Q_i}\exp[-\beta(V^\ddagger-\sum_i V_i)].
$$

`classical_tst` uses classical vibrational partitions; `harmonic_tst` uses
quantum harmonic vibrational partitions. Both use the same classical gas
translation/rotation convention. The integer `reaction_degeneracy=g` is a
reaction-path multiplicity, distinct from rotational symmetry numbers and
state electronic degeneracies. Supply these consistently to avoid double counting.

Wigner uses `kappa=1+(beta*abs(omega_b))**2/24`, with **angular frequency in atomic
units**, not cm⁻¹. Its default baseline is `harmonic_tst`; select
`tunnelling_baseline="classical_tst"` explicitly to change it. The report always
names this choice and warns when `beta*omega_b > 1`. Printing a Wigner number
below crossover does not make this weak-tunnelling expansion reliable there.

Eckart fits the imaginary frequency and **both** positive barrier heights.
When product stationary-point data are supplied, the default harmonic baseline
uses ZPE-corrected forward/reverse heights. The classical baseline uses
unmodified electronic heights. Alternatively, provide
`eckart_barriers_hartree=(forward, reverse)` explicitly in the desired energy
convention. These explicit heights override the automatic choice; the output
records that convention. No symmetric barrier is assumed when product data are
missing, and nonpositive effective heights are marked unavailable.

The exact one-dimensional Eckart transmission is integrated as

$$
\kappa_E=\beta e^{\beta V_f}
 \int_{\max(0,V_f-V_r)}^\infty P_E(E)e^{-\beta E}\,dE.
$$

The channel threshold is enforced. The shallow-barrier trigonometric branch is
included, not replaced with the hyperbolic branch. Logarithmic transmission and
scaled adaptive quadrature avoid subtracting nearly equal coshes or multiplying
an overflowing correction factor by an underflowing rate. The output preserves
`log_kappa`, `log_rate_au` and `log_rate_si`; a floating-point overflow is `null`
and an underflow is zero with an explicit `range_status`. Thus logs remain
available even when ordinary floating-point values cannot represent the result.

## Multidimensional thermal instanton rate

```python
from pyrpmd.instanton import RingPolymerAction, optimize_instanton
from pyrpmd.low_temperature import instanton_rate

orbit = optimize_instanton(
    potential, masses, temperature_kelvin, seed,
    transition_state=ts, retain_hessian=True,
)
action = RingPolymerAction(potential, masses, temperature_kelvin)
rate = instanton_rate(
    action, orbit, [reactant], rigid_body=True,
    symmetry_number=1, electronic_degeneracy=1, reaction_degeneracy=1,
)
```

Use the **same** active mask, Hessian displacement and energy reference in the
optimizer and action object when overriding their defaults. For a constrained
model, pass that mask to both and leave `rigid_body=False`.

`retain_hessian=True` retains the optimizer's final unweighted Cartesian action
Hessian. The rate postprocessor reuses it, with one PES evaluation to check the
path's stationarity and energy reference. A saved action Hessian can instead be
passed as `action_hessian=...`. Without a cache, it is computed once. The cache
is excluded from `InstantonResult.to_dict()`; the existing path-only JSON scope
and the old `instanton_rate_1d` API remain unchanged.

### Determinant and collective coordinates

Let `P` be the bead count, `dtau=beta/P`, `H_S` the Cartesian action Hessian, and
`K=M^(-1/2) H_S M^(-1/2)` its bead-major mass-weighted form. With
`B=sum_j,a m_a*(q[j,a]-q[j-1,a])**2`, the Gaussian ring-polymer expression is

$$
kQ_r=\frac{1}{\Delta\tau}\sqrt{\frac{B}{2\pi\Delta\tau}}
 Q_{\rm trans}^{\rm inst}Q_{\rm rot}^{\rm inst}
 \prod_{k}'\frac{1}{\sqrt{\Delta\tau\,|\lambda_k(K)|}}
 e^{-S_P}.
$$

Electronic and path-degeneracy factors multiply this expression. The prime
removes geometric global rigid modes and the imaginary-time phase mode, but
**retains the absolute value of the single negative eigenvalue**. Every
transverse fluctuation is included. This is not merely `exp(-S)` or the one-D
period-energy derivative multiplied by a guessed bath factor.

The phase direction is the mass-weighted Fourier derivative of the closed
path. Its residual is measured relative to the unstable curvature, not to the
largest spring eigenvalue. A poorly resolved phase mode, another internal zero
mode, a failed/nonstationary orbit or a wrong negative-mode count causes an
error. The default phase tolerance is a numerical guard, not a convergence
certificate. Dense calculations retain the existing 4096-active-bead-coordinate
limit and require cubic-cost diagonalization.

### Translation, rotation and reactant normalization

For a gas path, external factors belong to the **whole P-bead supermolecule at
`dtau`**, not to the physical molecule at `beta`. Its total mass is `P*M` and
its inertia tensor includes all beads about the global center of mass:

$$
Q_{\rm trans}/V=(M_{\rm super}/2\pi\Delta\tau)^{3/2},\quad
Q_{\rm rot}^{\rm linear}=\frac{2I}{\sigma\Delta\tau},\quad
Q_{\rm rot}^{\rm nonlinear}=\frac{\sqrt\pi}{\sigma}
 (2/\Delta\tau)^{3/2}\sqrt{I_1I_2I_3}.
$$

The extra bead powers cancel against the nonzero free ring modes. Reactant
harmonic partitions use the **same P and dtau**, with squared frequencies
`omega_k**2 + 4*sin(pi*n/P)**2/dtau**2`. Reactant rigid directions retain their
nonzero ring modes and use the corresponding supermolecule external factors.
For a separable transverse oscillator, this construction cancels the spectator
partition exactly at finite P rather than introducing a discretization bias.
For a stationary harmonic reactant, the finite product is evaluated through
the exact identity

$$
Q_{\rm vib,P}(\omega)=
\left[2\sinh\left(P\operatorname{asinh}\frac{\beta\omega}{2P}\right)\right]^{-1}.
$$

This removes work proportional to P from reactant partition evaluation. The
noncentroid rigid-mode product is $P^{-f_0}$ and cancels the supermolecule
external factor $P^{f_0}$ exactly. Reactant diagnostics therefore report the
physical-molecule translation and rotation factors with that cancellation
already applied. The instanton's external factors still describe the entire
noncollapsed supermolecule.

Only one total energy reference is subtracted from the **sum** of reactant
energies. It must match `action.energy_reference`. Reported rates are in s⁻¹
for a single reactant and m³/s for two gas reactants (number-density convention).
For `r` gas reactants, the length power is `3*(r-1)`. There is no hidden
Avogadro factor, standard pressure, standard concentration or cm³ conversion.
Translation and classical rotation are approximations appropriate to the
specified model; quantum rotor populations at very low T need another treatment.

Rate diagnostics report removed translation, rotation and phase modes
separately, relative residuals of the geometric rigid vectors, the retained
negative curvature, the positive fluctuation-mode count, the molecularity,
symmetry number, electronic degeneracy and reaction degeneracy. These expose
the normalization rather than relying on deletion of the smallest eigenvalues.

## Golden-rule reactant normalization

The [golden-rule workflow](instanton.md#golden-rule-nonadiabatic-rates) uses two
diabatic paths with durations `tau0` and `tau1`, whose sum is `beta`. Their
optimized time steps generally differ. A uniform-P harmonic partition cannot
be substituted without introducing a finite-P normalization error. The default
`partition_discretization="matched"` evaluates each reactant normal mode using
the same two primitive propagator grids as the rate numerator.

For an angular frequency `omega`, define

$$
u_i=\omega\Delta\tau_i,\quad
a_i=2P_i\operatorname{asinh}(u_i/2),\quad
A_i=1+u_i^2/4,\quad
c=\frac{A_0+A_1}{2\sqrt{A_0A_1}}.
$$

The two oscillator transfer matrices then give

$$
Q_{P_0,P_1}^{-2}
=4\sinh^2\!\left(\frac{a_0+a_1}{2}\right)
 +2(c-1)\sinh a_0\sinh a_1.
$$

`harmonic_segment_log_partition` evaluates this expression with logarithmic
sums and `c-1=2*sinh((log(A0)-log(A1))/4)**2`, avoiding cancellation or large
hyperbolic overflows. Its cost depends on the number of normal modes, not on
the number of beads. Equal time steps reduce exactly to the existing uniform
primitive harmonic formula. Independent cyclic Gaussian determinants test
the nonuniform formula, including the half potential weights at both hopping
nodes. A separable spectator oscillator identical on both electronic surfaces
cancels exactly at finite P when the matched reactant normalization is used.

The workflow computes the matched partition after each optimized time split.
It includes the reactant electronic degeneracy and the single energy factor
`-beta*(E_reactant-energy_reference)`. `continuous` and explicit `external`
normalizations are separate recorded options. Every golden-rule reactant is
bound or Cartesian-constrained and dimensionless; free gas translation and
rotation factors, including bimolecular standard states, are not supplied by
this implementation. Unlike the adiabatic thermal rate, no phase zero mode is
removed from a golden-rule path whose hopping points are already labelled.

## Ground-state tunnelling splittings

`pyrpmd.tunnelling.tunnelling_splitting` optimizes an **open** imaginary-time
path with fixed endpoint minima. Its duration is `imaginary_time_au`, not an
inverse temperature that can be interchanged with the thermal rate input.

```python
from pyrpmd.tunnelling import tunnelling_splitting

splitting = tunnelling_splitting(
    potential, masses, open_path, imaginary_time_au,
    active_mask=mask, equivalent_wells=True,
    endpoint_hessians=[left_cartesian_hessian, right_cartesian_hessian],
)
```

`equivalent_wells=True` explicitly asserts symmetry equivalence. The code also
checks endpoint stationarity, equal electronic energies and matching positive
harmonic spectra, but those numerical checks alone cannot prove the symmetry.
The open chain has fixed endpoints, so its harmonic reference uses **Dirichlet**
rather than periodic spring eigenvalues. After removing the approximate phase
mode, all kink fluctuation eigenvalues must be positive.

For the mass-weighted fluctuation operators `L`, the large-duration formula is

$$
\Delta=2\sqrt{\frac{2S_{\rm spring}}{2\pi}}
 \sqrt{\frac{\det L_{\rm well}}{\det' L_{\rm kink}}}\,e^{-S}.
$$

The collective-coordinate norm `2*S_spring` tends to the zero-energy action
`S0`. Output includes the splitting in hartree and cm⁻¹, its logarithm, action,
determinant ratio, optimized path and phase residual. A periodic thermal orbit
cannot be substituted for this open kink. Free-molecule rovibrational levels,
inequivalent wells and multiple-path symmetry patterns are **not** supplied by
this API. Do not remove free modes with a Cartesian mask and then label the
result as a fully rotationally invariant molecular splitting.

### Duration and grid scans in the same workflow

```python
from pyrpmd.workflows.instanton import run_instanton

report = run_instanton(
    potential, masses, open_path,
    mode="splitting",
    potential_identity={"model": "symmetric-bound-well", "parameters": parameters},
    active_mask=mask,
    bead_counts=[64, 128, 256],
    imaginary_times_au=[tau, 1.25*tau, 1.5*tau],
    splitting_options={"equivalent_wells": True, "gradient_tolerance": 1e-8},
    convergence_relative_tolerance=0.01,
    output="instanton-splitting.json",
)
```

For this mode, `bead_counts` denotes **interval counts at the shortest
duration**, and each open path has one additional endpoint node. The example
uses interval grids `[64,128,256]`, `[80,160,320]` and `[96,192,384]`: each
column increases duration at fixed time step, while each row refines the grid
at fixed duration. Noncommensurate durations/counts are rejected rather than
rounded silently. Seeds extend the fixed well tails and interpolate the kink
on its original physical imaginary-time axis.

The report separates `grid_convergence` and `duration_convergence`, requiring
two successful successive refinements. Grid checks must pass at the last
three durations; duration checks must pass at both of the finest time steps.
The last two refinements must each span at least a factor of 1.5 in interval
count and 1.1 in duration; tiny increments cannot produce a convergence flag.
`physical_converged` remains false. These are deterministic studies with an
independent scientific identity and atomic JSON output, using the same command
entry as thermal rates: [workflow input](instanton.md#unified-molecular-workflow-and-command-line).

Each splitting also reports actual force/Hessian work, whether the existing
L-BFGS solution needed Newton refinement, the endpoint harmonic frequencies,
the time step, `duration * smallest_frequency` and the splitting-to-frequency
ratio. The last ratio helps assess the dilute-instanton assumption; a finite
answer alone does not establish a well-isolated ground doublet. Analytic PES
Hessians are reused through the common provider interface for endpoint and
fluctuation calculations.

## Reproducibility and convergence

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python examples/instanton/low_temperature.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_low_temperature.py tests/test_tunnelling.py
```

Tests cover analytic Eckart scattering (including shallow barriers), adaptive
thermal integration, threshold/detailed-balance checks, classical and quantum
harmonic partitions, bead factors, linear/nonlinear rotors, global translation
and rotation invariance, symmetry factors, SI molecularity, separable transverse
cancellation, energy-origin invariance, cache reuse and backward compatibility.
The full determinant recovers the independent one-D instanton numerator within
0.6% at P=128; this is **discretization agreement with the semiclassical limit**,
not agreement with an exact quantum rate.

For `V=lambda*(x*x-a*a)**2`, the open-path tests compare with
`S0=4*a**3*sqrt(2*m*lambda)/3` and
`Delta=2*omega*sqrt(6*S0/pi)*exp(-S0)`, including the known fluctuation
prefactor. Independent Schrödinger finite-difference diagonalization checks the
remaining semiclassical error in a deeper well. A curved two-dimensional well
checks that the calculation is not restricted to a straight one-D path.

For production, independently vary the ring bead count, the PES Hessian step,
stationary gradient tolerance and reactant treatment. Splittings additionally
need an independent **duration study**, ideally keeping the time step fixed as
the tails are extended, followed by a grid refinement at fixed duration. A
longer path with an unchanged small bead count can make discretization worse.

## References

- Richardson and Althorpe, *J. Chem. Phys.* **131**, 214106 (2009),
  DOI: [10.1063/1.3267318](https://doi.org/10.1063/1.3267318).
- Richardson, *J. Chem. Phys.* **144**, 114106 (2016),
  DOI: [10.1063/1.4943866](https://doi.org/10.1063/1.4943866).
- Richardson and Althorpe, *J. Chem. Phys.* **134**, 054109 (2011),
  DOI: [10.1063/1.3530589](https://doi.org/10.1063/1.3530589).
- Eckart, *Phys. Rev.* **35**, 1303 (1930),
  DOI: [10.1103/PhysRev.35.1303](https://doi.org/10.1103/PhysRev.35.1303).
