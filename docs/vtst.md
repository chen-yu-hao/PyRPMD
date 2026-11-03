---
orphan: true
---

# Variational transition-state theory and SCT

PyRPMD provides a deterministic minimum-energy-path workflow for comparing
conventional harmonic transition-state theory (TST), canonical variational
transition-state theory (CVT), zero-curvature tunnelling (ZCT), and
small-curvature tunnelling (SCT) on the same potential-energy surface used by a
molecular RPMD rate input.

```bash
pyrpmd vtst examples/rate/research.inp --output runs/vtst \
  --temperature "200 K" --temperature "300 K" --temperature "500 K"
```

The command consumes the resolved `[system]`, `[potential]`, `[reaction]`, and
`[preparation]` sections of an ordinary `task = "rate"` input. It does not
change the RPMD input schema, existing run identities, or checkpoint formats.
The output directory must not already exist. Input validation and all PES work
finish before the complete result tree is atomically published from a sibling
temporary directory.

## Required input state

The workflow currently supports isolated, unconstrained gas-phase molecular
systems. Periodic cells, fixed atoms, and bond constraints are rejected. The
rate preparation must return a numerically validated first-order saddle; use
`preparation.ts_action = "validate"` or `"refine"`. Both MEP termini are
checked against the declared forming and breaking bonds and, when supplied, the
reactant and product fragment partitions. An inconsistent path fails instead
of being relabelled or accepted silently.

The CLI follows the existing bimolecular rate-input contract. The Python core
accepts one or more `PartitionSpecies` objects, so unimolecular partition and
rate normalization are supported by the numerical API even though the shared
rate-input schema does not yet expose a unimolecular reaction declaration.

## Atomic units and reaction coordinate

All internal quantities use atomic units: bohr, hartree, electron masses, and
atomic time with \(\hbar=1\). Input quantities are parsed only at the workflow
boundary. The path is integrated in isoinertial coordinates

\[
  x_i = \sqrt{m_i/\mu}\,q_i,
\]

where the default scaling mass \(\mu\) is one unified atomic mass unit. The
signed arc length \(s\) is therefore reported in bohr and the kinetic energy in
the path coordinate is \(\mu\dot{s}^2/2\).

Starting from the validated saddle, PyRPMD follows both signs of its unique
mass-weighted unstable mode. A normalized mass-scaled Heun predictor-corrector
with a monotonic-energy line search is used. Near a channel minimum, a local
directional Newton length prevents a fixed normalized step from oscillating in
a stiff bound mode. The step, initial displacement, maximum number of points,
force tolerance, and finite-difference Hessian displacement are explicit CLI
options and part of the VTST scientific fingerprint.

At every accepted path point the output retains the Cartesian geometry,
energy, gradient, and unweighted Cartesian Hessian. A callable PES
`hessian(x)` is authoritative. If it is absent, the Hessian is obtained by
central differences of forces. A malformed or failed analytic Hessian is never
silently replaced by finite differences.

## Generalized normal modes

At each \(s\), the unweighted Cartesian Hessian is transformed to the
mass-scaled basis. PyRPMD projects out the path tangent and the non-null global
translations and rotations, then diagonalizes the remaining Cartesian
subspace. This is the Cartesian projection convention; no curvilinear
vibrational-coordinate correction is applied. Every retained transverse mode
must be real and resolved. A changing mode count or an additional transverse
imaginary/zero mode is an explicit failure.

The local ground-state vibrationally adiabatic potential is

\[
  V_a^G(s) = V_{\mathrm{MEP}}(s)
             + \frac{1}{2}\sum_m \omega_m(s).
\]

Cartesian projection is robust and transparent, but its mode phases and
rotation-vibration separation can become less smooth for floppy or nearly
linear geometries. Such systems require an explicit path-step and projection
sensitivity study before interpreting SCT.

## Partition functions and CVT

Reactant and generalized-transition-state partitions use the same explicit
conventions as PyRPMD's harmonic low-temperature cross-checks:

- ideal-gas translation normalized per bohr cubed,
  \(\ln q_\mathrm{trans}=\tfrac32\ln[M/(2\pi\beta)]\);
- classical rigid-rotor rotation with a user-supplied symmetry number;
- quantum harmonic vibration including zero-point energy,
  \(\ln q_\mathrm{vib}=-\sum_m[\beta\omega_m/2+
  \ln(1-e^{-\beta\omega_m})]\);
- a user-supplied positive integer electronic degeneracy; and
- a user-supplied reaction-path degeneracy.

No symmetry number or electronic degeneracy is inferred from coordinates or
atom labels. Use `--reactant-symmetry` and
`--reactant-electronic-degeneracy` with one value per reactant fragment, plus
the corresponding transition-state and reaction-degeneracy options.

For a dividing surface at \(s\), the generalized free-energy barrier is

\[
 \Delta G(T,s) = V_{\mathrm{MEP}}(s)-V_R
 - k_BT\ln\left[g_\mathrm{path}\,
   \frac{Q^\ddagger(T,s)}{\prod_r Q_r(T)}\right],
\]

and

\[
 k^\mathrm{GT}(T,s)=\frac{k_BT}{h}\exp[-\beta\Delta G(T,s)].
\]

The conventional TST value is evaluated at the saddle \(s=0\). CVT selects the
dividing surface with the smallest generalized rate, equivalently the largest
\(\Delta G(T,s)\). Bimolecular results are reported in `m^3/s`; unimolecular
Python-API results are in `s^-1`.

## ZCT and SCT

Let \(V_\max\) be the maximum of the ground-state vibrationally adiabatic
potential and let \(E_\mathrm{th}\) be the higher of its two endpoint channel
energies. For \(E_\mathrm{th}<E<V_\max\), PyRPMD evaluates the semiclassical
action

\[
 \theta(E)=\int_{s_<}^{s_>}
 \sqrt{2\mu_\mathrm{eff}(s)[V_a^G(s)-E]}\,ds
\]

between the two turning points. The uniform ground-state transmission is

\[
 P(E)=\frac{1}{1+\exp[2\theta(E)]}.
\]

This gives \(P(V_\max)=1/2\) without a singular barrier-top branch. Above the
barrier, the reflected-energy continuation is used,

\[
 P(E)=1-P(2V_\max-E),
 \qquad V_\max<E<2V_\max-E_\mathrm{th},
\]

and \(P(E)=1\) at higher energy. Thus the thermal integral includes the
semiclassical nonclassical-reflection branch instead of replacing all
above-barrier transmission by one.

The canonical coefficient multiplying the CVT rate uses the CVT classical
threshold explicitly,

\[
 \kappa^{\mathrm{CVT/MT}}(T)=
 \beta e^{\beta V_a^G(s_*^\mathrm{CVT})}
 \int_{E_\mathrm{th}}^\infty e^{-\beta E}P(E)\,dE.
\]

ZCT sets \(\mu_\mathrm{eff}=\mu\), so the tunnelling path follows the MEP. SCT
uses the centrifugal-dominant small-curvature expression. With local curvature
components

\[
 B_{mF}(s)=L_m(s)^T\frac{d\hat{v}(s)}{ds}, \qquad
 \kappa(s)=\left[\sum_m B_{mF}(s)^2\right]^{1/2},
\]

PyRPMD forms

\[
 \bar\omega(s)=\left\{\sum_m
 \left[\frac{B_{mF}(s)}{\kappa(s)}\omega_m(s)\right]^2\right\}^{1/2},
 \qquad
 \bar t(s)=[\mu\bar\omega(s)]^{-1/2},
\]

and

\[
 \frac{\mu_\mathrm{eff}^{\mathrm{SCT}}(s)}{\mu}=
 \min\left\{
 \exp\left[-2\bar a(s)-\bar a(s)^2+
 \left(\frac{d\bar t}{ds}\right)^2\right],1\right\},
 \qquad \bar a(s)=|\kappa(s)\bar t(s)|.
\]

The normalized-gradient path direction is undefined exactly at a stationary
saddle. The SCT mass ratio there is therefore the linear interpolation of the
nearest nonstationary path values. The zero-curvature limit is handled
analytically: when \(\kappa=0\), the mass ratio is exactly one and SCT reduces
exactly to ZCT.

The implementation is a ground-state, harmonic, uniform-semiclassical
treatment. It does not include anharmonic transverse turning points,
large-curvature tunnelling, least-action optimization, or microcanonical
optimized multidimensional tunnelling. An analytic Eckart quantum transmission
probability is therefore not expected to equal this semiclassical result even
for an Eckart-shaped barrier; the validation separately checks the WKB action
against a continuous integral and records that distinction.

The transmission, reflection, curvature, effective-frequency, and saddle
interpolation equations follow J. L. Bao and D. G. Truhlar, *Chem. Soc. Rev.*
**46**, 7548-7596 (2017), DOI `10.1039/C7CS00602K`, Eqs. 79-86. The underlying
VTST/SCT conventions also follow A. Fernandez-Ramos, B. A. Ellingson,
B. C. Garrett, and D. G. Truhlar, *Reviews in Computational Chemistry* **23**,
125-232 (2007), DOI `10.1002/9780470116449.ch3`, especially Eqs. 159-174;
and the final centrifugal-dominant SCT formulation of Y.-P. Liu, D.-H. Lu,
A. Gonzalez-Lafont, D. G. Truhlar, and B. C. Garrett, *J. Am. Chem. Soc.*
**115**, 7806-7817 (1993), DOI `10.1021/ja00070a029`. The reaction-path
integration is an equivalent stable mass-scaled integrator rather than a
literal copy of M. Page and J. W. McIver, *J. Chem. Phys.* **88** (1988), DOI
`10.1063/1.454172`.

## Output and scientific identity

`OUTPUT/results/vtst.json` contains the complete path, Hessians, generalized
modes, free-energy profiles, tunnelling diagnostics, and rates.
`vtst_path.csv` contains one row per path point and `vtst_rates.csv` one row per
temperature. `manifest.json` records a versioned standalone physics identity,
source hashes, all numerical controls, partition conventions, and the resulting
SHA-256 fingerprint. No RNG is used and the calculation has no checkpoint or
resume state.

`physical_converged` is always `false`. The workflow also leaves
`numerical_converged` false because a single path does not establish numerical
convergence. A defensible study must independently vary at least:

1. MEP step and path extent;
2. saddle and endpoint force tolerances;
3. Hessian finite-difference displacement, unless an analytic Hessian is used;
4. tunnelling energy quadrature order; and
5. PES level, harmonic/rigid-rotor approximations, and the coordinate projection.

A practical minimum is to halve the MEP step twice and require the final two CVT
and CVT/SCT rate changes to remain below the reported tolerance. Endpoint force
flags are diagnostics only for bimolecular asymptotes: channel connectivity and
rate stability with increasing path extent must also be checked.
