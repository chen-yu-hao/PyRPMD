# Higher-order equilibrium paths and quantum estimators

PyRPMD provides a closed-path Takahashi–Imada (TI) action, an equilibrium
Monte Carlo sampler, isotope free energies and fractionation, heat capacity,
physical kinetic-energy estimators, and a primitive open-path sampler for a
one-dimensional nuclear momentum marginal. These are numerical Python APIs;
they reuse the existing batched physical PES contract.

The higher-order option in this implementation is **TI**, not PIGLET. No
generic Langevin thermostat is labelled PIGLET. Existing RPMD real-time
propagators remain governed by their physical Hamiltonian.

## Scope and units

Coordinates are bohr, masses are electron masses, potential energies are
hartree, momenta are atomic momenta, and `temperature_K` is kelvin (an explicit
string such as `"300 K"` is also accepted). Use `AMU_TO_AU` when starting with
isotopic masses in atomic mass units. The input PES is temperature independent
and mass independent and returns energy and physical force `-dV/dq` for
`[batch,atom,3]` configurations.

The sampler treats distinguishable nuclei, all Cartesian components, and an
unconstrained, normalizable confining potential. It does not implement nuclear
exchange, periodic winding sectors, periodic-volume normalization, frozen
degrees of freedom, or constraint measure corrections. For a molecular system
with free translation/rotation, the chosen confinement and normalization must
be specified before using these APIs. MC attempt counts are not physical time.

TI improves the **partition-function trace and consistent derivatives of that
trace** to fourth order for suitable smooth potentials. A raw coordinate
histogram from TI sampling is not automatically a fourth-order PMF, RDF, or
density. The TI force-squared action is not exposed as a `Potential` adapter,
and it must not be used to propagate physical RPMD recrossing trajectories.
`OpenPathAction(method="ti")` raises an error because TI's trace theorem does
not establish fourth-order off-diagonal density-matrix accuracy.

## Closed action and sampling

For P beads, beta = 1/(kB T), and hbar = 1 in atomic units, the dimensionless
closed action is

```{math}
S = \frac{P}{2\beta}\sum_{j,a}m_a|q_{j,a}-q_{j-1,a}|^2
  +\frac{\beta}{P}\sum_j V(q_j)
  +\frac{\beta^3}{24P^3}\sum_{j,a}\frac{|F_{j,a}|^2}{m_a}.
```

The third term is zero for `method="primitive"`. The free-particle trace
normalization contributes beta^(-3NP/2) and the corresponding mass factors.
`ActionTerms` includes those normalization derivatives in `energy`.

```python
import numpy as np
from pyrpmd.constants import AMU_TO_AU
from pyrpmd.path_integral import PathIntegralAction, sample_path_integral
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.quantum_statistics import energy_heat_capacity

action = PathIntegralAction(
    HarmonicPotential(k=0.04), [1.007825 * AMU_TO_AU], "300 K",
    method="takahashi_imada",
)
samples = sample_path_integral(
    action, np.zeros((16, 1, 3)),
    steps=30000, burn_in=3000, stride=5,
    centroid_step_bohr=0.1, internal_mix=0.5, seed=42,
)
thermodynamics = energy_heat_capacity(samples)
```

The sampler alternates symmetric centroid moves and preconditioned
Crank–Nicolson (pCN) moves of the free-ring internal modes. The latter preserve
the spring Gaussian exactly, so the Metropolis ratio contains only the
physical-plus-TI potential action. FFT mode transforms avoid dense normal-mode
matrices. `internal_mix` lies in (0,1]; it controls the fraction of a new
free-ring draw in each proposal. Acceptance rates for both move types are
returned. The caller retains ownership of the supplied potential, including
its cleanup/lifecycle.

## Internal energy and heat capacity

The implementation differentiates the action at fixed path coordinates:

```{math}
E = \frac{3NP}{2\beta}+\langle S'\rangle,\qquad
\frac{C_V}{k_B}=\frac{3NP}{2}-\beta^2\langle S''\rangle
              +\beta^2\operatorname{Var}(S').
```

Writing the three action terms as `S_spring`, `S_potential`, `S_TI`, the
per-configuration energy and second derivative are

```{math}
E(q)=\frac{3NP/2-S_{spring}+S_{potential}+3S_{TI}}{\beta},\qquad
S''(q)=\frac{2S_{spring}+6S_{TI}}{\beta^2}.
```

Thus the explicit beta dependence of the TI correction is retained. The heat
capacity is not computed from fluctuations of a fictitious ring-polymer
Hamiltonian. This thermodynamic estimator can have substantial variance at
large P; fewer beads improve discretization cost but do not guarantee lower
total sampling cost. Negative noisy heat-capacity estimates are preserved,
with errors, rather than clipped.

An optional scaled-coordinate estimator avoids the thermodynamic estimator's
spring-energy cancellation and much of its growth of variance with P:

```python
scaled = energy_heat_capacity(
    samples, estimator="scaled_coordinates", finite_difference_step=1e-3,
)
```

At a nearby beta', it maps fluctuations as
`q(beta') = qbar + sqrt(beta'/beta)*(q-qbar)`. The spring action is invariant,
and its coordinate Jacobian reduces the residual trace normalization to
`beta'^(-3N/2)`. With `A(beta')` the physical-plus-TI potential action at those
scaled coordinates, the consistent estimators are

```{math}
E=\frac{3N}{2\beta}+\langle A'\rangle,\qquad
\frac{C_V}{k_B}=\frac{3N}{2}-\beta^2\langle A''\rangle
  +\beta^2\operatorname{Var}(A').
```

Central finite differences use the requested **relative beta step** h and h/2,
with the finer result reported. `finite_difference` records both steps, the
coarse/fine changes in energy and heat capacity, and maximum derivative
changes. This requires four additional batched PES evaluations per saved
configuration. A path-specific reference potential is subtracted before
differentiation to reduce cancellation from the arbitrary PES energy zero.
Finite-difference bias and force/electronic-solver noise remain separate from
sampling error: inspect the step-sensitivity result and refine h when needed.
An excessively small h can amplify numerical noise. Agreement of two steps
is a diagnostic, not a guarantee of derivative convergence. The default
remains `estimator="thermodynamic"` for compatibility.

The scaled estimator also has sampling variance and can be noisy when the
true low-temperature heat capacity is very small. It does not guarantee a
fixed speedup or replace independent sampling/convergence checks. The example
reports both estimators on precisely the same MC configurations.

Errors use contiguous blocks. The default length is based on the largest
estimated statistical inefficiency among energy, its squared fluctuation and
the second derivative. At least four complete blocks are needed for reported
errors. Those errors are conditional on the blocks being long enough; use
independent seeds and longer runs to assess metastability and mixing.
`physical_converged` remains false until an independent convergence study
establishes the required accuracy.

`quantum_kinetic_energy(samples)` uses the primitive centroid-virial estimator
or, for TI, the consistent per-atom mass derivative

```{math}
K_a=\frac{3P/2-S_{spring,a}+S_{TI,a}}{\beta}.
```

The primitive centroid-virial formula is rejected for TI samples. The returned
per-atom quantum kinetic energies can be used in mass thermodynamic integration.

## Isotope free energies and equilibrium fractionation

```python
from pyrpmd.quantum_statistics import (
    isotope_substitution_free_energy, isotope_fractionation,
)

phase_A = isotope_substitution_free_energy(samples_A, new_masses_A)
phase_B = isotope_substitution_free_energy(samples_B, new_masses_B)
result = isotope_fractionation(phase_A, phase_B)
```

`samples_A` and `samples_B` represent independent equilibrium ensembles of
the original isotope. Both phases must use the same temperature and mass
substitutions. `new_masses_*` includes every atom, with unchanged entries for
atoms that are not substituted. The reported convention is

```{math}
\alpha_{A/B}=\frac{(new/old)_A}{(new/old)_B},\qquad
\log\alpha_{A/B}=\log\frac{Z_A^{new}}{Z_A^{old}}
                -\log\frac{Z_B^{new}}{Z_B^{old}}.
```

The default scaled-coordinate estimator maps each atom's fluctuations as
`q' = qbar + sqrt(m_old/m_new)*(q-qbar)`. This preserves the spring action.
The kinetic normalization and coordinate Jacobian leave the prefactor
`prod(m_new/m_old)**(3/2)`. The target physical potential and TI force-squared
term are reevaluated with the new coordinates and masses. Including only the
primitive action in this reweighting would give incorrect TI isotope effects.

Results include stable log partition ratios, substitution free energies,
weight ESS, ESS fraction, maximum weight, a correlation-adjusted ESS, and
conditional block errors. `isotope_fractionation` rejects failed overlap
diagnostics by default. `allow_poor_overlap=True` is available for diagnostic
output only. Good ESS does not prove overlap: unsampled tails are invisible.
For difficult substitutions, sample intermediate masses and use
`fractionation_mass_integration(masses, K_A, K_B, T)`, which integrates
`beta*(K_A-K_B) d(log m)`; refine the mass grid as well as sampling.

These are equilibrium isotope effects, not kinetic isotope effects from rate
ratios. Any KIE requires the separate rate workflow, including its dynamical
correction and convergence checks.

## Physical momentum distribution from open paths

```python
from pyrpmd.path_integral import OpenPathAction, sample_open_path
from pyrpmd.quantum_statistics import momentum_distribution_from_open_paths

open_action = OpenPathAction(potential, masses_au, "300 K", atom=0, axis=0)
# 32 imaginary-time links require 33 configurations.
open_samples = sample_open_path(
    open_action, np.zeros((33, len(masses_au), 3)),
    steps=30000, burn_in=3000, stride=5, seed=43,
)
momentum = momentum_distribution_from_open_paths(
    open_samples, np.linspace(-12.0, 12.0, 241), bandwidth_bohr=0.05,
)
```

Only the selected atom's selected Cartesian component is opened. All other
endpoint components coincide. This boundary condition gives its
one-dimensional momentum **marginal**. Opening all three components and then
using just one displacement component would give a different quantity.
The primitive action uses P spring links and trapezoidal endpoint potential
weights. The opened component uses free-chain cosine modes; closed components
use ring modes. This is actual off-diagonal density-matrix sampling.

For the normalized sampled end-to-end density g(s), the physical momentum
density is

```{math}
n(p)=\frac{\int ds\,g(s)\cos(ps)}{2\pi g(0)}.
```

The factor g(0) is essential: the Fourier transform of a unit-normalized
open-path histogram alone is not a unit-normalized momentum density.
The implementation applies the same symmetric Gaussian kernel to g(0) and
the Fourier transform. The bandwidth is explicit in the result; its default
uses a sample-size rule with the estimated effective sample count. Fourier
temporaries are bounded by 256 momentum points times 4096 samples, and
jackknife calculations group adjacent blocks into at most 128 blocks.
After choosing the bandwidth, block lengths are checked against the origin
density kernel and up to nine representative cosine transforms as well as
the displacement and its square. Uncertainty remains conditional on capturing
the correlation times of all Fourier observables; a few probes cannot prove
that every grid point or hidden metastable state is adequately sampled.

The result reports the integral over the supplied momentum grid, negative
area, and conditional error at each grid point. It never clips negative
statistical oscillations or normalizes away missing momentum tails. Refine
the bandwidth, momentum grid range, P and sampling separately. The sampler
does not assume the momentum distribution is Gaussian. Closed-ring positions
and fictitious sampling momenta are rejected as inputs to this estimator.

## Independent harmonic validation

Run the complete analytic and MC example:

```bash
python examples/quantum_statistics/harmonic_validation.py --output quantum-report.json
pytest -q tests/test_path_integral.py tests/test_quantum_statistics.py
```

The cyclic harmonic Gaussian determinant is evaluated independently of the
samplers. For one Cartesian mode with `beta*hbar*omega=8`, absolute log-partition
errors are:

| P | Primitive | Takahashi–Imada |
|---:|---:|---:|
| 4 | 0.475037 | 0.0534500 |
| 8 | 0.150423 | 0.00481974 |
| 16 | 0.0405650 | 0.000334897 |
| 32 | 0.0103511 | 0.0000215154 |
| 64 | 0.00260135 | 0.00000135410 |
| 128 | 0.000651193 | 0.0000000847787 |

The asymptotic error ratios upon doubling P approach 4 and 16 respectively.
In this harmonic trace comparison, TI with 16 beads is more accurate than
primitive with 128 beads. This is a discretization comparison for a specified
observable and model, not a general eight-fold wall-clock speedup claim.

Tests also compare energy, heat capacity, isotope partition ratios and kinetic
energy with independent finite-P Gaussian ensemble averages, verify the
open-path end-to-end variance against a separate matrix inverse, and check the
Fourier normalization against an analytic Gaussian. The example uses actual
MC samples and records conditional uncertainties and model limitations.

## Primary references

- M. Takahashi and M. Imada, *Monte Carlo Calculation of Quantum Systems. II.
  Higher Order Correction*, J. Phys. Soc. Jpn. 53, 3765 (1984).
  [doi:10.1143/JPSJ.53.3765](https://doi.org/10.1143/JPSJ.53.3765).
- T. M. Yamamoto, *Path-integral virial estimator based on the scaling of
  fluctuation coordinates*, J. Chem. Phys. 123, 104101 (2005).
  [doi:10.1063/1.2013257](https://doi.org/10.1063/1.2013257).
- M. Ceriotti and T. E. Markland, *Efficient methods and practical guidelines
  for simulating isotope effects*, J. Chem. Phys. 138, 014112 (2013).
  [doi:10.1063/1.4772676](https://doi.org/10.1063/1.4772676).
- B. Cheng and M. Ceriotti, *Direct path integral estimators for isotope
  fractionation ratios*, J. Chem. Phys. 141, 244112 (2014).
  [doi:10.1063/1.4904293](https://doi.org/10.1063/1.4904293).
- D. M. Ceperley, *Path integrals in the theory of condensed helium*,
  Rev. Mod. Phys. 67, 279 (1995).
  [doi:10.1103/RevModPhys.67.279](https://doi.org/10.1103/RevModPhys.67.279).
- L. Lin, J. A. Morrone, R. Car and M. Parrinello, *Displaced Path Integral
  Formulation for the Momentum Distribution of Quantum Particles*,
  Phys. Rev. Lett. 105, 110602 (2010).
  [doi:10.1103/PhysRevLett.105.110602](https://doi.org/10.1103/PhysRevLett.105.110602).

The implementation uses direct open-path sampling; the last reference supplies
the physical density-matrix/Fourier context, not a claim that its displaced-path
free-energy algorithm has been implemented here.
