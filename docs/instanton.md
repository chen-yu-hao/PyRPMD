# Ring-polymer instanton cross-check

`pyrpmd.instanton` adds a low-temperature **stationary imaginary-time path**
calculation using the existing batched potential interface. It contains a
Cartesian path solver and a quantitative **one-dimensional** thermal rate
cross-check. This is a separate calculation from real-time RPMD propagation.

## Supported calculations and boundaries

| Calculation | Current support |
| --- | --- |
| Physical PES energy/force calls | Existing `Potential.evaluate([P,N,3])`, in atomic units |
| Cartesian instanton action, gradient, Hessian | Arbitrary active Cartesian subset; finite differences of PES forces |
| Thermal periodic orbit | Folded half-ring stationary-point solve; asymmetric barriers supported |
| Crossover temperature | Stationary TS with exactly one negative active mass-weighted Hessian mode |
| Full-ring mode diagnostics | Negative modes, near-zero modes, phase residual, stationary gradient |
| Quantitative rate | Exactly one active Cartesian degree of freedom, explicit reactant partition function |
| Multidimensional rate prefactor | Not implemented; multidimensional outputs describe the path and action |
| Bead convergence | Explicit repeated calculations; example below compares 32, 64 and 128 beads |

No transverse vibrational, rotational, translational, molecular symmetry, or
three-dimensional bimolecular standard-state factor is guessed. An optimized
action by itself is **not a multidimensional rate**. `physical_converged=false`
is retained in results; numerical stationarity does not establish bead, PES,
partition-function or semiclassical accuracy. Periodic paths must be supplied
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

The physical Hessian is obtained from central differences of forces, evaluating
all beads together. Spring derivatives are analytic. The full-ring Hessian is
computed after optimization. Valid numerical results require a noncollapsed
orbit, a stationary active gradient, and exactly one negative mode. The phase
mode is only approximately zero at finite $P$; its spectral-derivative residual
is reported for convergence inspection. The local nonlinear solver needs a
reasonable path seed and does not prove uniqueness or locate every tunneling
path. Dense Hessians are limited to 4096 active bead coordinates; this initial
implementation is intended for small-system cross-checks.

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
    gradient_tolerance=1e-8,
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
