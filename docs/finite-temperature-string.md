# Finite-temperature strings in centroid CV space

`pyrpmd.adaptive_string.adaptive_mean_force_string` remains the deterministic
Euclidean geometry optimizer: it accepts a supplied free-energy-gradient
oracle. `pyrpmd.advanced_cv.PathCV` remains a differentiable progress/distance
coordinate relative to supplied Cartesian reference images. Neither existing
object performs thermal sampling.

The new `pyrpmd.finite_temperature_string` module adds restrained sampling and
the CV mass metric while reusing the existing arc-length reparameterization.
It implements the restrained-ensemble CV-string iteration described by
[Maragliano et al., J. Chem. Phys. 125, 024106 (2006)](https://doi.org/10.1063/1.2212942).
The current implementation keeps the number of images and both endpoints
fixed. It supports nonperiodic centroid CVs with full-rank derivatives.

## Physical estimator and limitations

For a harmonic restraint centered at image z with physical stiffness K,
the sampler measures

```{math}
g_K(z)=K\big[z-\langle\xi\rangle_{K,z}\big],\qquad
M_K(z)=\left\langle G\,m^{-1}\,G^T\right\rangle_{K,z}.
```

Here `G = jacobian.sum(axis=1)` is the centroid Cartesian derivative obtained
from the complete `jacobian[D,P,N,3]` contract. The bias contributes `P W` to
the sum-beads Hamiltonian, which is sampled at inverse temperature `beta/P`.
The mean force therefore has physical energy units without an extra bead
factor. The Cartesian restrained ensemble already includes the nonlinear CV's
Jacobian measure; adding another radial/Fixman bias would change the target.

At finite K, `g_K` is exactly the gradient of the Gaussian-smoothed marginal
free energy, assuming exact equilibrium sampling. The mean metric and resulting
path approach their constrained-CV limits as the restraints stiffen. The method
does **not** remove finite-restraint smoothing automatically. Test stiffness,
timestep and image spacing independently before interpreting an energy barrier.

Each iteration projects `M_K g_K` perpendicular to the current string, moves the
interior images and reparameterizes by Euclidean arc length in the **declared CV
coordinates**. CV units and scaling matter. A path in a reduced set of CVs does
not by itself define a kinetic rate or guarantee an isocommittor surface.

## Sampling API

```python
from pyrpmd.finite_temperature_string import (
    RestrainedStringSampler, finite_temperature_string,
)

# One independent Dynamics and State per image; CV dimension is >=2.
sampler = RestrainedStringSampler(
    dynamics, cv, states, stiffness=50.0,
    equilibration_steps=500, sampling_steps=10000,
    sample_interval=5, block_size=100, gamma_centroid=0.1,
)
result = finite_temperature_string(
    images, sampler, step=0.01, tolerance=0.001,
    max_iterations=50, checkpoint="thermal-string.npz",
)
```

`dynamics` must contain independent ordinary `Dynamics` objects and independent
RNG objects with distinct initial states. Identically seeded image streams are
rejected. The sampler uses explicit PILE-L with Cayley or exact normal-mode
free propagation. Its supported scope excludes atom constraints, configured
thermostat/pressure plans, force splitting and periodic CV seams; these fail
before propagation. All images must share masses, temperature and bead count.
Sampling controls are in atomic units. Each iteration performs fresh
equilibration after moving the restraint and keeps complete time blocks.

`result.free_energy` integrates the sampled gradient along the returned path,
using image zero as the zero of energy. It is a **path profile of the restrained
free energy**, not a histogram PMF in an independently defined path-progress
coordinate. Finite image spacing adds quadrature error. The reported standard
error includes reused-node covariance in the line integral and assumes separate
image simulations are independent. It is conditional on the final image
locations and does not include uncertainty from selecting/adapting the path.

The example `examples/finite_temperature_string.py` integrates out a hidden
harmonic coordinate analytically. Its minimum free-energy valley differs from
the minimum-energy valley, making the thermal contribution directly visible.
Small example defaults demonstrate the API without claiming convergence.

## Errors, convergence and restart

Mean-force and metric uncertainty is estimated with contiguous blocks; the
metric-force drift error retains their within-block covariance. The slowest
resolved CV/amplitude statistical inefficiency is reported as an effective
sample count. `correlation_resolved` requires at least 20 blocks, each at least
five estimated correlation times long. This remains a diagnostic, not proof
that all slow coordinates were sampled.

`converged` is a geometric criterion on the final images, including a selected
multiple of the perpendicular block error. Stiffness bias, equilibration,
sampling, timestep and path discretization require separate studies. The final residual and history refer to the returned images.

Checkpoints are atomic, integrity-checked NPZ archives at complete sampling
iteration boundaries. They contain the string, its current estimates, each
image's q/p/clock, propagation/operator state, RNG state and scientific settings.
An interrupted sampling batch is rolled back; replay starts from the previous
durable boundary. Set `Dynamics.potential_identity` for every image. Built-in CVs
receive semantic identities; unsupported/custom CV objects require an explicit
`cv_identity` string describing their complete mathematical configuration.

```python
continued = finite_temperature_string(
    images, fresh_sampler, step=0.01, tolerance=0.001,
    max_iterations=100, checkpoint="thermal-string.npz", resume=True,
)
```

The initial images and all scientific settings must match; only the total
iteration budget may be extended. The fresh sampler's temporary RNG seeds are
replaced by the saved random states. Ordinary restarts do not repeat sampling
at already committed images.

External exact or Monte Carlo sampling can return `StringStatistics` to
`finite_temperature_string` without the bundled dynamics sampler. The utility
`restrained_string_statistics` reduces time-ordered CV values and mass metrics.
Checkpointing an external sampler is rejected because its hidden state and RNG
cannot be verified by this module.

## Connecting the optimized path to a path CV

The optimized images live in CV space. Existing `PathCV` expects Cartesian
reference configurations `[images,N,3]`; therefore CV image coordinates cannot
be passed to it directly. Choose representative restrained configurations near
each converged CV image, check their ordering/alignment, and use their centroid
geometries as the Cartesian reference path. Subsequent umbrella/MBAR sampling
along that `PathCV` estimates its own explicitly chosen probability measure.
