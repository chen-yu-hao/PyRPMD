# Rates for nonlinear centroid coordinates

`task = "site_rate"` supports a differentiable **scalar centroid coordinate**,
including bond distances, bond differences, angles, and compatible Python CVs.
It reports a basin-normalized, first-order escape rate in `s^-1`. This works for
an isolated unimolecular system or a system in a fixed periodic cell with all
atoms mobile. The name `site_rate` is retained for input compatibility.

The existing Cartesian, projection, and fractional-coordinate route keeps its
constant-metric estimator and checkpoint identity. Nonlinear coordinates use
the same umbrella/MBAR sampler, constrained Cayley–RATTLE parent trajectories,
and paired unconstrained RPMD children. Their surface measure requires an
additional metric weight.

## Why the mass metric is necessary

Write the coordinate as $s(\bar{\mathbf q})$, the dividing surface as $s=s^*$,
and its centroid mass metric as

$$
z(\bar{\mathbf q}) = \sum_i \frac{|\nabla_i s|^2}{m_i}.
$$

With physical bead masses and the standard sum-over-beads Hamiltonian, a fresh
unconstrained Maxwell draw gives
$\operatorname{var}(\dot s\mid\mathbf q)=z/\beta$. The ordinary CV-density PMF
therefore gives the basin-normalized QTST expression

$$
k_{\rm QTST} =
\frac{\rho(s^*)}{\int_{\rm basin}\rho(s)\,ds}
\frac{\langle\sqrt z\rangle_{\delta(s-s^*)}}{\sqrt{2\pi\beta}}.
$$

The fixed-surface RATTLE configuration ensemble is proportional to
$\delta(s-s^*)\sqrt z\,e^{-\beta_P U_P}$. Thus its samples are converted to the
conditional delta-surface ensemble by the blue-moon weight $w=z^{-1/2}$:

$$
\langle A\rangle_\delta =
\frac{\langle wA\rangle_c}{\langle w\rangle_c},
\qquad
\langle\sqrt z\rangle_\delta=\frac{1}{\langle w\rangle_c}.
$$

The implementation evaluates $z$ at each saved parent. It uses the harmonic
mean of $\sqrt z$ in the QTST prefactor and multiplies both the paired-child
signed numerator and its initial absolute-velocity denominator by $w$:

$$
\kappa(t)=
\frac{\langle w\dot s_0[h(s_t^+-s^*)-h(s_t^--s^*)]\rangle_c}
     {\langle w|\dot s_0|\rangle_c},
\qquad k(t)=k_{\rm QTST}\kappa(t).
$$

Here the direction flag is included in both $\dot s_0$ and the product-side
test. The two children begin with opposite **unconstrained** Maxwell momenta.
Thermostatting or projecting their momenta would alter the reactive flux.

The PMF is already obtained from ordinary umbrella density. No geometric
mean-force correction or extra distance Jacobian is added to that PMF. The
blue-moon weight is applied to the constrained parents and child fluxes only.

For a smooth monotone relabeling $\eta=f(s)$, the surface density changes by
$1/|f'(s^*)|$ while $\sqrt z$ changes by $|f'(s^*)|$. The rate is invariant in
the continuum limit. Finite histogram bins still require convergence tests.

## Input and execution

For a bond-difference coordinate, use the ordinary CV declaration and specify
a local basin whose boundary is the dividing surface:

```toml
task = "site_rate"

[[cvs]]
name = "transfer"
kind = "bond_difference"
atoms = [0, 1, 2]
estimator = "centroid"

[umbrella]
cvs = ["transfer"]
centers = [-0.8, -0.4, 0.0, 0.4]
stiffness = 0.01

[analysis]
method = "mbar"
measure = "cv_density"
bins = 80
ranges = [[-1.2, 0.8]]
bootstrap = 100

[site_rate]
cv = "transfer"
basin = ["-0.8 bohr", "0.0 bohr"]
dividing_surface = "0.0 bohr"
direction = 1
```

This declaration means $r_{01}-r_{12}$ and uses zero-based atom indices.
Numeric CV values use their declared input units; explicit quantities keep
their own units. A negative direction makes the lower basin boundary the
dividing surface. Umbrellas must bracket both the full basin and the surface.

The complete executable example is
[`examples/rate/general_coordinate.inp`](../examples/rate/general_coordinate.inp):

```bash
pyrpmd run examples/rate/general_coordinate.inp --output runs/general-coordinate
pyrpmd resume runs/general-coordinate
```

This short example uses a confined harmonic model and artificial masses to
exercise the scientific estimator and restart behavior. It is not a model of
a particular chemical reaction or a converged physical rate.

A Python CV uses the existing factory protocol. It must declare
`estimator = "centroid"`, one scalar dimension, canonical output units, and
`periods = [None]`. Its full-ring Jacobian must obey
$\partial s/\partial q_j=(\nabla s)/P$. Setup checks this identity on a
non-collapsed ring before either sampler runs. The plugin author remains
responsible for smoothness, accurate derivatives, and this contract throughout
the sampled region. Bead-average and arbitrary full-ring dividing surfaces
require different constrained dynamics and are rejected by this workflow.

For existing histogram data and independently sampled constrained parent
metrics, the public estimator is also available directly:

```python
from pyrpmd.workflows.site_rate import estimate_site_qtst

qtst = estimate_site_qtst(
    histogram, beta, None, basin, dividing_surface,
    constrained_mass_metrics=parent_metrics,
)
```

`parent_metrics` must come from the canonical constrained surface ensemble.
Metrics from unconstrained umbrella frames require a different average.

## Uncertainty and output

The output includes the per-parent inverse-metric weights, their effective
sample size, the average inverse square root, and the effective surface metric.
The last quantity is $1/\langle w\rangle_c^2$; it is **not** an arithmetic mean
of $z$. Preparation leaves the nonlinear surface prefactor unset until the
parents have actually been sampled.

The shared recrossing bootstrap keeps all child pairs of one parent together
and resamples entire parent chains and correlated blocks. The block length is
at least the estimated correlation length of $w$, as well as the transmission
observable. Both $\kappa$ and $\langle w\rangle_c$ use exactly the same parent
resample. Independent umbrella-block draws are then combined with that joint
draw. This preserves the covariance of the surface prefactor and recrossing
correction. The primary total-rate uncertainty still requires at least two
independent parent chains and complete PMF bootstrap support.

`site_qtst.json` records the measure and metric diagnostics. The final
`site_qtst.npz` includes `metric_parent_ids`, `parent_inverse_sqrt_metric`, and
joint QTST bootstrap draws. `recrossing.npz` stores the corresponding total
rate draws. Checkpointed children retain their raw signed flux, and the
deterministic metric weight is applied when the result is read or resumed.

## Scientific scope and validation

The dividing surface must be differentiable and have a positive finite metric
throughout its sampled region. Singular bond/angle configurations fail with an
error. Periodic coordinates use a continuous local representation; minimum-image
switching seams and wrapped angular coordinates are outside this differentiable
surface assumption. The current workflow requires all atoms mobile, a fixed
cell if periodic, one constrained centroid CV, fixed local umbrellas, and
Andersen constrained-parent sampling. It does not add rigid-body constraints,
fixed-layer metrics, a vector dividing surface, or a gas concentration factor.

The tests verify:

- An ellipse with an angle-dependent mass metric against independent surface
  quadrature; an arithmetic-metric approximation is detectably incorrect.
- QTST invariance under a nonlinear monotone coordinate transformation as the
  histogram bins become fine.
- A synthetic case where correlated metric and transmission fluctuations
  cancel exactly; independent bootstrap draws would fail this check.
- Singular-metric and incompatible-plugin rejection, plus nonlinear parent
  and child checkpoint replay.
- A complete bond-difference umbrella → constrained parent → child → rate
  calculation and resume, while preserving the existing linear tests.

These are correctness regressions. A finite-time plateau does not establish
bead, timestep, system-size, dividing-surface, or statistical convergence; the
workflow continues to report `converged = false` until those separate studies
are supplied.

## References

1. M. Sprik and G. Ciccotti, “Free energy from constrained molecular dynamics,”
   *J. Chem. Phys.* **109**, 7737–7744 (1998),
   [doi:10.1063/1.477419](https://doi.org/10.1063/1.477419).
   This supplies the constrained/conditional ensemble relation.
2. C. Li, L. Zhang, B. Jiang and H. Guo, “Extending ring polymer molecular dynamics
   rate theory to reactions with non-separable reactants,”
   [arXiv:2503.22964](https://arxiv.org/abs/2503.22964).
   This motivates the basin-normalized single-dividing-surface formulation for
   unimolecular and condensed-phase processes.
3. [CP2K constrained-dynamics documentation](https://github.com/cp2k/cp2k/blob/master/docs/methods/sampling/constrained_dynamics.md)
   gives explicit constant-distance and variable bond-difference metrics and
   explains why arbitrary CVs require the blue-moon correction.
