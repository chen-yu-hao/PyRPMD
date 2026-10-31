# Thermal NRPMD sampling and rate curves

The Python thermal API prepares nuclei and electronic mapping variables from
an SEO-projected thermal trace, propagates them together, and calculates nuclear
or electronic rate correlation functions. It supports finite, real symmetric
diabatic matrices and their Cartesian gradients. The ordinary
`nonadiabatic.method = "nrpmd"` collision workflow continues to use its focused
incident-state preparation; it does not acquire an electronic thermal ensemble
by changing the collision-energy distribution.

## Thermal preparation

The implementation follows the position/momentum SEO projection of Richardson
and Thoss [1]. With `P` beads, write the full matrix as
`V(q) = V0(q) I + Vt(q)`, where `V0 = trace(V)/nstates`. Define

\[
M_b=\exp[-\beta V_t(q_b)/(2P)],\qquad
W=\prod_b (y_{b-1}^{T}M_bx_b)(x_b^{T}M_by_b).
\]

The Monte Carlo target is

\[
|W|\exp\!\left[-\sum_b(x_b^2+y_b^2)-\frac{\beta}{P}
\left(U_{\rm spring}+\sum_bV_0(q_b)\right)\right].
\]

Nuclear momenta are independent Gaussian draws with variance `m*P/beta`.
Retain the initial sign of `W` for every correlation function. The sampled
mapping populations fluctuate; they are not initialized to unit occupation of
one electronic state. Each nuclear or mapping proposal uses the Metropolis
acceptance ratio, including the proposal-density correction for the Gaussian
mapping refresh. Mapping exponential factors are shifted and evaluated in log
form to avoid a common electronic energy offset overflowing the weight.

`ThermalNRPMDSampler` implements centroid displacements, local bead moves, and
independent mapping refreshes. The two displacement scales are in bohr. Monitor
acceptance and correlation lengths, and run independent chains. `burn_in` and
`stride` are Monte Carlo sweep counts and have no physical time interpretation.

```python
import numpy as np
from pyrpmd.thermal_nrpmd import ThermalNRPMDSampler, ThermalNRPMDDynamics

# matrix_pes.evaluate(q) supplies full V[B,S,S] and dV/dq[B,S,S,N,3].
sampler = ThermalNRPMDSampler(
    matrix_pes, masses, beta, q_initial, nstates=2,
    seed=20261031, nuclear_step=0.1, centroid_step=0.2,
)
parents = sampler.sample(1000, burn_in=5000, stride=10)
dynamics = ThermalNRPMDDynamics(
    matrix_pes, masses, beta, len(q_initial), dt=0.5,
    nstates=2, integrator="cayley-bcb",
)
```

These numbers show argument conventions; an appropriate timestep and sampling
budget depend on the supplied physical model. Reuse the same provider object
for preparation and propagation. Mismatched masses, temperature, bead count,
state count or provider are rejected. `snapshot()` returns a detached sampler
continuation state; `restore(snapshot)` checks the parameter set and reevaluates
the provider before replacing the running state. The caller is responsible for
keeping the provider and its physical parameters unchanged across restoration.

## Real-time Hamiltonian

`ThermalNRPMDDynamics` applies the scalar `V0` force once per bead and propagates
the traceless electronic matrix with the existing exact MInt substep and nuclear
free-ring kernel. This is necessary because thermal mapping norms fluctuate.
Applying the entire common scalar surface through a fluctuating mapping norm
would give the wrong nuclear force. For one state, the mapping matrix is zero
and nuclear propagation reduces exactly to ordinary RPMD, independent of the
sampled mapping variables.

The thermal target is **not** the classical Boltzmann factor of this propagation
Hamiltonian. In particular, sampling `exp(-beta/P * dynamics.energy(state))`
would admit inverted mapping surfaces and can be nonnormalizable. The scalar
split affects the approximate multistate dynamics; this implementation specifies
the smooth trace split. It does not claim equivalence to alternative
state-independent/state-dependent partitions.

NRPMD is an approximate real-time method. Its harmonic-oscillator mapping
dynamics generally do not preserve the exact initial quantum Boltzmann
distribution [2]. Accurate equilibrium preparation therefore does not by itself
establish a quantitatively accurate nonadiabatic rate.

## Nuclear and electronic rate estimators

`thermal_flux_side` accepts a smooth scalar coordinate of the nuclear centroid.
The callback returns `(s, gradient)`, where the gradient has shape `[N,3]`, and
`s > 0` defines the product direction. The finite-width side function is
`h=(1+tanh(s/width))/2`. The code computes its initial time derivative from the
centroid velocity and returns

\[
k_{\rm nuc}(t)=
\frac{\langle \operatorname{sgn}(W)\,\dot h(0)h(t)\rangle_{|W|}}
{\langle\operatorname{sgn}(W)[1-h(0)]\rangle_{|W|}}.
\]

```python
from pyrpmd.thermal_nrpmd import thermal_flux_side

def surface(centroid):
    gradient = np.zeros_like(centroid)
    gradient[0, 0] = 1.0
    return centroid[0, 0] - dividing_position, gradient

nuclear = thermal_flux_side(
    parents, dynamics, surface, steps=500, stride=5,
    smoothing_width=0.1,
)
```

The width has the units of `s`. Check width and dividing-surface dependence in
addition to the usual bead and timestep convergence. The returned quantity is
a finite-time smooth-flux rate curve with explicit reactant normalization; no
gas translational volume or bimolecular standard state is inserted.

For electronic-state kinetics, `thermal_population_rate` calculates
`d C_RP(t)/dt / p_R`. The initial factor is the SEO projector insertion

\[
A_s=\frac1P\sum_b
\frac{(y_{b-1})_s(M_bx_b)_s}{y_{b-1}^TM_bx_b},
\]

and the final factor is the ordinary MMST population. Its derivative is
evaluated analytically from the mapping equations, without differentiating a
noisy sampled time curve. State indices refer to one fixed diabatic basis.

```python
from pyrpmd.thermal_nrpmd import thermal_population_rate

electronic = thermal_population_rate(
    parents, dynamics, reactant_state=0, product_state=1,
    steps=500, stride=5,
)
```

Both APIs propagate copies of the saved parents. Signed populations and
trajectory contributions remain signed, without clipping or assignment to a
single surface. The initial statistical sign remains attached to the trajectory
throughout propagation. The rate curve uses the ratio of signed sums, which
cancels the partition-function denominator without losing the shared reactant
normalization.

## Interpreting results

Each `ThermalRateCorrelation` retains the times and per-parent numerator,
reactant denominator and signs. `rate_curve` is in inverse atomic time; divide
by `pyrpmd.constants.ATOMIC_TIME_S` to obtain inverse seconds. A small average
sign or an invalid signed reactant population withholds the normalized curve.
The default average-sign threshold is `0.05`; it is a numerical guard, not a
statistical proof that a larger average sign is adequately resolved.

`summarize()` reports sign and normalization diagnostics without selecting a
plateau. To inspect a candidate interval, supply its endpoints in atomic time
and an independently justified parent block size:

```python
diagnostics = electronic.summarize(
    start=20.0, stop=50.0, block_size=50, relative_tolerance=0.1,
)
```

The point estimate uses every parent. The uncertainty uses a delete-one-block
jackknife of the joint numerator/denominator; incomplete final blocks are
excluded only from that uncertainty calculation. All time points in the chosen
interval must stay near its mean, so an oscillating curve cannot pass solely
because its two half-means agree. The diagnostic plateau flag additionally
requires at least eight blocks and sufficiently small uncertainty. It remains
conditional on the chosen block length and one chain's mixing.

A physical rate requires independently checked sampling, sign convergence,
basin definitions, intermediate-time kinetic behavior, timestep, beads, and
agreement across chains. Long-time population recurrences or relaxation are
not automatically a rate plateau. Results retain `physical_converged=False`.
These APIs currently cover finite Cartesian model systems; they do not provide
complex-phase thermal sampling, fixed-atom constraints, periodic winding,
distributed thermal sampling, or a new general `task="rate"` input mode.

## Runnable example and checks

```bash
python examples/nonadiabatic/thermal_rates.py --output runs/thermal-nrpmd.json
pytest tests/test_thermal_nrpmd.py -q
```

The example runs both estimators on a confined two-state teaching model. It
records finite-time curves, acceptance counts, signs and explicit unavailable
results; it intentionally does not assign a rate constant to the short run.

Independent tests integrate the mapping projection by Gaussian quadrature and
compare with direct matrix-exponential traces and projector insertions for
noncommuting matrices. Other checks cover thermal Gaussian/SEO moments,
Boltzmann electronic populations, the one-state RPMD limit, force finite
differences, analytical unitary population flux, free-streaming nuclear flux,
uncoupled zero electronic flux, reversible propagation, joint ratio errors,
failure rollback and reproducible continuation.

## References

1. J. O. Richardson and M. Thoss, “Communication: Nonadiabatic ring-polymer
   molecular dynamics,” *J. Chem. Phys.* **139**, 031102 (2013).
   [DOI: 10.1063/1.4816124](https://doi.org/10.1063/1.4816124),
   especially Eqs. 4–14.
2. D. Bossion, S. N. Chowdhury and P. Huo, “Non-adiabatic ring polymer
   molecular dynamics with spin mapping variables,” *J. Chem. Phys.* **154**,
   184106 (2021). [DOI: 10.1063/5.0051456](https://doi.org/10.1063/5.0051456);
   [author preprint](https://arxiv.org/abs/2103.14119).
