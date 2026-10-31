# KC-RPMD, MV-RPMD and the isomorphic Hamiltonian

These Python APIs implement three distinct model dynamics. They share the
existing nuclear ring propagator and matrix-PES contract. Their electronic
Hamiltonians and equilibrium weights differ.

| Method | Entry points | Implemented domain |
| --- | --- | --- |
| MV-RPMD | `MVRPMDState`, `MVRPMDDynamics`, `MVRPMDSampler` | Fixed real diabatic basis, multiple states, signed projected Wigner distribution |
| Original KC-RPMD | `KCCalibration`, `KCRPMDState`, `KCRPMDDynamics`, `KCRPMDSampler` | Two states, positive coupling, explicitly supplied constraint parameters and crossing calibration |
| Isomorphic RPMD | `IsomorphicPotential`, `IsomorphicEhrenfestState`, `IsomorphicEhrenfestDynamics` | Two real states, whole-ring trace-matching potential, explicit Ehrenfest composition |

These are CPU model APIs. Their nuclear free-ring map can use the compiled
kernel. They do not select a production CLI workflow, provide a real-system
matrix potential, or establish a converged chemical rate.

## Common conventions

Use atomic units, `beta = 1/(k_B*T)`, physical nuclear masses and arrays
`q,p[P,N,3]`. The providers return the **full** matrix and its positive nuclear
gradients through `DiabaticResult`. A common scalar energy must appear on every
diagonal. The methods internally separate that scalar for numerical stability;
users do not subtract it themselves.

The propagated nuclear energy is the sum-bead Hamiltonian. Its sampling
temperature is `P/beta`. Matrix derivatives are analytic, including the Frechet
derivative of each matrix exponential. Scaled prefix/suffix products avoid
overflow and evaluate all bead derivatives with linear bead cost. Providers
are borrowed, so the caller retains responsibility for closing them.

## MV-RPMD

The projected matrix path is

\[
\Gamma=\prod_b[(z_bz_b^\dagger-I/2)e^{-\beta V(q_b)/P}],
\quad z_b=x_b+i y_b,\qquad \Theta=\operatorname{Re}\operatorname{Tr}\Gamma.
\]

The implemented Hamiltonian is [1]

\[
H=T+U_{\mathrm{spring}}+
\frac{P}{\beta}\left[\sum_b(x_b^2+y_b^2)-\ln|\Theta|\right].
\]

`MVRPMDSampler` samples the absolute weight; observables require
`mean(sign * observable) / mean(sign)`. Report the mean sign and sampling
uncertainty. Signed mapping populations are available through
`dynamics.populations(state)` and can fall outside `[0,1]`.

The implementation uses exact electronic matrix exponentials at finite P.
Its nonseparable electronic flow is integrated with implicit midpoint steps,
symmetrically composed with the free ring. Failed nonlinear solves, unresolved
trace nodes and detected sign changes stop propagation without changing the
input state. Reduce the timestep and verify convergence near nodes. This
model sampler does not solve the sign problem.

## Original KC-RPMD

The original primitive transfer matrix has diagonal elements
`exp(-beta*V_i/P)` and off-diagonal elements
`-beta*K/P * exp(-beta*V_i/P)`. The implementation separates the two constant
electronic paths from kinked paths and applies the original centroid constraint
and finite-b square-well restraint [2]:

\[
D=f(y,-1)W_0+f(y,1)W_1+
f(y,0)\sqrt{a/\pi}\,\eta e^{-a w(\bar q)^2}W_{\mathrm{kink}},
\quad w=(V_0-V_1)/K.
\]

The sum-convention electronic potential is `-P*log(D)/beta`. An augmented
transfer path accumulates kink weights directly; it avoids subtracting nearly
equal no-kink and total traces at weak coupling.

Supply `a`, `b` and `KCCalibration` explicitly. The original calibration uses

\[
\eta=\langle|\nabla w|\rangle_c,\quad
m_y=\frac{\beta^3}{2\pi^3}
\left(\frac{\eta}{\langle|K|^{-1}\rangle_c}\right)^2,
\]

with crossing measure proportional to
`delta(w)*K**2*exp(-beta*V_0)`. `state.p_y` is `P*m_y*dy/dt`, so the
implemented auxiliary mass is `P*m_y`. The configuration sampler and Gaussian
momentum draws use that same convention.

Converge P, `a`, `b`, timestep and crossing averages. This implementation
requires `0 < a <= 1`, `b > 0`, P >= 2, positive K and `beta*K/P < 1`.
The finite-b restraint retains the original absolute-value corner at each
sector center; the force there uses its symmetric value. No published KC rate
or calibration-table reproduction is claimed, and later KC variants are not
silently substituted.

## Isomorphic potential and iso-Ehrenfest dynamics

The two-state potential preserves the finite-P electronic trace [3]:

\[
\operatorname{Tr}e^{-\beta V_{\mathrm{iso}}(\mathbf q)}
=\operatorname{Tr}\prod_b e^{-\beta V(q_b)/P}.
\]

Its diagonals are bead averages; its off-diagonal coupling follows the
trace-matching `acosh` expression. `IsomorphicResult.gradients` has shape
`[2,2,P,N,3]`, because this potential couples the entire ring. It must not be
passed as a separable scalar bead potential.

`IsomorphicEhrenfestDynamics` evolves one normalized two-state wavefunction
with `V_iso`. The nuclear force and conserved sum energy contain `P*V_iso`.
Electronic rotation and the associated nuclear impulse are integrated together.
This explicitly chooses Ehrenfest dynamics; the trace identity alone does not
guarantee a correct thermal rate or a stationary thermal electronic ensemble
under that choice.

A coupling must retain one nonzero sign at all beads and at the centroid.
Completely decoupled surfaces and a single bead are supported. Identical bead
matrices are returned exactly, including extremely weak couplings. For a
general path whose coupling cannot be resolved after trace subtraction, the
API raises an error instead of returning a coupling determined by roundoff.

## Run the examples and checks

```bash
python examples/nonadiabatic/method_models.py
python -m pytest -q tests/test_isomorphic.py tests/test_mv_rpmd.py \
  tests/test_kc_rpmd.py tests/test_nonadiabatic_method_invariants.py
```

The runnable script demonstrates propagation, energy diagnostics, populations,
KC calibration and both samplers. Its short sampler runs are demonstrations;
`model_equilibrated` and `physical_converged` remain false.

Independent tests integrate the signed MV weight and population estimator by
Gaussian quadrature, enumerate KC electronic paths, compare the isomorphic
trace against direct matrix exponentials, and differentiate every implemented
force numerically. Additional checks cover one-state/collapsed limits,
reversibility, second-order energy errors, native/NumPy agreement, weak-coupling
conditioning, provider failures, interrupted sampling and state ownership.

## Primary references

1. N. Ananth, *J. Chem. Phys.* **139**, 124102 (2013), Eqs. 25–35.
   [Author preprint](https://arxiv.org/abs/1309.4476).
2. A. R. Menzeleev, F. Bell and T. F. Miller III, *J. Chem. Phys.* **140**,
   064103 (2014), Eqs. 7, 19, 25–32 and Appendix B.
   [DOI](https://doi.org/10.1063/1.4863919),
   [author preprint](https://arxiv.org/abs/1401.3033).
3. X. Tao, P. Shushkov and T. F. Miller III, *J. Chem. Phys.* **148**, 102327
   (2018), Eqs. 13–19. [DOI](https://doi.org/10.1063/1.5005544),
   [author preprint](https://arxiv.org/abs/1709.06722).

