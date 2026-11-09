# When does a subspace correction transfer?

This page derives the conditions under which the multiplicative subspace correction of
[Subspace rate correction](subspace-correction.md),

$$
k \approx k_{\mathrm{RPMD}}(H)\,C_S,\qquad C_S=\frac{k_{\mathrm{QM}}(H_S)}{k_{\mathrm{RPMD}}(H_S)},
$$

is accurate, and states what the tests in `benchmarks/subspace_theory` and the H + H₂
benchmark found. Nothing here marks a rate as physically converged.

## Transfer error

Let $C(X)=k_{\mathrm{QM}}(X)/k_{\mathrm{RPMD}}(X)$ for a Hamiltonian $X$. The error of the
corrected rate is

$$
\Delta \equiv \ln C(H)-\ln C(H_S) = G_{\mathrm{QM}}-G_{\mathrm{RPMD}},\qquad
G_X=\ln\frac{k_X(H)}{k_X(H_S)},
$$

with both rates of each method in the same units. $G_X$ is the environment response of
method $X$: by how much the coordinates left out of $S$ change its rate. **$\Delta$ is small when
quantum mechanics and RPMD respond to the environment in the same way, not when the response
is small.** A subspace model can miss the absolute rate by a large factor ($|G_{\mathrm{QM}}|\gg 1$)
and still carry the RPMD defect.

## Exact cases

**Separable environment.** If $H=H_S\otimes 1+1\otimes H_E$, the quantum flux–side function
factorizes and $Z_E$ cancels against the reactant partition function, so
$k_{\mathrm{QM}}(H)=k_{\mathrm{QM}}(H_S)$. The ring-polymer Hamiltonian also separates; the centroid
constraint and the flux act only on $S$, so $k_{\mathrm{RPMD}}(H)=k_{\mathrm{RPMD}}(H_S)$ for every bead
number $P$. Hence $\Delta=0$ at all temperatures and bead numbers.

**Quadratic Hamiltonians.** A parabolic barrier coupled bilinearly to harmonic modes is
separable in its normal modes (one unstable mode $u$, stable modes). Both the exact quantum rate
and the RPMD rate are independent of the dividing surface, so both may be evaluated with a
dividing surface in $u$. Then $C(H)=C(H_u)$, and RPMD is exact for a parabolic barrier above its
crossover temperature (Craig and Manolopoulos, J. Chem. Phys. 122, 084106 (2005)), so
$C(H)=1$. The same holds for any quadratic $H_S$, so $\Delta=0$. **A nonzero transfer error
needs anharmonicity along the reaction coordinate together with coupling to the environment**;
neither alone produces one.

## First-order coupling theorem

Write $H=H_{S0}\otimes 1+1\otimes H_E+\lambda W$ with a potential coupling
$W=\sum_i a_i(s)\,b_i(y)$, where $s$ are the subspace and $y$ the environment coordinates.

*Quantum rate.* Every $\lambda$-derivative of $e^{-\beta H}$ and of $e^{\pm iHt}$ (Duhamel) inserts
$W$ at one complex time. At $\lambda=0$ the environment factor of such a term is
$\operatorname{Tr}_E[e^{-\beta H_E}b_i(\tau)]/Z_E=\langle b_i\rangle_\beta$, independent of $\tau$ because
$e^{-\beta H_E}$ commutes with $H_E$. Hence

$$
k_{\mathrm{QM}}(H)=k_{\mathrm{QM}}(\bar H)+O(\lambda^2),\qquad
\bar H=H_{S0}+\lambda\sum_i a_i(s)\,\langle b_i\rangle_\beta .
$$

*RPMD rate.* The RPMD rate is a classical flux–side function in ring-polymer phase space. At
first order, $W_P=\sum_j\sum_i a_i(s_j)b_i(y_j)$ enters once, either in the Boltzmann weight or as
a force on the $s$ beads along the unperturbed trajectory. The environment beads move under their
free ring-polymer dynamics, which preserves their stationary distribution, so the environment
factor is the $P$-bead average $\langle b_i\rangle_\beta^{(P)}$. A perturbation of the environment
trajectory feeds back on $s$ only at second order. Therefore

$$
k_{\mathrm{RPMD},P}(H)=k_{\mathrm{RPMD},P}(\bar H_P)+O(\lambda^2),\qquad
\bar H_P=H_{S0}+\lambda\sum_i a_i(s)\,\langle b_i\rangle_\beta^{(P)} .
$$

**Consequence.** A projection whose $V_S(s)$ equals $V_{S0}+\lambda\sum_i a_i\langle b_i\rangle_\beta$ to first
order gives $\Delta=O(\lambda^2)$, up to the finite-$P$ difference
$\langle b\rangle^{(P)}-\langle b\rangle$ (next section).

- For a harmonic environment whose Hessian depends on $s$, the Hellmann–Feynman theorem for
  the free energy gives
  $\partial_\lambda F_E=\tfrac12\operatorname{Tr}[K_1\langle yy^{\mathsf T}\rangle_\beta]=\bar W$. So the harmonic
  quantum free energy $F_E(s)=\beta^{-1}\sum_k\ln[2\sinh(\beta\hbar\omega_k(s)/2)]$ is the first-order
  correct projection: `environment="free_energy"` in `project_linear_subspace`,
  `project_path_subspace` and the pipeline (`[subspace] environment`).
- The zero-point projection (`"adiabatic"`) uses the ground-state average instead. Its coupling
  is wrong by the factor $\coth(\beta\hbar\omega/2)-1$: 0.31 at $\beta\hbar\omega=2$, 0.04 at 4, 0.005 at 6.
- The frozen projection misses $\langle y^2\rangle$ entirely.
- A displacement coupling $-\omega^2 y\,c(s)$ has $\langle y\rangle=0$. Its effect, the valley shift
  and the curvature, starts at second order.

How much a first-order error $\delta V_S$ in the projection matters is set by the *elasticity*
introduced below: it changes $\ln C_S$ by about $\varepsilon\,\beta\,\delta V_S$ at the barrier, not by
$\beta\,\delta V_S$.

## Static effects cancel; what remains

With the $P$-bead RPMD-TST rate $k^{\ddagger}_P$, define $\kappa_{\mathrm{QM}}=k_{\mathrm{QM}}/k^{\ddagger}_P$
and $\kappa_{\mathrm{RP}}=k_{\mathrm{RPMD},P}/k^{\ddagger}_P$. Then, identically,

$$
\Delta=\ln\frac{\kappa_{\mathrm{QM}}(H)}{\kappa_{\mathrm{RP}}(H)}-\ln\frac{\kappa_{\mathrm{QM}}(H_S)}{\kappa_{\mathrm{RP}}(H_S)} .
$$

Everything static drops out of each $C$ separately: the centroid free-energy barrier,
zero-point energies, the imaginary-time part of corner cutting, and the environment's static
non-adiabaticity. Only the environment's influence on RPMD's *dynamical* error matters.

For the curvature (displacement) coupling $\tfrac12\omega^2(y-c(s))^2$ the static part can be made
explicit. Integrating the environment ring polymer out for a fixed $s$ path gives, in the free
ring-polymer normal modes with frequencies $\omega_n=2\omega_P\sin(\pi n/P)$,

$$
U_E[\{s_j\}]=\frac12\sum_{n}\frac{\omega^2\omega_n^2}{\omega^2+\omega_n^2}\,|\tilde c_n|^2
=\frac12\sum_n\Big(\omega_n^2-\frac{\omega_n^4}{\omega^2}+\dots\Big)|\tilde c_n|^2,
$$

where $\tilde c_n$ are the normal-mode components of $c(s_j)$. The centroid term vanishes: the
environment follows statically. The $\omega_n^2$ term turns the $s$ springs into springs along
the valley, which is the arc-length metric that `project_path_subspace` uses. The
$\omega_n^4/\omega^2$ term is the leading imaginary-time corner-cutting correction. All of these
are static path-integral terms. They are sampled exactly by RPMD-TST and therefore cancel in
$\Delta$.

Curvature also changes the real-time dynamics near the barrier, and the 1D path model
represents it only through the normal-cut zero-point energy. Neither cancels. Because a
displacement coupling has no first-order term, the theorem above predicts
$\Delta=O(h^2)$ for a valley displaced by $h$. The tests confirm this. The coefficient is
temperature independent for a symmetric barrier, $\Delta=(0.148\pm0.006)(\kappa\rho)^2$, and
somewhat smaller for an asymmetric one, $(0.112\pm0.007)(\kappa\rho)^2$.
So curvature × width measures the transfer error itself, not only the size of $G$.

**Elasticity estimate.** Suppose the full system behaves near the barrier like a member $H_{\mathrm{eff}}$
of the same family of subspace models, with $k_{\mathrm{QM}}(H_{\mathrm{eff}})=k_{\mathrm{QM}}(H)$. Then

$$
\Delta\approx\varepsilon\,G_{\mathrm{QM}},\qquad \varepsilon=\frac{\partial\ln C}{\partial\ln k_{\mathrm{QM}}}\Big|_{\text{model family}} .
$$

$C$ varies slowly where $k$ varies exponentially, so $|\varepsilon|\ll 1$ is expected. $\varepsilon$ can
be measured on subspace models alone, for example by scaling the barrier
(`benchmarks/subspace_theory/elasticity.py`). For 1D Eckart barriers it is $-0.004$ to $-0.037$
between 600 and 200 K. The estimate bounds errors of the projected potential that act like a
change of the barrier, such as a thermal error of the zero-point projection. It does **not**
describe curvature: for $h=0.9$ bohr at 200 K it predicts $\Delta=-0.08$, while the
measured value is $+0.16$. Corner cutting is not a barrier-height change.

## Finite bead number

The adiabatic and free-energy projections put the environment's converged ($P\to\infty$) free
energy into $V_S$. A $P$-bead ring polymer of $H$ carries an additional environment bead error,
which $C_S$ at matched $P$ does not contain. For harmonic environment modes the error is exact
in closed form. A $P$-bead ring polymer of a mode with $u=\beta\hbar\omega$ has

$$
Z_P(u)=\frac{1}{2\sinh\!\big(P\,\operatorname{asinh}(u/2P)\big)},
$$

which is $1/u$ for $P=1$ and $1/[2\sinh(u/2)]$ for $P\to\infty$. Hence, at the RPMD-TST level,

$$
k_{\mathrm{RPMD},P}(H)\approx k_{\mathrm{RPMD},\infty}(H)\,B_P,\qquad
B_P=\frac{\prod_{\ddagger}Z_P/Z_\infty}{\prod_{r}Z_P/Z_\infty}
\approx\exp\!\Big[-\frac{\sum_r u_r^3-\sum_\ddagger u_\ddagger^3}{48P^2}\Big]
$$

(the large-$P$ form holds for $u\gg1$). Use `harmonic_bead_factor` and
`apply_subspace_correction(..., bead_factor=B_P)`, or converge $P$. To keep $|\ln B_P|<\delta$
one needs

$$
P\gtrsim\sqrt{\frac{\sum_r u_r^3-\sum_\ddagger u_\ddagger^3}{48\,\delta}} .
$$

The required $P$ grows as $u^{3/2}$, so a rule $P\propto\beta\hbar\omega$ fails at low temperature.
For H₂ at 200 K ($u_r=31.7$, TS stretch $u_\ddagger=14.8$), 5% requires $P\approx 100$.

The first-order theorem gives an equivalent statement: matched-$P$ transfer is first-order exact
when the RPMD half of $C_S$ uses the $P$-bead environment free energy
$-\beta^{-1}\ln Z_P(\beta\hbar\omega(s))$ and the quantum half uses the converged one. At the TST
level this is the same as dividing by $B_P$.

## Below the crossover temperature

For $T<T_c$, RPMD-TST is related to semiclassical instanton theory: it shares the instanton
action and differs in the prefactor (Richardson and Althorpe, J. Chem. Phys. 131, 214106 (2009)).
In that regime the identity above makes $\Delta$ a difference of prefactor errors,
$\ln[A_{\mathrm{inst}}/A_{\mathrm{RP}}]_H-\ln[A_{\mathrm{inst}}/A_{\mathrm{RP}}]_{H_S}$. These errors depend on
where the instanton sits relative to the dividing surface (its asymmetry), not on the environment
frequencies. This suggests a small $\Delta$ for symmetric barriers even with strong corner
cutting. **The tests do not bear this out.** The curvature transfer error is the same at 1.45,
0.73 and 0.48 $T_c$ and for symmetric and asymmetric barriers. Most of it comes from the change
of the 1D model's own RPMD error, $\ln C_S$, as its normal-cut zero-point energy reshapes
$V_S$, and not from deep tunnelling.

## Practical criteria

1. Project with `environment="free_energy"` whenever an environment mode has
   $\beta\hbar\omega\lesssim4$; it is first-order exact, the zero-point projection is not. The effect
   on $C_S$ is reduced by the elasticity.
2. Converge $P$ for the stiffest environment mode, or divide the full RPMD rate by $B_P$:
   $P\gtrsim[(\sum_r u^3-\sum_\ddagger u^3)/(48\delta)]^{1/2}$.
3. Curvature × width predicts the transfer error of a 1D path model:
   $\Delta\approx(0.11\text{–}0.15)(\kappa\rho)^2$ on the tested models. That is 3–4% at the
   default threshold $\kappa\rho=0.5$, and keeping $|\Delta|<5\%$ needs $\kappa\rho\lesssim0.6$. The coefficient comes from
   two model families and is not universal. The deciding test remains the nested sensitivity
   $\ln(C_{d+1}/C_d)$, a direct estimate of $\Delta$ for the smaller subspace.
4. Errors of $V_S$ that act like a barrier change reach $C_S$ reduced by the elasticity
   $|\varepsilon|\lesssim0.04$. A subspace with a poor absolute rate can still transfer, but only
   if its error is of that kind.
5. Converge the energy-truncated DVR. The walls must lie about six quanta of the stiffest
   transverse mode above the valley floor, ridge included, which is far above 30 kT at low
   temperature; check `truncation_margin` and set `truncation_min`.

## Tests

All numbers are in
[docs/validation/2026-11-09-subspace-transfer](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-09-subspace-transfer/README.md).
The study used 2D Eckart models with a transverse vibration and 1D path projections, at 600,
300 and 200 K (1.45, 0.73 and 0.48 $T_c$), computed on GitHub Actions.

| Statement | Test | Result |
| --- | --- | --- |
| Separable environment: $\Delta=0$ | 6 points, $h=0$ | all within 1.5σ of zero |
| First-order theorem: the free-energy projection is first-order exact | $G_{\mathrm{QM}}$ of the frequency family | zero point +0.129 / +0.027 / +0.043, free energy −0.0001 / −0.006 / −0.001 |
| Projection errors reach $C_S$ reduced by $\varepsilon$ | zero-point vs free-energy $C_S$; 1D barrier scaling | $\Delta$ differs by ≤ 0.02 although $k$ differs by up to 14%; $\lvert\varepsilon\rvert\le0.04$ |
| Displacement coupling: $\Delta=O(h^2)$, no linear term | 18 curvature points | $\Delta=0.21h^2$ (symmetric), $0.16h^2$ (asymmetric) |
| Elasticity estimate for curvature | $h=0.9$ bohr, 200 K | **fails**: predicts −0.08, measured +0.16 |
| Small $\Delta$ for symmetric barriers below $T_c$ | curvature, 0.48 $T_c$ | **not borne out**: $\Delta$ the same as at 1.45 $T_c$ |
| Finite-$P$ environment error is $\ln B_P$ | frequency family; $P$ = 8–48 | $\chi^2/n$ of $\Delta$: 5.1 → 0.77 after $\ln B_P$; quantitative for $\lvert\ln B_P\rvert\lesssim0.15$ |
| Truncated DVR: walls ≥ 6 transverse quanta | separable and curved models, H + H₂, Cl + HBr | separable 2D = 1D to 4×10⁻⁶ at 0.07 Eh; H + H₂ rates in the earlier report 3–13% high at 0.04 Eh |
