# The instanton anchor: what a subspace correction has to carry

[When does a subspace correction transfer?](subspace-transfer-theory.md) shows that the error of

$$
k\approx k_{\mathrm{RPMD}}(H)\,C_S,\qquad C_S=\varepsilon(H_S),\qquad \varepsilon(X)=\frac{k_{\mathrm{QM}}(X)}{k_{\mathrm{RPMD}}(X)},
$$

is $\Delta=\ln\varepsilon(H)-\ln\varepsilon(H_S)$ and that only RPMD's *dynamical* error enters it.
This page derives the leading part of $\ln\varepsilon$ in closed form, shows that it is a property
of the full-dimensional instanton that needs neither a reaction path nor sampling, and turns
that into a candidate correction, the *anchor*. The tests are in
`benchmarks/subspace_instanton`. Nothing here marks a rate as physically converged.

> **Status: tested on model potentials, not used by any correction.** On four
> two-dimensional valley models with exact quantum and RPMD rates
> ([report](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-14-subspace-instanton-anchor/README.md)) the steepest-descent factor is the leading part of the RPMD defect below
> the crossover temperature, and the skeleton anchor derived below agrees with the
> transfer error of the free-energy reduction where the naive anchor fails. The anchor is not
> quantitative near the crossover temperature for reductions that change it, and a path-free
> estimate is less accurate than the subspace correction itself.
> See [What the tests show](#what-the-tests-show).

Units: $\hbar=1$ and mass-weighted coordinates unless a formula shows them.

## Two regimes of the RPMD defect

Let $\lambda^\ddagger$ be the unstable frequency of the saddle of $H$ and
$T_c=\hbar\lambda^\ddagger/2\pi k_B$ its crossover temperature.

**Above $T_c$** the only saddle of the ring-polymer potential is the collapsed polymer at the
classical saddle. For a quadratic saddle RPMD is exact
([transfer theory](subspace-transfer-theory.md), quadratic Hamiltonians), so

$$
\ln\varepsilon=O(\text{anharmonicity})\qquad(T>T_c).
$$

**Below $T_c$** the saddle of the ring-polymer potential

$$
U_N(\mathbf x)=\sum_{i=1}^N V(x_i)+\frac{\omega_N^2}{2}\sum_i|x_i-x_{i+1}|^2,\qquad \omega_N=\frac{N}{\beta\hbar},
$$

is the instanton $\tilde{\mathbf x}$. Its Hessian has one negative eigenvalue $-\eta_1^2$, one zero
eigenvalue (cyclic permutation of the beads) and $Nf-2$ positive ones $\eta_k^2$.

## Steepest descent: $\varepsilon_{\mathrm{SD}}=2\pi/(\beta\hbar\eta_1)$

Evaluate both rates by steepest descent about $\tilde{\mathbf x}$. Every factor is common to
the two except that of the unstable mode $q_1$ (the result is that of Richardson and Althorpe,
J. Chem. Phys. 131, 214106 (2009); the short derivation is given because the correction below
rests on it).

*Quantum rate.* The instanton rate is $k=\frac{2}{\beta\hbar}\,\mathrm{Im}Z^\ddagger/Z_r$. The
unstable mode contributes its momentum integral and the analytically continued Gaussian,

$$
\frac{2}{\beta\hbar}\cdot\Big(\frac{m}{2\pi\beta_N\hbar^2}\Big)^{1/2}\cdot\frac12\Big(\frac{2\pi}{\beta_N m\eta_1^2}\Big)^{1/2}
=\frac{1}{\beta\hbar}\,\frac{1}{\beta_N\hbar\,\eta_1},\qquad\beta_N=\beta/N .
$$

*RPMD.* The ring polymer is a classical system at $\beta_N$. In the same harmonic
approximation, the surface $q_1=0$ is not recrossed, and the RPMD rate does not depend on the
dividing surface, so it equals the transition-state flux through $q_1=0$. The unstable mode
contributes

$$
\frac{1}{2\pi\hbar}\int dp_1\,\frac{p_1}{m}\,\theta(p_1)\,e^{-\beta_N p_1^2/2m}=\frac{1}{2\pi\hbar\,\beta_N}.
$$

The ratio is

$$
\boxed{\;\varepsilon_{\mathrm{SD}}(\beta)=\frac{2\pi}{\beta\hbar\,\eta_1(\beta)}\;}\qquad(T<T_c),\qquad \varepsilon_{\mathrm{SD}}=1\quad(T\ge T_c).
$$

At $T_c$ the instanton collapses, $\eta_1\to\lambda^\ddagger$, and $\varepsilon_{\mathrm{SD}}\to1$
continuously. Write

$$
\ln\varepsilon(\beta)=A(\beta)+R(\beta),\qquad A=\ln\varepsilon_{\mathrm{SD}},
$$

so that $R$ is everything beyond steepest descent: anharmonic fluctuations about the instanton
below $T_c$, the whole defect above it, and the smoothing of the kink of $A$ at $T_c$.

**$\eta_1$ is a property of the full system.** It follows from one instanton optimization in
all coordinates. It is local to the barrier: no path to the asymptotes, no reference slice, no
absorber, no sampling. A slow bath, a curved path and a changing zero-point energy all enter
through the instanton itself.

## Transfer error and the anchored correction

For a subspace model $H_S$ with its own instanton,

$$
\Delta=(A_H-A_S)+(R_H-R_S),\qquad A_H-A_S=\ln\frac{\eta_1^{S}}{\eta_1^{H}}\quad(T<T_c\text{ of both}).
$$

The first term is computable before any rate is solved. Removing it would give a correction
that is right to steepest-descent order for any subspace. But which instanton of $H_S$ has to
be compared with that of $H$ needs care.

**Dressing and skeleton.** $A_H$ belongs to the saddle of the *bare* ring-polymer potential.
A transverse zero-point or free energy is not in that saddle: it enters both rates through
fluctuation factors, which cancel in $\varepsilon_{\mathrm{SD}}$. A reduction that adds the
transverse free energy $F_E(s)$ to its potential (`fe`, the adiabatic projection) moves its own
instanton by that term, so $A_H-A_S$ then contains a first-order effect of the frequency
coupling. The [first-order coupling theorem](subspace-transfer-theory.md) says the opposite:
the free-energy projection transfers exactly to first order, for the quantum and the RPMD rate
alike. Model M1 shows the conflict in its purest form: the transverse mode decouples from the
instanton at quadratic order, $A_H$ is exactly that of the bare barrier, and
$A_H-A_{\texttt{fe}}=+0.07$ to $+0.08$ below $T_c$ although the theorem predicts no first-order
transfer error. The steepest-descent expansion treats $F_E\propto\hbar$ as a higher-order term; a
zero-point change of several $k_BT$ is not small in that sense.

The consistent comparison is therefore with the **skeleton** $S^0$ of the reduction: the same
path and mass with the bare potential, without the transverse free energy. This gives two
candidate corrections,

$$
\ln C_{\mathrm{anch}}^{\mathrm{skel}}=\ln C_S+A_H-A_{S^0},\qquad
\ln C_{\mathrm{anch}}^{\mathrm{naive}}=\ln C_S+A_H-A_{S},
$$

of which the first is the one consistent with the first-order theorem: the dressing transfers
by the theorem, and the anchor corrects the skeleton, i.e. the path, the mass and the crossover
temperature of the reduction. The measured transfer errors follow the skeleton anchor and
not the naive one ([What the tests show](#what-the-tests-show)).

- If $H_S$ is exact, both anchors vanish.
- Above both crossovers every $A$ is zero and $C_{\mathrm{anch}}=C_S$.
- If the crossover temperatures of $H$ and of the skeleton differ, the kinks of the two $A$
  sit at different temperatures and the anchor is uncontrolled between them. The bare potential
  on the minimum-energy path has the crossover of $H$ (the path leaves the saddle along the
  unstable mode); an SCT mass raises it, a bare-mass model of a dressed coordinate shifts it.

## The remainder and the symmetric Eckart barrier

For $V=V_0\,\mathrm{sech}^2(x/a)$ put $x=ay$. Newton's equation in the inverted potential
becomes $\ddot y=\tfrac12\omega_b^2\,(\mathrm{sech}^2y)'$ with $\omega_b^2=2V_0/ma^2$: the orbit and
its stability depend on $u=\beta\hbar\omega_b$ only. Hence

$$
A_{\mathrm{Eck}}=A_{\mathrm{Eck}}(u),\qquad R_{\mathrm{Eck}}=R_{\mathrm{Eck}}(u,v),\qquad v=\frac{V_0}{\hbar\omega_b},
$$

and all dependence on the barrier height is in the remainder. $1/v$ measures the
anharmonicity in units of $\hbar$ (the quartic coefficient at the top is
$4m^2\omega_b^4/V_0$), so $R\to0$ as $v\to\infty$ at fixed $u\ne2\pi$.

A path-free estimate replaces the remainder of $H$ by that of the Eckart barrier with the
same crossover temperature and the vibrationally adiabatic barrier height
$V_{\mathrm{ad}}^\ddagger$:

$$
\ln C_{\mathrm{Eck}}=A_H(\beta)+R_{\mathrm{Eck}}\!\Big(\beta\hbar\lambda^\ddagger,\ \frac{V_{\mathrm{ad}}^\ddagger}{\hbar\lambda^\ddagger}\Big).
$$

It needs the saddle, its frequencies and the instanton, and one 1D calculation (or a table of
$R_{\mathrm{Eck}}$). It assumes a symmetric barrier. On the valley models it is less accurate
than the subspace correction of the free-energy reduction
([What the tests show](#what-the-tests-show)).

## Which coordinates the instanton involves

Integrating a harmonic mode $\tfrac12\omega^2(y-c(s))^2$ out of the ring polymer leaves
([transfer theory](subspace-transfer-theory.md))

$$
U_E=\frac12\sum_n K_n\,|\tilde c_n|^2,\qquad K_n=\frac{\omega^2\omega_n^2}{\omega^2+\omega_n^2},
$$

with the ring-polymer normal-mode frequencies $\omega_n\approx2\pi n/\beta\hbar$. Two limits are
local:

- **adiabatic**, $\omega\gg\omega_1$: $K_n\to\omega_n^2$, the springs act along the valley, which
  is the arc-length model on the minimum-energy path;
- **sudden**, $\omega\ll\omega_1$: $K_n\to\omega^2$ for every $n\ne0$,
  $U_E=\tfrac12\omega^2\sum_i(c(s_i)-\bar c)^2$: the mode is frozen at the bead average during
  the tunnelling event and the dominant path runs along $s$.

In between ($\omega\approx2\pi/\beta\hbar$, i.e. $\omega\approx\lambda^\ddagger$ near $T_c$) the
action is non-local and no one-dimensional potential model represents it. The instanton of $H$
contains all three cases, so $A_H$ does; a subspace model on the minimum-energy path contains
only the first. The ratio $\omega_k/\lambda^\ddagger$ therefore classifies the modes: well above
one, fold into $V_S$; near or below one, put in $S$ or rely on the anchor.

## What the tests show

`benchmarks/subspace_instanton` has four two-dimensional valley models (M1: the transverse
zero-point energy changes along a straight path; M2: a curved path; M3: both; M4: a slow bath
displaced between reactants and products), an exact reference (a recursive Green's-function
$N(E)$ with exact leads) and this repository's RPMD solver. The model family and the reference
method re-implement those of a code package supplied for comparison. Tables, errors and
commands are in the [validation report](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-14-subspace-instanton-anchor/README.md); temperatures are 1.25, 0.8 and 0.55 $T_c$.
Reductions: `fe` is the free-energy (adiabatic) projection on the minimum-energy path, `mep`
its skeleton, `fe+sct` the same with the small-curvature mass, `bare` the bare barrier with
the bare mass (M4).

**The steepest-descent factor.** Below $T_c$ the measured $\ln\varepsilon(H)$ is 0.21 to 0.53
and the remainder $R_H$ lies between $-0.04$ and $+0.04$. For the symmetric Eckart barrier:

| $u=\beta\hbar\omega_b$ | $v$ | $\ln\varepsilon$ (RPMD here) | $A(u)$ | $R$ |
| --- | --- | --- | --- | --- |
| 5.03 ($1.25\,T_c$) | 1.75, 2.1, 2.70 | 0.146, 0.125, 0.107 | 0 | 0.146, 0.125, 0.107 (±0.004) |
| 6.0 | 2.1 | 0.175 | 0 | 0.175 ± 0.005 |
| 7.85 ($0.8\,T_c$) | 1.75, 2.1, 2.70 | 0.305, 0.287, 0.251 | 0.211 | 0.094, 0.076, 0.040 (±0.008) |
| 10.5 | 2.1 | 0.481 | 0.453 | 0.028 ± 0.010 |
| 11.4 ($0.55\,T_c$) | 1.75, 2.1, 2.70 | 0.554, 0.543, 0.542 | 0.518 | 0.036, 0.026, 0.024 (±0.013) |
| 12.0 | 2.1 | 0.607 | 0.555 | 0.052 ± 0.012 |
| 13.5 | 2.1 | 0.658 | 0.640 | 0.018 ± 0.014 |

- $A$ is the leading term well below $T_c$ only. The remainder is largest near the crossover
  (0.18 at $u=6$) and still 0.04 to 0.09 at $0.8\,T_c$, where $A$ is 0.21.
- $A_{\mathrm{Eck}}$ depends on $u$ only (identical to nine digits for $v=0.8$ and 3.6), and
  the remainder shrinks with $v$ at fixed $u$, as the scaling argument requires.
- The comparison package's independent table gives $\varepsilon=1.184$ at $(u,v)=(6,2.1)$ and
  1.512 at $(9,1.2)$; the runs here give $1.192\pm0.006$ and $1.514\pm0.011$. Its values of
  $\ln\varepsilon$ at $v=2.1$, $u=10.5$, 12 and 13.5 are 0.459, 0.558 and 0.641, which is 0.02,
  0.05 and 0.02 below the runs here (errors 0.010 to 0.014; those of the table are not known).

**Naive and skeleton anchor.** Transfer errors $\Delta=\ln\varepsilon(H)-\ln C_S$ of the
free-energy reduction and what is left after each anchor (one standard error):

| model | $T/T_c$ | $\Delta$ (`fe`) | skeleton anchor | $\Delta-$anchor | naive anchor | $\Delta-$anchor |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.8 | −0.001 ± 0.008 | 0.000 | −0.001 | +0.083 | −0.085 |
| | 0.55 | +0.004 ± 0.017 | 0.000 | +0.004 | +0.070 | −0.066 |
| M2 | 0.8 | +0.020 ± 0.010 | +0.021 | −0.001 | +0.077 | −0.056 |
| | 0.55 | +0.033 ± 0.019 | +0.024 | +0.009 | +0.033 | 0.000 |
| M3 | 0.8 | +0.034 ± 0.008 | +0.025 | +0.009 | +0.134 | −0.100 |
| | 0.55 | +0.054 ± 0.017 | +0.033 | +0.021 | +0.143 | −0.089 |
| M4 | 0.8 | +0.004 ± 0.015 | 0.000 | +0.004 | +0.008 | −0.004 |
| | 0.55 | −0.007 ± 0.021 | +0.008 | −0.015 | +0.022 | −0.029 |

- The naive anchor fails wherever the zero-point energy changes along the path (M1, M3), by
  4 to 12 standard errors. The skeleton-anchored correction has no detectable error at any
  of the eight points (root mean square 0.010).
- For this reduction the anchor is small, because the skeleton on the minimum-energy path has
  the steepest-descent factor of the full system to 0.04. The plain $C_S$ is already within
  0.054 of the exact defect at all twelve points (root mean square 0.025), where the quantum
  rate of the reduction on its own is off by up to 0.35.
- Above $T_c$ there is no anchor, and the curved paths keep a transfer error of the same size:
  $+0.025\pm0.005$ (M2) and $+0.036\pm0.005$ (M3) at $1.25\,T_c$. The agreement below $T_c$
  therefore does not show that the anchor is the whole transfer error of a curved path.

**Skeletons with another crossover temperature.** The small-curvature mass raises the barrier
frequency of the skeleton by 25% (M2) and 20% (M3); the bare-mass model of M4 has one 25%
above that of the saddle.

| model | $T/T_c$ | reduction | $\Delta$ | skeleton anchor | $\Delta-$anchor |
| --- | --- | --- | --- | --- | --- |
| M2 | 0.8 | `fe+sct` | −0.088 ± 0.011 | −0.150 | +0.063 |
| | 0.55 | | −0.064 ± 0.018 | −0.016 | −0.048 |
| M3 | 0.8 | `fe+sct` | −0.084 ± 0.009 | −0.197 | +0.113 |
| | 0.55 | | −0.075 ± 0.018 | −0.053 | −0.022 |
| M4 | 0.8 | `fe+sct` | −0.065 ± 0.015 | −0.089 | +0.024 |
| | 0.55 | | −0.124 ± 0.019 | −0.136 | +0.012 |
| M4 | 0.8 | `bare` | −0.185 ± 0.016 | −0.199 | +0.014 |
| | 0.55 | | −0.219 ± 0.020 | −0.198 | −0.021 |

- Where the skeleton has a higher crossover temperature and the system is near its own, the
  anchor overshoots: at $0.8\,T_c$ it is twice the measured transfer error for M2 and M3. The
  two models sit at different reduced temperatures there, and near a crossover the
  remainder, which the anchor neglects, is as large as the steepest-descent factor.
- The bare-mass model is anchored within errors at both temperatures. At $1.25\,T_c$, its own
  crossover, its transfer error is $-0.067\pm0.011$ and no anchor exists.
- The small-curvature mass overshoots the quantum rate by factors of 2 to 4, but $C_S$ of
  `fe+sct` is off by 0.06 to 0.12 only: $C_S$ is far less sensitive to the reduction than the
  rate of the reduction is. It is still too large on every curved model below $T_c$, and `fe`
  without that mass is closer to the exact defect at all six of those points.

**Path-free estimate.** $\ln C_{\mathrm{Eck}}$ agrees with the exact defect within 0.01 for
M4 and for M2 at two of three temperatures, and is 0.02 to 0.10 too large for M1 and M3:
root mean square 0.050 over twelve points, against 0.025 for $C_S$ of `fe`. Where the
zero-point energy changes along the path, the remainder is not that of an Eckart barrier with
the adiabatic barrier height.

**What follows for the subspace correction.**

- The free-energy projection on the minimum-energy path is the reduction to use; its $C_S$
  needs no anchor at the accuracy of these tests (0.02 to 0.05 on curved paths).
- An anchor has to compare with the skeleton. Comparing the instanton of the full system with
  that of a model dressed with zero-point energy is wrong by the size of the dressing.
- $A_H$ is a cheap diagnostic of how large the RPMD defect is well below $T_c$;
  `pyrpmd.instanton.steepest_descent_rpmd_defect` returns it for an optimized
  [instanton](instanton.md#steepest-descent-rpmd-defect). It is not a correction: the
  remainder is as large as $A_H$ near the crossover.
- No correction in PyRPMD uses an anchor.

**Limits of the tests.** Two-dimensional models, one symmetric barrier, a harmonic transverse
mode, no wells or rotations; three temperatures down to $0.55\,T_c$; statistical errors of the
transfer errors 0.005 to 0.02. Two SCT instantons at $0.45\,T_c$ (M2, M3) have a second
unstable mode; that temperature has no RPMD rates.
