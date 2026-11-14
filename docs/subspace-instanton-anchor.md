# The instanton anchor: what a subspace correction has to carry

[When does a subspace correction transfer?](subspace-transfer-theory.md) shows that the error of

$$
k\approx k_{\mathrm{RPMD}}(H)\,C_S,\qquad C_S=\varepsilon(H_S),\qquad \varepsilon(X)=\frac{k_{\mathrm{QM}}(X)}{k_{\mathrm{RPMD}}(X)},
$$

is $\Delta=\ln\varepsilon(H)-\ln\varepsilon(H_S)$ and that only RPMD's *dynamical* error enters it.
This page derives the leading part of $\ln\varepsilon$ in closed form, shows that it is a property
of the full-dimensional instanton that needs neither a reaction path nor sampling, and turns
that into a correction that is right to leading order for any subspace. The tests are in
`benchmarks/subspace_instanton`. Nothing here marks a rate as physically converged.

> **Status: draft.** The steepest-descent factor is checked against independent rates
> (below). The anchored corrections are predictions; their RPMD tests on the valley models are
> still running and are not reported here. See [Status of the tests](#status-of-the-tests).

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
temperature of the reduction. Which of the two the measured transfer error follows is what the
RPMD tests decide.

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
$R_{\mathrm{Eck}}$). It assumes a symmetric barrier.

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

## Status of the tests

`benchmarks/subspace_instanton` re-implements the model family and the exact reference of a
code package supplied for comparison (2D valley models M1–M4; a recursive Green's-function
$N(E)$ with exact leads) and uses this repository's RPMD solver.

**Checked**

| Statement | Test | Result |
| --- | --- | --- |
| Exact reference | Green's-function rate vs the analytic Eckart transmission, 5 points | relative difference $10^{-6}$ to $4\times10^{-4}$ |
| RPMD solver | $\varepsilon$ of the Eckart barrier vs the package's independent C kernel | $(u,v)=(6,2.1)$: $1.185\pm0.007$ vs 1.184; $(9,1.2)$: $1.512\pm0.012$ vs 1.512 |
| $\varepsilon_{\mathrm{SD}}=2\pi/\beta\hbar\eta_1$ | package's table of $\varepsilon(u,v)$ vs $\eta_1$ computed here | $v=2.1$, $u=10.5,12,13.5$: $\ln\varepsilon=0.459,0.558,0.641$ vs $A=0.453,0.555,0.640$ |
| $A_{\mathrm{Eck}}$ depends on $u$ only | $v=0.8$ to 3.6 | identical to four digits |
| $R_{\mathrm{Eck}}$ shrinks with $1/v$ | same table | $u=3$: $R\,v=0.12$ to 0.14; $u=9$: $R=0.13$ ($v=0.8$) to 0.01 ($v=3.6$) |

**Deterministic results for the valley models** ($G=\ln[k_{\mathrm{QM}}(H)/k_{\mathrm{QM}}(H_S)]$,
exact; `results/qm.json`, `results/skeleton.json`):

| model | $T/T_c$ | $G$: `mep` | `fe` | `fe+sct` | `bare` | $A_H$ | $A_H-A_{S^0}$: `mep` | `mep+sct` | `bare` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 (zero-point change) | 0.8 | +2.09 | −0.013 | – | – | 0.211 | 0.000 | – | – |
| | 0.55 | +2.28 | −0.019 | – | – | 0.518 | 0.000 | – | – |
| M2 (curved path) | 0.8 | −0.063 | +0.184 | −0.870 | – | 0.212 | +0.021 | −0.150 | – |
| | 0.55 | −0.048 | +0.301 | −1.475 | – | 0.492 | +0.024 | −0.016 | – |
| M3 (both) | 0.8 | +2.09 | +0.195 | −0.731 | – | 0.219 | +0.025 | −0.197 | – |
| | 0.55 | +2.33 | +0.347 | −1.351 | – | 0.496 | +0.033 | −0.053 | – |
| M4 (slow bath) | 0.8 | −0.585 | +0.037 | −0.652 | −2.48 | 0.200 | 0.000 | −0.089 | −0.199 |
| | 0.55 | −0.655 | +0.059 | −1.489 | −3.96 | 0.481 | +0.008 | −0.136 | −0.198 |

- The free-energy model reproduces the exact rate to 2% for M1 and 6% for the slow bath M4,
  but is 18–35% low on the curved paths M2 and M3, where the transverse frequency is only 1.5
  times the barrier frequency.
- The SCT mass overshoots by a factor 2 to 4 on every curved model: its adiabatic assumption
  does not hold at these frequency ratios.
- The skeleton on the minimum-energy path has the steepest-descent factor of the full system
  to 0.03 or better on all four models; the SCT and bare-mass skeletons do not, mostly because
  their crossover temperatures differ from that of $H$.

**Not yet tested** (RPMD runs in progress): the exact $\varepsilon(H)$ of M1–M4 and with it
every statement about transfer errors and the two anchors; the Eckart remainder as a
path-free estimate. Two SCT instantons at $0.45\,T_c$ (M2, M3) have a second unstable mode and
are not used.
