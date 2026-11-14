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
\Delta=\underbrace{A_H-A_S}_{\Delta_{\mathrm{SD}}}+\,(R_H-R_S),\qquad
\Delta_{\mathrm{SD}}=\ln\frac{\eta_1^{S}}{\eta_1^{H}}\quad(T<T_c\text{ of both}).
$$

$\Delta_{\mathrm{SD}}$ is computable before any rate is solved. It is a prediction of the
transfer error, and removing it gives the **instanton-anchored correction**

$$
\boxed{\;\ln C_{\mathrm{anch}}=\ln C_S+A_H(\beta)-A_S(\beta)\;}
$$

- If $H_S$ is exact, $A_S=A_H$ and nothing changes.
- Below both crossovers $C_{\mathrm{anch}}$ is correct to steepest-descent order **for any
  subspace**, including one whose path, mass or bath treatment is wrong. The subspace then only
  has to transfer the remainder, $R_S\approx R_H$.
- Above both crossovers $A_H=A_S=0$ and $C_{\mathrm{anch}}=C_S$.
- If the two crossover temperatures differ, the kinks of $A_H$ and $A_S$ sit at different
  temperatures and the anchor is uncontrolled between them. A subspace built on the
  minimum-energy path has the right crossover for the bare potential (the path leaves the
  saddle along the unstable mode); a zero-point or free-energy term and the SCT mass shift it.

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
