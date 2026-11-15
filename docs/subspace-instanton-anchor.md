# The instanton anchor: what a subspace correction has to carry

[When does a subspace correction transfer?](subspace-transfer-theory.md) shows that the error of

$$
k\approx k_{\mathrm{RPMD}}(H)\,C_S,\qquad C_S=\varepsilon(H_S),\qquad \varepsilon(X)=\frac{k_{\mathrm{QM}}(X)}{k_{\mathrm{RPMD}}(X)},
$$

is $\Delta=\ln\varepsilon(H)-\ln\varepsilon(H_S)$ and that only RPMD's *dynamical* error enters it.
This page derives the leading part of $\ln\varepsilon$ in closed form, shows that it is a property
of the full-dimensional instanton that needs neither a reaction path nor sampling, and turns
that into a correction that is right to leading order for any subspace. The tests are in
`benchmarks/subspace_instanton` (2D valley models M1–M4) and `benchmarks/subspace_complex`
(6D–10D curved/torsional, pre-reaction well, submerged ion–dipole complex, and 9D 3-atom
Cartesian LEPS + ion complex surfaces). Nothing here marks a rate as physically converged.

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

## Validation on 2D valley models (`benchmarks/subspace_instanton`)

`benchmarks/subspace_instanton` re-implements the model family and the exact reference of a
code package supplied for comparison (2D valley models M1–M4; a recursive Green's-function
$N(E)$ with exact leads) and uses this repository's RPMD solver across multi-seed runs
([validation report](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-14-subspace-instanton-anchor/README.md))
and 95 sharded full, reduced and Eckart RPMD tasks (`results/rpmd.json`, `report.py`).

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
  but is 17–29% low on the curved paths M2 and M3 (transverse frequency 1.1 to 1.8 times the
  barrier frequency).
- The SCT mass overshoots the rate by a factor of about 2 at $0.8\,T_c$ and 4 at $0.55\,T_c$ on
  every curved model: its adiabatic assumption does not hold at these frequency ratios.
- The skeleton on the minimum-energy path has the steepest-descent factor of the full system
  to 0.04 or better on all four models at every temperature computed; the SCT and bare-mass
  skeletons do not, mostly because their crossover temperatures differ from that of $H$.

**Full 2D RPMD defect and path-free matched Eckart estimate** (`results/rpmd.json`):

| Model | $T/T_c$ | $\varepsilon(H)$ | $\ln\varepsilon(H)$ | $A_H$ | $R_H=\ln\varepsilon(H)-A_H$ | $\ln C_{\mathrm{Eck}}$ | $\ln\varepsilon(H)-\ln C_{\mathrm{Eck}}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1.25 | $1.125\pm0.005$ | $+0.118\pm0.004$ | $0.000$ | $+0.118$ | $+0.149$ | $-0.031\pm0.008$ |
| | 1.00 | $1.177\pm0.005$ | $+0.163\pm0.004$ | $0.000$ | $+0.163$ | $+0.222$ | $-0.059\pm0.011$ |
| | 0.80 | $1.262\pm0.008$ | $+0.233\pm0.006$ | $0.211$ | $+0.022$ | $+0.280$ | $-0.047\pm0.017$ |
| | 0.65 | $1.441\pm0.016$ | $+0.365\pm0.011$ | $0.387$ | $-0.022$ | $+0.442$ | $-0.077\pm0.021$ |
| | 0.55 | $1.600\pm0.030$ | $+0.470\pm0.019$ | $0.518$ | $-0.048$ | $+0.544$ | $-0.074\pm0.028$ |
| M2 | 1.25 | $1.125\pm0.006$ | $+0.118\pm0.006$ | $0.000$ | $+0.118$ | $+0.135$ | $-0.017\pm0.009$ |
| | 1.00 | $1.191\pm0.009$ | $+0.174\pm0.007$ | $0.000$ | $+0.174$ | $+0.193$ | $-0.018\pm0.014$ |
| | 0.80 | $1.270\pm0.012$ | $+0.239\pm0.010$ | $0.212$ | $+0.027$ | $+0.301$ | $-0.062\pm0.018$ |
| | 0.65 | $1.477\pm0.022$ | $+0.390\pm0.015$ | $0.370$ | $+0.020$ | $+0.367$ | $+0.023\pm0.025$ |
| | 0.55 | $1.706\pm0.035$ | $+0.534\pm0.020$ | $0.492$ | $+0.042$ | $+0.493$ | $+0.041\pm0.028$ |
| M3 | 1.25 | $1.135\pm0.007$ | $+0.127\pm0.006$ | $0.000$ | $+0.127$ | $+0.149$ | $-0.022\pm0.009$ |
| | 1.00 | $1.177\pm0.008$ | $+0.163\pm0.007$ | $0.000$ | $+0.163$ | $+0.222$ | $-0.059\pm0.012$ |
| | 0.80 | $1.240\pm0.011$ | $+0.215\pm0.009$ | $0.219$ | $-0.004$ | $+0.289$ | $-0.074\pm0.018$ |
| | 0.65 | $1.390\pm0.018$ | $+0.329\pm0.013$ | $0.376$ | $-0.047$ | $+0.431$ | $-0.102\pm0.022$ |
| | 0.55 | $1.576\pm0.032$ | $+0.455\pm0.020$ | $0.496$ | $-0.041$ | $+0.523$ | $-0.068\pm0.029$ |
| M4 | 1.25 | $1.115\pm0.016$ | $+0.109\pm0.014$ | $0.000$ | $+0.109$ | $+0.097$ | $+0.012\pm0.016$ |
| | 1.00 | $1.157\pm0.019$ | $+0.146\pm0.016$ | $0.000$ | $+0.146$ | $+0.118$ | $+0.028\pm0.021$ |
| | 0.80 | $1.305\pm0.024$ | $+0.266\pm0.018$ | $0.200$ | $+0.066$ | $+0.239$ | $+0.027\pm0.026$ |
| | 0.65 | $1.489\pm0.029$ | $+0.398\pm0.019$ | $0.358$ | $+0.040$ | $+0.355$ | $+0.043\pm0.026$ |
| | 0.55 | $1.647\pm0.034$ | $+0.499\pm0.021$ | $0.481$ | $+0.017$ | $+0.496$ | $+0.003\pm0.033$ |

Across all 20 points, $A_H$ captures $\ln\varepsilon(H)$ below $T_c$ to $|R_H|\le 0.066$, and
the path-free matched Eckart estimate $\ln C_{\mathrm{Eck}}$ achieves an RMS log-rate error of
$0.051$ (maximum $0.102$; RMS $0.060$ across the 12 points below $T_c$).

**Aggregate transfer-error residuals below $T_c$** ($T/T_c\in\{0.8, 0.65, 0.55\}$,
$\Delta=\ln\varepsilon(H)-\ln C_S$):

| Subset | $N$ | RMS $|\Delta|$ | RMS skel $\Delta-(A_H-A_{S^0})$ | RMS naive $\Delta-(A_H-A_S)$ | Max $|\Delta|$ | Max skel | Max naive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All reductions | 36 | 0.076 | **0.048** | 0.069 | 0.227 | **0.108** | 0.173 |
| M1 `fe` (1st-order coupling) | 3 | 0.024 | **0.024** | 0.091 | 0.034 | **0.034** | 0.104 |
| `mep` | 12 | 0.044 | 0.054 | 0.054 | 0.102 | 0.102 | 0.102 |
| `fe` | 12 | 0.034 | **0.025** | 0.081 | 0.066 | **0.048** | 0.173 |
| `fe+sct` | 9 | 0.085 | **0.062** | 0.076 | 0.147 | **0.108** | 0.145 |
| `bare` | 3 | 0.188 | **0.048** | 0.048 | 0.227 | **0.078** | 0.078 |

- **Skeleton vs naive anchor (`M1 fe` and `M3 fe`).** On M1 `fe`, the unanchored free-energy
  projection already transfers to within statistical precision (RMS $0.024$). Because
  $A_H = A_{S^0}$ identically, the skeleton anchor adds zero spurious shift (RMS $0.024$),
  whereas the naive anchor $A_H - A_S$ double-counts the transverse zero-point barrier softening
  and degrades the residual nearly fourfold to RMS $0.091$ (and to $-0.173\pm0.018$ on M3 `fe`
  at $0.65\,T_c$, vs $-0.017\pm0.018$ for the skeleton anchor).
- **Slow bath and SCT reductions (`M4 bare` and `fe+sct`).** When the subspace skeleton misses
  the true crossover temperature or effective tunnelling mass, the skeleton anchor removes the
  leading error: on M4 `bare` it cuts the RMS residual fourfold from $0.188$ to $0.048$ (maximum
  from $0.227$ to $0.078$), and on `fe+sct` below $T_c$ from $0.085$ to $0.062$.
- **Crossover-mismatch and multi-mode gates.** At $T = T_c(H)$ ($T/T_c = 1.00$), `fe+sct` on
  M2/M3 and `bare` on M4 already sit below $T_c(S^0)$ while $A_H = 0$, producing an
  uncontrolled shift ($A_H - A_{S^0} = -0.21$ to $-0.25$); `instanton_anchor` flags this
  regime (`between_crossovers = True`, `valid = False`) unless explicitly overridden. At deeper
  tunnelling ($0.45\,T_c$), two SCT instantons (M2, M3) acquire a second negative Hessian mode
  and are likewise rejected by the one-negative-mode gate (`valid = False`).

## Validation on high-dimensional complex PESs (`benchmarks/subspace_complex`)

`benchmarks/subspace_complex` extends the test suite from symmetric 2D valleys to seven 6D–10D
cases (`results/complex_pes.json`, `report.py`) spanning four complex reaction topologies:

1. **Non-planar curved + slow-torsion valley (`twisted_10d_220K`, $D=10$, $T/T_c = 0.69$).**
   A 3D core with planar bend in $y$ and out-of-plane torsion + barrier-top mode softening in $z$
   ($\omega_z \approx 0.70\,\omega_b$), embedded with seven bath modes and rotated across all ten
   coordinates by $Q \in SO(10)$, checked against exact 3D `flux_eigenstates` quantum and 3D RPMD
   rates ($\ln C_{\mathrm{exact}} = +0.348 \pm 0.017$, $A_H = +0.376$).
2. **Barrier preceded by a pre-reaction van der Waals well (`prereaction_well_8d`, $D=8$,
   $200\,\mathrm{K}$ at $0.58\,T_c$ and $175\,\mathrm{K}$ at $0.51\,T_c$).** Exercises constrained
   $-\nabla\xi$ well-climb path continuation (`accepted = "reactant_reference"`), dome-bounded
   P-RFO instanton optimization across the pre-reaction well inflection point, and 1D/SCT/2D-RPH
   anchoring in deep tunnelling.
3. **Deep pre- and post-reaction ion–dipole complexes (`prereaction_complex_8d_200K`, $D=8$,
   $V_{\mathrm{TS}} > 0$, and `submerged_complex_6d_200K`, $D=6$, submerged barrier
   $V_{\mathrm{TS}} = -0.0030\,\mathrm{E_h} < V_{\mathrm{asymptote}} = 0$).** Exercises
   topographic-prominence saddle selection, product-well termination (`accepted = "product_well"`),
   and connected-valley 2D saddle localization.
4. **Full 9D 3-atom Cartesian molecular complex (`leps_ion_complex_9d`, $D=9$, `rigid_body = True`,
   `RPMDrateCoordinate`, $220\,\mathrm{K}$ at $0.58\,T_c$ and $180\,\mathrm{K}$ at $0.48\,T_c$).**
   3-atom LEPS transfer surface in 3D space ($\mathrm{H + TH \to HT + H}$) with damped
   $-C_{\mathrm{ion}}/(r^4 + R_0^4)$ pre-reaction complexes and exact 3D translational/rotational
   invariance, cross-checking 9D `full_instanton_defect` against standalone
   `pyrpmd.instanton.optimize_instanton`.

**9D 3-atom Cartesian molecular results** (`results/complex_pes.json`):

| Case | $T$ (K) | $T/T_c$ | $A_H$ (9D) | $A_H - A_{\mathrm{opt}}$ | Model | $\ln C_S$ | $A_H - A_{S^0}$ | $\ln C_S^{\mathrm{skel}}$ | $\ln(C_{\mathrm{RPH}}/C_S)$ | $\ln(C_{\mathrm{RPH}}^{\mathrm{skel}}/C_S^{\mathrm{skel}})$ |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `leps_ion_complex_9d_220K` | 220 | 0.58 | $+0.376$ | $+5.7\times 10^{-6}$ | `1d` | $+0.013$ | $+0.041$ | $+0.053$ | $+0.138$ | $+0.098$ |
| | | | | | `1d_sct` | $+0.278$ | $-0.051$ | $+0.227$ | $-0.127$ | $-0.076$ |
| | | | | | `rph` | $+0.151$ | $+0.001$ | $+0.152$ | $0.000$ | $0.000$ |
| `leps_ion_complex_9d_180K` | 180 | 0.48 | $+0.394$ | $+8.0\times 10^{-6}$ | `1d` | $+0.042$ | $+0.024$ | $+0.066$ | $-0.147$ | $-0.170$ |
| | | | | | `1d_sct` | $-0.869$ | $+0.001$ | $-0.868$ | $+0.764$ | $+0.764$ |
| | | | | | `rph` | $-0.105$ | $+0.001$ | $-0.104$ | $0.000$ | $0.000$ |

**Key findings on high-dimensional complex PESs:**

- **Strict non-planar hierarchy (`twisted_10d_220K`).** On the 10D non-planar curved + torsional
  surface, 2D RPH captures the dominant planar bend ($\text{captured\_curvature} = 0.977$) while
  leaving the slow out-of-plane torsion in the adiabatic bath, yielding the strict steepest-descent
  hierarchy $A_H(10\mathrm{D}) = +0.376 > A_{\mathrm{RPH}^0} = +0.347 > A_{1\mathrm{D}^0} = +0.260$.
  2D RPH reproduces $\ln C_{\mathrm{exact}}$ to $-0.006 \pm 0.025$ unanchored and $-0.035 \pm 0.025$
  skeleton-anchored, while `1d_sct` develops a multi-mode instanton instability (`anchor_valid = False`).
- **9D Cartesian rigid-body invariance and corner cutting (`leps_ion_complex_9d`).** Starting from
  an off-saddle geometry in 9D atomic Cartesians, `full_instanton_defect(..., rigid_body=True)`
  agrees with standalone `optimize_instanton` to $5.7\times 10^{-6}$ ($220\,\mathrm{K}$) and
  $8.0\times 10^{-6}$ ($180\,\mathrm{K}$). Because 2D RPH unrolls the collinear transfer plane
  ($\text{captured\_curvature} = 1.000$), its skeleton shift vanishes ($+0.0006$ at $220\,\mathrm{K}$,
  $+0.0010$ at $180\,\mathrm{K}$), whereas the 1D MEP skeleton misses corner cutting ($+0.041$ at
  $220\,\mathrm{K}$, reducing the 1D-vs-RPH gap from $+0.138$ to $+0.098$), and `1d_sct` breaks down
  severely at $180\,\mathrm{K}$ ($\ln C_S = -0.869$ vs $-0.105$ for 2D RPH).
- **Submerged barriers and deep complexes (`prereaction_complex_8d` and `submerged_complex_6d`).**
  When $V_{\mathrm{TS}} < V_{\mathrm{asymptote}}$ (`submerged_complex_6d_200K`), centroid PMF
  umbrella integration lowers the free energy at the submerged central saddle below the asymptotic
  reactant reference while short-time recrossing trajectories trap transiently inside the flanking
  complexes, causing a four-order-of-magnitude breakdown of the raw short-time RPMD plateau rate
  ($\ln C_{\mathrm{exact}} \ll 0$). Even in this extreme regime, `1d` + skeleton anchoring improves
  the log-rate transfer error by $+0.095$ (whereas `1d_sct` overshoots RPMD by an order of magnitude
  and misses $\ln C_{\mathrm{exact}}$ by $\sim 3$ log units). On the positive-barrier complex
  `prereaction_complex_8d_200K`, 2D RPH captures the complex-to-barrier bend and matches
  $\ln C_{\mathrm{exact}}$ to $\sim 0.11$ log units while 1D misses by $\sim 0.85$ log units.

## API and workflow usage

- `pyrpmd.instanton.steepest_descent_defect(result, *, beta=None, temperature_kelvin=None)`
  (and `pyrpmd.instanton.steepest_descent_rpmd_defect(result)`) evaluates
  $\varepsilon_{\mathrm{SD}} = 2\pi/(\beta\hbar\eta_1)$ from a converged `InstantonResult`
  below $T_c$ (`steepest_descent_defect` also returns `factor = 1.0, log_factor = 0.0` for
  `result=None` at or above $T_c$).
- `SubspaceModel.skeleton()` returns a bare-potential `SubspaceModel` on the same axes, masses
  and dividing surface (`bare_values` is recorded with zero extra PES evaluations by
  `project_linear_subspace`, `project_path_subspace` and `project_rph_values(..., return_bare=True)`).
- `pyrpmd.analysis.subspace_instanton_defect(model, beta, *, use_skeleton=False, nbeads=128)`
  and `pyrpmd.analysis.full_instanton_defect(energy, gradient, masses, beta, saddle_guess, ...)`
  compute $A_S$ / $A_{S^0}$ and $A_H$ with dome-bounded partitioned rational-function
  optimization (P-RFO) on the folded half-ring action, rigid-body projection (`rigid_body=True`),
  and sparse `LinearOperator` + `eigsh` spectra when `nbeads * D > 2048`.
- `pyrpmd.analysis.instanton_anchor(correction, full_defect, subspace_defect)` and
  `apply_subspace_correction(..., anchor_log_shift=...)` apply $\ln C_S + A_H - A_{S^0}$ and
  enforce the crossover-consistency and single-negative-mode validity checks.
- In TOML configurations, set `anchor = "skeleton"` under `[rate.subspace_correction]` (or under
  `[subspace]` for `pyrpmd subspace-correct`) to enable skeleton anchoring; the default
  `anchor = "none"` leaves existing workflows unchanged. Anchored records set
  `approximation = "instanton-anchored multiplicative subspace defect"` while preserving
  `exact_general_rate_claim = false` and `physical_converged = false`.
