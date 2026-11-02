# H + CH₄ independent surface reference: feasibility review

**Status: mathematical design only. The reference estimator is unimplemented,
unregistered and unvalidated. No PES evaluation, dynamics, partition estimate
or reaction-rate estimate was performed for this review.** The separate
[geometry checks](geometry-checks.json) test coordinate algebra, not statistical
or physical convergence. All reviewed scientific source hashes were unchanged.

An independent classical reference is feasible. Integrating out one incoming-H
radius converts the exact flux surface into a regular integral over methane
shapes and two incoming-direction angles. Removing overall rotation leaves
**11 surface dimensions and 9 isolated-methane dimensions**. This eliminates
the need to traverse 111 umbrella windows, but it does not reduce methane to
the two-dimensional classical H₃ oracle. Its cost and statistical efficiency
remain unknown until an excluded, bounded energy/weight pilot is registered.

The recommended first target is the **finite-R∞ ratio of two flux-surface
integrals evaluated with the same six-atom PES**. That reproduces the existing
rate definition without assuming that an isolated CH₄ potential or its absolute
partition normalization has already been independently established.

## 1. Fixed inputs and comparison surfaces

The reviewed conventions are the frozen
[HCH4 protocol](../../../../../benchmarks/hch4/reproduction_protocol.json),
[input generator](../../../../../benchmarks/hch4/reproduction_input.py),
[geometry](../../../../../benchmarks/hch4/geometries.json),
[PES receipt](../../../../../benchmarks/hch4/source.json), and
[production CV](../../../../../src/pyrpmd/cv.py). The upstream CBE/PES-2008
source is pinned to RPMDrate commit
[`8c1828d2302aacf09a15faf5c7fc50a18d4c113e`](https://github.com/GreenGroup/RPMDrate/tree/8c1828d2302aacf09a15faf5c7fc50a18d4c113e).
No external PES source or binary is included in this review.

The primary comparison is Table 2 of Suleimanov, Allen and Green,
[CPC 184, 833–840 (2013)](https://doi.org/10.1016/j.cpc.2012.10.017), at 300 K:

| Beads | Published surface ξ‡ | TST/QTST | κ | Full rate |
|---|---:|---:|---:|---:|
| 1 | 0.999 | 6.53 × 10⁻²² | 0.762 | 4.98 × 10⁻²² |
| 128 | 1.007 | 1.71 × 10⁻¹⁹ | 0.659 | 1.13 × 10⁻¹⁹ |

Rates have units cm³ molecule⁻¹ s⁻¹. These are different PMF maxima for the two
bead counts. The [author-uploaded manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics)
is readable. The [MIT record](https://dspace.mit.edu/entities/publication/a4555c62-26a0-4500-8953-5dd1905583cc)
identifies an original manuscript; its PDF endpoint returned HTTP 429 during
this review, so no inaccessible final-version content is asserted. Existing
[paper/source differences](../../other-reactions/hch4/reproduction-plan.md)
remain applicable, including isotope/unit constants, R∞ and window endpoints.

The atom order is H,C,H,H,H,H. Carbon is atom 1, incoming H is atom 5, and
methane H atoms are 0,2,3,4, using zero-based indices. Let their common mass be
\(m\), the carbon mass be \(M\), and

\[
M_B=M+4m,\qquad \mu=\frac{m M_B}{m+M_B},\qquad L=R_\infty=30.
\]

The explicit isotope masses are H=1.00782503207 u and C=12 u. With the current
recorded PyRPMD constants at 300 K, \(\mu=1728.4892968817187\) electron masses
and \(\beta=1052.5834160132904\) hartree⁻¹. The reference bond difference comes
from the **exact distributed coordinates**, not rounded paper distances or a
newly optimized saddle:

\[
c=r_f^\ddagger-r_b^\ddagger
 =1.649352979111344-2.6272862155199657
 =-0.9779332364086217\ {\rm bohr}.
\]

For methane H label \(h\), define

\[
s_0=L-|q_5-R_B|,\qquad
s_h=c+|q_h-q_C|-|q_5-q_h|,\qquad
s_1=\max_{h\in\{0,2,3,4\}}s_h,\qquad
\xi=\frac{s_0}{s_0-s_1}.
\]

There is one paired forming/breaking bond and four equivalent channels. The
configured electronic weight and symmetry multiplier are both 1. All four
channels are already represented by the maximum; they are not nuclear
permutation exchange.

## 2. Exact flux measure and radial elimination

This section is an independent derivation from the frozen Cartesian CV and
canonical Maxwell momentum distribution. All coordinates, masses, energies
and times below are in atomic units.

For a fixed published surface \(\alpha\), introduce

\[
S_\alpha=(1-\alpha)s_0+\alpha s_1,\qquad
f_S=\sqrt{\frac{\nabla S_\alpha^T\mathsf M^{-1}\nabla S_\alpha}{2\pi\beta}}.
\]

Writing \(D=s_0-s_1\), the identity \(\xi-\alpha=S_\alpha/D\) gives, on a
nonsingular surface,

\[
\delta(\xi-\alpha)f_\xi=\delta(S_\alpha)f_S.
\]

The \(|D|\) from the delta function exactly cancels its inverse in the flux
metric. Keeping one factor but omitting the other changes the rate. The
intended reactant-connected component has \(D>0\); on these positive-α
surfaces this is equivalent to \(|q_5-R_B|<L\). Any disconnected component
across the CV pole requires explicit exclusion/analysis in a new domain
definition, rather than silent inclusion by a root solver.

Set carbon at the origin without removing any physical degrees of freedom:
\(b_h=q_h-q_C\), \(x=q_5-q_C\), and
\(d=(m/M_B)\sum_h b_h\). For one candidate active H, write

\[
x=b_h+\rho u,\quad \rho>0,\quad |u|=1,\quad
r=|b_h|,\quad v=b_h-d,\quad R(\rho)=|v+\rho u|.
\]

The branch surface is

\[
S_h(\rho)=(1-\alpha)[L-R(\rho)]+\alpha(c+r-\rho),
\quad
\frac{dS_h}{d\rho}=-\alpha-(1-\alpha)u\cdot u_R,
\quad u_R=\frac{v+\rho u}{R}.
\]

Since \(\alpha>|1-\alpha|\) at both 0.999 and 1.007, the derivative is
strictly negative. Its magnitude is at least 0.998 and 1.000 respectively.
There is **at most one positive root**, and it exists exactly when
\(S_h(0)>0\), excluding zero-measure boundary singularities. If this condition
fails, that shape/direction contributes zero; it must not be replaced by a
nearby root or a renormalized accepted draw. A bracket is

\[
0<\rho_h\le\frac{S_h(0)}{\alpha-|1-\alpha|}.
\]

The root uses only geometry, not a PES call. Bisection or a safeguarded scalar
solver avoids spurious roots from squaring the equation. The three-dimensional
incoming measure becomes

\[
d^3x\,\delta(S_h)
 =d\Omega_u\,\frac{\rho_h^2}{|dS_h/d\rho|}.
\]

### Mass metric

Let \(u_b=b_h/r\). For the selected branch, the nonzero gradients of \(s_h\)
are \(-u\) on incoming H, \(u_b+u\) on the active H, and \(-u_b\) on carbon.
The entry-surface gradient is \(-u_R\) on incoming H and
\((m_i/M_B)u_R\) on each methane atom. Therefore

\[
G_\alpha=\frac{(1-\alpha)^2}{\mu}
 +\alpha^2\left[\frac{3+2u_b\cdot u}{m}+\frac1M\right]
 +\frac{2\alpha(1-\alpha)}{\mu}u_R\cdot u,
\qquad f_S=\sqrt{\frac{G_\alpha}{2\pi\beta}}.
\]

The methane COM terms containing \(u_b\) cancel in the cross term. Replacing
this metric by an incoming-H reduced mass alone is incorrect away from the
entry surface. For other isotope patterns, use the full mass-weighted
Cartesian metric and re-evaluate channel symmetry; this equal-H expression
does not authorize a four-channel shortcut for inequivalent isotopes.

### Channels and normalization

Let \(I_h\) select points where the candidate branch is the active maximum,
with a deterministic tie convention on the measure-zero seams. Let
\(I_{\cal D}\) represent a separately justified physical/molecular domain.
After dividing out the overall translation volume, the flux numerator is

\[
N_\alpha=\sum_h\int d^{12}b\,d\Omega_u\;
 I_h I_{\cal D}\,
 \frac{\rho_h^2 f_S}{|dS_h/d\rho|}
 e^{-\beta V(b,x_h)}.
\]

This **sum already counts all four channels**. Alternatively, four times one
channel is exact only after proving the equal-mass PES, domain, measure and
proposal preserve the required permutation symmetry, and retaining \(I_h\)
on that channel. Dropping \(I_h\) counts branch surfaces that lie beyond the
maximum surface. The diagnostic includes an explicit non-equilibrium
counterexample; it does not estimate that region's physical weight.

All Cartesian momenta were integrated analytically. Total translation and
their common Gaussian normalizers cancel between numerator and denominator.
No additional 4!, molecular rotor symmetry number, inertia determinant,
Planck factor, or bead-dependent mass factor should be appended to these
consistently labeled configurational ratios. A different fundamental domain
is possible only with its multiplicity applied consistently to both sides.

## 3. Reactant normalization: retain the finite-R∞ definition first

The isolated-methane expression is

\[
Z_B=\int d^{12}b\;I_B e^{-\beta V_B(b)},\qquad
k_{\rm TST,1}(\alpha)=\frac{N_\alpha}{Z_B}.
\]

It is valid when the free incoming atom decouples and \(V_B\), including its
energy convention and molecular domain, is well defined. The existing adapter
exposes a six-atom PES, so simply calling a five-atom CH₄ potential is not an
available validated operation. A remote sixth atom approximates an isolated
potential only after separation/orientation controls over thermally relevant
shapes.

An immediately well-defined comparison with the current rate workflow uses
the entry-surface integral instead:

\[
N_0=\frac{L^2}{\sqrt{2\pi\beta\mu}}
 \int d^{12}b\,d\Omega\;I_0
 e^{-\beta V(b,d+L u)},
\]

\[
F_\infty=\frac{4\pi L^2}{\sqrt{2\pi\beta\mu}},\qquad
k_{\rm TST,1}(\alpha)=F_\infty\frac{N_\alpha}{N_0}.
\]

This retains the same six atoms, PES and finite-R∞ convention on both sides.
When the interaction is negligible, \(N_0=F_\infty Z_B\), recovering the
isolated formula. The known prefactor here is
3.3450362638130775 bohr³/atomic-time, or
2.0492225575240845 × 10⁻⁸ cm³ molecule⁻¹ s⁻¹. It is fixed by mass, temperature
and R∞, never calibrated using the reference rate. A finite-R∞ calculation
still needs separation controls before claiming an asymptotic physical rate.

Dimensional check: \(N_\alpha\) has length¹⁵/time, while \(Z_B\) has length¹²;
their ratio has length³/time. \(N_\alpha/N_0\) is dimensionless.

## 4. Removing global rotation without losing a Jacobian

An explicit chart fixes an active methane H at \(b_h=r e_z\) and a specified
spectator at \(b_j=s(\sqrt{1-v^2},0,v)\), with \(r,s>0\), \(-1<v<1\).
Let the remaining two methane-H vectors be \(a,b\in\mathbb R^3\). Integrating
the overall SO(3) rotation gives

\[
d^{12}b=8\pi^2 r^2s^2\,dr\,ds\,dv\,d^3a\,d^3b.
\]

The common \(8\pi^2\) cancels. The radial factors do not. Nine internal
methane coordinates plus two incoming angles give eleven surface coordinates.
The finite-R∞ denominator also has eleven coordinates before any justified
asymptotic angular simplification. Near-collinear chart boundaries have zero
measure but must not cause numerical division errors.

Both mirror versions of a labeled tetrahedral well must be represented.
A normalized mixture can cover them. A single local harmonic well is not an
automatic symmetry quotient. Reflection or permutation restrictions require
an exact domain proof and matching multiplicities. The geometry check
independently finite-differences the full twelve-coordinate Euler-angle map;
its determinant agrees with \(r^2s^2\sin\theta\), and integrating the Euler
angles gives the stated factor.

## 5. A plausible computational route, with explicit stop conditions

Ordinary methane equilibrium samples are a poor direct proposal for the
surface: the active C–H bond is substantially stretched. Moving the incoming
atom to the root alone does not make that rare methane stretch common. Using
that estimator without tail/overlap controls could produce a precise-looking
answer with unseen dominant weights.

A proposed implementation should instead use separately normalized, full-
support reference distributions near the methane and constrained-surface
regions. A local Hessian may define proposal scales; it does not replace the
PES Boltzmann weight. The target contains the exact chart Jacobian, root
factor, metric and channel indicator. A direction proposal can combine a
known-normalized directional distribution with a uniform-sphere component.
Mixtures should include the label/mirror sectors and sufficiently broad tails.

Direct importance sampling or randomized quasi-Monte Carlo is an initial
efficiency diagnostic. If weights are unstable, a reference-to-PES bridge with
adjacent-state BAR is a possible alternative. Each reference normalizer must
be known. If its support is truncated by a physical domain, the accepted
reference probability is a separate normalization factor, not silently one.
Invalid draws can contribute zero to an unconditional importance average;
discarding them and dividing by only the accepted count changes the integral.
Unseen tails remain a limitation even when empirical ESS is large.

There is a genuine domain prerequisite. Integrating methane over all
\(\mathbb R^{12}\) includes arbitrarily separated dissociation products and
does not give a finite bound-molecule partition function. The existing PES
adapter's 0.2-bohr minimum-pair guard prevents a numerical singularity; it is
not a validated molecular-basin definition or a guarantee of reliable global
repulsion. Before numerical normalization, declare bound-methane and reactive
surface domains, audit the PES at their boundaries, and demonstrate negligible
boundary sensitivity at the required precision. The H₃ cutoff cannot simply
be copied. If relevant weight reaches unreliable PES regions, stop and report
the model/domain limitation rather than selecting a cutoff to match the paper.

The PES has an unshifted negative energy origin. Evaluate weights in log space
with an explicit common energy offset; do not exponentiate large absolute
\(-\beta V\) values. If separate reference offsets are used, their exact
Boltzmann factors must appear in the final ratio. The pinned energy/gradient
rounding discrepancy remains unchanged: static energy integration and later
force-driven dynamics must retain that stated model qualification.

### Proposed next gate — not adopted or run

1. Independently review the root/metric/channel and rotational measures. The
   geometry-only gate in this directory is an initial algebra check; add
   analytic Gaussian integrals and controlled symmetry tests when a real
   estimator is implemented.
2. Register a separate energy-only feasibility pilot with excluded seeds,
   all sources hashed, a fixed proposal construction, and ceilings of one
   CPU, 60 process-CPU seconds and fewer than 1000 evaluated PES geometries.
   A candidate allocation is 256 draws for each of the three targets
   \(N_0,N_{.999},N_{1.007}\): at most 768 P1 energy evaluations, with every
   rejected/zero weight retained and no uncounted proposal/Hessian/PES work.
   Proposals may be constructed from existing coordinates without new energy
   calls. These counts are proposed resource limits, not convergence claims
   or authorization to execute. The exact finite-domain/proposal design
   requires independent support and normalization review before execution.
3. Report actual energy-call time, weight ESS/max weight, independent replica
   or randomized-scramble dispersion, mirror/channel coverage, and sensitivity
   to wider proposals/domain boundaries. A timeout is an incomplete pilot,
   not an accuracy pass. Select any subsequent fixed budget/bridge solely
   from these diagnostics, before looking at paper agreement.
4. Validate the P1 result at ξ=.999 and separately calculate ξ=1.007. Preserve
   covariance when the same shapes estimate both surfaces. Use whole
   independent chains/scrambles as uncertainty units; quadrature nodes,
   individual correlated configurations and paired surfaces are not extra
   independent replicates. The frozen original HCH4 protocol remains intact;
   an alternative estimator needs its own reviewed registration.
5. Only after this classical normalization succeeds, assess the quantum
   extension below. A full-rate claim also needs new matched-surface
   transmission coefficients and the existing time-step/plateau checks.

### What the measured timing does and does not establish

The retained [native measurements](../../other-reactions/hch4/native/verification.json)
and [budget calculation](../../other-reactions/hch4/reproduction-plan.md) give:

| Existing operation | P1 | P128 |
|---|---:|---:|
| Native biased MD step | 17.5–24.5 μs | 0.957–1.061 ms |
| 26.64 billion umbrella steps | 130–181 CPU h | 7085–7854 CPU h |

These calls include forces, propagation, CV/Hessian/bias evaluation and
observations. They are **not timings of energy-only surface importance
sampling**, and they do not predict its effective sample size. The direct
method's cost is approximately the number of evaluated geometries times the
measured energy-call cost, plus proposal/root/analysis overhead. A P-bead
proposal ordinarily evaluates P bead energies, possibly in one batch. Neither
the required proposal count nor that implementation's timing has been
measured. No factor-of-speedup or affordable P128 precision is established by
the present algebra.

## 6. Quantum extension and the different-surface normalization trap

With normalized zero-centroid free-ring priors and a fixed physical domain,
define \(N_{P,\alpha}\) by replacing the classical Boltzmann factor in the
centroid surface integral with
\(\mathbb E_\eta[I_{\cal D}\exp(-\beta P^{-1}\sum_j V(q_c+\eta_j))]\).
The centroid root, channel indicator and flux metric are unchanged. Then the
finite-R∞ identity is

\[
k_{{\rm QTST},P}(\alpha)
=k_{{\rm TST},1}(\alpha)
\frac{N_{P,\alpha}/N_{1,\alpha}}{N_{P,0}/N_{1,0}}.
\]

In the verified asymptotic limit, the denominator correction can be written
as \(Z_{B,P}/Z_{B,1}\). Without that verification, retain the six-atom entry
integral. A full-support energy bridge can estimate these corrections, with
free-ring pCN moves and exact reference/domain normalizers. This is a proposed
method, not an implemented extension of the H₃ sampler.

The quantum surface is **1.007**, while the paper classical surface is
**0.999**. Multiplying a correction at 1.007 by the classical rate at 0.999 is
wrong unless the extra factor
\(k_{{\rm TST},1}(1.007)/k_{{\rm TST},1}(.999)\) is included. Likewise a
transmission coefficient at one surface cannot be reused at the other.
Surface invariance applies to a converged QTST×κ product, not to its separate
components.

For a free-ring construction, use four methane Jacobi vectors in an explicitly
declared H ordering, with reduced masses

\[
\mu_j=\frac{m[M+(j-1)m]}{M+jm},\quad j=1,\ldots,4,
\qquad \mu_5=\frac{m(M+4m)}{M+5m}
\]

for the incoming atom relative to methane. The spring action of each internal
vector is \(P\mu_j\sum_k|\eta_{k+1}-\eta_k|^2/(2\beta)\). The total COM
free modes integrate out. At P128 the six-atom surface has 1905 noncentroid
internal scalar coordinates, in addition to its centroid shape coordinates;
isolated methane has 1524. Normalize the free priors explicitly so no hidden
P powers or mass determinants are introduced. At P1 every quantum correction
must equal one exactly. Independently solvable multidimensional harmonic
models are essential normalization tests.

Methane has nine internal vibrational degrees of freedom, so the diatomic
radial-spectrum/transfer oracle used for H₃ reactants is unavailable. Both
surface and reactant corrections need validation; their possibly small
individual values do not disappear because their ratio is moderate. More
dimensions, C–H stretches, noncentroid reactive modes and channel/domain
restrictions can obstruct mixing. H₃'s successful λ grid, pCN scale, sample
count or CPU cost is not evidence that those choices work here. A separate
fixed overlap/mixing calibration and P64/P128/P256 controls remain necessary.

## 7. Completed scope and reproducibility

[geometry_checks.py](geometry_checks.py) performs 128 synthetic perturbed-
geometry root/CV checks, twelve independent coordinate-map Jacobian checks,
and a separate channel-exclusion counterexample. It uses no PES or dynamics.
Maximum errors were 1.24×10⁻¹⁰ relative for the flux metric against Cartesian
finite differences, 1.11×10⁻¹⁰ relative for the gauge Jacobian, and below
7×10⁻¹⁶ for coarea/surface identities. Nine scientific input/source hashes
match before and after. The initial attempt made an unnecessarily strong
synthetic-coverage assertion that a small near-saddle set would contain an
inactive branch; it did not. That [observation](initial-diagnostic-observation.json)
is retained, and the exclusion case is now separately labeled without any
inference about its Boltzmann weight.

A second reviewer independently checked the mass metric, monotonic-root
condition, channel count and matching rotation/permutation convention. This
is an algebra review only. It supplies neither a molecular-domain validation
nor a rate, statistical precision, or performance result.

Replay the geometry-only diagnostic from the original source checkout with:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python \
  docs/validation/literature-convergence-2026-11-01/review/hch4-reference-feasibility/geometry_checks.py
```

It prints a new report; keep the frozen recorded report unchanged. The
[review receipt](review.json) and [publication inventory](publication-inventory.json)
bind this design and its bounded checks. There is a valid route to implement,
but the stopping point here is deliberate: **no classical or quantum H + CH₄
literature agreement follows until a molecular-domain and statistical
normalization calculation has actually been completed.**
