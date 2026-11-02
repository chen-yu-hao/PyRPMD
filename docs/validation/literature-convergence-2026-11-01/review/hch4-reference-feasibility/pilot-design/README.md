# Proposed P1 finite-R∞ cost and weight pilot

**Draft for independent measure/support review. Not implemented, adopted,
registered or executed. No H + CH₄ rate is claimed.** This proposal is separate
from the frozen [feasibility review](../README.md). It would evaluate at most
768 P1 geometries with the already pinned CBE/PES-2008 library, on one CPU with
a 60-process-CPU-second ceiling. It contains no dynamics or umbrella sampling.
The exact settings are in [proposal.json](proposal.json).

Its purpose is to measure the actual cost and weight concentration of direct
importance estimates of the finite-R∞ entry surface and the two published
transition surfaces. It targets an explicitly truncated mathematical domain.
The truncation has not been shown to approximate the molecular partition
function. Even a numerically quiet pilot cannot establish paper agreement,
physical-domain convergence, an asymptotic rate, or a viable quantum budget.

## Targets and fixed domain

Use the frozen H,C,H,H,H,H order, masses, exact reference bond difference,
four-channel CV, temperature 300 K and R∞=30 bohr in the feasibility review.
Keep α=0.999 and 1.007. At α=0 the entry surface is R=30, with the incoming
atom at \(x=d+30u\). Each transition branch uses the unique positive root
\(x=b_h+\rho_h u\) derived in the review.

The provisional finite domain is identical for all methane labels:

\[
1.4\le |b_i|\le4.5\ {\rm bohr}\quad(i=0,2,3,4),
\qquad \min_{i<j}|q_i-q_j|\ge0.7\ {\rm bohr}.
\]

The minimum-pair bound applies to all six atoms after constructing the surface
point. Transition points additionally require a positive radial root, the
selected branch to be the active maximum and \(s_0-s_1>0\). Entry points do
not have an active-channel restriction: the full entry sphere appears once.
On these declared bounds the entry component is automatically the intended
one: \(|d|\le(4m/M_B)4.5<1.133\) bohr, so every incoming-H/methane-H
distance at R=30 exceeds 24.367 bohr and every \(s_h<-20.84\) bohr. Thus
\(s_0-s_1=-s_1>0\) there, without another selection factor.

These numbers are **provisional pilot bounds**, selected before energies.
They are not fitted to a rate, inferred from a successful short MD path, or
described as a validated physical PES domain. They include the ordinary
reactant and saddle geometries and give a finite integration region. Their
boundary mass, reliability of the PES inside them and omitted molecular
partition weight remain open. Suspect energies or failures are preserved;
there is no after-the-fact deletion, energy clipping or cutoff adjustment.
A later domain change would be a new separately registered pilot.

## Full-support shape proposal

For candidate active H \(h\), choose the smallest other methane-H label as
the spectator \(j\). Order the remaining two labels increasingly as \(a,b\).
Use the nine-dimensional chart

\[
z=(r,s,v,a_x,a_y,a_z,b_x,b_y,b_z),\qquad
b_h=(0,0,r),\quad b_j=(s\sqrt{1-v^2},0,sv).
\]

The shape Jacobian after removing the common overall rotation factor is
\(J(z)=r^2s^2\). The domain lies inside

\[
B=[1.4,4.5]^2\times[-1,1]\times[-4.5,4.5]^6,
\quad |B|=(4.5-1.4)^2\,2\,(9)^6.
\]

Use the normalized mixture on all of \(\mathbb R^9\)

\[
q_z(z)=0.20\,\frac{\mathbf 1_B(z)}{|B|}
       +0.40\,\mathcal N(z;\mu_+,\Sigma)
       +0.40\,\mathcal N(z;\mu_-,\Sigma).
\]

Both Gaussians are **untruncated**. Draws outside the chart or target domain
are saved with zero weight and remain in the sample count. The negative-mirror
mean reverses only \(a_y,b_y\). The uniform component guarantees strictly
positive density everywhere in the finite target domain; the reflected
Gaussians place finite sampling effort near both labeled local wells. This
mathematical support guarantee does not imply adequate tail sampling with
256 points. Always divide by the **full mixture density**, not the density of
the selected mixture component.

For transition proposals, obtain \(\mu_+\) from the exact official input:
exchange its abstracted H label 3 with candidate label \(h\), translate carbon
to zero, align \(b_h\) to +z, and rotate the chosen spectator into the xz
half-plane with positive x. The remaining coordinates give \(\mu_+\).
No PES optimization or Hessian is evaluated. The diagonal standard deviations
are \((.15,.12,.12,.20,.20,.20,.20,.20,.20)\), with lengths in bohr and the
third coordinate dimensionless. The same proposal serves both α values.

For entry proposals, use a fixed h=3,j=0 chart and an exact regular tetrahedron
with C–H radius \(r_0=1.0890\) Å converted with the frozen constants:

\[
b_3=(0,0,r_0),\quad
b_0=r_0(\sqrt{8/9},0,-1/3),
\]

\[
b_2=r_0(\sqrt{8/9}\cos(2\pi/3),
             \sqrt{8/9}\sin(2\pi/3),-1/3),
\]

and \(b_4\) uses azimuth \(-2\pi/3\). The entry standard deviations are
\((.12,.12,.12,.20,.20,.20,.20,.20,.20)\). The reference radius is a proposal
center from the existing geometry receipt, not a substituted physical
potential or an assumed harmonic partition function.

## Direction proposal

At the entry surface use uniform \(q_{u,0}=1/(4\pi)\). At each transition
branch use

\[
q_{u,\ddagger}(u)=\frac{0.20}{4\pi}
  +0.80\,\frac{8}{4\pi\sinh8}\,e^{8u_z},\quad u\in S^2.
\]

The second component is a normalized von Mises–Fisher density about the
outward active C–H axis. Its finite concentration is a fixed proposal choice,
not a constraint on the physical approach angle. Both components cover the
whole sphere. A draw from the second component can use
\(u_z=1+\log[U+(1-U)e^{-16}]/8\) and an independent uniform azimuth. Use the
full mixture density in the importance denominator.

## Exact finite-domain importance weights

Use a shared energy offset \(E_*=-0.7\) hartree and compute all densities and
weights in log space. It multiplies every integral by the same constant and
cancels from ratios. No energy clipping or fitted normalization is permitted.

For a draw from candidate channel \(h\), the transition weight is

\[
w_{h,\alpha}=
\frac{J(z)\,\rho_h^2}{q_z(z)q_{u,\ddagger}(u)}
\frac{\sqrt{G_\alpha/(2\pi\beta)}}{|dS_h/d\rho|}
\exp[-\beta(V(q)-E_*)]
\;I_{\cal D}\,I_h.
\]

The [review](../README.md) supplies the analytic \(G_\alpha\) and root
derivative. A missing root, inactive branch, excluded domain or invalid chart
gives exactly zero weight **without replacement**. A PES exception or
nonfinite energy is a failed pilot with the offending geometry retained; it
is not a zero-weight observation. The adapter may compute forces internally,
but only its energy defines the statistical weight.

For the entry surface the weight is

\[
w_0=\frac{J(z)L^2}{q_{z,0}(z)q_{u,0}(u)\sqrt{2\pi\beta\mu}}
\exp[-\beta(V(q)-E_*)]I_{\cal D}.
\]

The expectation of each channel weight is
\(e^{\beta E_*}N_{h,\alpha}/(8\pi^2)\); the entry expectation is
\(e^{\beta E_*}N_0/(8\pi^2)\). Those common factors cancel. Each of the four
channel integrals is sampled explicitly. The estimator sums their four sample
means and divides by the entry mean. **Do not multiply this sum by four**.
This construction does not require a numerical assumption that the four
channel PES values are exactly equal.

The finite-domain ratio corresponds formally to
\(F_\infty N_\alpha/N_0\). The pilot report should expose cost, zero-weight
reasons, log-integral/weight diagnostics and paired surface differences;
it should omit a converted physical-rate claim and any paper-agreement gate.
It must not use the CPC target numbers to select proposals, bounds or budget.

## Fixed ledger, cost ceiling and uncertainty scope

Eight excluded diagnostic seeds are 2026111201 through 2026111208. Within each
seed, draw eight transition proposals for each of the four channel charts.
Evaluate both α values on the same latent shape/direction, for at most 64 PES
geometries. Draw 32 separate entry proposals, for at most 32 more. Thus the
full fixed pilot has 256 candidate evaluations for each of its three targets
and a strict maximum of **768 evaluated P1 geometries**. Invalid target points
reduce actual PES calls but never the denominator of a sample mean.

Before the first PES evaluation, materialize and hash the full latent-draw
ledger, all reconstructed candidate geometries and all geometric zero-weight
reasons. Bind the evaluator/model/library, proposal and source hashes. All
proposal work uses existing geometry and performs no hidden PES calls. Count
evaluated geometries, not merely Python batch invocations. Preserve attempted
calls and their CPU durations even if they fail.

Run sequentially on one CPU with all BLAS/OpenMP threads set to one. Check a
55-process-CPU-second admission stop before every new PES call, with a
60-process-CPU-second hard ceiling and a 120-second wall supervisor retaining
the partial ledger if execution stalls. Any
ceiling reached produces an **incomplete** report; do not replace seeds,
fill the budget selectively, or label partial completion as success. No
original runtime, binary, source, protocol, or running campaign is modified.

For complete output, record actual CPU seconds per evaluated geometry,
maximum normalized weight, empirical weight ESS, mixture/channel/mirror
coverage, all zero-weight reasons, and the eight independent seed summaries.
TS surface estimates are paired because they share latent draws. The entry
estimate and both TS estimates within each seed must stay together in any
ratio/paired uncertainty calculation. Individual importance points, channel
labels and the paired α evaluations are not independent study replications.

With only eight draws per channel per seed, these statistics diagnose gross
inefficiency and cost; they are not a trustworthy precision certificate.
No observed maximum weight bounds unseen tails. An all-zero denominator or
numerator is reported as unresolved, without epsilon substitution. A later
fixed-budget refinement or bridge design must be chosen from efficiency,
domain and support diagnostics, with all pilot outputs excluded from its
primary result.

## Required review before any execution

The independent reviewer should verify the rotation-chart determinant,
full-mixture densities and finite-domain support; the positive-root and
active-channel restriction; the common finite-R∞ entry normalization;
the different α surfaces and paired-draw accounting; and the strict 768-call
ledger. The existing geometry-only proof is useful but does not validate
this proposed sampler or its finite molecular domain. Implemented code would
need separate bounded synthetic tests and source freeze before approval to
evaluate the first PES geometry.
