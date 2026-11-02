# H + CH₄ classical flux normalization

**Conclusion:** the known entry prefactor **2.0492225575240845 × 10⁻⁸ cm³ molecule⁻¹ s⁻¹** is correct for the existing finite-entry, flux-weighted numerator/denominator convention at 300 K and \(L=30\) bohr. Its role is to restore the entry-sphere flux scale to a dimensionless ratio. It is not a fitted barrier factor. No additional factor four, factor one-half, mass determinant, rotor symmetry factor or product-H₂ factorial belongs in that ratio.

This conclusion concerns normalization algebra. The two excluded importance pilots do not establish an absolute physical rate: their precision, unseen tails, provisional molecular domains and finite-separation errors remain unresolved. This note reads no pilot observations, computes no pilot rate, changes no scientific source, and authorizes no sampling.

The derivation and calculator do not call the producer's normalization functions. Their author also implemented the pilot helpers, so this independence is mathematical/calculational, not independent authorship; a separate author's read-review is requested before adopting a new numerical reference design.

## 1. Conventions checked against the primary method

The author-uploaded [CPC manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics) defines the interpolating coordinate in Eq. 10, the bimolecular prefactor in Eq. 12, and the flux-weighted umbrella measure in Eqs. 15–16. Its Table 1 gives four H-abstraction channels and \(L=30\) bohr. Table 2 uses \(\xi^\ddagger=0.999\) for P1 and 1.007 for P128. The [MIT record](https://dspace.mit.edu/entities/publication/a4555c62-26a0-4500-8953-5dd1905583cc) identifies the original manuscript and the published DOI, [10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017). Its PDF endpoint returned HTTP 429 during this review.

The mathematical derivation below is independent of the paper's target rate values. It audits the actual [original surface helper](../../../../../benchmarks/hch4/reference_pilot.py), [efficiency helper](../../../../../benchmarks/hch4/efficiency_pilot.py), [frozen constants/proposal](../hch4-reference-feasibility/pilot-design/proposal.json), and [reproduction conventions](../../../../../benchmarks/hch4/reproduction_protocol.json).

The atomic labels are H,C,H,H,H,H: carbon is 1; incoming H is 5; bound H labels are \(h\in\{0,2,3,4\}\). Let their common H mass be \(m\), carbon mass be \(M\), methane mass be \(M_B=M+4m\), and relative mass be

\[
\mu=\frac{mM_B}{m+M_B}.
\]

The declared isotope masses are H = 1.00782503207 u and C = 12 u. The surface uses

\[
s_0=L-|q_5-R_B|,\quad
s_h=c+|q_h-q_C|-|q_5-q_h|,\quad
s_1=\max_h s_h,\quad
\xi=\frac{s_0}{s_0-s_1}.
\]

Here \(c=r_f^\ddagger-r_b^\ddagger\) comes from the exact pinned input geometry. A single-bead evaluation at 1.007 remains a classical calculation on a different surface; the coordinate value alone does not make it a P128 ensemble.

## 2. Start from a positive classical flux

For a translation-invariant dividing function \(S(q)\), use the Cartesian Hamiltonian

\[
H(p,q)=\sum_i\frac{|p_i|^2}{2m_i}+V(q),\qquad
\dot S=\sum_i\nabla_i S\cdot\frac{p_i}{m_i}.
\]

Under normalized canonical Maxwell momenta, \(\dot S\) is a zero-mean scalar Gaussian with variance

\[
\sigma_S^2=\frac{G_S}{\beta},\qquad
G_S=\sum_i\frac{|\nabla_i S|^2}{m_i}.
\]

Therefore its **unconditional positive flux**, including the probability of the positive half, is

\[
\langle \dot S\,\Theta(\dot S)\rangle_p
=\frac{\sigma_S}{\sqrt{2\pi}}
=\sqrt{\frac{G_S}{2\pi\beta}} \equiv f_S.
\]

It is neither the positive-half conditional mean nor the mean three-dimensional speed. Substituting either without its corresponding probability/geometric factor creates a factor-two or factor-four error.

To see the absolute units without choosing internal momenta, consider one A atom and one B molecule in a large volume \(\mathcal V\). With a consistent labeled reactant basin, their separated configurational partition scales as \(\mathcal V^2 Z_B\), while the surface flux scales as \(\mathcal V N_S\). The same six-atom Cartesian momentum Gaussian and the same phase-space normalization multiply both. Multiplying the single-pair crossing frequency by \(\mathcal V\) to form a bimolecular rate gives

\[
k_{\rm TST}(S)=\frac{N_S}{Z_B},\qquad
N_S=\int d^{12}b\,d^3x\;
I_{\mathcal D}\delta(S)\,f_S\,e^{-\beta V}.
\]

Here \(b_h=q_h-q_C\), \(x=q_5-q_C\), and \(Z_B\) includes a specified bound-methane basin and the matching separated-atom energy convention. A full-space integral over all dissociated methane configurations is not a finite molecular partition function. The domain and energy reference are substantive physical inputs, not multiplicative symmetry corrections.

## 3. Translation and carbon's kinetic contribution

The map from all positions to \((q_C,b_0,b_2,b_3,b_4,x)\) has unit Jacobian. Integrating \(q_C\) removes one translation volume. Replacing \(q_C\) by the total center of mass is a shape-dependent shift with another unit Jacobian. Neither operation introduces an extra mass determinant.

Putting carbon at the coordinate origin **does not immobilize carbon physically**. Its contribution to \(G_S\) remains. Integrating physical Maxwell momenta before removing coordinate translation makes this distinction explicit.

Even conditioning the Gaussian velocities on zero total linear momentum leaves a translation-invariant flux variance unchanged:

\[
G_{S\,|\,P_{\rm tot}=0}
=G_S-\frac{\left|\sum_i\nabla_i S\right|^2}{M_{\rm tot}}
=G_S.
\]

The final equality uses \(\sum_i\nabla_iS=0\). The common center-of-mass momentum normalization cancels if that convention is applied consistently to numerator and denominator. Removing overall coordinate rotation later likewise does not impose zero angular momentum. Rotational kinetic degrees of freedom have already been integrated in the Maxwell calculation.

## 4. Interpolating-coordinate coarea and the complete mass metric

For the requested coordinate value \(\alpha>0\), define

\[
D=s_0-s_1,\qquad S_\alpha=(1-\alpha)s_0+\alpha s_1.
\]

Since \(\xi-\alpha=S_\alpha/D\), on the nonsingular surface

\[
\delta(\xi-\alpha)=|D|\delta(S_\alpha),\qquad
f_\xi=\frac{f_{S_\alpha}}{|D|},\qquad
\boxed{\delta(\xi-\alpha)f_\xi=\delta(S_\alpha)f_{S_\alpha}}.
\]

The delta-function Jacobian and the flux-metric factor cancel exactly. This identity is on the surface; replacing the two gradients indiscriminately away from it is not valid. The existing target keeps the reactant-connected component \(D>0\) and excludes its pole.

For one candidate active H, let \(u_b\) point from C to that H, \(u\) from that H to incoming H, and \(u_R\) from methane COM to incoming H. The nonzero branch gradients of \(s_h\) are \(-u\) on incoming H, \(u_b+u\) on active H and \(-u_b\) on C. The entry gradients are \(-u_R\) on incoming H and \((m_i/M_B)u_R\) on each methane atom. Direct summation gives

\[
G_{00}=\frac1\mu,\qquad
G_{11}=\frac{3+2u_b\cdot u}{m}+\frac1M,\qquad
G_{01}=\frac{u_R\cdot u}{\mu}.
\]

In the cross term, the carbon and active-H terms involving \(u_R\cdot u_b\) cancel. Thus

\[
\boxed{
G_\alpha=\frac{(1-\alpha)^2}{\mu}
+\alpha^2\!\left[\frac{3+2u_b\cdot u}{m}+\frac1M\right]
+\frac{2\alpha(1-\alpha)}{\mu}\,u_R\cdot u.}
\]

This is the implemented expression. The \(1/M\) term survives fixing the carbon coordinate origin. The sign of \(1-\alpha\) is retained for \(\alpha=1.007\). Replacing this metric by \(1/\mu\) at the transition surface is incorrect.

Write incoming position as \(x=b_h+\rho u\). With \(d=(m/M_B)\sum_h b_h\), the branch root satisfies

\[
S_h(\rho)=(1-\alpha)\,[L-|b_h-d+\rho u|]
+\alpha(c+|b_h|-\rho)=0,
\]

\[
\partial_\rho S_h=-\alpha-(1-\alpha)\,u\cdot u_R.
\]

For both declared alpha values the derivative is strictly negative. A positive root exists exactly when \(S_h(0)>0\). With \(I_h\) selecting the active maximum, radial integration gives

\[
N_\alpha=
\sum_h\int d^{12}b\,d\Omega_u\;
I_{\mathcal D}I_h\,
\frac{\rho_h^2}{|\partial_\rho S_h|}
\sqrt{\frac{G_\alpha}{2\pi\beta}}\,
e^{-\beta V(b,b_h+\rho_hu)}.
\]

Nonexistent roots, inactive channels and excluded-domain draws contribute zero to the unconditional integral. Renormalizing only accepted draws changes the target.

## 5. Overall rotation and four-channel counting

For each channel's fixed label chart, put its chosen H at \(r e_z\), its selected spectator at \(s(\sqrt{1-v^2},0,v)\), and leave the other two vectors \(a,b\) free. Away from zero-measure chart singularities,

\[
d^{12}b=8\pi^2\,r^2s^2\,dr\,ds\,dv\,d^3a\,d^3b.
\]

The same simultaneous SO(3) volume \(8\pi^2\) appears in every numerator chart and the one entry chart, so it cancels. The shape factor \(r^2s^2\) does not. The incoming direction sphere is a separate relative orientation; its solid angle must not be confused with the removed overall rotation.

The active-max indicators partition the dividing surface. The existing helper sums four separate channel integrals, each retaining its indicator. This already counts the four equivalent abstractions. Summing the four chart means and then multiplying by four would overcount; replacing their sum by a pooled mean without its chart-count correction would undercount. A factor-four shortcut from one chart would require a separate exact permutation/domain proof.

Consistently labeling methane does not omit its physical symmetry factor. Applying a \(1/4!\) indistinguishability factor to the same labeled reactant and flux integrals cancels it. If all five possible free-H label basins were instead summed with a \(1/5!\) factor, the same bookkeeping is recovered. Mixing those conventions on opposite sides is invalid.

A molecular rotor treatment may use methane's symmetry number, but the present calculation integrates Cartesian labeled coordinates and their full rotational volume. One cannot append a rotor symmetry factor to only one side. The two mirror sectors that remain after proper-rotation removal must have support; they are not a new factor to attach after integrating them.

There is also **no extra H₂ product \(1/2!\)**. A forward rate counts a reaction crossing from the specified H + CH₄ reactant ensemble, not a product partition function. The two H atoms in the product do not create a second copy of that same phase-space crossing. Product partition symmetry factors belong in a consistently defined reverse-rate/equilibrium calculation; appending one here changes the forward flux. The configured electronic multiplier remains one, separately from all nuclear-label and channel bookkeeping.

## 6. The finite-entry denominator and the prefactor

At the entry surface, \(S_0=L-R\) and \(G_0=1/\mu\). Therefore

\[
N_0(L)=\frac{L^2}{\sqrt{2\pi\beta\mu}}
\int d^{12}b\,d\Omega\;
I_0\,e^{-\beta V(b,d+Lu)}.
\]

Define the orientation-averaged finite-entry partition

\[
Z_B^{(L)}=\frac1{4\pi}\int d^{12}b\,d\Omega\;
I_0e^{-\beta V(b,d+Lu)}.
\]

Then exactly, for this declared finite-entry domain,

\[
\boxed{
N_0(L)=F_L Z_B^{(L)},\quad
F_L=\frac{4\pi L^2}{\sqrt{2\pi\beta\mu}},\quad
k_L(\alpha)=F_L\frac{N_\alpha}{N_0(L)}
=\frac{N_\alpha}{Z_B^{(L)}}.}
\]

The code's entry proposal is normalized per solid angle. Dividing a draw's integrand by that proposal integrates over \(d\Omega\); it does not calculate an angular average. In particular, the uniform entry component has density \(1/(4\pi)\), so that inverse density already restores the full sphere inside \(N_0\). The prefactor above is correct for this integral convention. If the denominator were redefined as an angular average instead, its corresponding prefactor would have to change.

The PMF obtained from the declared flux-weighted measure obeys
\(\exp[-\beta(W(\alpha)-W(0))]=N_\alpha/N_0\).
This is the connection to the paper's expression and to the production rate workflow. An ordinary configurational PMF without the \(f_\xi\) weight is a different measure; this prefactor alone cannot repair it.

When the sixth atom decouples uniformly over the relevant bound-methane configurations, \(Z_B^{(L)}\) approaches the bound reactant partition with the matching separated-atom energy. Only then does \(k_L\) become the asymptotic physical TST rate for the fixed surface and molecular domain. Agreement of a few entry configurations is not a bound on missing tails.

For a separation control, compare \(N_0(L)/L^2\), or equivalently \(Z_B^{(L)}\). Replacing an entry denominator at 30 bohr by one at 60 bohr while keeping the 30-bohr prefactor would introduce an erroneous factor four. One may keep the transition surface fixed at its original \(L\) while using a separately checked remote entry partition. If instead \(L\) is changed inside \(S_\alpha\) while holding \(\alpha\ne1\) fixed, the physical transition surface also moves; a static TST change is then not purely a reactant-normalization effect.

## 7. Units and deterministic cross-checks

Using the declared current constants,

| Quantity | Value |
|---|---:|
| \(\mu\), electron masses | 1728.4892968817187 |
| \(\beta\), hartree⁻¹ | 1052.5834160132904 |
| \(F_L\), bohr³ per atomic time | 3.3450362638130775 |
| \(F_L\), cm³ molecule⁻¹ s⁻¹ | \(2.0492225575240845\times10^{-8}\) |

Since \(a_0^3/t_{\rm au}\) converts to cm³/s, the atomic-unit conversion is simply

\[
{\cal C}_{\rm rate}=\frac{a_0^3}{t_{\rm au}}\,10^6.
\]

No Avogadro factor appears for the number-density convention cm³ molecule⁻¹ s⁻¹. Converting that coefficient to cm³ mol⁻¹ s⁻¹ would multiply by \(N_A\). The source distribution's console and saved-file unit conventions must not be conflated.

Dimensions provide a separate check: \(N_\alpha\) and \(N_0\) have units length¹⁵/time; \(Z_B^{(L)}\) has length¹²; their quotient is length³/time. A common energy offset multiplies both surface integrals by the same exponential and cancels. An offset used only on one side would not.

The [deterministic check program](check_normalization.py) recomputes the known prefactor through atomic units and independently through SI. It also integrates normalized one-dimensional Gaussian flux and the Maxwell speed law:

\[
4\pi L^2\langle (-v_R)_+\rangle
=\pi L^2\langle |v_{\rm rel}|\rangle.
\]

This is the absorbing-sphere collision-rate identity and fixes the one-sided-flux and sphere-area factors without a molecular potential. Twelve unit-vector algebra cases independently compare the closed-form mass metric with full Cartesian gradient summation, including the total-COM conditioning cancellation. All checks pass; [the receipt](algebra-checks.json) records constants, source identities and limits. No molecular geometry was evaluated, and no pilot observations or rate estimates were read.

The pinned legacy distribution uses slightly different unit constants and a converted input separation approximately 30.000158 bohr; the current proposal explicitly adopts the table's 30 bohr and current constants. Those choices are recorded, not corrected by a fitted scale. The many digits above establish internal consistency of the current convention, not many-digit reproduction of an old simulation.

## 8. What remains before an absolute physical-rate claim

The remaining work is substantive: justify the bound-methane/reactive domains, demonstrate adequate support and tail control, establish finite-entry convergence with the correct sphere normalization, and obtain sufficiently precise complete independent estimates. The excluded pilots remain excluded. A full reaction rate additionally requires a matched-surface transmission coefficient; a classical calculation at 0.999 cannot be combined with a quantum correction or transmission coefficient at 1.007 without the appropriate separately established surface relation.

There is no normalization-based reason to alter the existing prefactor, append a channel/product-symmetry factor, or rescale the PES. This note is an algebra and source-convention review, not a numerical validation of the H + CH₄ rate.
