# H + H2 umbrella normalization and pilot diagnosis

The CPC 2013 H + H2 umbrella coefficient is **30 hartree at 300 K in
PyRPMD's input convention**. Dividing it by the bead count would change the
physical umbrella. This conclusion follows from the original RPMDrate code,
independently of the observed pilot rates. No production force or PMF formula
was changed during this check.

## Primary source audit

The inspected original repository is `GreenGroup/RPMDrate`, pinned to commit
`8c1828d2302aacf09a15faf5c7fc50a18d4c113e`. The following links preserve the
reviewed source version:

| Source | Relevant implementation |
| --- | --- |
| [`rpmdrate/_main.f90`, `add_umbrella_potential`, lines 688–716](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/rpmdrate/_main.f90#L688-L716) | Adds one half of `kforce` times the squared displacement to **each** bead's potential energy; adds `kforce * delta * dxi` to each bead's gradient. |
| [`rpmdrate/_main.f90`, `get_reaction_coordinate`, lines 783–840](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/rpmdrate/_main.f90#L783-L840) | Computes `dxi` with respect to the centroid coordinates. The umbrella reaction coordinate is `s0 / (s0 - s1)`. |
| [`rpmdrate/_main.f90`, `sample_momentum`, lines 911–932](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/rpmdrate/_main.f90#L911-L932) | Uses inverse temperature `beta / Nbeads` for bead momenta. |
| [`rpmdrate/main.py`, lines 677–705](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/rpmdrate/main.py#L677-L705) | Gaussian umbrella integration subtracts `kforce * (xi - xi_window)` from the physical-temperature density derivative. It does not divide `kforce` by the bead count. |
| [`examples/H+H2/input.py`, lines 32–65](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/examples/H%2BH2/input.py#L32-L65) | Selects `kforce = 0.1 * T`, a 0.1 fs timestep, GLE thermostat, 5 ps initial configuration generation, and 200 sampling segments per window with 20 ps equilibration and 100 ps production per segment. |

The corresponding publication is Y. V. Suleimanov, J. W. Allen, and W. H.
Green, *RPMDrate: Bimolecular chemical reaction rates from ring polymer
molecular dynamics*, Computer Physics Communications **184**, 833–840 (2013),
[DOI 10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).
Tables 1 and 2 contain the calculation parameters and the rate reference,
respectively. The source registry records the inspected table values and
their retrieval locations.

## Energy, probability, and force

Write the centroid as $\bar q=P^{-1}\sum_j q_j$, the physical inverse
temperature as $\beta=(k_BT)^{-1}$, and the physical umbrella as

$$
W_i(\bar q)=\tfrac12 k_i[\xi(\bar q)-\xi_i]^2.
$$

Both programs use the summed bead Hamiltonian with sampling weight
$\exp(-\beta H_P/P)$. The original loop adds $W_i$ to each bead energy,
so the full Hamiltonian receives

$$
\Delta H_P=P W_i.
$$

Consequently the umbrella factor in the equilibrium distribution is
$\exp(-\beta W_i)$. Its force on each bead is

$$
-\frac{\partial(P W_i)}{\partial q_j}
=-k_i(\xi-\xi_i)\frac{\partial\xi}{\partial\bar q}.
$$

The factor of $P$ in the energy cancels the factor $1/P$ from differentiating
the centroid. RPMDrate stores the positive energy gradient, whereas PyRPMD
returns its negative as the force. `src/pyrpmd/bias.py::UmbrellaBias` implements
exactly this energy and chain rule; its CV Jacobian is with respect to bead
coordinates. The matching Gaussian integration equation is

$$
\frac{dA}{d\xi}
=\sum_i w_i(\xi)\left[
\frac{\xi-\langle\xi\rangle_i}{\beta\,\operatorname{var}_i(\xi)}
-k_i(\xi-\xi_i)\right],
$$

where $w_i$ are normalized sample-count-weighted Gaussian window densities.
This is the expression in both the original source and
`src/pyrpmd/analysis/fes.py::umbrella_integration`.

At 300 K, `0.1 * T` is 30 hartree. With 128 beads the coefficient of
$\tfrac12(\xi-\xi_i)^2$ in the **total** Hamiltonian is 3840 hartree; the input
coefficient remains 30 hartree. The flat-free-energy harmonic width
$\sigma_\xi=\sqrt{k_BT/k_i}$ is about 0.00563. The exploratory umbrella
`100*kBT`, approximately 0.0950 hartree, instead has width 0.1. Actual widths
also depend on the underlying free energy, but these numbers explain why the
two coefficient choices have very different sampling behavior. Neither a
bead-count division nor choosing the coefficient that gives a favorable rate
is a valid normalization fix.

## What the existing short trajectories show

The two 300 K, 128-bead, 30-bohr pilots used two replicas per window, each
with 50 fs equilibration and 100 fs production. They also used Andersen
thermostatting and Cayley propagation, whereas the historical example uses
GLE and the original normal-mode Verlet implementation. These are independent
numerical and sampling protocols with matching target physical parameters;
they do not reproduce the full historical protocol. The public input's
sampling budget is 400 times longer per equilibration segment, 1000 times
longer per production segment, and has 100 times as many segments per window
as these two-replica pilots. A different sampling method need not require
exactly that budget, but equilibration and convergence must be demonstrated.

The compact calculation outputs and this diagnostic are retained under
`docs/validation/tier1-2026-10-31/`. The diagnostic
`pmf-sampling-diagnostic.json` identifies the original checkpoint streams by
path and by a digest of the ordered filenames and file hashes. Recomputing
Gaussian umbrella integration from all samples reproduces both saved PMF
barriers exactly. Restricting the same records to temporal halves or to each
replica exposes instability:

| Diagnostic | Exploratory 0.0950-hartree bias | Source-matched 30-hartree bias |
| --- | ---: | ---: |
| Saved full-data PMF barrier / hartree | 0.0107258 | 0.0260140 |
| Replica 0 barrier / hartree | 0.0109510 | 0.0230143 |
| Replica 1 barrier / hartree | 0.0114112 | 0.0305352 |
| First-half barrier at original dividing surface / hartree | 0.0210437 | 0.0273160 |
| Second-half barrier at original dividing surface / hartree | 0.0176483 | 0.0255875 |
| Mean estimated independent samples per stream | 4.78 | 10.54 |
| Streams with fewer than 20 estimated independent samples | 222 of 222 | 213 of 222 |

Each replica has its own maximum in the replica rows. The temporal-half rows
hold the original dividing surface fixed. These diagnostics are correlated
subsamples of short trajectories, not new uncertainty estimates or
convergence tests. In particular, the first/second halves must not be chosen
afterward to improve agreement. Gaussian integration is nonlinear in the
sampled means and variances, so a pooled estimate need not fall between
estimates from the halves when the distributions change or are poorly
resolved.

The source-matched bias's replica barriers differ by approximately 7.9
$k_BT$, which alone changes a rate exponentially. The pilot transmission
coefficients, approximately 0.576 and 0.617, vary much less than the QTST
factors. Thus the observed disagreement is dominated by the PMF estimates.
Using the pilots' common reactant prefactor, the published QTST value
$2.51\times10^{-16}$ cm³ molecule⁻¹ s⁻¹ corresponds to a barrier of about
0.0174714 hartree. This number is **derived with that prefactor**, not a
separately published barrier value.

A subsequent [same-sample UI/MBAR diagnostic](../../docs/validation/tier1-2026-10-31/pmf-estimator-diagnostic.json)
holds the original dividing surface fixed and gives a source-matched-bias MBAR
barrier of 0.0259357 hartree, compared with 0.0260140 hartree from UI. This
changes the QTST factor by about 8.6%, far less than the discrepancy from the
published value. Both estimators use the same short trajectories, so their
agreement does not establish equilibrium sampling. The broad-bias pilot lacks
supported MBAR endpoints and therefore has no admissible MBAR rate comparison.

The data establish that these pilots cannot validate the published rate.
They do not establish that under-sampling is the only possible source of
disagreement. The source audit rules out the proposed factor-of-128 umbrella
normalization change. The next discriminating work is to equilibrate the
internal and orientational degrees of freedom, extend independently seeded
window sampling, inspect stationarity and non-Gaussian distributions, repeat
the UI/MBAR comparison with those longer samples, and then complete the timestep,
bead-count, reactant-distance, and recrossing convergence checks. Changing
production mathematics to make either short pilot fit the table is not
supported by this evidence.
