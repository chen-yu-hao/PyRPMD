# Other requested reaction benchmarks: source audit

This document records primary-paper checks made on 2026-11-01. These are
reference-data checks, not newly simulated H + CH₄, OH + H₂, F + H₂ or
isotope rate constants. Exact PES assets, electronic-state conventions and
independent convergence campaigns remain prerequisites for those comparisons.
The machine-readable reference catalog remains
`benchmarks/published_rates/references.json`.

## H + CH₄

[Suleimanov, Allen and Green, CPC 184, 833–840 (2013)](https://doi.org/10.1016/j.cpc.2012.10.017),
[author manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics),
Table 2, gives the following 300 K results. All rates use cm³ molecule⁻¹ s⁻¹.

| Treatment | TST/QTST | κ | Final rate |
|---|---:|---:|---:|
| Classical | 6.53 × 10⁻²² | 0.762 | 4.98 × 10⁻²² |
| RPMD, 128 beads | 1.71 × 10⁻¹⁹ | 0.659 | 1.13 × 10⁻¹⁹ |

The four abstraction arrangements are already included. The potential is the
historical Espinosa-García modified/recalibrated Jordan–Gilbert surface cited
there, not an arbitrary methane force field. The later
[2015 TRPMD paper](https://www.tjhh.uk/pubs/hel15trpmd.pdf),
DOI 10.1063/1.4928599, instead uses PES-2008. Those surfaces cannot be
interchanged while treating the difference as an implementation error.

The isotope study is [DOI 10.1063/1.4793394](https://doi.org/10.1063/1.4793394),
with an [MIT author-manuscript record](https://dspace.mit.edu/entities/publication/2ccd972a-f59c-4564-8ecc-90680e7e0b16).
Its complete numerical tables were not retrieved in this audit. No new
numbers were guessed from its abstract. Exact historical PES parameters and
permission to use the source remain missing for a new matched simulation.

## OH + H₂

[Novikov, Suleimanov and Shapeev, DOI 10.1039/C8CP06037A](https://doi.org/10.1039/C8CP06037A)
has an accessible [primary manuscript, arXiv:1805.11924v3](https://arxiv.org/pdf/1805.11924).
Table 1 separates the NN1 reference PES from the learned MTP. Its **NN1**
columns give:

| Temperature | Beads | TST/QTST | κ | Final rate |
|---|---:|---:|---:|---:|
| 300 K | 1 | 5.74 × 10⁻¹⁶ | 0.613 | 3.52 × 10⁻¹⁶ |
| 300 K | 128 | 2.37 × 10⁻¹⁴ | 0.528 | 1.25 × 10⁻¹⁴ |
| 1000 K | 1 | 2.78 × 10⁻¹² | 0.666 | 1.85 × 10⁻¹² |
| 1000 K | 16 | 3.72 × 10⁻¹² | 0.599 | 2.23 × 10⁻¹² |

These are the uncorrected single-surface values used by the existing catalog.
They must not be mixed with the electronic statistical correction in
[DOI 10.1039/C7CP05266A](https://doi.org/10.1039/C7CP05266A),
[primary author manuscript](https://www.researchgate.net/publication/320300566_A_ring_polymer_molecular_dynamics_study_of_the_OH_H_2_D_2_reaction).
That paper's Eq. 3 uses the 140 cm⁻¹ OH spin–orbit splitting. Its Table 3
corrected 300 K values are 8.28 × 10⁻¹⁵ on NN1 and 7.22 × 10⁻¹⁵ on YZCL2.
Applying the electronic correction a second time would introduce an error.

An unresolved source discrepancy is retained explicitly: the retrieved 2017
manuscript's Table 2 labels a 1000 K NN1 RPMD rate of 3.75 × 10⁻¹², whereas
the 2018 Table 1 gives 2.23 × 10⁻¹² with QTST 3.72 × 10⁻¹² and κ = 0.599.
The latter product is internally consistent. This audit does not silently
correct the older table or attribute the difference to a typo; a comparison
must identify the chosen source/version. No new OH + H₂ trajectory is
claimed because the exact NN1/YZCL2 model assets were not available here.

## F + H₂

The original rate paper is [DOI 10.1063/1.3127145](https://doi.org/10.1063/1.3127145).
Its [Oxford record](https://www.chem.ox.ac.uk/publication/34713/pubmed)
confirms the reaction scope. A scientific erratum exists at
[DOI 10.1063/1.3464477](https://doi.org/10.1063/1.3464477); the complete
corrected numerical table was not retrieved. The separate 2009 publisher's
note, DOI 10.1063/1.3147536, concerns journal-section placement and must not be
confused with that 2010 erratum.

The [2015 primary PDF](https://www.tjhh.uk/pubs/hel15trpmd.pdf),
DOI 10.1063/1.4928599, explicitly identifies the Stark–Werner PES. Table II
uses 384 beads at 200 K, 256 at 300 K and 64 at 500 K, R∞ = 30 bohr and two
product arrangements. The zero-friction data are the RPMD limit of that
TRPMD comparison. The other friction values are different dynamics. An
audited numerical table or digitization with uncertainty, the exact SW
implementation and convention checks are still needed; no F + H₂ rate has
been invented or marked reproduced.

## Mu/H/D + H₂

[Suleimanov et al., PCCP 15, 3655–3665 (2013)](https://pubs.rsc.org/en/content/articlelanding/2013/cp/c2cp44364c),
DOI 10.1039/C2CP44364C, is the primary isotopologue study over 200–1000 K.
It covers H, D, Mu and Heμ + H₂ and compares RPMD with quantum dynamics and
other methods. The new H₃ campaign uses the ordinary-H mass only. This audit
did not obtain a full numerical isotope table, so D/Mu reference records
remain explicitly unavailable. The earlier Mu + H₂ paper is
[DOI 10.1021/jz201702q](https://doi.org/10.1021/jz201702q).

The 2015 TRPMD test **D + MuH** is a different reaction from **Mu + H₂**;
its table cannot stand in for the requested isotope benchmark. Changing
masses in a general driver is not evidence that the corresponding isotope
reactant partition function, bead count, sampling and rate have converged.

## What would make these comparisons executable

Each case needs a versioned PES with a confirmed use/redistribution status,
independent energy/force test geometries, exact isotope masses and electronic
weights, then a committed physical input and a documented compute budget.
The same independent-seed, temporal-stationarity, bead, timestep, PMF and
transmission-horizon checks used for H₃ should be applied before publishing
agreement. A missing model or table is recorded as missing, not converted to
a passing or failing numerical comparison.
