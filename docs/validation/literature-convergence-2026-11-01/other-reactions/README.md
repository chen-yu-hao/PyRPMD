# Other reaction benchmarks: new source inventory and correction

This is an additive source audit made during the H₃ convergence work. It records
newly located primary sources and one small H + CH₄ electronic-PES validation.
It does not supersede the saved H₃ numerical campaigns or turn missing thermal
rates into successful reproductions. Machine-readable details are in
[source-inventory.json](source-inventory.json).

## H + CH₄: correct the previous PES distinction

The [earlier source audit](../../literature-audit-2026-11-01/h3/other-reaction-references.md)
described the CPC example as a modified Jordan–Gilbert surface and then treated
the later PES-2008 as a different model. **That distinction was not justified.**
Section 4 and reference 49 of the
[CPC primary manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics)
identify Corchado, Bravo and Espinosa-García, JCP 130, 184314 (2009),
[DOI 10.1063/1.3132223](https://doi.org/10.1063/1.3132223).
The [official RPMDrate source](https://github.com/GreenGroup/RPMDrate/tree/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/examples/H%2BCH4)
contains `ch52008.f` with that same reference. The modified Jordan–Gilbert
description refers to the functional form of CBE/PES-2008, not evidence for a
different historical surface.

The pinned directory also contains `utility.f`, the f2py signature, Makefile
and input geometry, including all four equivalent abstraction arrangements.
The source/interface metadata are in
[the benchmark manifest](../../../../benchmarks/hch4/source.json).
A new C-compatible wrapper builds the two unchanged Fortran files without
requiring legacy f2py. A bounded real-source check gives a Table III barrier
of **15.0142843 kcal/mol**, compared with the primary paper's **15.01 kcal/mol**.
Its force checks explicitly retain the upstream 2.06223 ppm energy/gradient
rounding inconsistency and 0.0840463% legacy energy-unit scale difference.
See [the full electronic validation](hch4/README.md).

This removes the missing-source obstacle for a matched future H + CH₄
calculation. It does not supply equilibrated six-atom ensembles, a transmission
coefficient, isotope convergence or the published thermal rates. The old
CPC table remains a reference, not a measured result in this project.

## F + H₂: distinct public author distributions

Werner's [official potential page](https://www.itheoc.uni-stuttgart.de/research/werner/research/h2f/potential/)
provides separate **SW1 without spin–orbit correction** and **HSW with
spin–orbit correction**, plus later LWA models. The direct archive links are
recorded in the inventory. The SW primary paper is Stark and Werner,
[DOI 10.1063/1.471372](https://doi.org/10.1063/1.471372).
The archives were located but not downloaded, compiled or hash-pinned in this
audit. No explicit archive license or force-interface contract was verified.

The [author publication list](https://manolopoulos.chem.ox.ac.uk/publications.html)
links both the 2009 RPMD paper, DOI 10.1063/1.3127145, and its 2010 scientific
erratum, DOI 10.1063/1.3464477. The corrected numerical table was still not
retrieved. The separate 2009 publisher note is not that scientific correction.
Before comparing a rate, a case must fix the exact SW1/HSW variant, spin–orbit
treatment, electronic population factor, temperature and isotope masses.
The [2015 primary TRPMD paper](https://www.tjhh.uk/pubs/hel15trpmd.pdf)
provides another explicitly identified SW calculation, but nonzero-friction
results cannot be treated as ordinary RPMD and a plotted rate needs audited
digitization uncertainty if no numerical table is available.

## OH + H₂: the missing asset is model-specific

The NN1 model is Chen, Xu, Xu and Zhang,
[DOI 10.1063/1.4801658](https://doi.org/10.1063/1.4801658).
Its exact network weights and matching evaluation source were not retrieved.
The [official MCTDH Hamiltonian documentation](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/doc.86/mctdh/ham_docu.html)
does identify both YZCL1 and YZCL2 and an `-i yzcl` integration option. It places
external PES code in a supplementary `addsurf` distribution; the linked package
endpoint timed out and the linked `funcsrf.F` was missing during this audit.
That is a concrete acquisition route, not a completed build or a claim that
the potential is unavailable everywhere.

As a guard against using an easy-to-download substitute, the
[ChemPotPy OH3 directory](https://github.com/shuyinan/chempotpy/tree/4716d22e87ae01770cad9ec995c4a02a8e479296/chempotpy/OH3)
was inspected: it lists the older SE ground-state model and the FFW1/FFW2
multistate reactive-quenching models, not NN1 or YZCL2. Its FH2 directory likewise
does not list SW1/HSW. A matching chemical formula does not establish matching
potential parameters or electronic-state treatment.

The existing 2017/2018 OH + H₂ reference discrepancy and the distinction between
uncorrected single-surface rates and the OH electronic statistical correction
remain unresolved exactly as recorded in the earlier audit. No corrected
reference value is invented here.
