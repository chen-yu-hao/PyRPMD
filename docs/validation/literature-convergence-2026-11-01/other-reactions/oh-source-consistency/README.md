# OH + H₂: source consistency and exact-model availability

The 300 K uncorrected NN1 reference is consistent across the two retrieved
author manuscripts. Their printed 1000 K final-rate entries disagree. No
reference number has been changed to resolve this discrepancy, and no
OH + H₂ trajectory or numerical reproduction is reported here.

| Source and location | Temperature | Printed uncorrected final rate, cm³ molecule⁻¹ s⁻¹ |
| --- | ---: | ---: |
| 2017, Table 2, NN1 | 300 K | 1.25 × 10⁻¹⁴ |
| 2018, Table 1, NN1, P128 | 300 K | 1.25 × 10⁻¹⁴ |
| 2017, Table 2, NN1 | 1000 K | 3.75 × 10⁻¹² |
| 2018, Table 1, NN1, P16 | 1000 K | 2.23 × 10⁻¹² |

The [2017 author manuscript](https://www.researchgate.net/publication/320300566_A_ring_polymer_molecular_dynamics_study_of_the_OH_H_2_D_2_reaction)
identifies NN1 with [DOI 10.1063/1.4801658](https://doi.org/10.1063/1.4801658).
Its Table 2 explicitly labels the observable as the final RPMD rate.
Its later electronic correction divides the uncorrected result by
`1 + exp(-hc × 140 cm⁻¹ / kBT)`; corrected and uncorrected tables must stay
separate. The text describes 48 beads at high temperature, without assigning
a bead count individually to each Table 2 entry.

The [2018 author manuscript, arXiv v3](https://arxiv.org/pdf/1805.11924)
was checked in both extracted text and the rendered Table 1 on PDF page 11.
At 1000 K it explicitly lists P16, QTST `3.72e-12`, κ `0.599`, and final rate
`2.23e-12`; the displayed product rounds consistently. The 2017 printed
final rate is close to the 2018 QTST factor, but that observation does not
establish a typo or license replacing either source entry. The final rates
differ by 40.53% relative to the 2017 entry. Different bead counts, sampling,
or reporting errors have not been established as the cause.

The existing registry retains the explicitly decomposed 2018 targets and
the separately identified 2017 spin-orbit-corrected targets. An automatic
cross-paper equivalence assertion is not warranted by these sources.

## Available model inputs

The exact NN1 Fortran implementation and trained weights were not obtained
in this check. The NN1 paper's indexed abstract identifies the model, but
does not supply executable weights. Searches of the public author and
publisher routes did not yield a matched model archive. A different
publication's public code link returned GitHub 404; its relevance to NN1
was never established. No alternate OH₃ surface was substituted.

The [machine-readable receipt](check.json) records successful readings,
failed retrievals, and this limited conclusion. Failed publisher/PDF
routes were not retried through an alternate endpoint. No person was
contacted and no copyrighted full paper was added to the repository.
