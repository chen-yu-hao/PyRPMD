# H + CH₄ normalization: completed algebra review

The known prefactor **2.0492225575240845 × 10⁻⁸ cm³ molecule⁻¹ s⁻¹ is correct** for the existing 300 K, 30 bohr, finite-entry flux-surface ratio. The separate [independent-author review](independent-review.json) found no remaining blocker within that algebra/source-convention scope. It does not authorize a new calculation or validate a physical reaction rate.

The [derivation](README.md) starts from the full Cartesian canonical momentum integral and shows the positive-flux, translational/COM, global-rotation, coarea and mass-metric cancellations. The existing four active-max channel integrals already count four abstractions. Neither an additional factor four nor a product-H₂ factor one-half belongs in the forward ratio. The full transition-surface mass metric, including carbon, remains inside its numerator.

The entry relation is **N₀(L) = F_L Z_B^(L)**, where Z_B^(L) is the orientation-averaged finite-entry partition. Therefore **F_L Nα/N₀(L) = Nα/Z_B^(L)**. It becomes an asymptotic physical normalization only after the molecular domain, unseen tails and entry separation are controlled. The known prefactor cannot cure those unresolved numerical/physical requirements.

The [deterministic calculation receipt](algebra-checks.json) independently agrees through SI and atomic units to 1.95 × 10⁻¹³ relative, and verifies the Maxwell absorbing-sphere flux identity and twelve full-gradient mass-metric algebra cases. The author executed that small calculator once; the second reviewer inspected its code and result without rerunning it. No pilot observations, converted pilot rates, PES calls, random draws, molecular-geometry evaluations or dynamics were used by either review.

The derivation's author also implemented the pilot helpers; this authorship limit remains explicit. The independent second author verified the mathematical statements, the three exact subject files and eleven original source/definition identities. The excluded pilots remain excluded, and no old evidence or scientific source was changed.

## Frozen files

- [Derivation](README.md), [calculator](check_normalization.py) and [algebra result](algebra-checks.json).
- [Primary-source access record](sources.json): author-uploaded CPC manuscript, institutional bibliographic record, and the inaccessible MIT PDF endpoint.
- [Author review receipt](review-note.json) and [independent-author review](independent-review.json).
- [Publication inventory](publication-inventory.json): seven text/code/JSON files plus this inventory, with no raw NPZ, model library or external PES source.
