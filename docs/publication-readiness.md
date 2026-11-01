# Publication readiness

PyRPMD needs complete, independently reviewable scientific records before a
manuscript can claim the requested benchmark reproductions. The
[benchmark inventory](literature-validation.md) is the current source of
truth. Passing interface tests or publishing wheels does not establish a
thermal rate, a tunnelling isotope effect, liquid diffusion or an IR spectrum.

## Choose the claim before the venue

| Venue | Appropriate core contribution | Evidence to assemble |
| --- | --- | --- |
| JOSS | Maintained research software with a clear need, design rationale, documentation and research use | Open development history, installation/test/reuse evidence, comparison with existing packages, archived release and complete attribution |
| JCTC | A methodological or computational advance with convincing chemical applications | A precise new contribution, matched physical benchmarks, accuracy/cost/convergence analysis and reproducible supporting data |
| CPC | A substantial computational program and its implementation for substantive physical problems | Program description, algorithm/architecture rationale, reproducible examples, performance/scaling evidence, licensed code/data archive |

This is a fit assessment, not an editorial acceptance prediction. The current
[JCTC author guidelines](https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce)
include computational packages presenting innovative methods and emphasize
methodological advances; a feature list alone does not establish that claim.
[CPC's publisher description](https://shop.elsevier.com/journals/computer-physics-communications/0010-4655)
describes its computer-program track and archival program library. Confirm
current detailed submission requirements with the chosen venue when the
manuscript is ready.

## JOSS timing and disclosure

As checked on 2026-11-01, the official
[JOSS submission criteria](https://joss.readthedocs.io/en/latest/submitting.html)
require more than six months of public development, demonstrated research
use and iterative open-source work. A newly created public repository cannot
meet that history requirement immediately. Repository creation metadata is
not proof of the date code first became public; preserve actual history and
community evidence. This is a readiness constraint, not something a draft
paper or an automated benchmark can remove.

The [review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
also require a statement of need, state-of-field comparison, software-design
discussion, research-impact evidence and disclosure of AI assistance. Record
the code/documentation assistance used in this work and the human review and
scientific verification actually performed. Do not assert human review or
external adoption that has not happened. Authors must manage their own
editorial/reviewer correspondence under the journal's current policies.

## Evidence package for the first submission

Prepare a fixed release with tagged source, installable artifacts, all model
and third-party licenses, an archival DOI when available, the computational
environment and exact rerun commands. Archive reference provenance and raw
observations as well as final plots. A reader should be able to recreate a
successful numerical result and inspect failed/unresolved cases.

The manuscript should separate numerical verification, physical convergence,
agreement with a matching implementation and differences from exact quantum
dynamics. Report failure rates, confidence intervals, compute budgets and
unresolved limitations. Show cost at a fixed error target when making an
efficiency claim; provider dispatch counts alone are not speedup evidence.

Assign real authorship/affiliations, contributions, funding, conflicts and
approval of the final claims before submission. No author list, submission,
archival DOI, publication acceptance or external adoption has been invented or
created by these Tier 3 changes.

## Manuscript structure once the data exist

1. State the scientific use case and how PyRPMD relates to RPMDrate, i-PI and
   the external force engines. Explain the reasons for a separate workflow.
2. Describe scientific conventions, propagation and sampling, unit boundaries,
   checkpoint identity and evidence-based reporting.
3. Present the completed benchmark matrix, including every matched model,
   uncertainty, convergence sweep and unsuccessful or out-of-scope case.
4. Present installation, interoperability, reproducibility and performance
   measurements on stated hardware.
5. State limitations, availability and contributor/AI disclosures.

For JOSS, adapt this material to its software-paper scope. For JCTC/CPC,
develop the methods and scientific evaluation to support the contribution
claimed. A future manuscript should draw its numerical claims from archived
results, not from this planning document.
