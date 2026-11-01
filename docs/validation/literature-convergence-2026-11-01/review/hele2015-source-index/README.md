# Hele–Suleimanov 2015: source and graph audit

This additive source record identifies published RPMD points in **Figures 4–7 of JCP 143, 074107 (2015), DOI [10.1063/1.4928599](https://doi.org/10.1063/1.4928599)**. Section II B identifies zero friction (λ = 0) with RPMD. These are literature graph coordinates, with no new PyRPMD trajectories or physical pass.

The author [publication index](https://www.tjhh.uk/pubs.html) links the [published PDF](https://www.tjhh.uk/pubs/hel15trpmd.pdf) and [arXiv submission](https://arxiv.org/abs/1506.07460v2). The pinned arXiv v2 source archive contains seven original figure PDFs and one TeX file. It contains no raw rate table, simulation input deck, seeds or trajectory archive. Its TeX was read as text and never executed. Original PDF/TeX bytes remain external to Git; the retrieval identities are recorded here.

## Zero-friction graph points

Rates below are in cm³ molecule⁻¹ s⁻¹. Three significant figures are display precision, not a claim about statistical accuracy. The final column perturbs the marker coordinate by **±1 point in the original figure PDF**, adding the maximum fitted log-axis residual. It is a coordinate-sensitivity range; it is **neither a paper confidence interval nor a physical equivalence tolerance**. The figures do not supply a statistical uncertainty.

| Reaction as published | T / K | Beads | Extracted λ = 0 rate | Coordinate-sensitivity range |
| --- | ---: | ---: | ---: | --- |
| H + H2 | 200 | 128 | 1.180e-18 | [1.155e-18, 1.206e-18] |
| H + H2 | 300 | 128 | 1.948e-16 | [1.907e-16, 1.991e-16] |
| H + H2 | 500 | 128 | 2.548e-14 | [2.494e-14, 2.604e-14] |
| D + MuH | 200 | 512 | 1.122e-14 | [1.107e-14, 1.137e-14] |
| D + MuH | 300 | 512 | 1.140e-13 | [1.124e-13, 1.155e-13] |
| D + MuH | 500 | 512 | 1.295e-12 | [1.277e-12, 1.312e-12] |
| H + CH4 | 250 | 192 | 4.016e-21 | [3.923e-21, 4.112e-21] |
| H + CH4 | 300 | 128 | 1.139e-19 | [1.112e-19, 1.166e-19] |
| H + CH4 | 500 | 64 | 2.590e-16 | [2.530e-16, 2.652e-16] |
| F + H2 | 200 | 384 | 1.930e-11 | [1.925e-11, 1.936e-11] |
| F + H2 | 300 | 256 | 2.861e-11 | [2.852e-11, 2.869e-11] |
| F + H2 | 500 | 64 | 5.356e-11 | [5.341e-11, 5.372e-11] |

## Extraction and checks

The [offline extractor](../../../../../benchmarks/literature/hele2015_digitization.py) first checks exact SHA256 identities for both journal and author-source PDFs, the source archive, and TeX. It pairs each printed rate label with its physical tick, fits **log(rate)** against downward PDF coordinates, verifies the linear λ axis, and identifies the two arms of every red data cross. All six declared friction values at all three temperatures are retained in [the numerical record](attempt-1/digitization.json); the single legend cross is excluded by coordinate membership. The λ = 0 column is selected from that fixed complete set.

For F + H₂, a separate extraction of the scaled, typeset journal Figure 7 agrees with the original author figure to less than 3 parts per million at all three temperatures. This checks coordinate calibration and source correspondence; it does not bound statistical uncertainty in the original simulations. The [execution record](attempt-1/execution.json) preserves the actual successful, bounded read-only run and exact extractor snapshot. No PES or random generator was called.

The root reviewer visually inspected journal pages 074107-7, 074107-8 and 074107-9, including temperature labels, axis exponents, Table II and the distinction between red TRPMD points, black quantum lines and green fitted curves. An independent source review is tracked separately when complete.

## Conditions and remaining gaps

[source-protocol.json](source-protocol.json) transcribes the relevant Table II parameters and links each fact to its printed page. For F + H₂ the paper names Stark–Werner, uses 384/256/64 beads at 200/300/500 K, two equivalent channels, a 30 bohr entry surface and 0.2 ps children. **D + MuH** uses BKMP2, 512 beads and a 0.03 fs child time step; it is a different reaction from Mu + H₂.

Exact 2015 masses/constants, the distributed SW source variant and any electronic-state/postprocessing factor are still unpinned. The authors' original run decks and numerical uncertainty have not been recovered. Therefore every entry remains ineligible for an exact physical-reproduction assertion.

This independent 2015 source leaves the registered CPC targets and the unresolved 2010 F + H₂ erratum unchanged. A matching literature number does not establish a second independently reproduced experiment. None of these data changes an ongoing calculation, a convergence threshold, or a previously failed gate.

## Reproduce the source audit

Obtain the exact public sources identified by [retrieval.json](retrieval.json), [source-retrieval.json](source-retrieval.json) and [selected-source-members.json](selected-source-members.json). With the optional PyMuPDF reader installed, run the extractor with `--source-dir` pointing at those local bytes and `--output` naming a new JSON file. It rejects an existing output and any changed source hash. No document code, embedded script, TeX compiler, potential, molecular dynamics or benchmark analyzer is executed.
