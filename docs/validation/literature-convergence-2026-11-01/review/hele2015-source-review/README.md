# Independent check of the 2015 F + H₂ graph

The three zero-friction points in the original Figure 7 agree with the root
source audit to **0.181 parts per million** when read independently with
PDFMiner/pdfplumber and standard-library arithmetic. This confirms the graph
transcription. It supplies no new molecular calculation, physical agreement
test, or uncertainty estimate for the authors' simulations.

The [review receipt](independent-review.json) binds the exact root source
record, extractor, protocol and README. The original article, TeX and figure
bytes remain external. None is copied into this review's publication inventory.

## Source conditions

The [author PDF](https://www.tjhh.uk/pubs/hel15trpmd.pdf) identifies zero
friction with RPMD. Figure 7 uses red crosses for the computed series, black
lines for quantum references and a green fit. The independent visual check
confirmed the red λ = 0 crosses and temperature assignments. Its seven labeled
rate ticks form a **logarithmic** axis.

Table II gives F + H₂ at 200/300/500 K with 384/256/64 beads, two channels,
30 bohr separation, ξ‡ = 0.9671/0.9885/0.9947, 0.1 fs child steps, 200 fs
children and 500,000 child trajectories. The text names Stark–Werner and cites
JCP 104, 6515 (1996). It does not identify exact executable PES bytes here.
D + MuH remains a distinct reaction from Mu + H₂. The TeX's dynamic date
macro must not be used to relabel the 2015 publication as a modern paper.

## Independent numerical check

[check_fh2.py](check_fh2.py) does not import or execute the root's PyMuPDF
extractor. It uses another PDF reader to locate all seven physical major ticks,
match their printed labels, locate the two arms of each red zero-friction cross,
and fit natural log rate against downward PDF position. The visible temperature
labels independently determine the row assignments.

For tick positions \(y_j\), the calculation fits
\(\log k_j = a + b y_j\), then evaluates \(\exp(a+b y)\) at each cross center.
It also retains two-point log calculations using the axis endpoints and the
nearest two major ticks. Those checks make the coordinate interpolation explicit
without reusing the producer's fitting function or drawing-object indices.

| T / K | Independent graph value / cm³ s⁻¹ | Relative difference from root's reader |
| ---: | ---: | ---: |
| 200 | 1.9301814926 × 10⁻¹¹ | +0.0260 ppm |
| 300 | 2.8608203988 × 10⁻¹¹ | −0.0961 ppm |
| 500 | 5.3562365762 × 10⁻¹¹ | −0.1807 ppm |

These extra digits expose the coordinate calculation and cross-reader check;
they are not physical accuracy. The 200 K cross is slightly below the lowest
labeled major tick. Its estimate therefore includes a short log-axis
extrapolation within the drawn axis, rather than a bracketed interpolation.

The [complete check record](checks.json) retains tick coordinates, fit
residuals, cross centers, all three calculation routes and source hashes.
The [execution receipt](execution.json) records the successful single-CPU,
read-only invocation. The source script and inputs remained unchanged. No PES,
random generator, molecular engine or benchmark analyzer was loaded.

## Limits of this result

The ±1 original-PDF-point sensitivity used by the source audit describes
coordinate sensitivity. It is neither a reported paper confidence interval nor
a simulation acceptance margin. The paper's statistical uncertainty remains
unknown. Agreement between two readers, and the root's separate typeset/original
cross-check, cannot establish that missing uncertainty.

Exact masses/constants, the SW implementation variant and any electronic-state
or postprocessing convention remain unresolved. A future physical comparison
must pin these separately. This review does not infer them from the graph,
resolve the 2010 erratum, replace a registered CPC target, or change any ongoing
simulation or failed convergence gate.

This is an independently authored **source check**, not an independent molecular
experiment. Only the F + H₂ three-point extraction and the listed source
conditions received this additional numerical review; the other three reactions
were not independently re-digitized here.
