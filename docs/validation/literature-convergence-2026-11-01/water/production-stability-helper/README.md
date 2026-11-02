# Descriptive production-stability supplement

The [sealed definition](../production-stability-preregistration.json) resolves
the original protocol's unspecified split of 25 production segments before any
complete-cohort diffusion results are inspected. The balanced diagnostic uses
segments 0–11 and 13–24. Segment 12 remains in all primary means and in the
display of all 25 segment estimates for every seed. Both the VACF integral to
2 ps and the MSD slope over 1–2 ps are reported.

For each of the eight seeds, the difference is its late 12-segment mean minus
its early 12-segment mean. The reported interval is the mean difference plus or
minus `t.ppf(0.975, 7) * std(differences, ddof=1) / sqrt(8)`. The eight paired
chain differences are the uncertainty units. Adjacent segments, individual
molecules and overlapping origins are not counted as independent replicas.
This is nominal small-sample inference conditional on the original shared
starting geometry and preparations. It introduces no new hard gate, sample
selection, stopping criterion or claim of paper agreement.

The separate helper reuses the frozen complete-cohort loader and uncertainty
function. It requires a report containing all eight complete original chains,
all 25 retained segments, the original protocol and compact-curve hashes,
unchanged primary chain means, and the sealed supplement. It additionally
requires the observed original coordinator's normal exit, released leases and
the same eight manifest identities. A complete report alone is insufficient
evidence that the original writer pool has exited.

From the repository root, after the complete original-cohort analysis and
observed exit receipt exist:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmarks/water/plot_production_stability.py ANALYSIS_DIR \
  --protocol docs/validation/literature-convergence-2026-11-01/water/protocol.json \
  --supplement docs/validation/literature-convergence-2026-11-01/water/production-stability-preregistration.json \
  --exit-proof ORIGINAL_NORMAL_EXIT_PROOF.json --output FIGURE_DIR
```

The outputs are PNG/SVG plots, all 400 endpoint/seed/segment records, 16
endpoint/seed half-window summaries, and a JSON record binding inputs, sources
and outputs. Original eligibility failures remain visible in the JSON and plot
caption. There is no reference-value test in this helper.

## Validation scope

[Six excluded synthetic cases passed](pytest.log); their
[JUnit record](pytest.xml) and [before/after source guard](gate.json) are retained.
They check paired-chain variance in the presence of large shared offsets,
middle-segment retention, rejection of missing/nonfinite estimates, mismatched
exit/manifest identity, unsuccessful exit or unreleased leases, and the full
plot/CSV/hash roundtrip. This gate performed no PES calls or molecular dynamics
and did not inspect actual water diffusion data. An independently generated
synthetic display was visually checked; it is not included as scientific data.

This is a diagnostic of the external OpenMM reference campaign. It does not
validate PyRPMD production dynamics, repair preparation failures, establish
liquid equilibration, or validate water IR spectra.
