# Complete-cohort transport figure

[plot_native_campaign.py](../../../../../benchmarks/water/plot_native_campaign.py) consumes the compact `report.json` and `curves.npz` produced by the frozen streaming analyzer. It produces a PNG, an SVG, an eight-row CSV and a provenance receipt. It does not read or alter an active trajectory or random state.

```sh
.venv/bin/python benchmarks/water/plot_native_campaign.py path/to/complete-analysis \
  --protocol docs/validation/literature-convergence-2026-11-01/water/protocol.json \
  --output path/to/final-figures
```

The helper requires the complete original eight-chain, classical H2O cohort: every registered seed exactly once, all registered NVE segments retained, no partial segment, and the original 100 ps preparation and 4 ps segment durations. The duration guard prevents the proposed later 500 ps preparation from being mislabeled with this figure's original 50–75/75–100 ps windows. The report/protocol and compact-curve checksums are validated before plotting.

Aggregate estimates and 95% Student-t intervals use eight seed-chain means and seven degrees of freedom. The helper independently recalculates that uncertainty and rejects a report that instead treats the 25 adjacent segments as independent replicas. The running-integral and MSD bands are pointwise intervals across the eight chain-mean curves, not simultaneous confidence bands.

The four panels display individual-chain and aggregate diffusion estimates, the cumulative VACF integral, the molecular COM MSD with its registered fitting interval, and the original preparation-energy difference. Failed seeds and all their registered segments remain in the figure and CSV. Other failed numerical checks are named in the footer. A paper-reference overlay is enabled only when the frozen analyzer enabled its literature comparison; a failed preparation check keeps that overlay disabled. Even a numerically eligible figure explicitly retains the external OpenMM-reference scope and does not certify physical convergence or reproduction. When shown, the paper's shaded interval is its reported **one standard error**, while our aggregate error bars are **95% Student-t intervals**.

The [test receipt](gate.json) and [JUnit record](pytest.xml) contain nine passing tests for completeness, duplicate/missing data, original preparation duration, independent-chain uncertainty, artifact checksums, curve/estimate consistency and paper-eligibility preservation. An actual incomplete campaign report was also refused before plotting. Two explicitly synthetic layouts were visually inspected to check the failed/eligible display branches; those synthetic images and data are excluded from scientific evidence. No real incomplete-cohort aggregate was generated.

The source and fixed validation evidence are enumerated in [publication-inventory.json](publication-inventory.json). This helper does not change any frozen integration source or the running experiment.
