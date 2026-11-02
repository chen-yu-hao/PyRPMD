# H + H₂ archived-pilot comparison rerun

This is a **fresh run of the numerical comparison**, using the actual
previously committed calculation manifests. No new H₃ dynamics, PES build,
PMF replay or sampling was performed. At comparison time the local environment
had no GNU Fortran compiler. Workspace-local GNU Fortran 16.2.0 was subsequently
installed and successfully compiled/executed a minimal program. The pinned BKMP2
source was not present, and automatic approval review denied its network fetch;
no alternate download was attempted. A fresh dynamics run therefore still
requires a locally supplied, checksum-matching PES source.

Commands run from the repository root:

```bash
python benchmarks/published_rates/compare.py \
  --reference-id h_h2_bkmp2_rpmd_300k \
  --calculation docs/validation/tier1-2026-10-31/broad-pilot/calculation.json \
  --output docs/validation/ecosystem-2026-11-01/h3-archived-comparison/broad-pilot.json
python benchmarks/published_rates/compare.py \
  --reference-id h_h2_bkmp2_rpmd_300k \
  --calculation docs/validation/tier1-2026-10-31/stiff-pilot/calculation.json \
  --output docs/validation/ecosystem-2026-11-01/h3-archived-comparison/stiff-pilot.json
```

Both reports retain `status="pilot_comparison"`, `gate_eligible=false` and
`passed=null`. They identify the missing bead/timestep/entrance-distance/sampling
convergence, missing plateau evidence and excessive rate uncertainty.

The old broad and stiff estimates have calculation/reference ratios of about
899 and 9.89e-5 respectively. These enormous disagreements are diagnostic
failures of a reproduction claim, not successful implementation validation.
The original PMF-support/stationarity discussion remains in
`docs/validation/tier1-2026-10-31/pmf-replay.md`. The raw PMF replay payloads
described there are not distributed, so no independent PMF estimator replay
is claimed here.

The new JSON files retain SHA256 identities of the input calculation manifests
and source registry. Their original-run paths are provenance strings from the
archived manifests, not locations accessed in this execution. The accompanying
`execution.json` records source and output hashes for this rerun.
