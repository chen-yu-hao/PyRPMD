# Literature reproduction campaign and evidence audit

`campaign.json` gives an explicit status, source and acceptance protocol for
H + CH₄, OH + H₂, F + H₂, H/D/Mu + H₂, H in Ni/Pd and q-TIP4P/F water.
Every case currently retains `status="not_reproduced"` for its full requested
convergence scope. The complete H + H₂ P128 reference now passes all eleven
fixed-P primary criteria and agrees with the paper; its supplementary parent
half check, dynamic bead refinement and production-umbrella validation remain
separate. The inventory records that successful fixed-P result explicitly
without treating it as evidence for the unresolved tests or other cases.

The campaign reuses the
[published-rate registry and comparator](../published_rates/README.md), the
[BKMP2 builder/driver](../h3/README.md) and the
[Ni EAM pilot](../materials/README.md). A separate
[author-cited Ni/H EAM4 static reference](../materials_reference/README.md)
now pins and checks the original routines without claiming diffusion or
promoting the unmatched NiAlH pilot. The
[water model and pilot](../water/README.md) add a new physical force field with
independent model checks. This is not another set of synthetic reaction rates.

## Inspect the inventory

```bash
python benchmarks/literature/evidence.py --output runs/literature-status.json
```

Inventory mode returns success when the inventory can be read, including when
no physics is validated. Inspect the explicit status fields; exit code alone
is never a reproduction claim.

## Audit an actual local evidence bundle

```bash
python benchmarks/literature/evidence.py \
  --bundle runs/reproduction/evidence-bundle.json \
  --relative-tolerance 0.05 \
  --output runs/reproduction/evidence-audit.json
```

The bundle describes one campaign case:

| Field | Content |
| --- | --- |
| `schema_version` | `1` |
| `case_id` | One exact ID in `campaign.json` |
| `identity` | An object containing `method`, `model_sha256`, `temperature_K` and `observable`; add isotope, density, channel and other fixed physical definitions here |
| `model` | `{ "path": "model.xml", "sha256": "ACTUAL_SHA256" }` for the real model/source/model-package bytes |
| `result` | Same path/hash structure, pointing to a JSON result with exactly matching `identity` |
| `convergence` | Map of control name to a path/hash object for a measured convergence record |

Paths are relative to the bundle's directory. The auditor reads the bytes,
checks each SHA256, refuses missing files and paths or symlinks that escape the
bundle, and refuses an output path that would replace an input. A model package
should include every parameter file and source needed to identify the PES.
Its hash identifies the supplied bytes; scientific review must establish that
these are the model that actually produced the trajectories and the correct
model for the paper.

Each convergence JSON contains `schema_version=1`, `control`, `unit`, the same
`identity`, and `baseline`/`refined` objects. Each of those objects contains:

- `parameter`: the actual numerical control setting.
- `replica_ids`: at least three nonempty, unique identifiers for independent runs.
- `values`: their finite positive estimates, in the case's stated unit.

Both groups must have disjoint IDs. Correlated time origins, Cartesian
components and beads are not independent replicas. This implementation uses an
**unpaired** comparison; paired studies need their covariance treated explicitly
and must not be relabelled as independent studies.

Beads and sampling duration must at least double, and timestep must at least
halve. Other controls must increase. For `electrostatic_accuracy`, use the
reciprocal of the solver error tolerance so that increasing it tightens the
calculation. Record the units and definitions of each control in the archived
run configurations. For `plateau` or `correlation_length`, the numerical record
compares estimates at longer physical horizons; inspect the full curves as well.

The auditor computes the mean and standard error from the replica values and
a two-sided **Welch 95% confidence interval for the change**. The entire interval
must lie within the declared relative-equivalence band around the refined mean.
Thus equal noisy means do not pass merely because a difference is insignificant.
Zero sample variance also withholds the check, because repeated/rounded values
do not establish statistical precision. The default 5% band is a project policy,
not a universal physical tolerance and not a confidence interval of a paper.
Fix it before running a campaign and justify it for the intended observable.

Missing evidence withholds the numeric gate. Malformed evidence, mismatched
identity or changed bytes are errors. In bundle mode the CLI exits 0 for passed
numeric checks and 1 for unresolved/failed numeric checks; invalid input raises
an error. **No mode sets `physical_validated=true`.**

## What the audit does and does not establish

The report separates `artifact_integrity_passed`,
`numeric_convergence_checks_passed`, `literature_agreement` (always unset here)
and `physical_validated` (always false here). It verifies local evidence bytes
and numerical arithmetic. It does not establish replica independence, correct
PES provenance, physical normalization, ensemble correctness or an appropriate
fit window. The case's `required_scientific_reviews` remains in every report.

After those reviews, use the existing published-rate comparator for matched
gas-phase values. Water source values are in `benchmarks/water/references.json`;
compare the finite-system result with the corresponding finite-system entry.
Quantum exact results and PA-CMD spectra measure approximation differences and
must remain separate from RPMD implementation checks. Archive the signed review,
raw trajectories/correlations, inputs, environment and source revision alongside
the numerical comparison. A green software CI job is not that scientific review.

`tests/test_literature_evidence.py` contains explicitly synthetic statistics to
test the gate, checksum rejection and unresolved states. Its numbers are never
registered as scientific benchmark results.
