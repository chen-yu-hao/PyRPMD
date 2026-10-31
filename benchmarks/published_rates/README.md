# Published-rate references and comparison harness

`references.json` records source-verified published values and explicit missing
reference entries. It is **not** a report that PyRPMD has reproduced all the
listed reactions. Each numeric record identifies its original paper/table,
PES, temperature, method, and statistical conventions. Unretrieved values stay
`null`; `status="not_available"` never becomes a numerical target.

The registry's RPMD rows are implementation references. Exact quantum rows
measure the physical error of an approximate dynamics method and are not
implementation pass/fail targets. Current H + H₂ quantum rows have unresolved
nuclear-spin/electronic/symmetry matching and are deliberately blocked from an
automatic numeric comparison. Their tabulated numbers remain available for
source auditing.

## Compare one calculation

Run from the repository root:

```bash
python benchmarks/published_rates/compare.py \
  --reference-id h_h2_bkmp2_rpmd_300k \
  --calculation runs/h3/published-comparison-manifest.json \
  --relative-tolerance 0.10 \
  --output runs/h3/published-comparison.json
```

The calculation manifest must be prepared from actual output, with a positive
finite candidate/final rate. The script does not propagate trajectories or
infer physical conventions from a directory name. It records the SHA-256 of
both input JSON files and prevents the output from replacing either input.
`--references` can select another explicitly reviewed registry using the same
schema.

Reusable Python entry points are `load_reference(id, registry_path=None)` and
`compare(reference, calculation, relative_tolerance=0.10)` in `compare.py`.
`load_reference` attaches the top-level registry rate unit to the selected case.

## Calculation manifest

These fields describe the **already reported final/candidate rate**, including
any already applied factors. Copying a reference's labels is appropriate only
after confirming that the calculation really uses those definitions.

| Field | Required meaning |
| --- | --- |
| `reaction` | Exact registry reaction and channel/isotope definition |
| `pes` | Exact PES/version string; unrelated surfaces cannot be compared |
| `method` | `"RPMD"` for an RPMD implementation reference; a resolved method name for QM comparisons; `reference_method` is an accepted alias |
| `temperature_K` | Positive finite physical temperature, matching the reference |
| `nuclear_spin_convention` | Matched distinguishable/para/ortho or other population convention |
| `electronic_convention` | Matched electronic-state/spin-orbit convention |
| `electronic_weight` | Positive finite already-applied electronic factor |
| `symmetry_convention` | Matched equivalent-channel/symmetry definition |
| `symmetry_factor` | Positive finite already-applied factor; no automatic second application |
| `rate_cm3_molecule_s` | Positive finite measured rate or explicitly identified pilot candidate |
| `rate_unit` | Exactly `"cm3 molecule-1 s-1"` |
| `provenance.source` | Calculation-result artifact path or URL |

Provide `convergence` as an object with boolean keys `beads`, `timestep`, `rinf`,
`sampling`, and `plateau`. A pilot should leave every unestablished check false.
Missing flags also withhold validation. For example:

```json
{
  "convergence": {
    "beads": false,
    "timestep": false,
    "rinf": false,
    "sampling": false,
    "plateau": false
  },
  "uncertainty": {
    "includes_pmf": false,
    "includes_recrossing": true,
    "standard_error_cm3_molecule_s": null,
    "independent_parent_chains": 1,
    "method": "conditional recrossing bootstrap; PMF uncertainty not yet included"
  },
  "provenance": {
    "source": "runs/h3/summary.json"
  }
}
```

This is only the diagnostic portion of the manifest; add the required identity
fields and the actual measured rate. `null` uncertainty is allowed for a pilot.
A reported zero standard error is retained but cannot establish sufficient
precision for the gate; a degenerate bootstrap is not proof of zero uncertainty.
A zero/negative/nonfinite rate is not an admissible comparison value and is
never replaced by a guessed rate.

For each convergence flag set to true, also provide
`convergence_evidence.<flag>.source` and `.criterion`, both nonempty descriptions
without unresolved/pending labels. The referenced artifact should contain the
actual parameter series, statistics, and decision criterion. For example, a
timestep evidence record should point to the matched full/half-step results
and state the selected tolerance; a plateau record should point to the
time-dependent transmission curve and its tail analysis. A boolean flag alone
does not enable the gate.

A complete rate uncertainty record requires:

- `includes_pmf=true` and `includes_recrossing=true`.
- A nonempty estimator `method`.
- At least two `independent_parent_chains`.
- A positive finite `standard_error_cm3_molecule_s`. Its relative value must
  be no greater than the requested comparison tolerance.

The harness checks these assertions and their metadata structure. It does not
open the referenced convergence files or verify that a stated criterion was
actually met. Every report retains `evidence_independently_verified=false`.
Review isotope masses, PES source hashes, channel definitions, and convergence
evidence before treating an eligible gate as a scientific validation.

## Result interpretation

| Status | Numeric comparison | `passed` |
| --- | --- | --- |
| `reference_not_available` | No ratio; missing values stay null | `null` |
| `incompatible_conventions` | No ratio; the report lists missing/mismatched definitions | `null` |
| `pilot_comparison` | Matched raw ratio and relative difference | `null` |
| `implementation_agreement` | Matched RPMD comparison with supplied convergence/uncertainty evidence | `true` or `false` |
| `quantum_accuracy_comparison` | Matched QM comparison, separately identifying method error | `null` |

The reported ratio is `calculation/reference`; the signed relative difference
is `ratio-1`. Neither electronic nor symmetry factors are applied again.
`within_tolerance` is a descriptive point-estimate test. It does not become
`passed` for a pilot, and it never creates an implementation failure for a QM
reference. A failed RPMD tolerance gate flags a discrepancy for investigation;
it does not, by itself, locate a software bug or prove statistical significance.
The default 10% tolerance is a configurable comparison policy, not a confidence
interval assigned to every publication.

The CLI exits with code 0 after generating a compatible report (including
pilots and QM comparisons), 1 for an eligible failed RPMD gate, and 2 for an
incompatible/unavailable reference or invalid input. Exit code 0 alone must not
be used as evidence of physical validation: inspect `gate_eligible` and
`passed`.

## Source scope and remaining work

The initial registry contains verified H + H₂ and H + CH₄ RPMD entries from
the [RPMDrate program paper](https://doi.org/10.1016/j.cpc.2012.10.017), OH + H₂
entries from the cited NN1 studies, and separately flagged H + H₂ quantum
tables. The JSON source and note fields distinguish corrected and uncorrected
OH rates and identify unresolved conventions. It does not substitute one PES
for another or apply a correction to an already corrected table entry.

Numerical/source verification remains incomplete for several requested isotope,
Cl + CH₄, F + H₂, Mu + H₂, and quantum methane cases. Their explicit pending
records prevent an absent benchmark from appearing as a success. Reproducing
the full reaction set also requires matched PES access and converged PyRPMD
calculations; registering a published number does not accomplish either.

`tests/test_published_rate_comparison.py` uses clearly labelled synthetic
calculation manifests to test matching, missing references, pilot withholding,
eligible pass/fail decisions, and QM separation. Those synthetic calculations
are unit-test fixtures, not added scientific results.

## Real H + H₂ pilot

After building the checksum-pinned BKMP2 library with `benchmarks/h3/build.py`,
the following runs the actual umbrella/recrossing workflow and writes a
calculation manifest. It is a diagnostic calculation, not a converged rate:

```bash
PYTHONPATH=src python benchmarks/published_rates/h3.py \
  --profile pilot --workers 2 --output runs/h3-published-pilot
python benchmarks/published_rates/compare.py \
  --reference-id h_h2_bkmp2_rpmd_300k \
  --calculation runs/h3-published-pilot/calculation.json \
  --output runs/h3-published-pilot/comparison.json
```

The source-matched physical settings are 300 K, 128 beads, 0.1 fs timestep,
30 bohr reactant separation, and 111 umbrellas with `kappa="30 hartree"`.
The [umbrella-normalization audit](umbrella-normalization.md) verifies that
the coefficient must **not** be divided by the bead count. The driver uses
Andersen/Cayley and a different initialization/sampling protocol from the
original GLE/Verlet calculation.

The `pilot` profile has only 50 fs equilibration and 100 fs production per
window/replica. The `extended` profile increases these to 1 ps and 2 ps,
respectively; this is a proposed next sampling diagnostic, not an assertion
that those times establish convergence. Compare separate replicas and time
halves before increasing recrossing work. The 2026-10-31 short-pilot results
have large PMF disagreement with the paper and are explicitly withheld from
validation; see the [evidence record](../../docs/validation/tier1-2026-10-31/README.md).
