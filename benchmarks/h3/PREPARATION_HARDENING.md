# Initial-preparation implementation hardening

This audit follows the action-QCT and isolated-switch study in PR #43. The
starting source tree is `8b4c1c8b77d1756619932662d86648ed5e854a82`, identical to
the tree merged into `main` as `c4fa743f552c1d71db35fab791a2e63409bc1c3e`.
The scope is implementation correctness and honest failure accounting. It does
not introduce a new state-selection theory or reinterpret any recorded outcome.

## Completed TODO list and acceptance criteria

The following list was fixed before implementation. All seven items are
implemented and verified. Each regression is in the corresponding
`tests/test_h3_*` module.

| Item | Reproduced problem | Required behavior |
| --- | --- | --- |
| 1. Physical input and request validation | Complex phase space lost its imaginary part on conversion; fractional counts, nonfinite budgets and off-grid durations reached initialization. An explicit override could change a supposedly fixed collision energy. | Validate real numerical inputs before conversion and requests before PES/build/output/worker side effects. Check timestep grids, known native capacities, integrator domains and declared energy conventions. |
| 2. Native finite-result contract | A second initial observation failure was ignored; hold integrals could overflow while the run still reported completion. | Check initialization references and every candidate observation/integral. Use a stable trapezoid and accept the phase point, integral and elapsed hold time together. On failure retain the last accepted state. |
| 3. Near-threshold action-QCT | At v=1 and actual total energy 0.800 eV, the EBK collision energy is positive but the harmonic paper-beta formula rejects the request. | Keep existing paper-beta values where defined. For otherwise valid P=1 QCT only, use and record an auxiliary placeholder; preserve phase, orientation, impact parameter and physical energy. |
| 4. Complete failure accounting | A prepared row with a missing or nonfinite mean silently disappeared. All-failed cases could vanish or crash reports; failed collisions were also described as unresolved. | Reject damaged successful rows, retain every attempted case and paired outcome, and distinguish failed from unresolved. Undefined SEM or absent numerical diagnostics remain unavailable. |
| 5. Comparable paired experiments | Matching IDs did not ensure matching timestep, beta, classifier, preparation labels or hold windows. | Check the full configuration and actual preparation/provenance/ledger metadata, with only declared subset-size and energy-retuning exceptions. Verify all independent reductions against the ledger. |
| 6. Current-generation analysis artifacts | Empty results left old CSVs or figures, and an old PDF could be packaged with new analysis. | Generate the complete managed output set in a staging directory; replace it only after generation succeeds. Empty outputs and no-success plots are explicit. Invalidate the fixed-name generated PDF on successful reanalysis. |
| 7. Independent archive validation | Nonstandard JSON numbers and inconsistent audit diagnostics could bypass the packaging checks. | Independently reject nonfinite/duplicate-key JSON, invalid counts, window/grid metadata and inconsistent reductions before publishing an archive. Preserve checksum verification and no-overwrite publication. |

## Compatibility and data semantics

Existing physically valid requests at the three registered production points
retain their beta values and random-number identities. Numerical tests compare
the original and new launch phase spaces element by element. The EBK action
equation, normal-mode splitting and outcome classifier are unchanged.

For P=1 QCT outside the paper-beta domain, the fallback is `1500 au`, recorded
with convention `qct_single_bead_placeholder_1500_au`. There are no internal
ring modes or spring forces at P=1; this auxiliary value is not a physical
temperature. Explicit beta overrides retain priority. A finite positive
**actual** collision energy is required before the fallback is usable. NE-RPMD
continues to require a valid auxiliary beta for its sampled ring ensemble.

The analyzer adds two attempt-level tables:

- `audit_cases.csv`: attempts, prepared and failed for each completed audit
  case, including a case with no successful preparation.
- `audit_paired_outcomes.csv`: attempted and successful pairs plus failures on
  either or both sides, including comparisons with no usable paired mean.

Hold means remain conditional on successful preparation. Missing or malformed
diagnostics in a row labeled `prepared` cause an error; they are not silently
dropped or relabeled. A single independent observation has a mean but no
estimable SEM. No successful observations means no estimated mean. Collision
failures and unresolved trajectories remain in all cross-section denominators
and unknown-outcome bounds.

Historical reports without `audit_cases` can still be rendered if their
statistics unambiguously contain all twelve audit cases with consistent counts.
If a case is missing, the renderer requires reanalysis rather than guessing
that no attempts occurred. Managed CSVs and plots are always regenerated;
reanalysis removes only the fixed-name `pyrpmd_state_preparation_report.pdf`,
which must then be rendered from the new JSON. Custom files are preserved.

## Reproduce the checks

The combined preparation and core state-selected regression run on
2026-10-31 passed **263 tests and 142 subtests**, with no skips. A read-only
reanalysis of the frozen study compared **10,074 existing numerical fields**
with the previous results and found **zero differences at tolerance 1e-12**.
It retained 21 cross-section estimates, 12 collision contrasts, 744 audit
statistics, 54 paired audit means, 15 numerical diagnostic rows and all six
unresolved endpoints. The new archive validators independently accepted all
12 audit ledgers and their 6,144 records. No production dynamics were rerun
and no old archive was rewritten.

Build the native extension and BKMP2 according to [SOURCE.md](SOURCE.md), then
run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
python -m pytest -q \
  tests/test_h3_diatomic_action.py tests/test_h3_diatomic_switch.py \
  tests/test_h3_preparation_study.py tests/test_h3_preparation_audit.py \
  tests/test_h3_preparation_analysis.py tests/test_h3_preparation_archive.py \
  tests/test_h3_preparation_report.py \
  tests/test_state_selected_input.py tests/test_state_selected_workflow.py \
  tests/test_state_selected_native.py tests/test_state_selected_check.py
```

The numerical gate includes real BKMP2, harmonic and Morse analytic checks,
an independent DOP853 solution, and normal-mode convergence for P=1,5,32,64.
Analysis/archive tests use tiny synthetic ledgers, including all-failed cases,
wrong window metadata, invalid JSON, relocated data and output-generation
failures. The optional PDF test requires ReportLab/Pillow and pypdf; CJK visual
verification uses an installed Chinese font as described in the protocol.

Two bounded end-to-end smoke calculations exercise the repaired production
entry points. Use fresh output directories; these samples do not enter the
registered study or provide cross-section estimates:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.h3.state_preparation_study \
  --kind action_qct --conditioning none --energy-convention fixed_total \
  --v 1 --etot 0.800 --samples 4 --id-offset 70000 --workers 2 \
  --output runs/preparation_hardening_smoke/action_qct_v1_0p800

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.h3.preparation_audit \
  --samples 2 --id-offset 920000 --point v1:0.90 \
  --variant switch200_none --workers 2 \
  --output runs/preparation_hardening_smoke/audit_switch200
```

The observed near-threshold launch has EBK internal energy
`0.7851776048192163 eV` and collision energy about `0.01482239518 eV`.
All four collision attempts completed as nonreactive, with no failed or
unresolved trajectories. Both P=32 audit preparations completed the 200 fs
switch and 100 fs hold. These are execution checks, not statistically useful
reaction probabilities or new claims of paper reproduction.

## Remaining boundaries

The benchmark runner still uses a one-shot output directory. Infrastructure
interruption retains completed JSONL lines but does not provide per-case resume;
the core dRPMD checkpoint workflow is separate. Failed or unresolved attempts
are not retried until a desired scientific result appears.

Historical run source snapshots and the original data archive remain immutable.
Reanalysis can read relocated run data, but repacking different producing-code
and analysis-code revisions requires separate provenance identities; the
existing single-code-commit archive contract is not bypassed by editing hashes.

Smooth switching remains experimental. Numerical completion, small work-balance
residuals and an energy mean near a quantum reference do not establish a pure
vibrational state. Bead, timestep, preparation-time, impact-parameter and
sampling convergence, and a validated three-fragment classifier, remain
separate scientific work described in [the study protocol](STATE_PREPARATION_STUDY.md).
