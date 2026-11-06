---
orphan: true
---

# NP6 materials/diffusion validation (2026-11-06)

This record covers the NP6 route on the PyRPMD checkout used for the pull
request.  It is an implementation check for the E0 legacy scheduling changes
and the E2 native diffusion opt-in; it is not a physical convergence study.

## Scope and environment

- Source: `chen-yu-hao/PyRPMD`, branch `agent/np6-materials-route`, based on
  the latest `main` after NP0, NP1 and NP2.
- Python: `Python 3.12.14`.
- Native extension: the checked-in extension reports
  `MOMENTUM_REFRESH_API=1` and `SAMPLING_BLOCK_API=1`.
- Test models: built-in free and harmonic controls only.  No claim is made
  about a production material potential, long-time diffusion, finite-size
  effects, or statistical convergence.

## Required commands

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests -k "materials or diffusion or native_pimd or pimd"
python -m pytest -q -m "not campaign"
```

The pull request description includes the raw pytest summary lines from the
same environment.  If a local environment lacks pytest or a compiled native
extension, report that fact rather than substituting a different test result.

The focused NP6 checks completed with these raw summaries:

```text
tests/test_np6_materials_native.py: 4 passed in 4.28s
tests/test_materials_native_blocks.py: 12 passed in 4.47s
tests/test_materials_workflows.py tests/test_materials_extensions.py tests/test_materials_stream_failures.py: 72 passed in 9.28s
```

The required targeted command completed with one expected stale assertion:

```text
1 failed, 466 passed, 43 skipped, 11132 deselected, 9 subtests passed in 56.27s
```

The only failure was the pre-NP6 assertion that native diffusion must be
rejected (`tests/test_native_pimd_followup.py::test_no_unsupported_native_workflow_fallback[change0]`).
The full non-campaign suite also completed its runnable tests; its two
environment/coordination failures were:

```text
2 failed, 9450 passed, 228 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1611.89s (0:26:51)
```

One failure is the pre-NP6 assertion
`test_native_pimd_followup.py::test_no_unsupported_native_workflow_fallback[change0]`,
which still expects every native `task="diffusion"` input to be rejected.  It
conflicts directly with the NP6 E2 route and needs the integrator's test update;
no production code path was hidden or skipped to make that assertion pass.  The
other is the sandbox's unrelated Unix-domain socket restriction in
`tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`
(`PermissionError: [Errno 1] Operation not permitted`).

## E0 legacy checks

The Andersen and eligible PILE-L paths were compared with their forced Python
fallbacks on the same fixed-cell built-in free-particle input.  The committed
sample arrays (`positions`, `velocities`, `observables`, `radii_gyration_squared`,
`sample_steps`, and `times`) matched exactly for the same seed and block
boundaries.  The tests also check a diffusion handoff at `neq`, where the first
production frame must retain the legacy observation clock.  Checkpoint/RNG
comparison and interrupted/resume checks use the same output schema; a changed
block length is an execution choice and must not change the committed E0
trajectory.

For the short controls, the legacy summary reports native sampling/production
blocks with `python_steps = 0` when the native extension is eligible.  Native
call counts are reported in the summary's `propagation` records; they are not
wall-clock performance claims for a real PES.

## E2 native diffusion checks

`examples/ensembles/diffusion_native.inp` selects fixed-cell native PILE-L
equilibration followed by native unthermostatted production.  The smoke run
checks that:

- the resolver rejects NPT, trajectory batching, unsupported thermostats and
  invalid production lengths before creating output;
- the output contains native journals, checkpoints, `trajectory.npz`, and
  `results/diffusion.json`;
- `propagation.python_steps == 0` and native block counts are nonzero;
- a completed `resume` reanalyzes the journals without rebuilding the PES; and
- the trajectory and summary use the materials diffusion field names.

The free-particle control is expected to report ballistic/drifting behavior
rather than a converged finite diffusion coefficient.  A harmonic control is
useful for a bounded-MSD regression, but neither is an empirical material
benchmark.

## Ten-thousand-step scheduling control

The following short timing control uses one free H atom, `P=4`,
`dt=0.1 fs`, `sample_interval=1000`, `checkpoint_steps=1000`,
`block_steps=256`, one replica and in-memory output.  The Python and native
timings were separate fresh runs on the validation host.  Native call counts
were measured by wrapping the corresponding extension entry point in a second
run; the wrapper itself is excluded from the reported timing.

| Legacy sampler | Backend | Steps | Native block calls | Python steps | Wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| PILE-L | `loop_backend="python"` | 10,000 | 0 | 10,000 | 4.567 |
| PILE-L | `loop_backend="auto"` | 10,000 | 40 `advance_sampling` | 0 | 0.093 |
| Andersen (`andersen_interval=10`) | `loop_backend="python"` | 10,000 | 0 | 10,000 | 4.920 |
| Andersen (`andersen_interval=10`) | `loop_backend="auto"` | 10,000 | 1,000 `advance` | 0 | 0.556 |

These numbers are implementation evidence for event/block scheduling, not a
portable performance claim.  Hardware, compiler, native extension build,
filesystem and optional providers can change wall time.  The call-count
invariant is the portable part: PILE-L is bounded by the 256-step block and
event boundaries, while Andersen stops at each historical refresh boundary.

## Limits and follow-up

Native diffusion currently supports fixed-cell PILE-L equilibration and the
native NVE production engine.  Native NPT remains under `task="pimd"`; rate,
FES, metadynamics, GPU/bead-MPI, trajectory batching and bond constraints are
outside NP6.  Existing tests that assert native diffusion is unsupported need
an integrator update in the branch that consumes this PR; those assertions are
not evidence that the new route failed.
