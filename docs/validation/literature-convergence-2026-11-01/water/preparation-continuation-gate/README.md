# Preparation-only continuation: implementation gate

**Prepared for independent review; no long preparation extension launched.**
The new runner is separate from the frozen native reference and the original
campaign scheduler. It can resume additional preparation from an explicitly
selected original checkpoint without changing any original output. Its generic
single-chain interface does not approve the eight-chain experiment or settle
the proposed structural uncertainty policy.

## Contract at the branch point

The original native run must be complete. The configuration explicitly binds its
whole manifest and its unique final-preparation checkpoint by SHA256. The source,
Python/OpenMM environment, serialized model and integrator, exact checkpoint
step, box and `active_segment=-1` must match. Selecting a nearby frame or a chosen
production endpoint is rejected. The original files are only read.

For the proposed original 100 ps boundary at step 400000, the saved RNG has not
performed the NVE momentum draw. The continuation performs exactly one ordinary
preparation Maxwell draw with `remove_com=False`, then advances on the existing
absolute 400-step refresh grid. The checkpoint stores the next pending refresh
and draw count. Both on-boundary and between-boundary restarts preserve that
phase. The runner stops at the target preparation step **before** any terminal
refresh or NVE draw. There is no NVE phase in its code.

Restoring a Context reuses the frozen version 2 validation contract: raw engine
positions and half-step velocities must match exactly; recomputed physical
forces have independent platform-dependent roundoff bounds; the canonical
momentum difference must equal the predicted half-step force conversion. A
deliberately changed force cannot pass merely because its momentum was changed
consistently. CPU force-cache roundoff means this is not a promise of bitwise
future-trajectory replay.

New manifests, model copies, and chunks use a separate output directory. An
initial branch checkpoint records the pre-refresh state; every later commit
records the external RNG, preparation-only phase, redundant raw state, Context
checkpoint and observation/refresh steps. Source and original-manifest hashes
are rechecked before committing each new chunk. A completed resume returns
without creating a Context or consuming another random number.

## Focused tests

Nine real-engine tests passed in the initial gate. The subsequent combined gate
passed **17 tests in 7.92 s**: the same nine plus eight analytic RDF-noise checks.
These are 17 distinct tests, not 26. The real-engine fixtures use eight molecules
and a short 0.1 fs Reference/CPU trajectory; no 216-molecule continuation was run.

The nine cases verify:

- Branching at a pending refresh agrees with uninterrupted preparation in
  canonical coordinates, momenta and external RNG.
- Split restarts at the refresh boundary and midway through a refresh period
  preserve the exact next draw and final RNG.
- Every preparation draw omits COM projection; the terminal step consumes no
  NVE draw; a completed resume creates no Context.
- Configuration preparation reads and hashes files without creating an engine;
  changed explicit source identity is rejected before engine creation.
- A self-consistent alteration of the serialized force model is rejected against
  the fresh frozen factory model.
- An original checkpoint labeled after an NVE draw is rejected.
- Actual CPU restoration satisfies the raw-state contract, while deliberately
  altered forces fail its independent roundoff bound.
- A corrupted saved refresh deadline cannot resume.

[gate.json](gate.json) records the source hashes, exact JUnit hashes and the
verification that the original frozen water sources and compiled binaries remain
unchanged. [initial-nine-tests.xml](initial-nine-tests.xml) and
[continuation-and-rdf-pytest.xml](continuation-and-rdf-pytest.xml) retain both gates.

Run the focused gate from the repository root:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python -m pytest -q tests/test_water_preparation_extension.py tests/test_water_rdf_sampling_noise.py
```

## Reviewable interface and remaining adoption work

The following operation only prepares a checksum-bound configuration; it creates
no engine and advances no dynamics. It must point to a completed original run.

```sh
python benchmarks/water/native_preparation_extension.py \
  --source-run ORIGINAL_COMPLETED_RUN \
  --additional-preparation-ps 400 \
  --prepare-config NEW_CONFIG.json
```

After a separate scientific adoption and all-eight checkpoint selection receipt,
the execution interface is `--config NEW_CONFIG.json --output NEW_RUN_DIRECTORY`
with a bounded `--wall-seconds` or `--max-steps`; later invocations use `--resume`.
It must have one writer per output directory, enforced by a scheduling lease
when used in a campaign. This module does not replace the cohort-level lease and
adoption checks. No long command or real extension configuration was issued as
part of this implementation gate.

Before adopting a long campaign, the complete original eight-chain assessment,
independent code/phase review, and the
[revised RDF uncertainty policy](../equilibration-extension-proposal-v2.md)
remain necessary. Passing the short implementation tests does not establish
equilibration, transport convergence, or agreement with the paper.
