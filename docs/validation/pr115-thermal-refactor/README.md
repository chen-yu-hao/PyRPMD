# PR #115 thermal-kernel and observation refactor

## Scope and immutable baseline

This follow-up applies to PR #115 head `9559e39ef586f7c26b6a103aa2ed51484180a271`.
The input was its existing GitHub Actions source distribution (artifact
`13404157745`), not reconstructed source. Its ZIP SHA-256 is
`9fa1be7c3b302f65f54726e91823473b642cbafd20ba8c462a7e21d0f2074e8c`.
The validation here does **not** describe a merge with current main/#132.
This record describes local validation before publication; it does not claim
remote CI or merge acceptance. Earlier PR evidence is not superseded by a
claim that the original review machine now passes every performance gate.

## Implementation

- Keep legacy MT normal generation serial, mode-major and unchanged. Preserve
  four real transforms for the default stream and the existing packed Philox
  transform convention; no RNG default, ABI, physics fingerprint or checkpoint
  format changes.
- Hold a tile's modal positions and momenta in the existing per-worker scratch,
  rather than repeatedly scattering and gathering through full-ring modal arrays.
  Complete FFT tiles have a compile-time width and disjoint scratch annotations;
  partial tiles and fixed atoms retain their own paths.
- Prepare full-step PILE decay/amplitude coefficients once and reuse the stream
  descriptors. Skip the fixed-atom scan only when every atom is mobile.
- Preserve `sqrt(theta*m)` **inside** the original kick expression. An attempted
  cached-root optimization was rejected: it produced 138/270 FMA mismatches.
  The final expression passes the matched FMA baseline matrix exactly.
- Add a private capability-gated position-only observer to whole-block Dynamics.
  It still checks energy on every step and preserves endpoint extrema, clock,
  transaction and failure behavior. It no longer computes unrequested centroid
  momenta/forces or allocates unused radius slots. Direct NativeEnsemble output
  and ordinary valid-prefix observers are unchanged. Old extensions use the
  original observation path.
- Record width is `18 + 3*N` doubles with positions, or `18` without positions,
  versus the previous `18 + 7*N`. At N=4096, position records are 57.1% smaller.
  The temporary-buffer regression checks the actual byte budget for both the
  compact path and old-extension fallback, rather than assuming the old layout.

## Reproduced correctness checks

| Build | Byte-exact cross-version cases | Mismatches |
| --- | ---: | ---: |
| Default OpenMP thermal build | 270 | 0 |
| AVX2/FMA thermal build | 270 | 0 |
| Serial thermal build | 258 | 0 |

The matrices compare array bytes (including signed zeros), RNG text, complete
state digests and observations. They cover P=1/2/3/7/8/16/32/64/128,
full/partial tiles, fixed atoms, PILE-L/TRPMD, both RNGs and eligible
1/2/4-thread cases. FMA and serial each use their own identically built baseline.

118 original-baseline checkpoint fixtures replay exactly at **both** requested
1 and 4 threads, spanning other thermostats/barostats and both integrators.
The 33 newly added observer tests pass against default, FMA-thermal and
serial-thermal builds. The observer/buffer-budget suite reports `58 passed`.

The expanded focused suite reports:

```
11 failed, 514 passed, 1 skipped in 45.42s
```

All 11 failures are the pre-existing absolute OS-thread-count tests. The
unmodified baseline independently reports `11 failed, 12 passed, 1 skipped`
for that file in the same environment. With OMP_NUM_THREADS absent, this
container starts Python with six threads **before NumPy/PyRPMD imports**;
one-thread propagation preserves that count. The original assertions were
not relaxed and these failures are not counted as passes. The one skip is
the serial-build-only test in the OpenMP build.

## Same-host timings

AMD EPYC 9V74 container; five CPUs in affinity but a four-CPU cgroup quota.
Python 3.13.5, NumPy 2.3.5, GCC 14.2.0. Normal extension builds use repository
default flags (no -march=native or fast-math). Both versions are built with the
same compiler and dependencies. No tests or builds run alongside final timing.

Five alternating baseline/candidate subprocess pairs, three timing repetitions
per case, eight warm-up steps excluded. P=32 below. Large cases use 16-step
calls; small cases use additional fixed call repetitions to avoid sub-millisecond
measurements. Both versions perform exactly the same requested observations.
Rates below are the median of five per-round rates; speedup is their ratio.
All per-round paired ratios, including noisy or unfavorable rounds, remain in
`summary.json`; raw repeats are retained in the supplied repair bundle.

| API | N | RNG | Threads | Positions | Baseline steps/s | Refactor steps/s | Ratio |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| dynamics | 512 | mt19937_64 | 1 | True | 368.23 | 404.97 | 1.100x |
| dynamics | 4096 | mt19937_64 | 1 | True | 42.97 | 49.55 | 1.153x |
| dynamics | 4096 | philox-v1 | 1 | True | 45.44 | 54.86 | 1.207x |
| dynamics | 4096 | philox-v1 | 4 | True | 117.05 | 138.37 | 1.182x |
| dynamics | 4096 | mt19937_64 | 1 | False | 43.43 | 49.46 | 1.139x |
| native | 4096 | philox-v1 | 1 | True | 46.37 | 48.84 | 1.053x |
| native | 4096 | philox-v1 | 4 | True | 128.32 | 133.04 | 1.037x |

Default MT PILE-L/NVE throughput ratios on **this** machine are
74.41% (N=512) and
68.90% (N=4096).
The public Philox ratio of 4-thread to 1-thread throughput medians is
2.522x, but two of five individual rounds fall below 2.5x.
Thus this is not a robust all-round 2.5x gate pass. Public scaling was already
above 2.5x for the baseline on this different machine; do not attribute that
cross-machine difference to this patch.

The default MT two-step outside/native profile ratio is
7.48% (baseline 6.80%).
Absolute execution is faster, but this overhead *fraction* did not improve.
Small-control median throughput did not regress; individual rounds remain noisy.

## Reproduction

Build baseline and candidate separately with one matched Python/compiler setup:

```bash
(cd /path/to/baseline && python setup.py build_ext --inplace)
(cd /path/to/candidate && python setup.py build_ext --inplace)
python /path/to/candidate/benchmarks/efficiency/thermal_refactor_compare.py \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --rounds 5 --repeats 3 --block-steps 16 --output paired-benchmark.json
PYTHONPATH=/path/to/baseline/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python /path/to/candidate/benchmarks/efficiency/thermal_refactor_fingerprint.py baseline.json
PYTHONPATH=/path/to/candidate/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python /path/to/candidate/benchmarks/efficiency/thermal_refactor_fingerprint.py candidate.json --reference baseline.json
```

The bundle also preserves exact build commands, original stdout/stderr logs,
checkpoint-replay reports, all fingerprint manifests and source hashes. Its
raw artifacts are separate from this concise repository record.

## Outstanding validation and disclosure

Full non-campaign pytest, Windows/macOS, strict Sphinx, remote CI and the
original acceptance-machine benchmarks were not run. Integration with latest
main must preserve #132 and be rebuilt/retested; this patch is not a replacement
main tree. No GPU/MPI speedup, physical convergence or independent human review
is claimed. `physical_converged` remains false. This refactor and the recorded
local experiments were produced with AI assistance.
