# NP2 sampling-block validation (2026-11-05)

This record compares the existing one-step PILE-L path with the native
sampling block on the built-in harmonic potential. Each row uses the same
initial `q`, `p`, and NumPy seed for the two measurements. The reported values
are wall-clock seconds per step for 32 steps; the block call uses
`record_energy=False` so the timing isolates noise generation and native
propagation. The call count is measured by wrapping the native entry point.

The managed execution environment exposed Linux 6.18.57 on x86-64, but did not
expose a CPU model or a reliable CPU-count value. The runs used Python 3.12.14,
NumPy 2.3.5, and the default OpenMP build. BKMP2 was not built or measured, so
these are harmonic microbenchmarks rather than production chemistry results.

| P | N | one-step (s/step) | native block (s/step) | native calls | completed | Python steps |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 1.96557250e-04 | 1.63926875e-05 | 1 | 32 | 0 |
| 4 | 64 | 2.06093594e-04 | 6.88412188e-05 | 1 | 32 | 0 |
| 8 | 3 | 1.44402313e-04 | 2.24012499e-05 | 1 | 32 | 0 |
| 8 | 64 | 2.59150781e-04 | 8.77691875e-05 | 1 | 32 | 0 |
| 32 | 3 | 1.37733406e-04 | 2.68343751e-05 | 1 | 32 | 0 |
| 32 | 64 | 6.41341969e-04 | 4.20438063e-04 | 1 | 32 | 0 |

Reproduce the table with:

```bash
PYTHONPATH=src python - <<'PY'
import time
import numpy as np
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.sampling_block import propagate_sampling_block

for beads, atoms in ((4, 3), (4, 64), (8, 3), (8, 64), (32, 3), (32, 64)):
    rng = np.random.default_rng(9000 + beads + atoms)
    q0 = rng.normal(scale=0.05, size=(beads, atoms, 3))
    p0 = rng.normal(scale=0.05, size=q0.shape)

    def make():
        return Dynamics(HarmonicPotential(0.4), np.ones(atoms), 3.0,
                        beads, 0.01, native=True, seed=17)

    # Warm up a separately constructed instance; all measured instances below
    # receive fresh copies of the same initial arrays.
    warm = make()
    propagate_sampling_block(warm, State(q0, p0), 2, record_energy=False)

    one_step = make()
    state = State(q0, p0)
    start = time.perf_counter()
    for _ in range(32):
        one_step.sample_step(state)
    one_step_time = (time.perf_counter() - start) / 32

    block = make()
    state = State(q0, p0)
    native = block.integrator._native
    original = native.advance_sampling
    calls = [0]
    def counted(*args):
        calls[0] += 1
        return original(*args)
    native.advance_sampling = counted
    start = time.perf_counter()
    result = propagate_sampling_block(block, state, 32, record_energy=False)
    block_time = (time.perf_counter() - start) / 32
    native.advance_sampling = original
    assert result.backend == "native" and result.completed == 32
    assert block.propagation_stats["python_steps"] == 0
    print(beads, atoms, one_step_time, block_time, calls[0],
          result.completed, block.propagation_stats["python_steps"])
PY
```

The correctness gate is `tests/test_sampling_block.py` plus the native sampling
tests. The no-bias cases compare `q`, `p`, clock, returned energies, and the
NumPy generator state with repeated one-step calls; fixed atoms, masks, the
unsupported `P=12` fallback, and a reduced noise budget are covered separately.
