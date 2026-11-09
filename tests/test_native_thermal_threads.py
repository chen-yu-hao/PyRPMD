"""Observe native worker threads in real, isolated propagation processes."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd import _native
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble, ensemble_from_config
from pyrpmd.potentials import HarmonicPotential


def run_workers(api, threads, *, workers=1, omp=None):
    if not Path("/proc/self/task").is_dir():
        pytest.skip("actual process thread counts require Linux /proc")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OMP_DYNAMIC": "FALSE", "OMP_THREAD_LIMIT": "8"}
    env.pop("OMP_NUM_THREADS", None)
    if omp is not None:
        env["OMP_NUM_THREADS"] = str(omp)
    with ExitStack() as stack:
        processes = [stack.enter_context(subprocess.Popen(
            [sys.executable, __file__, api, str(threads)], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )) for _ in range(workers)]
        try:
            for process in processes:
                assert process.stdout.readline().strip() == "ready"
            for process in processes:
                process.stdin.write("go\n")
                process.stdin.flush()
            results = []
            for process in processes:
                output, errors = process.communicate(timeout=120)
                assert process.returncode == 0, errors
                results.append(json.loads(output))
            return results
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()


@pytest.mark.parametrize("api", ["native", "native-loop", "python-loop", "native-pimd", "legacy-pimd"])
@pytest.mark.parametrize("omp", [None, 4])
def test_one_thread_overrides_openmp_environment(api, omp):
    assert run_workers(api, 1, omp=omp) == [{"before": 1, "after": 1}]


@pytest.mark.skipif(not _native.has_openmp(), reason="requires OpenMP")
@pytest.mark.parametrize("api", ["native", "native-loop", "python-loop", "legacy-pimd"])
def test_requested_threads_reach_real_parallel_regions(api):
    assert run_workers(api, 2, omp=1) == [{"before": 1, "after": 2}]


def test_four_trajectory_processes_do_not_oversubscribe():
    assert run_workers("native-loop", 1, workers=4) == [{"before": 1, "after": 1}]*4


@pytest.mark.parametrize("threads", [True, 0, -1, 1.5, "2", None, 2**31])
def test_invalid_thread_count_is_rejected(threads):
    with pytest.raises(ValueError, match="threads"):
        NativeEnsemble(np.zeros((1, 1, 3)), [1.], 2., .01, threads=threads)


@pytest.mark.skipif(_native.has_openmp(), reason="requires a serial build")
def test_serial_build_rejects_multiple_threads():
    with pytest.raises(RuntimeError, match="OpenMP"):
        NativeEnsemble(np.zeros((1, 1, 3)), [1.], 2., .01, threads=2)


def test_native_input_threads_are_execution_only():
    config = {"positions": [[.1, .2, .3]], "masses": [1.], "beads": 8,
              "beta": 2., "timestep": .01, "seed": 49}
    default = ensemble_from_config(config)
    explicit = ensemble_from_config({**config, "threads": 2 if _native.has_openmp() else 1})
    default.run(3)
    explicit.restore(default.state())
    default.run(5)
    explicit.run(5)
    assert default.state()["digest"] == explicit.state()["digest"]


def worker(api, threads):
    # Bound the failing pre-fix case without depending on the host core count.
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    before = len(list(Path("/proc/self/task").iterdir()))
    q = np.full((32, 512, 3), .1)
    print("ready", flush=True)
    assert sys.stdin.readline().strip() == "go"
    if api == "native":
        engine = NativeEnsemble(q, np.ones(512), 2., .01, threads=threads,
                                potential={"kind": "harmonic", "k": .5})
        engine.run(3)
    elif api.endswith("-pimd"):
        from pyrpmd.runner import run
        config = {
            "task": "pimd", "temperature": "300 K", "beads": 32, "seed": 49,
            "system": {"symbols": ["H"]*512, "geometry": q[0].tolist(), "geometry_unit": "bohr"},
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .01}},
            "dynamics": {"timestep": "0.1 autime", "loop_backend": "auto"},
            "sampling": {"engine": api.removesuffix("-pimd"), "thermostat": "pile-l",
                         "equilibration_steps": 2, "production_steps": 4, "sample_interval": 2},
            "parallel": {"threads_per_rank": threads},
            "output": {"checkpoint_steps": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            run(config, output=Path(directory)/"run")
    else:
        dynamics = Dynamics(HarmonicPotential(k=.5), np.ones(512), 2., 32, .01,
                            threads=threads, thermostat="pile-l",
                            loop_backend=api.removesuffix("-loop"))
        dynamics.propagate(State(q, np.zeros_like(q)), 3)
    print(json.dumps({"before": before, "after": len(list(Path("/proc/self/task").iterdir()))}))


if __name__ == "__main__":
    worker(sys.argv[1], int(sys.argv[2]))
