"""Endpoint reuse, trusted validation and per-step energies in block propagation."""

import ctypes
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.block_propagation import propagate_block
from pyrpmd.dynamics import Dynamics, State, evaluate_checked, native_capabilities
from pyrpmd.potentials import PESResult, PotentialError
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.potentials.python import PythonPotential


MASSES = np.array([1.0, 2.0])


def _anharmonic(q):
    q = np.asarray(q, dtype=float)
    energy = 0.5 * np.sum(q * q, axis=(1, 2)) + 0.1 * np.sum(q ** 4, axis=(1, 2))
    return energy, -(q + 0.4 * q ** 3)


class Counting:
    def __init__(self, metadata=None, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
        if metadata is not None:
            self.metadata = metadata

    def evaluate(self, q):
        self.calls += 1
        energy, forces = _anharmonic(q)
        status = np.zeros(len(q), dtype=np.int32)
        if self.fail_after is not None and self.calls > self.fail_after:
            status[:] = 3
        return energy, forces, status


def _state(nbeads):
    rng = np.random.default_rng(4)
    q = 0.3 * rng.normal(size=(nbeads, 2, 3))
    p = 0.5 * rng.normal(size=(nbeads, 2, 3))
    return State(q, p)


def _dynamics(provider, nbeads=4, **options):
    return Dynamics(PythonPotential(provider), MASSES, 2.0, nbeads, 0.05,
                    loop_backend="python", **options)


@pytest.mark.parametrize("nbeads", [1, 4])
@pytest.mark.parametrize("reuse", [False, True])
def test_python_block_matches_reference_steps_bitwise(nbeads, reuse):
    reference_provider, provider = Counting(), Counting()
    reference = _dynamics(reference_provider, nbeads)
    dynamics = _dynamics(provider, nbeads)
    expected, actual = _state(nbeads), _state(nbeads)
    energies, centroids = [], []
    for _ in range(9):
        reference.step(expected)
        energies.append(reference.energy(expected))
        centroids.append(expected.q.mean(axis=0))
    result = propagate_block(dynamics, actual, 9, reuse_forces=reuse,
                             diagnostics={"energy_parts": True, "ring_radii": True},
                             record_step_energies=True)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    np.testing.assert_array_equal(result.energies, energies)
    np.testing.assert_array_equal(result.step_energies, energies)
    np.testing.assert_array_equal(result.centroids, centroids)
    # Reference: two force calls per step plus one energy call per observation.
    assert reference_provider.calls == 27
    # The endpoint energy is reused; cached endpoint forces skip the next start.
    assert provider.calls == (10 if reuse else 18)
    parts = result.diagnostics["energy_parts"]
    np.testing.assert_allclose(parts.sum(axis=1), np.asarray(energies) / nbeads, rtol=1e-12)
    if nbeads == 1:
        assert np.all(parts[:, 2] == 0) and np.all(parts[:, 3] == 0)
        assert np.all(result.diagnostics["ring_radii"] == 0)
        kinetic = float(np.sum(actual.p * actual.p / (2 * MASSES[None, :, None])))
        assert parts[-1, 1] == kinetic


def test_sparse_observation_keeps_every_step_energy():
    dense_provider, sparse_provider = Counting(), Counting()
    dense, sparse = _dynamics(dense_provider), _dynamics(sparse_provider)
    first, second = _state(4), _state(4)
    observe = np.zeros(12, dtype=bool)
    observe[[3, 11]] = True
    full = propagate_block(dense, first, 12, reuse_forces=True)
    result = propagate_block(sparse, second, 12, observe=observe, reuse_forces=True,
                             record_step_energies=True)
    np.testing.assert_array_equal(result.step_energies, full.energies)
    np.testing.assert_array_equal(result.energies, full.energies[observe])
    np.testing.assert_array_equal(result.centroids, full.centroids[observe])
    np.testing.assert_array_equal(second.q, first.q)
    assert sparse_provider.calls == dense_provider.calls == 13


def test_failed_step_with_cached_force_rolls_back_and_invalidates_cache():
    provider = Counting(fail_after=6)
    dynamics = _dynamics(provider)
    state = _state(4)
    reference = _dynamics(Counting())
    expected = _state(4)
    for _ in range(5):
        reference.step(expected)
    result = propagate_block(dynamics, state, 10, reuse_forces=True, raise_on_error=False,
                             record_previous=True, record_step_energies=True)
    # Calls: 1 initial + 1 endpoint per completed step; the 6th endpoint fails.
    assert result.completed == 5 and "PES frames failed" in result.error
    assert len(result.step_energies) == 5
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert state.step == 5
    provider.fail_after = None
    before = provider.calls
    propagate_block(dynamics, state, 1, reuse_forces=True)
    # A new block never inherits a cache: start and endpoint are evaluated.
    assert provider.calls - before == 2


def test_stateful_provider_keeps_the_reference_evaluation_schedule():
    provider = Counting(metadata={"stateful": True})
    dynamics = _dynamics(provider)
    propagate_block(dynamics, _state(4), 5, reuse_forces=True)
    assert provider.calls == 15


def test_composed_executor_reports_step_energies_for_a_sparse_mask():
    dynamics = Dynamics(PythonPotential(Counting()), MASSES, 2.0, 4, 0.05, native=False,
                        thermostat="pile-l", seed=3)
    other = Dynamics(PythonPotential(Counting()), MASSES, 2.0, 4, 0.05, native=False,
                     thermostat="pile-l", seed=3)
    first, second = _state(4), _state(4)
    observe = np.zeros(6, dtype=bool)
    observe[-1] = True
    full = propagate_block(dynamics, first, 6)
    sparse = propagate_block(other, second, 6, observe=observe, record_step_energies=True)
    np.testing.assert_array_equal(sparse.step_energies, full.energies)
    np.testing.assert_array_equal(sparse.energies, full.energies[-1:])
    np.testing.assert_array_equal(sparse.steps, full.steps[-1:])
    np.testing.assert_array_equal(second.q, first.q)


class FailingFrame:
    def evaluate(self, q):
        energy, forces = _anharmonic(q)
        return PESResult(energy, forces, np.full(len(q), 2, dtype=np.int32),
                         ("bad",) * len(q))


def test_trusted_adapter_still_rejects_failed_frames():
    with pytest.raises(PotentialError, match="status=2: bad"):
        evaluate_checked(PythonPotential(FailingFrame()), np.zeros((2, 2, 3)))


def test_overridden_or_unknown_providers_keep_full_validation():
    class Override(PythonPotential):
        def evaluate(self, x):
            return PESResult(np.zeros(len(x)), np.full(x.shape, np.nan))

    with pytest.raises(PotentialError, match="nonfinite"):
        evaluate_checked(Override(Counting()), np.zeros((1, 2, 3)))
    instance = PythonPotential(Counting())
    instance.evaluate = lambda x: PESResult(np.zeros(len(x)), np.full(x.shape, np.inf))
    with pytest.raises(PotentialError, match="nonfinite"):
        evaluate_checked(instance, np.zeros((1, 2, 3)))

    class Plain:
        def evaluate(self, x):
            return np.zeros(len(x)), np.zeros((len(x), 3, 3))

    with pytest.raises(PotentialError, match="shape mismatch"):
        evaluate_checked(Plain(), np.zeros((1, 2, 3)))


@pytest.fixture(scope="module")
def batch_library(tmp_path_factory):
    if not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("step-energy-batch")
    source = directory / "pes.c"
    source.write_text(r'''
#include <stdint.h>
void batch(int32_t count, const double* q, double* e, double* f, int32_t* status) {
    for (int32_t b = 0; b < count; ++b) {
        e[b] = 0.; status[b] = 0;
        for (int j = 0; j < 6; ++j) {
            int i = 6*b+j; double x = q[i];
            e[b] += .5*x*x + .1*x*x*x*x;
            f[i] = -(x + .4*x*x*x);
        }
    }
}
''')
    library = directory / "pes.so"
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


def test_native_step_energies_match_observed_energies(batch_library):
    library = ctypes.CDLL(str(batch_library))
    function = library.batch
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int32, doubles, doubles, doubles,
                         np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
    function.restype = None
    lock = threading.RLock()

    class Batch:
        def evaluate(self, q):
            q = np.ascontiguousarray(q)
            e, f, s = np.empty(len(q)), np.empty_like(q), np.empty(len(q), np.int32)
            function(len(q), q, e, f, s)
            return e, f, s

        def native_binding(self):
            return NativePESBinding("fortran-batch-v1", function, 2, self, lock)

    def make():
        return Dynamics(PythonPotential(Batch()), MASSES, 2.0, 4, 0.05)

    dense, sparse = make(), make()
    first, second = _state(4), _state(4)
    observe = np.zeros(10, dtype=bool)
    observe[[4, 9]] = True
    full = propagate_block(dense, first, 10, reuse_forces=True)
    result = propagate_block(sparse, second, 10, observe=observe, reuse_forces=True,
                             record_step_energies=True)
    assert full.backend == result.backend == "native"
    np.testing.assert_array_equal(result.step_energies, full.energies)
    np.testing.assert_array_equal(result.energies, full.energies[observe])
    np.testing.assert_array_equal(second.q, first.q)
