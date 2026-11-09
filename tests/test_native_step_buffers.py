"""Double-buffered native steps, shared reaction coordinates and force reuse.

The native loop keeps its accepted state in either the caller's q/p arrays or
a private buffer. These checks exercise both parities of the final commit, a
refresh source aliasing the caller's momenta, the single reaction-coordinate
evaluation shared by umbrella, flux metric and observation, and endpoint-force
reuse in the whole-block ``Dynamics.propagate`` API.
"""
import ctypes
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
from pyrpmd.cv import CVStack, Distance, RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.propagation import PropagationError

pytestmark = pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("C compiler required")
    root = tmp_path_factory.mktemp("native-step-buffers")
    source = root / "pes.c"
    source.write_text(r'''
#include <stdint.h>
static int calls=0, fail_at=-1;
void reset(int fail) {calls=0;fail_at=fail;}
int get_calls(void) {return calls;}
void batch(int32_t B, const double* q, double* e, double* f, int32_t* status) {
  ++calls;
  for(int32_t b=0;b<B;++b){
    e[b]=0;status[b]=calls==fail_at?3:0;
    for(int j=0;j<9;++j){int i=9*b+j;e[b]+=.005*q[i]*q[i];f[i]=-.01*q[i];}
  }
}
''')
    library = root / "pes.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


class CountingPotential:
    def __init__(self, library, metadata=None):
        self.lib = ctypes.CDLL(str(library))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.fn = self.lib.batch
        self.fn.argtypes = [ctypes.c_int32, pointer, pointer, pointer,
                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
        self.fn.restype = None
        self.lib.reset.argtypes = [ctypes.c_int]
        self.lib.get_calls.restype = ctypes.c_int
        self.lock = threading.RLock()
        self.metadata = dict(metadata or {})
        self.lib.reset(-1)

    def evaluate(self, q):
        e = np.empty(len(q));f = np.empty_like(q);status = np.empty(len(q), dtype=np.int32)
        with self.lock:
            self.fn(len(q), q, e, f, status)
        return e, f, status

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.fn, 3, self, self.lock)


MASSES = np.array([1800., 1800., 1800.])
BETA = 220.


def start(beads):
    rng = np.random.default_rng(311)
    q = np.repeat(np.array([[-2., .1, 0], [0, 0, 0], [1.7, .2, .1]])[None], beads, axis=0)
    q += rng.normal(scale=.01, size=q.shape)
    return State(q, rng.normal(size=q.shape))


def coordinate():
    return RPMDrateCoordinate([[0], [1, 2]], MASSES, np.array([[-1.75, 0, 0], [0, 0, 0], [1.75, 0, 0]]),
                              [[0, 1]], [[1, 2]], 10, equivalent_paths=[[0, 2, 1]])


def umbrella_flux(cv):
    return CompositeBias(UmbrellaBias(cv, [0.4], 0.2), FluxMetricBias(cv, MASSES, BETA))


@pytest.mark.parametrize("beads", [1, 3, 8])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_failure_after_odd_or_even_accepted_steps_commits_exact_prefix(native_library, beads, accepted):
    pes = CountingPotential(native_library)
    dyn = Dynamics(pes, MASSES, BETA, beads, .2, seed=17)
    cv = coordinate();bias = umbrella_flux(cv)
    reference = start(beads)
    expected = dyn.propagate_block(reference, accepted, bias=bias, cv=cv, check_energy=True,
                                   backend="native", record_states=True)
    state = start(beads);q_buffer, p_buffer = state.q, state.p
    # One initial force, then one endpoint force per step: fail step accepted+1.
    pes.lib.reset(accepted + 2)
    with pytest.raises(PropagationError) as caught:
        dyn.propagate_block(state, 9, bias=bias, cv=cv, check_energy=True,
                            backend="native", record_states=True)
    result = caught.value.result
    assert result.completed == accepted and pes.lib.get_calls() == accepted + 2
    # The caller's arrays are updated in place and hold the accepted prefix.
    assert state.q is q_buffer and state.p is p_buffer
    np.testing.assert_array_equal(state.q, reference.q)
    np.testing.assert_array_equal(state.p, reference.p)
    for name in ("values", "energies", "centroids", "q", "p"):
        np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))
    assert state.step == reference.step and state.time == reference.time


@pytest.mark.parametrize("steps", [1, 2, 5, 6])
def test_block_parities_match_single_steps(native_library, steps):
    pes = CountingPotential(native_library)
    dyn = Dynamics(pes, MASSES, BETA, 4, .2, seed=17)
    cv = coordinate();bias = umbrella_flux(cv)
    whole, stepped = start(4), start(4)
    block = dyn.propagate_block(whole, steps, bias=bias, cv=cv, backend="native")
    values = [dyn.propagate_block(stepped, 1, bias=bias, cv=cv, backend="native").values for _ in range(steps)]
    np.testing.assert_array_equal(whole.q, stepped.q)
    np.testing.assert_array_equal(whole.p, stepped.p)
    np.testing.assert_array_equal(block.values, np.concatenate(values))


def test_refresh_momenta_aliasing_the_caller_momenta(native_library):
    pes = CountingPotential(native_library)
    dyn = Dynamics(pes, MASSES, BETA, 4, .2, seed=17)
    aliased, copied = start(4), start(4)
    offsets = np.array([3], dtype=np.int64)
    # A refresh source viewing state.p must still supply the *initial* values.
    dyn.propagate_block(aliased, 6, backend="native",
                        refresh={"offsets": offsets, "momenta": aliased.p[None]})
    dyn.propagate_block(copied, 6, backend="native",
                        refresh={"offsets": offsets, "momenta": copied.p[None].copy()})
    np.testing.assert_array_equal(aliased.q, copied.q)
    np.testing.assert_array_equal(aliased.p, copied.p)


def test_shared_coordinate_observations_match_python_values(native_library):
    pes = CountingPotential(native_library)
    dyn = Dynamics(pes, MASSES, BETA, 4, .2, seed=17)
    cv = coordinate()
    # The flux metric uses an equal but distinct coordinate object, and a
    # second, unrelated observation keeps the ordinary evaluation.
    bias = CompositeBias(UmbrellaBias(cv, [0.4], 0.2), FluxMetricBias(coordinate(), MASSES, BETA))
    observe = CVStack([cv, Distance([0, 2])])
    result = dyn.propagate_block(start(4), 12, bias=bias, cv=observe, backend="native",
                                 record_states=True)
    expected = np.array([observe.value(q) for q in result.q])
    np.testing.assert_allclose(result.values, expected, rtol=1e-13, atol=1e-13)
    reference = dyn.propagate_block(start(4), 12, bias=bias, cv=observe, backend="python",
                                    record_states=True)
    np.testing.assert_allclose(result.q, reference.q, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.values, reference.values, rtol=0, atol=1e-11)


@pytest.mark.parametrize("stateful", [False, True])
def test_propagate_reuses_endpoint_forces_unless_stateful(native_library, stateful):
    pes = CountingPotential(native_library, {"stateful": True} if stateful else None)
    dyn = Dynamics(pes, MASSES, BETA, 4, .2, seed=17)
    cv = coordinate();bias = umbrella_flux(cv)
    state = start(4)
    pes.lib.reset(-1)
    result = dyn.propagate(state, 7, bias=bias, observe=cv, sample_interval=2, backend="native")
    assert result.backend == "native"
    assert pes.lib.get_calls() == (14 if stateful else 8)
    # The reference schedule gives the identical trajectory and records.
    reference_pes = CountingPotential(native_library, {"stateful": True})
    reference_dyn = Dynamics(reference_pes, MASSES, BETA, 4, .2, seed=17)
    reference = start(4)
    expected = reference_dyn.propagate(reference, 7, bias=bias, observe=cv, sample_interval=2, backend="native")
    np.testing.assert_array_equal(state.q, reference.q)
    np.testing.assert_array_equal(state.p, reference.p)
    np.testing.assert_array_equal(result.energies, expected.energies)
    np.testing.assert_array_equal(result.values, expected.values)
    assert (result.initial_energy, result.minimum_energy, result.maximum_energy) == (
        expected.initial_energy, expected.minimum_energy, expected.maximum_energy)
