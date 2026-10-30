"""A two-atom BIND(C)-compatible provider works through both native APIs."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.cv import Distance
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials.native import NativePESBinding


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable"
)


@pytest.fixture(scope="module")
def two_atom_library(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("two-atom-fortran-batch")
    source = directory / "pes.c"
    source.write_text(r'''
#include <stdint.h>
/* Same ISO_C_BINDING signature as a Fortran batch provider; fixed N=2. */
void batch(int32_t count, const double* q, double* e, double* f, int32_t* status) {
    for (int32_t b = 0; b < count; ++b) {
        e[b] = 0.; status[b] = 0;
        for (int j = 0; j < 6; ++j) {
            int i = 6*b+j;
            e[b] += .5*q[i]*q[i];
            f[i] = -q[i];
        }
    }
}
''')
    library = directory / "pes.so"
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


class TwoAtomPotential:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(Path(path)))
        self.function = self.library.batch
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        statuses = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int32, doubles, doubles, doubles, statuses]
        self.function.restype = None
        self.lock = threading.RLock()

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.function, 2, self, self.lock)

    def evaluate(self, q):
        assert q.shape[1:] == (2, 3)
        energy = np.empty(len(q))
        force = np.empty_like(q)
        status = np.empty(len(q), dtype=np.int32)
        with self.lock:
            self.function(len(q), q, energy, force, status)
        return energy, force, status


def initial_state():
    rng = np.random.default_rng(110)
    q = rng.normal(scale=.1, size=(4, 2, 3))
    q[:, 1, 0] += 1.
    return State(q, rng.normal(scale=.2, size=q.shape))


@pytest.mark.parametrize("api", ["atomic", "prefix"])
def test_generic_fortran_binding_matches_python_for_both_apis(two_atom_library, api):
    dyn = Dynamics(TwoAtomPotential(two_atom_library), [2., 3.], 5., 4, .003)
    a = initial_state()
    b = a.copy()
    cv = Distance([0, 1])
    if api == "atomic":
        native = dyn.propagate(a, 17, backend="native", observe=cv)
        reference = dyn.propagate(b, 17, backend="python", observe=cv)
    else:
        native = dyn.propagate_block(a, 17, backend="native", cv=cv)
        reference = dyn.propagate_block(b, 17, backend="python", cv=cv)
    assert native.backend == "native"
    np.testing.assert_allclose(a.q, b.q, atol=3e-13, rtol=1e-12)
    np.testing.assert_allclose(a.p, b.p, atol=3e-13, rtol=1e-12)
    np.testing.assert_allclose(native.energies, reference.energies, atol=3e-12)
    np.testing.assert_allclose(native.values, reference.values, atol=3e-13)
    assert a.step == b.step == 17
    assert a.time == b.time


def test_generic_fortran_binding_thermalization_preserves_rng(two_atom_library):
    dn = Dynamics(TwoAtomPotential(two_atom_library), [2., 3.], 5., 4, .003,
                  seed=991, loop_backend="native", block_steps=4)
    dp = Dynamics(TwoAtomPotential(two_atom_library), [2., 3.], 5., 4, .003,
                  seed=991, loop_backend="python", block_steps=7)
    a = initial_state()
    b = a.copy()
    dn.thermalize(a, 19, interval=5)
    dp.thermalize(b, 19, interval=5)
    np.testing.assert_allclose(a.q, b.q, atol=3e-13, rtol=1e-12)
    np.testing.assert_allclose(a.p, b.p, atol=3e-13, rtol=1e-12)
    assert dn.rng.bit_generator.state == dp.rng.bit_generator.state
    assert a.step == b.step == 19
    assert a.time == b.time
    assert dn.propagation_stats["native_steps"] == 19
