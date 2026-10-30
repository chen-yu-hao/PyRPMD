"""Selected maps retain nonlinear RATTLE and real Fortran-PES semantics.

The harmonic batch fixture isolates the constraint map from chemistry. The
optional BKMP2 cases exercise the pinned authors' PES after the real-H3 CI
build; they skip explicitly when that independent library is absent.
"""
from pathlib import Path
import ctypes
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.cv import CVStack, Distance, RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State, INTEGRATORS, native_capabilities
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.workflows.rate import project_positions, project_momenta


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable"
)


class ThreeAtomHarmonic:
    """Compiled fixture with the same BIND(C) batch signature as Fortran."""

    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.batch
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        statuses = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int32, doubles, doubles, doubles, statuses]
        self.function.restype = None
        self.lock = threading.RLock()

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.function, 3, self, self.lock)

    def evaluate(self, q):
        assert q.shape[1:] == (3, 3)
        energy = np.empty(len(q))
        force = np.empty_like(q)
        status = np.empty(len(q), dtype=np.int32)
        with self.lock:
            self.function(len(q), q, energy, force, status)
        return energy, force, status


@pytest.fixture(scope="module")
def three_atom_pes(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("integrator-rattle-pes")
    source = directory / "pes.c"
    source.write_text(r'''
#include <stdint.h>
void batch(int32_t count, const double* q, double* e, double* f, int32_t* status) {
    for (int32_t b = 0; b < count; ++b) {
        e[b] = 0.; status[b] = 0;
        for (int j = 0; j < 9; ++j) {
            int i = 9*b+j;
            e[b] += .005*q[i]*q[i];
            f[i] = -.01*q[i];
        }
    }
}
''')
    library = directory / "pes.so"
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True)
    return ThreeAtomHarmonic(library)


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_nonlinear_centroid_rattle_matches_python_and_is_reversible(three_atom_pes, scheme):
    beads, beta, dt, count = 8, 220., .2, 25
    masses = np.full(3, 1800.)
    coordinate = RPMDrateCoordinate(
        [[0], [1, 2]], masses,
        [[-1.75, 0., 0.], [0., 0., 0.], [1.75, 0., 0.]],
        [[0, 1]], [[1, 2]], 10., equivalent_paths=[[0, 2, 1]],
    )
    surface = coordinate.dividing_surface(1.001)
    rng = np.random.default_rng(149)
    q = np.repeat([[[-2., .1, 0.], [0., 0., 0.], [1.7, .2, .1]]], beads, axis=0)
    q += rng.normal(scale=.01, size=q.shape)
    initial = State(q, rng.normal(size=q.shape))
    initial.q[:] = project_positions(initial.q, surface, masses)
    project_momenta(initial.p, initial.q, surface, masses)
    # Guard the intended coverage: this surface has curvature at the start.
    assert np.linalg.norm(surface.centroid_value_gradient_hessian(initial.q.mean(0))[2]) > .1
    native = Dynamics(three_atom_pes, masses, beta, beads, dt, integrator=scheme)
    python = Dynamics(three_atom_pes, masses, beta, beads, dt,
                      integrator=scheme, native=False)
    a, b = initial.copy(), initial.copy()
    na = native.propagate_block(a, count, backend="native", constraint=surface,
                                cv=surface, record_states=True)
    py = python.propagate_block(b, count, backend="python", constraint=surface,
                                cv=surface, record_states=True)
    assert na.backend == "native" and py.backend == "python"
    np.testing.assert_allclose(na.q, py.q, atol=3e-11, rtol=0)
    np.testing.assert_allclose(na.p, py.p, atol=3e-10, rtol=0)
    np.testing.assert_allclose(na.energies, py.energies, atol=3e-12, rtol=0)
    for result in (na, py):
        assert np.max(np.abs(result.values)) < 1e-10
        tangent = [np.sum(surface.centroid_value_gradient(q.mean(0))[1] *
                          p.mean(0) / masses[:, None])
                   for q, p in zip(result.q, result.p)]
        assert np.max(np.abs(tangent)) < 1e-12
    assert a.step == b.step == count and a.time == b.time
    for dynamics, final, backend in ((native, a, "native"), (python, b, "python")):
        final.p *= -1
        dynamics.propagate_block(final, count, backend=backend, constraint=surface)
        np.testing.assert_allclose(final.q, initial.q, atol=3e-11, rtol=0)
        np.testing.assert_allclose(final.p, -initial.p, atol=3e-10, rtol=0)


@pytest.mark.skipif(
    not (ROOT / "benchmarks/h3/build/libbkmp2.so").is_file(),
    reason="optional real BKMP2 PES has not been built",
)
@pytest.mark.parametrize("scheme", INTEGRATORS)
@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_real_bkmp2_selected_integrators_match_independent_python(scheme, api):
    from benchmarks.h3.pes import BKMP2Potential
    from benchmarks.loop_native.benchmark import initial, MASS_H, BETA, DT_AU

    beads, count = 16, 23
    q, p = initial(beads)
    a, b = State(q, p), State(q, p)
    native = Dynamics(BKMP2Potential(), MASS_H, BETA, beads, DT_AU, integrator=scheme)
    python = Dynamics(BKMP2Potential(), MASS_H, BETA, beads, DT_AU,
                      integrator=scheme, native=False)
    cv = CVStack([Distance([0, 1]), Distance([1, 2])])
    options = {"observe": cv} if api == "propagate" else {"cv": cv, "record_states": True}
    na = getattr(native, api)(a, count, backend="native", **options)
    py = getattr(python, api)(b, count, backend="python", **options)
    assert na.backend == "native" and py.backend == "python"
    np.testing.assert_allclose(a.q, b.q, atol=3e-11, rtol=1e-11)
    np.testing.assert_allclose(a.p, b.p, atol=3e-9, rtol=1e-10)
    np.testing.assert_allclose(na.values, py.values, atol=3e-11, rtol=1e-11)
    np.testing.assert_allclose(na.energies, py.energies, atol=2e-10, rtol=1e-11)
    np.testing.assert_array_equal(na.steps, py.steps)
    assert a.step == b.step == count and a.time == b.time
    assert np.isfinite(na.energies).all()
