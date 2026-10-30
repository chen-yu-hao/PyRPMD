"""Native descriptors must not bypass class-level Python mathematics."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
from pyrpmd.cv import Cartesian, COMDistance, CVStack, RPMDrateCoordinate
from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State, native_capabilities
from pyrpmd.potentials import CABIPotential, PythonPotential
from pyrpmd.potentials.python import import_object


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable"
)


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("native-class-overrides")
    cabi = directory / "harmonic.so"
    subprocess.run([
        compiler, "-shared", "-fPIC", "-I", str(ROOT / "native/include"),
        str(ROOT / "native/examples/harmonic_pes.c"), "-o", str(cabi),
    ], check=True)
    source = directory / "three_atom.c"
    source.write_text(r'''
#include <stdint.h>
void batch(int32_t count, const double* q, double* e, double* f, int32_t* status) {
    for (int32_t b = 0; b < count; ++b) {
        e[b] = 0.; status[b] = 0;
        for (int j = 0; j < 9; ++j) {
            int i = 9*b+j;
            e[b] += .5*q[i]*q[i];
            f[i] = -q[i];
        }
    }
}
''')
    fortran = directory / "three_atom.so"
    subprocess.run([
        compiler, "-shared", "-fPIC", str(source), "-o", str(fortran),
    ], check=True)
    return cabi, fortran


@pytest.fixture
def cabi_pes(libraries):
    pes = CABIPotential(libraries[0])
    yield pes
    pes.close()


@pytest.fixture
def fortran_adapter(libraries):
    # Exercise the real adapter methods with a small ISO_C_BINDING-signature
    # fixture. This is an adapter test, not a BKMP2 chemistry/provenance test.
    cls = import_object(str(ROOT / "benchmarks/h3/pes.py") + ":BKMP2Potential")
    pes = cls.__new__(cls)
    pes.lib = ctypes.CDLL(str(libraries[1]))
    pes.batch = pes.lib.batch
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    statuses = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    pes.batch.argtypes = [ctypes.c_int32, doubles, doubles, doubles, statuses]
    pes.batch.restype = None
    pes._lock = threading.RLock()
    return pes


def _state(natoms):
    if natoms == 1:
        q = np.array([[[.2, .1, -.1]]])
    else:
        q = np.array([[[-1.8, .1, -.2], [0., 0., 0.], [1.5, .2, .1]]])
    return State(q, np.full_like(q, .07))


def _coordinate():
    return RPMDrateCoordinate(
        [[0], [1, 2]], [1., 1., 1.],
        [[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
        [[0, 1]], [[1, 2]], 10.,
    )


def _advance(dyn, state, api, cv, bias=None, backend="auto"):
    options = {"observe" if api == "propagate" else "cv": cv}
    return getattr(dyn, api)(state, 3, bias=bias, backend=backend, **options)


CASES = (
    "dynamics_force", "dynamics_energy", "integrator_free_step",
    "cabi_evaluate", "python_evaluate", "cv_gradient", "cvstack_value",
    "umbrella_energy_force", "composite_energy_force", "fortran_evaluate",
    "rate_surfaces", "nested_com_hessian", "flux_energy_force",
)


def _changed_case(case, monkeypatch, cabi_pes, fortran_adapter):
    three_atoms = case in {
        "fortran_evaluate", "rate_surfaces", "nested_com_hessian", "flux_energy_force",
    }
    pes = fortran_adapter if three_atoms else cabi_pes
    if case == "python_evaluate":
        pes = PythonPotential(pes)
    natoms = 3 if three_atoms else 1
    cv = Cartesian(0, "x")
    bias = None
    if case == "cvstack_value":
        cv = CVStack([cv])
    if case in {"cv_gradient", "umbrella_energy_force", "composite_energy_force"}:
        bias = UmbrellaBias(cv, [.05], 2.)
        if case == "composite_energy_force":
            bias = CompositeBias(bias)
    if case in {"rate_surfaces", "nested_com_hessian", "flux_energy_force"}:
        coordinate = _coordinate()
        cv = coordinate.dividing_surface(.8)
        if case == "flux_energy_force":
            bias = FluxMetricBias(coordinate, [1., 1., 1.], 2.)
    targets = {
        "dynamics_force": (Dynamics, "_force"),
        "dynamics_energy": (Dynamics, "energy"),
        "integrator_free_step": (CayleyIntegrator, "free_step"),
        "cabi_evaluate": (CABIPotential, "evaluate"),
        "python_evaluate": (PythonPotential, "evaluate"),
        "cv_gradient": (Cartesian, "centroid_value_gradient"),
        "cvstack_value": (CVStack, "value"),
        "umbrella_energy_force": (UmbrellaBias, "energy_force"),
        "composite_energy_force": (CompositeBias, "energy_force"),
        "fortran_evaluate": (type(fortran_adapter), "evaluate"),
        "rate_surfaces": (RPMDrateCoordinate, "surfaces"),
        "nested_com_hessian": (COMDistance, "centroid_value_gradient_hessian"),
        "flux_energy_force": (FluxMetricBias, "energy_force"),
    }
    cls, name = targets[case]
    original = getattr(cls, name)
    calls = []

    def changed(self, *args, **kwargs):
        calls.append(name)
        result = original(self, *args, **kwargs)
        if name == "free_step":
            args[0][:, 0, 0] += .004
            return result
        if name == "evaluate":
            result.energy *= .5
            result.forces *= .5
            return result
        if name == "energy_force":
            return .5 * result[0], .5 * result[1]
        if name in {"centroid_value_gradient", "centroid_value_gradient_hessian"}:
            return (result[0] + .4, *result[1:])
        if name == "surfaces":
            return (*result[:3], result[3] + .05, *result[4:])
        return result * .5 if name == "_force" else result + .4

    monkeypatch.setattr(cls, name, changed)
    dyn = Dynamics(pes, np.ones(natoms), 2., 1, .01)
    return dyn, _state(natoms), cv, bias, calls


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ["auto", "native"])
def test_class_overrides_preserve_python_semantics(
    api, case, backend, monkeypatch, cabi_pes, fortran_adapter,
):
    dyn, actual, cv, bias, calls = _changed_case(
        case, monkeypatch, cabi_pes, fortran_adapter,
    )
    reference = actual.copy()
    if backend == "native":
        with pytest.raises(RuntimeError, match="custom|overridden"):
            _advance(dyn, actual, api, cv, bias, backend)
        assert not calls
        np.testing.assert_array_equal(actual.q, reference.q)
        np.testing.assert_array_equal(actual.p, reference.p)
        assert (actual.step, actual.time) == (reference.step, reference.time)
    else:
        result = _advance(dyn, actual, api, cv, bias, backend)
        assert result.backend == "python" and calls
        assert result.fallback_reason
        actual_calls = len(calls)
        expected = _advance(dyn, reference, api, cv, bias, "python")
        assert len(calls) == 2 * actual_calls
        np.testing.assert_array_equal(actual.q, reference.q)
        np.testing.assert_array_equal(actual.p, reference.p)
        np.testing.assert_array_equal(result.values, expected.values)
        np.testing.assert_array_equal(result.energies, expected.energies)
        assert (actual.step, actual.time) == (reference.step, reference.time)
        assert not dyn.can_fuse(bias=bias, observe=cv)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("provider", ["cabi", "python", "fortran"])
def test_unmodified_builtin_providers_keep_native_path(api, provider, cabi_pes, fortran_adapter):
    pes = {"cabi": cabi_pes, "python": PythonPotential(cabi_pes), "fortran": fortran_adapter}[provider]
    natoms = 3 if provider == "fortran" else 1
    dyn = Dynamics(pes, np.ones(natoms), 2., 1, .01)
    actual = _state(natoms)
    reference = actual.copy()
    cv = Cartesian(0, "x")
    result = _advance(dyn, actual, api, cv, backend="native")
    expected = _advance(dyn, reference, api, cv, backend="python")
    assert result.backend == "native"
    np.testing.assert_allclose(actual.q, reference.q, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.p, reference.p, atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.values, expected.values, atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.energies, expected.energies, atol=2e-14, rtol=0)


@pytest.mark.parametrize("target", ["dynamics", "integrator", "cabi", "python", "cv", "bias"])
def test_original_identities_are_captured_before_lazy_propagation_import(libraries, target):
    script = r'''
import sys
import numpy as np
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State
from pyrpmd.potentials import CABIPotential, PythonPotential
assert "pyrpmd.propagation" not in sys.modules
pes = CABIPotential(sys.argv[1])
if sys.argv[2] == "python":
    pes = PythonPotential(pes)
cv = Cartesian(0, "x")
bias = UmbrellaBias(cv, [0.], 2.)
cls, name = {
    "dynamics": (Dynamics, "_force"),
    "integrator": (CayleyIntegrator, "free_step"),
    "cabi": (CABIPotential, "evaluate"),
    "python": (PythonPotential, "evaluate"),
    "cv": (Cartesian, "centroid_value_gradient"),
    "bias": (UmbrellaBias, "energy_force"),
}[sys.argv[2]]
original = getattr(cls, name)
calls = []
def changed(self, *args, **kwargs):
    calls.append(True)
    return original(self, *args, **kwargs)
setattr(cls, name, changed)
dyn = Dynamics(pes, [1.], 2., 1, .01)
state = State([[[.2, 0., 0.]]], [[[.1, 0., 0.]]])
assert "pyrpmd.propagation" not in sys.modules
result = dyn.propagate(state, 2, observe=cv, bias=bias)
assert result.backend == "python" and calls
pes.close()
'''
    subprocess.run([sys.executable, "-c", script, str(libraries[0]), target], check=True)
