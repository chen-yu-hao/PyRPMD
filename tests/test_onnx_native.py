"""Real ONNX Runtime/C ABI integration. Optional locally; mandatory in its CI.

The harmonic ONNX fixtures are interface tests, not fitted chemical PESs.
"""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REQUIRED = os.environ.get("PYRPMD_REQUIRE_ONNX_TESTS") == "1"
LIBRARY = os.environ.get("PYRPMD_ONNX_LIBRARY", "")
if not LIBRARY or not Path(LIBRARY).is_file():
    if REQUIRED:
        raise RuntimeError("PYRPMD_ONNX_LIBRARY must name the compiled real ORT plugin")
    pytest.skip("optional native ONNX plugin not configured", allow_module_level=True)
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    if REQUIRED:
        raise
    pytest.skip("optional ONNX dependencies unavailable", allow_module_level=True)

from onnx import TensorProto, helper, numpy_helper
from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import CABIPotential, PotentialError, check_potential
from pyrpmd.potentials import cabi


def make_model(path, k=0.2, length="angstrom", energy="eV", *, dtype=TensorProto.DOUBLE,
               batch="B", metadata=True, independent=True, output_nan=False):
    dt = np.float64 if dtype == TensorProto.DOUBLE else np.float32
    inputs = [helper.make_tensor_value_info("q", dtype, [batch, 3, 3])]
    outputs = [helper.make_tensor_value_info("energy", dtype, [batch]),
               helper.make_tensor_value_info("forces", dtype, [batch, 3, 3])]
    initializers = [numpy_helper.from_array(np.asarray([1, 2], np.int64), "axes"),
                    numpy_helper.from_array(np.asarray(0.5*k, dt), "half_k"),
                    numpy_helper.from_array(np.asarray(-k if not output_nan else np.nan, dt), "minus_k")]
    nodes = [helper.make_node("Mul", ["q", "q"], ["square"]),
             helper.make_node("ReduceSum", ["square", "axes"], ["sum"], keepdims=0),
             helper.make_node("Mul", ["sum", "half_k"], ["energy"]),
             helper.make_node("Mul", ["q", "minus_k"], ["forces"])]
    graph = helper.make_graph(nodes, "harmonic-contract-test", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    if metadata:
        spec = {"schema": "pyrpmd.onnx/1", "batch_independent": independent,
                "atomic_numbers": [1, 1, 8], "state": {"charge": 0, "multiplicity": 1},
                "position_unit": length, "energy_unit": energy,
                "force_unit": f"{energy}/{length}", "derivative": "force",
                "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU}}
        helper.set_model_props(model, {"pyrpmd.onnx": json.dumps(spec)})
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def model(tmp_path):
    return make_model(tmp_path / "harmonic.onnx")


def potential(path, **options):
    return CABIPotential(LIBRARY, {"model": str(path), **options})


def data(batch=3):
    return np.random.default_rng(72).normal(0.1, 0.3, (batch, 3, 3))


@pytest.mark.parametrize("length,energy", [("angstrom", "eV"), ("bohr", "hartree"),
                                           ("bohr", "eV"), ("angstrom", "hartree")])
def test_batch_units_forces_and_resize(tmp_path, length, energy):
    path = make_model(tmp_path / "units.onnx", length=length, energy=energy)
    lf = BOHR_ANGSTROM if length == "angstrom" else 1.0
    ef = EV_TO_AU if energy == "eV" else 1.0
    with potential(path) as pes:
        for batch in [1, 3, 32, 128, 3, 1]:
            q = data(batch)
            got = pes.evaluate(q)
            np.testing.assert_allclose(got.energy, 0.1*np.sum((q*lf)**2, axis=(1, 2))*ef, rtol=2e-14)
            np.testing.assert_allclose(got.forces, -0.2*q*lf*lf*ef, rtol=2e-14)
            assert got.energy.dtype == got.forces.dtype == np.float64
            assert not got.status.any()
        check = check_potential(pes, data(), epsilon=1e-5)
        assert check["max_absolute_force_error"] < 1e-9
        assert pes.metadata["capabilities"]["stateful"] is False
        assert pes.metadata["capabilities"]["host_memory"] is True


def test_batch_is_independent_and_not_scaled_by_bead_count(model):
    with potential(model) as pes:
        q = data(32)
        got = pes.evaluate(q)
        individual = [pes.evaluate(x[None]) for x in q]
        np.testing.assert_allclose(got.energy, np.concatenate([x.energy for x in individual]), atol=1e-14)
        np.testing.assert_allclose(got.forces, np.concatenate([x.forces for x in individual]), atol=1e-14)
        q[7] += 0.5
        changed = pes.evaluate(q)
        keep = np.arange(32) != 7
        np.testing.assert_array_equal(got.energy[keep], changed.energy[keep])
        np.testing.assert_array_equal(got.forces[keep], changed.forces[keep])


def test_independent_models_and_load_once(tmp_path):
    a = make_model(tmp_path / "a.onnx", k=0.2)
    b = make_model(tmp_path / "b.onnx", k=0.6)
    with potential(a) as pa, potential(b) as pb:
        a.unlink()
        b.unlink()
        # A session still works after the source model has disappeared.
        for batch in (1, 32, 3):
            ea, eb = pa.evaluate(data(batch)), pb.evaluate(data(batch))
            np.testing.assert_allclose(eb.energy, 3*ea.energy, rtol=1e-14)
            np.testing.assert_allclose(eb.forces, 3*ea.forces, rtol=1e-14)
    pa.close()
    with pytest.raises(PotentialError, match="closed"):
        pa.evaluate(data())


@pytest.mark.parametrize("options", [{"dtype": TensorProto.FLOAT}, {"batch": 8},
                                    {"metadata": False}, {"independent": False}])
def test_invalid_models_fail_at_initialization(tmp_path, options):
    path = make_model(tmp_path / "invalid.onnx", **options)
    with pytest.raises(PotentialError):
        potential(path)


@pytest.mark.parametrize("options", [{"typo": True}, {"device": "gpu"},
                                     {"intra_op_threads": 0}, {"intra_op_threads": True},
                                     {"atomic_numbers": [8, 1, 1]}, {"multiplicity": 3},
                                     {"natoms": 4}])
def test_invalid_configuration_fails(model, options):
    with pytest.raises(PotentialError):
        potential(model, **options)


def test_cuda_request_never_silently_becomes_cpu(model):
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("test requires the CPU-only runtime used by CI")
    with pytest.raises(PotentialError, match="CUDA"):
        potential(model, device="cuda:0")


def test_nonfinite_output_is_explicit_failure(tmp_path):
    path = make_model(tmp_path / "nan.onnx", output_nan=True)
    with potential(path) as pes:
        result = pes.evaluate(data())
        assert np.all(result.status == 3)
        assert np.isnan(result.energy).all() and np.isnan(result.forces).all()
        assert all("nonfinite" in msg for msg in result.messages)


def _raw_evaluate(pes, q, **changes):
    b, n, _ = q.shape
    e = np.empty(b)
    f = np.empty_like(q)
    s = np.empty(b, np.int32)
    inp = cabi.Input(1, C.sizeof(cabi.Input), 1, 1, b, n, cabi._ptr(q), q.size)
    out = cabi.Output(1, C.sizeof(cabi.Output), cabi._ptr(e), b, cabi._ptr(f), f.size,
                      s.ctypes.data_as(C.POINTER(C.c_int32)), b, None, 0, 0)
    for name, value in changes.items():
        setattr(inp, name, value)
    errbuf = C.create_string_buffer(1024)
    err = cabi.Error(C.addressof(errbuf), len(errbuf), 0)
    rc = pes._evaluate(pes._context, C.byref(inp), C.byref(out), C.byref(err))
    return rc, errbuf.value, e, f, s


@pytest.mark.parametrize("change", [{"abi_version": 7}, {"positions_count": 1},
                                   {"nstructures": -1}, {"nstructures": 2**63-1},
                                   {"layout": 4}, {"scalar_type": 3}])
def test_malformed_abi_never_runs_or_throws(model, change):
    with potential(model) as pes:
        rc, message, *_ = _raw_evaluate(pes, data(), **change)
        assert rc != 0 and message


def test_nonfinite_input_and_metadata_sizing(model):
    with potential(model) as pes:
        q = data()
        q[0, 0, 0] = np.inf
        rc, message, e, f, status = _raw_evaluate(pes, q)
        assert rc != 0 and b"nonfinite" in message
        assert np.isnan(e).all() and np.isnan(f).all()
        needed = cabi.I64()
        buf = C.create_string_buffer(b"zzzz", 5)
        rc = pes._metadata(pes._context, C.addressof(buf), 5, C.byref(needed), None)
        assert rc == 4 and needed.value > 5 and buf.value == b""


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("beads", [1, 8, 32])
def test_real_native_propagation_without_python_pes_callbacks(model, scheme, beads):
    assert native_capabilities()["available"], "CI must build the actual C++ integrator"
    with potential(model) as pes:
        dyn = Dynamics(pes, [1800., 1800., 29000.], 220., beads, 0.1, integrator=scheme)
        q = data(beads)
        p = np.random.default_rng(19).normal(size=q.shape)
        a, b = State(q.copy(), p.copy()), State(q.copy(), p.copy())
        calls = []
        def profile(frame, event, arg):
            if event == "call" and frame.f_code is CABIPotential.evaluate.__code__:
                calls.append(1)
        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            actual = dyn.propagate_block(a, 25, backend="native", record_states=True)
        finally:
            sys.setprofile(previous)
        assert not calls, "native block must not invoke the Python PES adapter"
        reference = dyn.propagate_block(b, 25, backend="python", record_states=True)
        assert actual.backend == "native"
        np.testing.assert_allclose(actual.q, reference.q, atol=2e-11, rtol=1e-11)
        np.testing.assert_allclose(actual.p, reference.p, atol=2e-10, rtol=1e-11)
        np.testing.assert_allclose(actual.energies, reference.energies, atol=2e-11, rtol=1e-11)


def test_openmp_context_isolation(tmp_path):
    executable = os.environ.get("PYRPMD_ONNX_CONTEXT_TEST")
    if not executable:
        if REQUIRED:
            pytest.fail("native OpenMP context test executable not configured")
        pytest.skip("OpenMP context executable not configured")
    a = make_model(tmp_path / "a.onnx", k=0.2)
    b = make_model(tmp_path / "b.onnx", k=0.6)
    result = subprocess.run([executable, str(a), str(b)], check=True,
                            capture_output=True, text=True, timeout=60)
    assert "passed" in result.stdout
