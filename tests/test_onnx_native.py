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
        assert isinstance(pes.metadata["capabilities"]["float64_cuda_scatter_nd"], bool)


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


def _make_scatter_nd_model(path, *, reduction="add", slice_update=False,
                           negative_indices=False, out_of_bounds=False,
                           mixed_float32=False, omit_reduction_attr=False):
    """Build a dynamic-batch [B, 3, 3] float64 ONNX model exercising ScatterND."""
    inputs = [helper.make_tensor_value_info("q", TensorProto.DOUBLE, ["B", 3, 3])]
    outputs = [helper.make_tensor_value_info("energy", TensorProto.DOUBLE, ["B"]),
               helper.make_tensor_value_info("forces", TensorProto.DOUBLE, ["B", 3, 3])]
    if slice_update:
        atom_idx = np.array([-1, 0, -2, -1] if negative_indices else [2, 0, 1, 2], dtype=np.int64)
        if out_of_bounds:
            atom_idx = np.array([2, 0, 5, 2], dtype=np.int64)
        gather_idx = np.array([0, 1, 2, 0], dtype=np.int64)
    else:
        atom_idx = np.array([[-1, 0], [0, -2], [-2, -1], [-1, 0]] if negative_indices
                            else [[2, 0], [0, 1], [1, 2], [2, 0]], dtype=np.int64)
        if out_of_bounds:
            atom_idx = np.array([[2, 0], [0, 1], [4, 2], [2, 0]], dtype=np.int64)
        gather_idx = np.array([0, 4, 8, 2], dtype=np.int64)

    num_u = int(atom_idx.shape[0])
    idx_tail = int(atom_idx.shape[1]) if atom_idx.ndim == 2 else 1
    inits = [
        numpy_helper.from_array(np.array(True, dtype=np.bool_), "true_b"),
        numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), "s_st"),
        numpy_helper.from_array(np.array([1 << 62, 1, 1], dtype=np.int64), "s_en"),
        numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "s_ax"),
        numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), "sh_b1"),
        numpy_helper.from_array(np.array([-1, 1, 1], dtype=np.int64), "sh_b11"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "axis0"),
        numpy_helper.from_array(np.zeros((1, num_u, 1), dtype=np.int64), "z_1u1"),
        numpy_helper.from_array(atom_idx.reshape(1, num_u, idx_tail), "atom_idx_1uk"),
        numpy_helper.from_array(np.zeros((1, num_u, idx_tail), dtype=np.int64), "z_1uk"),
        numpy_helper.from_array(gather_idx, "gather_idx"),
        numpy_helper.from_array(np.array([-1, 9], dtype=np.int64), "sh_b9"),
        numpy_helper.from_array(np.array(0.25, dtype=np.float64), "quarter"),
        numpy_helper.from_array(np.array(-0.5, dtype=np.float64), "scale_u"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes_12"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float64), "tenth"),
    ]
    nodes = [
        helper.make_node("Slice", ["q", "s_st", "s_en", "s_ax"], ["q_b11"]),
        helper.make_node("Reshape", ["q_b11", "sh_b1"], ["q_b1"]),
        helper.make_node("Cast", ["q_b1"], ["b1_i64"], to=TensorProto.INT64),
        helper.make_node("Where", ["true_b", "one_i64", "b1_i64"], ["ones_b1"]),
        helper.make_node("CumSum", ["ones_b1", "axis0"], ["b_idx_b1"], exclusive=1),
        helper.make_node("Reshape", ["b_idx_b1", "sh_b11"], ["b_idx_b11"]),
        helper.make_node("Add", ["b_idx_b11", "z_1u1"], ["b_idx_bu1"]),
        helper.make_node("Mul", ["b_idx_b11", "z_1uk"], ["zero_buk"]),
        helper.make_node("Add", ["zero_buk", "atom_idx_1uk"], ["atom_buk"]),
        helper.make_node("Concat", ["b_idx_bu1", "atom_buk"], ["scatter_idx"], axis=2),
        helper.make_node("Mul", ["q", "quarter"], ["base_data"]),
    ]
    if slice_update:
        nodes.extend([
            helper.make_node("Gather", ["q", "gather_idx"], ["gathered"], axis=1),
            helper.make_node("Mul", ["gathered", "scale_u"], ["updates"]),
        ])
    else:
        nodes.extend([
            helper.make_node("Reshape", ["q", "sh_b9"], ["q_flat"]),
            helper.make_node("Gather", ["q_flat", "gather_idx"], ["gathered"], axis=1),
            helper.make_node("Mul", ["gathered", "scale_u"], ["updates"]),
        ])

    scatter_kwargs = {} if omit_reduction_attr else {"reduction": reduction}
    nodes.append(helper.make_node("ScatterND", ["base_data", "scatter_idx", "updates"],
                                  ["scattered"], **scatter_kwargs))

    if mixed_float32:
        inits.extend([
            numpy_helper.from_array(np.array([0], dtype=np.int64), "f32_col"),
            numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "z_12_f32"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "ax_1"),
            numpy_helper.from_array(np.array(0.125, dtype=np.float32), "w_f32"),
            numpy_helper.from_array(np.array([-1, 1, 2], dtype=np.int64), "sh_b12"),
        ])
        nodes.extend([
            helper.make_node("Cast", ["q_b1"], ["q_b1_f32"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["q_b1_f32", "z_12_f32"], ["base_f32"]),
            helper.make_node("Mul", ["b_idx_b1", "f32_col"], ["zero_col"]),
            helper.make_node("Concat", ["b_idx_b1", "zero_col"], ["idx_f32_2d"], axis=1),
            helper.make_node("Reshape", ["idx_f32_2d", "sh_b12"], ["idx_f32"]),
            helper.make_node("Mul", ["q_b1_f32", "w_f32"], ["upd_f32"]),
            helper.make_node("ScatterND", ["base_f32", "idx_f32", "upd_f32"],
                             ["out_f32"], reduction="add"),
            helper.make_node("ReduceSum", ["out_f32", "ax_1"], ["sum_f32"], keepdims=0),
            helper.make_node("Cast", ["sum_f32"], ["sum_f32_f64"], to=TensorProto.DOUBLE),
            helper.make_node("Mul", ["scattered", "scattered"], ["sq_forces"]),
            helper.make_node("ReduceSum", ["sq_forces", "axes_12"], ["e_base"], keepdims=0),
            helper.make_node("Mul", ["e_base", "tenth"], ["e_scaled"]),
            helper.make_node("Add", ["e_scaled", "sum_f32_f64"], ["energy"]),
            helper.make_node("Identity", ["scattered"], ["forces"]),
        ])
    else:
        nodes.extend([
            helper.make_node("Mul", ["scattered", "scattered"], ["sq_forces"]),
            helper.make_node("ReduceSum", ["sq_forces", "axes_12"], ["e_base"], keepdims=0),
            helper.make_node("Mul", ["e_base", "tenth"], ["energy"]),
            helper.make_node("Identity", ["scattered"], ["forces"]),
        ])

    used = {inp for n in nodes for inp in n.input}
    inits = [t for t in inits if t.name in used]
    graph = helper.make_graph(nodes, "scatter-nd-float64-test", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    spec = {
        "schema": "pyrpmd.onnx/1", "batch_independent": True,
        "atomic_numbers": [1, 1, 8], "state": {"charge": 0, "multiplicity": 1},
        "position_unit": "bohr", "energy_unit": "hartree",
        "force_unit": "hartree/bohr", "derivative": "force",
        "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU},
    }
    helper.set_model_props(model, {"pyrpmd.onnx": json.dumps(spec)})
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _scatter_nd_reference(q, *, reduction="add", slice_update=False,
                          negative_indices=False, mixed_float32=False):
    base = 0.25 * q.copy()
    if slice_update:
        atom_idx = [-1, 0, -2, -1] if negative_indices else [2, 0, 1, 2]
        gather_idx = [0, 1, 2, 0]
        updates = -0.5 * q[:, gather_idx, :]
        for u, a in enumerate(atom_idx):
            idx = a + 3 if a < 0 else a
            val = updates[:, u, :]
            if reduction == "none":
                base[:, idx, :] = val
            elif reduction == "add":
                base[:, idx, :] += val
            elif reduction == "mul":
                base[:, idx, :] *= val
            elif reduction == "max":
                base[:, idx, :] = np.maximum(base[:, idx, :], val)
            elif reduction == "min":
                base[:, idx, :] = np.minimum(base[:, idx, :], val)
    else:
        atom_idx = ([[-1, 0], [0, -2], [-2, -1], [-1, 0]] if negative_indices
                    else [[2, 0], [0, 1], [1, 2], [2, 0]])
        gather_idx = [0, 4, 8, 2]
        updates = -0.5 * q.reshape(len(q), 9)[:, gather_idx]
        for u, (a, c) in enumerate(atom_idx):
            ia = a + 3 if a < 0 else a
            ic = c + 3 if c < 0 else c
            val = updates[:, u]
            if reduction == "none":
                base[:, ia, ic] = val
            elif reduction == "add":
                base[:, ia, ic] += val
            elif reduction == "mul":
                base[:, ia, ic] *= val
            elif reduction == "max":
                base[:, ia, ic] = np.maximum(base[:, ia, ic], val)
            elif reduction == "min":
                base[:, ia, ic] = np.minimum(base[:, ia, ic], val)
    energy = 0.1 * np.sum(base * base, axis=(1, 2))
    if mixed_float32:
        f32_term = (np.float32(0.125) * q[:, 0, 0].astype(np.float32)).astype(np.float64)
        energy = energy + f32_term
    return energy, base


@pytest.mark.parametrize("reduction,slice_update,negative_indices", [
    ("add", True, False),
    ("add", False, True),
    ("none", True, False),
    ("default_none", True, False),
    ("mul", False, False),
    ("max", True, True),
    ("min", False, True),
])
def test_scatter_nd_float64_reductions_shapes_and_cuda_parity(
    tmp_path, reduction, slice_update, negative_indices,
):
    # Note: reduction="none" with duplicate indices is order-dependent in parallel;
    # use distinct indices when reduction == "none".
    path = tmp_path / f"scatter_{reduction}_{int(slice_update)}_{int(negative_indices)}.onnx"
    if reduction in ("none", "default_none"):
        inputs = [helper.make_tensor_value_info("q", TensorProto.DOUBLE, ["B", 3, 3])]
        outputs = [helper.make_tensor_value_info("energy", TensorProto.DOUBLE, ["B"]),
                   helper.make_tensor_value_info("forces", TensorProto.DOUBLE, ["B", 3, 3])]
        inits = [
            numpy_helper.from_array(np.array(True, dtype=np.bool_), "true_b"),
            numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), "s_st"),
            numpy_helper.from_array(np.array([1 << 62, 1, 1], dtype=np.int64), "s_en"),
            numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), "s_ax"),
            numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), "sh_b1"),
            numpy_helper.from_array(np.array([-1, 1, 1], dtype=np.int64), "sh_b11"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i64"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), "axis0"),
            numpy_helper.from_array(np.zeros((1, 2, 1), dtype=np.int64), "z_121"),
            numpy_helper.from_array(np.array([[[2], [0]]], dtype=np.int64), "atom_idx_121"),
            numpy_helper.from_array(np.array([0, 2], dtype=np.int64), "gather_idx"),
            numpy_helper.from_array(np.array(0.25, dtype=np.float64), "quarter"),
            numpy_helper.from_array(np.array(-0.5, dtype=np.float64), "scale_u"),
            numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes_12"),
            numpy_helper.from_array(np.array(0.1, dtype=np.float64), "tenth"),
        ]
        nodes = [
            helper.make_node("Slice", ["q", "s_st", "s_en", "s_ax"], ["q_b11"]),
            helper.make_node("Reshape", ["q_b11", "sh_b1"], ["q_b1"]),
            helper.make_node("Cast", ["q_b1"], ["b1_i64"], to=TensorProto.INT64),
            helper.make_node("Where", ["true_b", "one_i64", "b1_i64"], ["ones_b1"]),
            helper.make_node("CumSum", ["ones_b1", "axis0"], ["b_idx_b1"], exclusive=1),
            helper.make_node("Reshape", ["b_idx_b1", "sh_b11"], ["b_idx_b11"]),
            helper.make_node("Add", ["b_idx_b11", "z_121"], ["b_idx_b21"]),
            helper.make_node("Mul", ["b_idx_b11", "z_121"], ["zero_b21"]),
            helper.make_node("Add", ["zero_b21", "atom_idx_121"], ["atom_b21"]),
            helper.make_node("Concat", ["b_idx_b21", "atom_b21"], ["scatter_idx"], axis=2),
            helper.make_node("Mul", ["q", "quarter"], ["base_data"]),
            helper.make_node("Gather", ["q", "gather_idx"], ["gathered"], axis=1),
            helper.make_node("Mul", ["gathered", "scale_u"], ["updates"]),
            helper.make_node("ScatterND", ["base_data", "scatter_idx", "updates"],
                             ["scattered"], **({} if reduction == "default_none" else {"reduction": "none"})),
            helper.make_node("Mul", ["scattered", "scattered"], ["sq_forces"]),
            helper.make_node("ReduceSum", ["sq_forces", "axes_12"], ["e_base"], keepdims=0),
            helper.make_node("Mul", ["e_base", "tenth"], ["energy"]),
            helper.make_node("Identity", ["scattered"], ["forces"]),
        ]
        graph = helper.make_graph(nodes, "scatter-none-distinct", inputs, outputs, inits)
        mdl = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        mdl.ir_version = 10
        spec = {
            "schema": "pyrpmd.onnx/1", "batch_independent": True,
            "atomic_numbers": [1, 1, 8], "state": {"charge": 0, "multiplicity": 1},
            "position_unit": "bohr", "energy_unit": "hartree",
            "force_unit": "hartree/bohr", "derivative": "force",
            "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU},
        }
        helper.set_model_props(mdl, {"pyrpmd.onnx": json.dumps(spec)})
        onnx.save(onnx.shape_inference.infer_shapes(mdl, check_type=True, strict_mode=True), path)

        def ref_fn(q):
            base = 0.25 * q.copy()
            base[:, 2, :] = -0.5 * q[:, 0, :]
            base[:, 0, :] = -0.5 * q[:, 2, :]
            return 0.1 * np.sum(base * base, axis=(1, 2)), base
    else:
        _make_scatter_nd_model(
            path, reduction=reduction, slice_update=slice_update,
            negative_indices=negative_indices,
        )

        def ref_fn(q):
            return _scatter_nd_reference(
                q, reduction=reduction, slice_update=slice_update,
                negative_indices=negative_indices,
            )

    with potential(path, device="cpu") as cpu_pes:
        for batch in (1, 8, 32, 128, 3, 1):
            q = data(batch)
            exp_e, exp_f = ref_fn(q)
            got_cpu = cpu_pes.evaluate(q)
            np.testing.assert_allclose(got_cpu.energy, exp_e, atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(got_cpu.forces, exp_f, atol=1e-13, rtol=1e-13)

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            assert cpu_pes.metadata["capabilities"]["float64_cuda_scatter_nd"] is False
            return

        if not cpu_pes.metadata["capabilities"]["float64_cuda_scatter_nd"]:
            with pytest.raises(PotentialError, match="PYRPMD_ONNX_CUDA_KERNELS=ON"):
                potential(path, device="cuda:0", allow_cpu_fallback=False)
            return

        with potential(path, device="cuda:0", allow_cpu_fallback=False) as cuda_pes:
            assert cuda_pes.metadata["execution_device"] == "cuda:0"
            assert cuda_pes.metadata["capabilities"]["allow_cpu_fallback"] is False
            assert cuda_pes.metadata["capabilities"]["float64_cuda_scatter_nd"] is True
            for batch in (1, 8, 32, 128, 3, 1):
                q = data(batch)
                exp_e, exp_f = ref_fn(q)
                got_cpu = cpu_pes.evaluate(q)
                got_cuda = cuda_pes.evaluate(q)
                np.testing.assert_allclose(got_cuda.energy, exp_e, atol=1e-13, rtol=1e-13)
                np.testing.assert_allclose(got_cuda.forces, exp_f, atol=1e-13, rtol=1e-13)
                np.testing.assert_allclose(got_cuda.energy, got_cpu.energy, atol=1e-13, rtol=1e-13)
                np.testing.assert_allclose(got_cuda.forces, got_cpu.forces, atol=1e-13, rtol=1e-13)


def test_scatter_nd_mixed_float32_and_float64_domains(tmp_path):
    path = _make_scatter_nd_model(
        tmp_path / "scatter_mixed.onnx", reduction="add", slice_update=True, mixed_float32=True,
    )
    with potential(path, device="cpu") as cpu_pes:
        q = data(16)
        exp_e, exp_f = _scatter_nd_reference(q, reduction="add", slice_update=True, mixed_float32=True)
        got_cpu = cpu_pes.evaluate(q)
        np.testing.assert_allclose(got_cpu.energy, exp_e, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(got_cpu.forces, exp_f, atol=1e-13, rtol=1e-13)
        if ("CUDAExecutionProvider" in ort.get_available_providers()
                and cpu_pes.metadata["capabilities"]["float64_cuda_scatter_nd"]):
            with potential(path, device="cuda:0", allow_cpu_fallback=False) as cuda_pes:
                for batch in (1, 8, 32):
                    qb = data(batch)
                    ee, ef = _scatter_nd_reference(
                        qb, reduction="add", slice_update=True, mixed_float32=True,
                    )
                    got_cuda = cuda_pes.evaluate(qb)
                    np.testing.assert_allclose(got_cuda.energy, ee, atol=1e-13, rtol=1e-13)
                    np.testing.assert_allclose(got_cuda.forces, ef, atol=1e-13, rtol=1e-13)


def test_scatter_nd_out_of_bounds_index_rejected(tmp_path):
    bad_path = _make_scatter_nd_model(
        tmp_path / "scatter_oob.onnx", reduction="add", slice_update=True, out_of_bounds=True,
    )
    with potential(bad_path, device="cpu") as cpu_pes:
        with pytest.raises(PotentialError):
            cpu_pes.evaluate(data(4))
        if ("CUDAExecutionProvider" in ort.get_available_providers()
                and cpu_pes.metadata["capabilities"]["float64_cuda_scatter_nd"]):
            with potential(bad_path, device="cuda:0", allow_cpu_fallback=False) as cuda_pes:
                with pytest.raises(PotentialError, match="out of bounds"):
                    cuda_pes.evaluate(data(4))
            good_path = _make_scatter_nd_model(
                tmp_path / "scatter_ok.onnx", reduction="add", slice_update=True,
            )
            with potential(good_path, device="cuda:0", allow_cpu_fallback=False) as clean_pes:
                got = clean_pes.evaluate(data(8))
                assert not got.status.any()


def test_public_runner_end_to_end_scatter_nd_cabi(tmp_path):
    from pyrpmd.constants import KB_AU
    from pyrpmd.runner import run

    scatter_onnx = _make_scatter_nd_model(
        tmp_path / "runner_scatter.onnx", reduction="add", slice_update=True,
    )

    def make_job(device):
        return {
            "task": "drpmd",
            "temperature": 1.0 / KB_AU,
            "beads": 4,
            "seed": 202,
            "native": True,
            "system": {
                "symbols": ["H", "H", "O"],
                "masses": [1800.0, 1800.0, 29000.0],
                "geometry": [[0.0, 0.0, -5.0], [0.0, 0.0, 1.8], [0.0, 0.0, 0.0]],
                "fragments": [[0], [1, 2]],
            },
            "potential": {
                "provider": "cabi",
                "library": str(LIBRARY),
                "params": {
                    "model": str(scatter_onnx),
                    "device": device,
                    "allow_cpu_fallback": False,
                },
            },
            "dynamics": {"timestep": 0.01},
            "sampling": {
                "equilibration_steps": 2,
                "production_steps": 4,
                "interval_steps": 2,
                "andersen_interval": 2,
            },
            "collision": {"bmax": 0.5, "trajectories": 2, "max_steps": 12},
            "outcomes": {"bond_cutoffs": {"H-H": 1.5, "H-O": 2.5}, "persistence_steps": 2},
            "output": {"checkpoint_steps": 6},
        }

    cpu_summary = run(make_job("cpu"), output=tmp_path / "run_cpu")
    assert cpu_summary["trajectories"] == 2
    assert (tmp_path / "run_cpu" / "results" / "summary.json").is_file()

    with potential(scatter_onnx, device="cpu") as probe:
        has_cuda_scatter = probe.metadata["capabilities"]["float64_cuda_scatter_nd"]
    if "CUDAExecutionProvider" in ort.get_available_providers() and has_cuda_scatter:
        cuda_summary = run(make_job("cuda:0"), output=tmp_path / "run_cuda")
        assert cuda_summary["trajectories"] == 2
        for cpu_rec, cuda_rec in zip(cpu_summary["records"], cuda_summary["records"]):
            assert cpu_rec["status"] == cuda_rec["status"]
            assert cpu_rec["channel"] == cuda_rec["channel"]
            assert cpu_rec["steps"] == cuda_rec["steps"]
