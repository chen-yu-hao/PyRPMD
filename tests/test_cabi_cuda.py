"""C ABI CUDA device-pointer extension (pes_api_cuda_v1.h) and ONNX GPU tests.

CPU-only contract tests (struct layout, host-only v1/v2 compatibility and
fail-fast diagnostics, malformed CUDA symbol/metadata rejection, context
cleanup) run on every machine with a C11 compiler. Real CUDA kernel, stream
ordering, ABI v2 strain-virial, and ONNX Runtime IoBinding tests execute on a
CUDA GPU (or fail explicitly when ``PYRPMD_REQUIRE_CUDA_CABI_TESTS=1``).
"""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.potentials import CABIPotential, PotentialError
from pyrpmd.potentials import cabi

ROOT = Path(__file__).resolve().parents[1]
REQUIRE_CUDA_CABI = os.environ.get("PYRPMD_REQUIRE_CUDA_CABI_TESTS") == "1"
ONNX_LIBRARY = os.environ.get("PYRPMD_ONNX_LIBRARY", "")


def _compile_c11(directory: Path, name: str, source_text: str | None = None, *, extra_sources=()) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        if REQUIRE_CUDA_CABI:
            raise RuntimeError("C11 compiler 'cc' is required for C ABI tests")
        pytest.skip("C compiler required for C ABI tests")
    sources = []
    if source_text is not None:
        src = directory / f"{name}.c"
        src.write_text(source_text, encoding="utf-8")
        sources.append(str(src))
    sources.extend(str(p) for p in extra_sources)
    output = directory / f"{name}.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-fPIC",
            "-shared",
            "-I",
            str(ROOT / "native/include"),
            *sources,
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def _require_cuda_torch():
    try:
        import torch
    except ImportError:
        if REQUIRE_CUDA_CABI:
            raise
        pytest.skip("PyTorch not installed")
    if not torch.cuda.is_available():
        if REQUIRE_CUDA_CABI:
            raise RuntimeError("CUDA GPU required when PYRPMD_REQUIRE_CUDA_CABI_TESTS=1")
        pytest.skip("CUDA hardware/runtime unavailable")
    return torch


def _compile_nvcc(directory: Path, name: str, cuda_source: str) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        if REQUIRE_CUDA_CABI:
            raise RuntimeError("nvcc is required when PYRPMD_REQUIRE_CUDA_CABI_TESTS=1")
        pytest.skip("nvcc unavailable for custom CUDA ABI plugin compilation")
    src = directory / f"{name}.cu"
    src.write_text(cuda_source, encoding="utf-8")
    output = directory / f"{name}.so"
    subprocess.run(
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC,-Wall,-Wextra",
            "-shared",
            "-I",
            str(ROOT / "native/include"),
            str(src),
            "-lcudart",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def _require_onnx_cuda():
    torch = _require_cuda_torch()
    if not ONNX_LIBRARY or not Path(ONNX_LIBRARY).is_file():
        if REQUIRE_CUDA_CABI:
            raise RuntimeError("PYRPMD_ONNX_LIBRARY must point to CUDA-enabled libpyrpmd_onnx.so")
        pytest.skip("PYRPMD_ONNX_LIBRARY not configured")
    try:
        import onnx
    except ImportError:
        if REQUIRE_CUDA_CABI:
            raise
        pytest.skip("onnx Python package unavailable")
    return torch, onnx


def _make_onnx_harmonic(
    onnx_mod,
    path: Path,
    *,
    k: float = 0.24,
    length: str = "angstrom",
    energy: str = "eV",
    natoms: int = 3,
    atomic_numbers=(1, 1, 8),
    output_nan: bool = False,
    bounded_gaussian: bool = False,
) -> Path:
    from onnx import TensorProto, helper, numpy_helper

    inputs = [helper.make_tensor_value_info("q", TensorProto.DOUBLE, ["B", natoms, 3])]
    outputs = [
        helper.make_tensor_value_info("energy", TensorProto.DOUBLE, ["B"]),
        helper.make_tensor_value_info("forces", TensorProto.DOUBLE, ["B", natoms, 3]),
    ]
    if bounded_gaussian:
        # V(q) = k * exp(-sum(q^2)), F(q) = 2 * k * q * exp(-sum(q^2)).
        # Note: exp(-inf) == 0 and inf * 0 == NaN, or for V(q) = sum(exp(-q^2)),
        # F(q) = 2 * q * exp(-q^2); even if an activation saturates to 0 at +inf,
        # device input finiteness must still flag +inf/-inf coordinates!
        # Construct V(q) = sum_{i,a} k * Sigmoid(-q_{i,a}^2) with zero force node
        # test: to keep exact physics F = -dV/dq while also having bounded outputs
        # at +inf: for q = +inf, exp(-q^2) -> 0 and 1/(1 + q^2) -> 0!
        # Let V(q) = sum_{i,a} 0.5 * k * (q_{i,a}^2 / (1 + q_{i,a}^2)),
        # dV/dq = k * q / (1 + q^2)^2 -> at q = +inf, q^2/(1+q^2) = NaN in IEEE,
        # whereas V(q) = sum(k * Exp(-Abs(q))), F(q) = k * Sign(q) * Exp(-Abs(q))
        # evaluates to FINITE 0 at q = +inf in pure IEEE754 arithmetic!
        initializers = [
            numpy_helper.from_array(np.asarray([1, 2], np.int64), "axes"),
            numpy_helper.from_array(np.asarray(k, np.float64), "k_val"),
            numpy_helper.from_array(np.asarray(-1.0, np.float64), "neg_one"),
        ]
        nodes = [
            helper.make_node("Abs", ["q"], ["abs_q"]),
            helper.make_node("Mul", ["abs_q", "neg_one"], ["neg_abs_q"]),
            helper.make_node("Exp", ["neg_abs_q"], ["exp_term"]),
            helper.make_node("Mul", ["exp_term", "k_val"], ["term_e"]),
            helper.make_node("ReduceSum", ["term_e", "axes"], ["energy"], keepdims=0),
            helper.make_node("Sign", ["q"], ["sign_q"]),
            helper.make_node("Mul", ["sign_q", "term_e"], ["forces"]),
        ]
    else:
        initializers = [
            numpy_helper.from_array(np.asarray([1, 2], np.int64), "axes"),
            numpy_helper.from_array(np.asarray(0.5 * k, np.float64), "half_k"),
            numpy_helper.from_array(
                np.asarray(-k if not output_nan else np.nan, np.float64), "minus_k"
            ),
        ]
        nodes = [
            helper.make_node("Mul", ["q", "q"], ["square"]),
            helper.make_node("ReduceSum", ["square", "axes"], ["sum"], keepdims=0),
            helper.make_node("Mul", ["sum", "half_k"], ["energy"]),
            helper.make_node("Mul", ["q", "minus_k"], ["forces"]),
        ]
    graph = helper.make_graph(nodes, "cabi-cuda-test", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    spec = {
        "schema": "pyrpmd.onnx/1",
        "batch_independent": True,
        "atomic_numbers": list(atomic_numbers),
        "state": {"charge": 0, "multiplicity": 1},
        "position_unit": length,
        "energy_unit": energy,
        "force_unit": f"{energy}/{length}",
        "derivative": "force",
        "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU},
    }
    helper.set_model_props(model, {"pyrpmd.onnx": json.dumps(spec)})
    onnx_mod.checker.check_model(model)
    onnx_mod.save(model, path)
    return path


def _make_onnx_anharmonic_pair(onnx_mod, path: Path, *, length: str = "angstrom", energy: str = "eV") -> Path:
    """Smooth nonlinear 3-atom PES with exact analytical force graph for parity checks."""
    from onnx import TensorProto, helper, numpy_helper

    # V(q) = sum_{i,a} (0.15 * q^2 + 0.04 * q^4), F(q) = -(0.30 * q + 0.16 * q^3)
    inputs = [helper.make_tensor_value_info("q", TensorProto.DOUBLE, ["B", 3, 3])]
    outputs = [
        helper.make_tensor_value_info("energy", TensorProto.DOUBLE, ["B"]),
        helper.make_tensor_value_info("forces", TensorProto.DOUBLE, ["B", 3, 3]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray([1, 2], np.int64), "axes"),
        numpy_helper.from_array(np.asarray(0.15, np.float64), "c2"),
        numpy_helper.from_array(np.asarray(0.04, np.float64), "c4"),
        numpy_helper.from_array(np.asarray(-0.30, np.float64), "fc1"),
        numpy_helper.from_array(np.asarray(-0.16, np.float64), "fc3"),
    ]
    nodes = [
        helper.make_node("Mul", ["q", "q"], ["q2"]),
        helper.make_node("Mul", ["q2", "q"], ["q3"]),
        helper.make_node("Mul", ["q2", "q2"], ["q4"]),
        helper.make_node("Mul", ["q2", "c2"], ["e2"]),
        helper.make_node("Mul", ["q4", "c4"], ["e4"]),
        helper.make_node("Add", ["e2", "e4"], ["eterm"]),
        helper.make_node("ReduceSum", ["eterm", "axes"], ["energy"], keepdims=0),
        helper.make_node("Mul", ["q", "fc1"], ["f1"]),
        helper.make_node("Mul", ["q3", "fc3"], ["f3"]),
        helper.make_node("Add", ["f1", "f3"], ["forces"]),
    ]
    graph = helper.make_graph(nodes, "anharmonic-cuda-test", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    spec = {
        "schema": "pyrpmd.onnx/1",
        "batch_independent": True,
        "atomic_numbers": [1, 1, 8],
        "state": {"charge": 0, "multiplicity": 1},
        "position_unit": length,
        "energy_unit": energy,
        "force_unit": f"{energy}/{length}",
        "derivative": "force",
        "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU},
    }
    helper.set_model_props(model, {"pyrpmd.onnx": json.dumps(spec)})
    onnx_mod.checker.check_model(model)
    onnx_mod.save(model, path)
    return path


# ---------------------------------------------------------------------------
# 1. Pure C11 ABI layout & host-only backward compatibility / fail-fast tests
# ---------------------------------------------------------------------------


def test_cuda_header_c11_layout_matches_ctypes_without_cuda_sdk(tmp_path):
    """pes_api_cuda_v1.h compiles as strict C11 with no CUDA headers and matches ctypes."""
    lib_path = _compile_c11(
        tmp_path,
        "cuda_layout_probe",
        """
        #include "pyrpmd/pes_api_cuda_v1.h"
        #include <stddef.h>

        void probe_cuda_abi_v1(int64_t *out) {
            out[0] = PYRPMD_PES_CUDA_ABI_VERSION_V1;
            out[1] = (int64_t)sizeof(pyrpmd_cuda_error_v1);
            out[2] = (int64_t)sizeof(pyrpmd_pes_cuda_input_v1);
            out[3] = (int64_t)sizeof(pyrpmd_pes_cuda_output_v1);

            out[4] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, cuda_abi_version);
            out[5] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, struct_size);
            out[6] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, layout);
            out[7] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, scalar_type);
            out[8] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, device_id);
            out[9] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, flags);
            out[10] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, cuda_stream);
            out[11] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, nstructures);
            out[12] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, natoms);
            out[13] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, d_positions);
            out[14] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, positions_count);
            out[15] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, d_cells);
            out[16] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, cells_count);
            out[17] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, d_pbc);
            out[18] = (int64_t)offsetof(pyrpmd_pes_cuda_input_v1, pbc_count);

            out[19] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, cuda_abi_version);
            out[20] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, struct_size);
            out[21] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, d_energies);
            out[22] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, energies_count);
            out[23] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, d_forces);
            out[24] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, forces_count);
            out[25] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, d_virials);
            out[26] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, virials_count);
            out[27] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, d_structure_status);
            out[28] = (int64_t)offsetof(pyrpmd_pes_cuda_output_v1, structure_status_count);
        }
        """,
    )
    dll = C.CDLL(str(lib_path))
    buf = (C.c_int64 * 29)()
    dll.probe_cuda_abi_v1(buf)
    vals = list(buf)

    assert vals[0] == 1
    assert vals[1] == C.sizeof(cabi.Error)
    assert vals[2] == C.sizeof(cabi.CudaInputV1)
    assert vals[3] == C.sizeof(cabi.CudaOutputV1)

    in_fields = [
        "cuda_abi_version",
        "struct_size",
        "layout",
        "scalar_type",
        "device_id",
        "flags",
        "cuda_stream",
        "nstructures",
        "natoms",
        "d_positions",
        "positions_count",
        "d_cells",
        "cells_count",
        "d_pbc",
        "pbc_count",
    ]
    for idx, name in enumerate(in_fields, start=4):
        assert vals[idx] == getattr(cabi.CudaInputV1, name).offset, f"offset mismatch for CudaInputV1.{name}"

    out_fields = [
        "cuda_abi_version",
        "struct_size",
        "d_energies",
        "energies_count",
        "d_forces",
        "forces_count",
        "d_virials",
        "virials_count",
        "d_structure_status",
        "structure_status_count",
    ]
    for idx, name in enumerate(out_fields, start=19):
        assert vals[idx] == getattr(cabi.CudaOutputV1, name).offset, f"offset mismatch for CudaOutputV1.{name}"


def test_host_only_v1_and_v2_work_unchanged_and_fail_fast_on_as_torch_pes(tmp_path):
    v1_lib = _compile_c11(
        tmp_path, "host_v1", extra_sources=[ROOT / "native/examples/harmonic_pes.c"]
    )
    v2_lib = _compile_c11(
        tmp_path, "host_v2", extra_sources=[ROOT / "native/examples/periodic_cosine_pes.c"]
    )
    # Also test a host-only v1 library that explicitly sets batch_independent=true,
    # ensuring make_workflow_torch_pes reaches as_torch_pes() and fails fast with
    # the CUDA device-pointer diagnostic rather than falling back to NumPy.
    v1_indep_lib = _compile_c11(
        tmp_path,
        "host_v1_indep",
        """
        #include "pyrpmd/pes_api_v1.h"
        #include <stdlib.h>
        #include <string.h>

        struct pyrpmd_pes_context_v1 { int ok; };
        static const char META[] =
            "{\\"schema\\":\\"pyrpmd.pes/1\\",\\"abi_version\\":1,"
            "\\"provider_name\\":\\"host-indep\\",\\"provider_version\\":\\"1\\","
            "\\"natoms\\":1,\\"atomic_numbers\\":[1],"
            "\\"state\\":{\\"charge\\":0,\\"multiplicity\\":1},"
            "\\"layout\\":\\"B,N,3:C\\",\\"dtype\\":\\"float64\\","
            "\\"position_unit\\":\\"bohr\\",\\"energy_unit\\":\\"hartree\\","
            "\\"force_unit\\":\\"hartree/bohr\\",\\"boundary_conditions\\":\\"nonperiodic\\","
            "\\"batch_independent\\":true,"
            "\\"capabilities\\":{\\"stateful\\":false,\\"host_memory\\":true,"
            "\\"cuda_device_memory\\":false,\\"execution_device\\":\\"cpu\\"}}";

        int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
        pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1 *opt,
                                          pyrpmd_pes_context_v1 **out, pyrpmd_error_v1 *err) {
            (void)opt; (void)err;
            *out = (pyrpmd_pes_context_v1 *)calloc(1, sizeof(**out));
            return *out ? PYRPMD_OK_V1 : PYRPMD_INTERNAL_ERROR_V1;
        }
        pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1 *ctx, char *buf,
                                            int64_t cap, int64_t *req, pyrpmd_error_v1 *err) {
            (void)ctx; (void)err;
            *req = (int64_t)sizeof(META);
            if (cap < *req) return PYRPMD_BUFFER_TOO_SMALL_V1;
            memcpy(buf, META, sizeof(META));
            return PYRPMD_OK_V1;
        }
        pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1 *ctx,
                                            const pyrpmd_pes_input_v1 *x,
                                            pyrpmd_pes_output_v1 *y, pyrpmd_error_v1 *err) {
            (void)ctx; (void)err;
            for (int64_t i = 0; i < x->nstructures; ++i) {
                y->energies[i] = 0.0;
                y->structure_status[i] = 0;
                for (int j = 0; j < 3; ++j) y->forces[3*i + j] = 0.0;
            }
            return PYRPMD_OK_V1;
        }
        void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *ctx) { free(ctx); }
        """,
    )

    q_v1 = np.ones((4, 1, 3), dtype=np.float64) * 0.25
    with CABIPotential(v1_lib) as pes1:
        assert pes1.abi_version == 1
        assert pes1.device == "cpu"
        assert pes1._evaluate_cuda is None
        res1 = pes1.evaluate(q_v1)
        assert res1.energy.shape == (4,) and np.isfinite(res1.energy).all()
        with pytest.raises(PotentialError, match="CUDA device-pointer"):
            pes1.as_torch_pes()
        with pytest.raises(PotentialError, match="CUDA device-pointer"):
            pes1.evaluate_tensor(q_v1)

    cell = np.eye(3) * 5.0
    sys_v2 = {"symbols": ["H"], "atomic_numbers": [1], "cell": cell, "pbc": [True, True, True]}
    q_v2 = np.zeros((4, 1, 3), dtype=np.float64)
    with CABIPotential(v2_lib, system=sys_v2) as pes2:
        assert pes2.abi_version == 2
        assert pes2.device == "cpu"
        assert pes2._evaluate_cuda is None
        res2 = pes2.evaluate(q_v2)
        assert res2.virial is not None and res2.virial.shape == (4, 3, 3)
        with pytest.raises(PotentialError, match="CUDA device-pointer"):
            pes2.as_torch_pes()
        with pytest.raises(PotentialError, match="CUDA device-pointer"):
            pes2.evaluate_tensor(q_v2)

    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes

    with CABIPotential(v1_indep_lib) as pes_indep:
        with pytest.raises(PotentialError, match="no host/NumPy fallback"):
            make_workflow_torch_pes(pes_indep)
    with CABIPotential(v1_lib) as pes1_wf:
        with pytest.raises((PotentialError, ValueError)):
            make_workflow_torch_pes(pes1_wf)


@pytest.mark.parametrize(
    "case_kind,match",
    [
        ("only_version", "Malformed PES CUDA ABI"),
        ("only_evaluate", "Malformed PES CUDA ABI"),
        ("bad_cuda_version", "unsupported PES CUDA ABI version"),
        ("cap_true_no_symbols", "without exporting pyrpmd_pes_evaluate_cuda_v1"),
        ("cap_true_cpu_device", "execution_device='cuda:<nonnegative index>'"),
        ("cap_true_bare_cuda", "execution_device='cuda:<nonnegative index>'"),
        ("cap_non_bool", "capabilities.cuda_device_memory must be boolean"),
        ("conflicting_devices", "disagree"),
    ],
)
def test_malformed_cuda_symbol_and_metadata_combinations_clean_up_context(tmp_path, case_kind, match):
    export_ver = case_kind not in ("only_evaluate", "cap_true_no_symbols")
    export_eval = case_kind not in ("only_version", "cap_true_no_symbols")
    cuda_ver = 2 if case_kind == "bad_cuda_version" else 1

    if case_kind == "cap_non_bool":
        cap_json = '"cuda_device_memory":"yes","execution_device":"cuda:0"'
        top_dev = ""
    elif case_kind == "cap_true_cpu_device":
        cap_json = '"cuda_device_memory":true,"execution_device":"cpu"'
        top_dev = ""
    elif case_kind == "cap_true_bare_cuda":
        cap_json = '"cuda_device_memory":true,"execution_device":"cuda"'
        top_dev = ""
    elif case_kind == "conflicting_devices":
        cap_json = '"cuda_device_memory":true,"execution_device":"cuda:0"'
        top_dev = '"execution_device":"cuda:1",'
    else:
        cap_json = '"cuda_device_memory":true,"execution_device":"cuda:0"'
        top_dev = ""

    meta_c = (
        '{"schema":"pyrpmd.pes/1","abi_version":1,'
        '"provider_name":"malformed","provider_version":"1",'
        '"natoms":1,"atomic_numbers":[1],'
        '"state":{"charge":0,"multiplicity":1},'
        '"layout":"B,N,3:C","dtype":"float64",'
        '"position_unit":"bohr","energy_unit":"hartree",'
        '"force_unit":"hartree/bohr","boundary_conditions":"nonperiodic",'
        + top_dev
        + '"capabilities":{"stateful":false,'
        + cap_json
        + "}}"
    ).replace('"', '\\"')

    extra_c = ""
    if export_ver:
        extra_c += f"\nint32_t pyrpmd_pes_cuda_abi_version_v1(void) {{ return {cuda_ver}; }}\n"
    if export_eval:
        extra_c += """
        pyrpmd_cuda_rc_v1 pyrpmd_pes_evaluate_cuda_v1(
            void *ctx,
            const pyrpmd_pes_cuda_input_v1 *x,
            pyrpmd_pes_cuda_output_v1 *y,
            pyrpmd_cuda_error_v1 *err) {
            (void)ctx; (void)x; (void)y; (void)err;
            return 0;
        }
        """

    lib_path = _compile_c11(
        tmp_path,
        f"malformed_{case_kind}",
        f"""
        #include "pyrpmd/pes_api_v1.h"
        #include "pyrpmd/pes_api_cuda_v1.h"
        #include <stdlib.h>
        #include <string.h>

        static int LIVE_CONTEXTS = 0;
        int live_contexts(void) {{ return LIVE_CONTEXTS; }}

        struct pyrpmd_pes_context_v1 {{ int alive; }};
        static const char META[] = "{meta_c}";

        int32_t pyrpmd_pes_abi_version_v1(void) {{ return 1; }}
        pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1 *opt,
                                          pyrpmd_pes_context_v1 **out, pyrpmd_error_v1 *err) {{
            (void)opt; (void)err;
            *out = (pyrpmd_pes_context_v1 *)calloc(1, sizeof(**out));
            if (!*out) return PYRPMD_INTERNAL_ERROR_V1;
            LIVE_CONTEXTS += 1;
            return PYRPMD_OK_V1;
        }}
        pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1 *ctx, char *buf,
                                            int64_t cap, int64_t *req, pyrpmd_error_v1 *err) {{
            (void)ctx; (void)err;
            *req = (int64_t)sizeof(META);
            if (cap < *req) return PYRPMD_BUFFER_TOO_SMALL_V1;
            memcpy(buf, META, sizeof(META));
            return PYRPMD_OK_V1;
        }}
        pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1 *ctx,
                                            const pyrpmd_pes_input_v1 *x,
                                            pyrpmd_pes_output_v1 *y, pyrpmd_error_v1 *err) {{
            (void)ctx; (void)x; (void)y; (void)err;
            return PYRPMD_OK_V1;
        }}
        void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *ctx) {{
            if (ctx) {{
                LIVE_CONTEXTS -= 1;
                free(ctx);
            }}
        }}
        {extra_c}
        """,
    )
    probe = C.CDLL(str(lib_path))
    probe.live_contexts.restype = C.c_int
    with pytest.raises(PotentialError, match=match):
        CABIPotential(lib_path)
    assert probe.live_contexts() == 0, "failed CABIPotential construction must destroy created C context"


def test_torch_materials_rejects_host_only_cabi_without_creating_output(tmp_path):
    torch = pytest.importorskip("torch")
    from pyrpmd.runner import prepare, run

    v1_lib = _compile_c11(
        tmp_path, "wf_host_v1", extra_sources=[ROOT / "native/examples/harmonic_pes.c"]
    )
    job = {
        "task": "pimd",
        "temperature": "300 K",
        "beads": 4,
        "seed": 11,
        "system": {
            "symbols": ["H"],
            "masses": [1836.0],
            "geometry": [[0.1, 0.2, -0.1]],
            "geometry_unit": "bohr",
        },
        "potential": {"provider": "cabi", "library": str(v1_lib), "params": {}},
        "dynamics": {"timestep": 0.04, "integrator": "cayley-bcb"},
        "sampling": {
            "engine": "torch",
            "device": "cpu",
            "thermostat": "pile-l",
            "integrator": "bcocb",
            "gamma_centroid": 0.5,
            "equilibration_steps": 2,
            "production_steps": 6,
            "sample_interval": 2,
            "replicas": 2,
        },
        "output": {"checkpoint_steps": 4, "chunk_frames": 3},
    }
    # 1. CPU sampling.device with C ABI provider is rejected before creating output.
    out_dir = tmp_path / "no-output"
    with pytest.raises(ValueError, match="require a CUDA device"):
        run(job, output=out_dir)
    assert not out_dir.exists()

    # 2. CUDA sampling.device passes structural check without loading the library,
    #    and fails fast at numerical check / run when the library is host-only.
    job["sampling"]["device"] = "cuda:0"
    assert prepare(job, numerical=False)["ready"] is True
    if torch.cuda.is_available():
        job["output"]["directory"] = str(out_dir)
        with pytest.raises((PotentialError, ValueError)):
            prepare(job, numerical=True)
        assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 2. Real CUDA GPU ABI v2 periodic-cosine + strain-virial plugin tests
# ---------------------------------------------------------------------------

_CUDA_V2_PERIODIC_SOURCE = r"""
#include "pyrpmd/pes_api_v2.h"
#include "pyrpmd/pes_api_cuda_v1.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct pyrpmd_pes_context_v2 {
    int device_id;
};

static constexpr double TAU = 6.283185307179586476925286766559;

static const char META_TEMPLATE[] =
    "{\"schema\":\"pyrpmd.pes/2\",\"abi_version\":2,"
    "\"provider_name\":\"periodic-cosine-cuda-v2\",\"provider_version\":\"1\","
    "\"natoms\":1,\"atomic_numbers\":[1],"
    "\"state\":{\"charge\":0,\"multiplicity\":1},"
    "\"layout\":\"B,N,3:C\",\"dtype\":\"float64\","
    "\"position_unit\":\"bohr\",\"energy_unit\":\"hartree\","
    "\"force_unit\":\"hartree/bohr\",\"boundary_conditions\":\"periodic\","
    "\"cell_layout\":\"B,3,3:C:rows\",\"pbc_layout\":\"B,3:C\","
    "\"virial_unit\":\"hartree\",\"virial_layout\":\"B,3,3:C\","
    "\"virial_convention\":\"minus_dE_dstrain\","
    "\"batch_independent\":true,"
    "\"execution_device\":\"cuda:0\","
    "\"capabilities\":{\"native_batch\":true,\"virial\":true,"
    "\"stateful\":false,\"host_memory\":true,\"cuda_device_memory\":true,"
    "\"thread_safety\":\"context_serial\",\"requires_bead_affinity\":false,"
    "\"execution_device\":\"cuda:0\"}}";

static void set_v2_err(pyrpmd_error_v2 *err, const char *msg) {
    if (!err) return;
    err->required_capacity = static_cast<int64_t>(std::strlen(msg)) + 1;
    if (err->message && err->capacity > 0)
        std::snprintf(err->message, static_cast<size_t>(err->capacity), "%s", msg);
}

static void set_cuda_err(pyrpmd_cuda_error_v1 *err, const char *msg) {
    if (!err) return;
    err->required_capacity = static_cast<int64_t>(std::strlen(msg)) + 1;
    if (err->message && err->capacity > 0)
        std::snprintf(err->message, static_cast<size_t>(err->capacity), "%s", msg);
}

__host__ __device__ static bool invert3x3(const double *h, double *inv) {
    double det = h[0]*(h[4]*h[8]-h[5]*h[7])
               - h[1]*(h[3]*h[8]-h[5]*h[6])
               + h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (!isfinite(det) || det == 0.0) return false;
    inv[0]=(h[4]*h[8]-h[5]*h[7])/det;
    inv[1]=(h[2]*h[7]-h[1]*h[8])/det;
    inv[2]=(h[1]*h[5]-h[2]*h[4])/det;
    inv[3]=(h[5]*h[6]-h[3]*h[8])/det;
    inv[4]=(h[0]*h[8]-h[2]*h[6])/det;
    inv[5]=(h[2]*h[3]-h[0]*h[5])/det;
    inv[6]=(h[3]*h[7]-h[4]*h[6])/det;
    inv[7]=(h[1]*h[6]-h[0]*h[7])/det;
    inv[8]=(h[0]*h[4]-h[1]*h[3])/det;
    for (int j = 0; j < 9; ++j) {
        if (!isfinite(h[j]) || !isfinite(inv[j])) return false;
    }
    return true;
}

__host__ __device__ static void eval_one_frame(
    int64_t i,
    const double *positions,
    const double *cells,
    const int32_t *pbc,
    double *energies,
    double *forces,
    double *virials,
    int32_t *status) {
    const double *q = positions + 3 * i;
    double *f = forces + 3 * i;
    double *w = virials ? (virials + 9 * i) : nullptr;
    double energy = 0.0;
    double inv[9];
    bool valid = true;
    for (int j = 0; j < 3; ++j) {
        f[j] = 0.0;
        if (!isfinite(q[j])) valid = false;
    }
    if (w) {
        for (int j = 0; j < 9; ++j) w[j] = 0.0;
    }
    if (!cells) {
        for (int j = 0; j < 3; ++j) {
            if (pbc[3 * i + j] != 0) valid = false;
            energy += 0.5 * q[j] * q[j];
            f[j] = -q[j];
            if (w) {
                for (int k = 0; k < 3; ++k) w[3 * j + k] = -q[j] * q[k];
            }
        }
    } else {
        const double *h = cells + 9 * i;
        for (int j = 0; j < 3; ++j) {
            if (pbc[3 * i + j] != 0 && pbc[3 * i + j] != 1) valid = false;
        }
        valid = valid && invert3x3(h, inv);
        if (valid) {
            for (int a = 0; a < 3; ++a) {
                double s = 0.0, norm2 = 0.0;
                for (int j = 0; j < 3; ++j) {
                    s += q[j] * inv[3 * j + a];
                    norm2 += h[3 * a + j] * h[3 * a + j];
                }
                double g = pbc[3 * i + a] ? (1.0 - cos(TAU * s)) / (TAU * TAU) : 0.5 * s * s;
                double deriv = pbc[3 * i + a] ? sin(TAU * s) / TAU : s;
                energy += norm2 * g;
                for (int j = 0; j < 3; ++j) {
                    f[j] -= norm2 * deriv * inv[3 * j + a];
                    if (w) {
                        for (int k = 0; k < 3; ++k)
                            w[3 * j + k] -= 2.0 * g * h[3 * a + j] * h[3 * a + k];
                    }
                }
            }
        }
    }
    valid = valid && isfinite(energy);
    for (int j = 0; j < 3; ++j) valid = valid && isfinite(f[j]);
    if (w) {
        for (int j = 0; j < 9; ++j) valid = valid && isfinite(w[j]);
    }
    if (status) {
        status[i] = valid ? PYRPMD_CUDA_STRUCTURE_OK_V1 : PYRPMD_CUDA_STRUCTURE_NONFINITE_V1;
    }
    energies[i] = valid ? energy : NAN;
    if (!valid) {
        for (int j = 0; j < 3; ++j) f[j] = NAN;
        if (w) {
            for (int j = 0; j < 9; ++j) w[j] = NAN;
        }
    }
}

__global__ static void periodic_cosine_cuda_kernel(
    int64_t b,
    const double *positions,
    const double *cells,
    const int32_t *pbc,
    double *energies,
    double *forces,
    double *virials,
    int32_t *status) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < b) {
        eval_one_frame(i, positions, cells, pbc, energies, forces, virials, status);
    }
}

extern "C" int32_t pyrpmd_pes_abi_version_v2(void) { return 2; }
extern "C" int32_t pyrpmd_pes_cuda_abi_version_v1(void) { return 1; }

extern "C" pyrpmd_rc_v2 pyrpmd_pes_create_v2(
    const pyrpmd_create_options_v2 *options,
    pyrpmd_pes_context_v2 **out,
    pyrpmd_error_v2 *error) {
    if (!out) return PYRPMD_INVALID_ARGUMENT_V2;
    *out = nullptr;
    if (!options || options->abi_version != 2 ||
        options->struct_size != static_cast<int32_t>(sizeof(*options)))
        return PYRPMD_ABI_MISMATCH_V2;
    auto *ctx = static_cast<pyrpmd_pes_context_v2 *>(std::calloc(1, sizeof(pyrpmd_pes_context_v2)));
    if (!ctx) return PYRPMD_INTERNAL_ERROR_V2;
    ctx->device_id = 0;
    *out = ctx;
    set_v2_err(error, "");
    return PYRPMD_OK_V2;
}

extern "C" pyrpmd_rc_v2 pyrpmd_pes_metadata_v2(
    const pyrpmd_pes_context_v2 *context,
    char *out,
    int64_t capacity,
    int64_t *required,
    pyrpmd_error_v2 *error) {
    if (!context || !required || capacity < 0) return PYRPMD_INVALID_ARGUMENT_V2;
    *required = static_cast<int64_t>(sizeof(META_TEMPLATE));
    if (capacity < *required) {
        if (out && capacity > 0) out[0] = '\0';
        return PYRPMD_BUFFER_TOO_SMALL_V2;
    }
    if (!out) return PYRPMD_INVALID_ARGUMENT_V2;
    std::memcpy(out, META_TEMPLATE, sizeof(META_TEMPLATE));
    set_v2_err(error, "");
    return PYRPMD_OK_V2;
}

extern "C" pyrpmd_rc_v2 pyrpmd_pes_evaluate_v2(
    pyrpmd_pes_context_v2 *context,
    const pyrpmd_pes_input_v2 *x,
    pyrpmd_pes_output_v2 *y,
    pyrpmd_error_v2 *error) {
    if (!context || !x || !y) return PYRPMD_INVALID_ARGUMENT_V2;
    if (x->abi_version != 2 || y->abi_version != 2 ||
        x->struct_size != static_cast<int32_t>(sizeof(*x)) ||
        y->struct_size != static_cast<int32_t>(sizeof(*y)))
        return PYRPMD_ABI_MISMATCH_V2;
    int64_t b = x->nstructures;
    if (b < 1 || x->natoms != 1 || x->positions_count != 3 * b || x->pbc_count != 3 * b ||
        y->energies_count != b || y->forces_count != 3 * b || y->structure_status_count != b)
        return PYRPMD_INVALID_ARGUMENT_V2;
    for (int64_t i = 0; i < b; ++i) {
        eval_one_frame(i, x->positions, x->cells, x->pbc,
                       y->energies, y->forces, y->virials, y->structure_status);
        if (y->structure_messages && y->structure_message_stride > 0) {
            std::snprintf(
                y->structure_messages + i * y->structure_message_stride,
                static_cast<size_t>(y->structure_message_stride),
                "%s",
                y->structure_status[i] == 0 ? "" : "nonfinite geometry or result");
        }
    }
    set_v2_err(error, "");
    return PYRPMD_OK_V2;
}

extern "C" pyrpmd_cuda_rc_v1 pyrpmd_pes_evaluate_cuda_v1(
    void *raw_context,
    const pyrpmd_pes_cuda_input_v1 *x,
    pyrpmd_pes_cuda_output_v1 *y,
    pyrpmd_cuda_error_v1 *error) {
    auto *context = static_cast<pyrpmd_pes_context_v2 *>(raw_context);
    if (!context || !x || !y) {
        set_cuda_err(error, "null pointer");
        return PYRPMD_CUDA_INVALID_ARGUMENT_V1;
    }
    if (x->cuda_abi_version != 1 || y->cuda_abi_version != 1 ||
        x->struct_size != static_cast<int32_t>(sizeof(*x)) ||
        y->struct_size != static_cast<int32_t>(sizeof(*y))) {
        set_cuda_err(error, "CUDA ABI version/size mismatch");
        return PYRPMD_CUDA_ABI_MISMATCH_V1;
    }
    if (x->device_id != context->device_id || x->flags != 0 ||
        x->layout != 1 || x->scalar_type != 1) {
        set_cuda_err(error, "invalid device_id/flags/layout/scalar_type");
        return PYRPMD_CUDA_INVALID_ARGUMENT_V1;
    }
    int64_t b = x->nstructures;
    if (b < 1 || x->natoms != 1 || !x->d_positions || x->positions_count != 3 * b ||
        !x->d_pbc || x->pbc_count != 3 * b || !y->d_energies || y->energies_count != b ||
        !y->d_forces || y->forces_count != 3 * b) {
        set_cuda_err(error, "invalid CUDA v2 counts or null buffers");
        return PYRPMD_CUDA_INVALID_ARGUMENT_V1;
    }
    if ((x->d_cells && x->cells_count != 9 * b) || (!x->d_cells && x->cells_count != 0) ||
        (y->d_virials && y->virials_count != 9 * b) || (!y->d_virials && y->virials_count != 0) ||
        (y->d_structure_status && y->structure_status_count != b) ||
        (!y->d_structure_status && y->structure_status_count != 0)) {
        set_cuda_err(error, "invalid optional cell/virial/status counts");
        return PYRPMD_CUDA_INVALID_ARGUMENT_V1;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (prev != context->device_id) cudaSetDevice(context->device_id);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(x->cuda_stream);
    int threads = 128;
    int blocks = static_cast<int>((b + threads - 1) / threads);
    periodic_cosine_cuda_kernel<<<blocks, threads, 0, stream>>>(
        b, x->d_positions, x->d_cells, x->d_pbc,
        y->d_energies, y->d_forces, y->d_virials, y->d_structure_status);
    cudaError_t launch_err = cudaGetLastError();
    if (prev != context->device_id) cudaSetDevice(prev);
    if (launch_err != cudaSuccess) {
        set_cuda_err(error, cudaGetErrorString(launch_err));
        return PYRPMD_CUDA_BACKEND_FAILURE_V1;
    }
    set_cuda_err(error, "");
    return PYRPMD_CUDA_OK_V1;
}

extern "C" void pyrpmd_pes_destroy_v2(pyrpmd_pes_context_v2 *context) {
    std::free(context);
}
"""


@pytest.fixture(scope="module")
def cuda_v2_library(tmp_path_factory):
    _require_cuda_torch()
    return _compile_nvcc(tmp_path_factory.mktemp("cabi_cuda_v2"), "periodic_cuda_v2", _CUDA_V2_PERIODIC_SOURCE)


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize(
    "cell_mode,pbc",
    [
        ("triclinic", [True, True, True]),
        ("triclinic", [True, True, False]),
        ("none", [False, False, False]),
    ],
)
def test_cuda_v2_device_pointer_matches_host_v2_energies_forces_and_virials(
    cuda_v2_library, batch, cell_mode, pbc
):
    torch = _require_cuda_torch()
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
    from pyrpmd.torch_backend import TorchDynamics, TorchState

    cell = (
        None
        if cell_mode == "none"
        else np.array([[6.0, 0.25, -0.1], [1.1, 5.2, 0.3], [0.35, -0.4, 4.4]], dtype=np.float64)
    )
    system = {
        "symbols": ["H"],
        "atomic_numbers": [1],
        "charge": 0,
        "multiplicity": 1,
        "cell": cell,
        "pbc": pbc,
    }
    rng = np.random.default_rng(104 + batch)
    q_np = rng.normal(0.15, 0.65, size=(batch, 1, 3)).astype(np.float64)

    with CABIPotential(cuda_v2_library, system=system) as pes:
        assert pes.abi_version == 2
        assert pes.device == "cuda:0"
        assert pes.has_virial is True

        host_res = pes.evaluate(q_np)
        q_gpu = torch.tensor(q_np, dtype=torch.float64, device="cuda:0")
        gpu_res, gpu_status = pes.evaluate_tensor(q_gpu, return_status=True)

        assert gpu_status.dtype == torch.int32 and gpu_status.device.type == "cuda"
        assert torch.all(gpu_status == 0).item()
        assert gpu_res.virial is not None and gpu_res.virial.device.type == "cuda"

        e_err = np.max(np.abs(gpu_res.energy.cpu().numpy() - host_res.energy))
        f_err = np.max(np.abs(gpu_res.forces.cpu().numpy() - host_res.forces))
        w_err = np.max(np.abs(gpu_res.virial.cpu().numpy() - host_res.virial))
        assert e_err < 1e-13, f"energy parity error {e_err} >= 1e-13"
        assert f_err < 1e-13, f"force parity error {f_err} >= 1e-13"
        assert w_err < 1e-13, f"virial parity error {w_err} >= 1e-13"

        # Verify bind_system updates cached CUDA cell/PBC tensors cleanly.
        if cell is not None:
            scaled_sys = {**system, "cell": cell * 1.07}
            pes.bind_system(scaled_sys)
            host_rebound = pes.evaluate(q_np)
            bound = make_workflow_torch_pes(pes, device="cuda:0")
            gpu_rebound = bound.evaluate_tensor(q_gpu)
            assert np.max(np.abs(gpu_rebound.energy.cpu().numpy() - host_rebound.energy)) < 1e-13
            assert np.max(np.abs(gpu_rebound.forces.cpu().numpy() - host_rebound.forces)) < 1e-13
            assert np.max(np.abs(gpu_rebound.virial.cpu().numpy() - host_rebound.virial)) < 1e-13

            if batch == 8:
                dyn = TorchDynamics(bound, [1836.15], beta=800.0, nbeads=8, dt=0.2)
                state = TorchState(
                    q_gpu.reshape(1, 8, 1, 3).clone(),
                    dyn.sample_momenta(1),
                )
                dyn.propagate(state, 10)
                assert torch.isfinite(state.q).all().item()


# ---------------------------------------------------------------------------
# 3. Real libpyrpmd_onnx.so CUDA IoBinding, TorchDynamics & stream tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize(
    "length,energy",
    [
        ("angstrom", "eV"),
        ("bohr", "hartree"),
        ("bohr", "eV"),
        ("angstrom", "hartree"),
    ],
)
def test_onnx_cuda_device_pointer_matches_host_v1_within_1e13(tmp_path, batch, length, energy):
    torch, onnx_mod = _require_onnx_cuda()
    model_path = _make_onnx_anharmonic_pair(
        onnx_mod, tmp_path / "anharmonic.onnx", length=length, energy=energy
    )
    rng = np.random.default_rng(203 + batch)
    q_np = rng.normal(0.05, 0.35, size=(batch, 3, 3)).astype(np.float64)

    with CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cpu"}) as cpu_pes, \
         CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cuda:0"}) as gpu_pes:
        assert gpu_pes.device == "cuda:0"
        assert gpu_pes.metadata["batch_independent"] is True
        assert gpu_pes.metadata["capabilities"]["cuda_device_memory"] is True
        assert gpu_pes.metadata["execution_device"] == "cuda:0"

        host_cpu = cpu_pes.evaluate(q_np)
        host_gpu_v1 = gpu_pes.evaluate(q_np)

        q_gpu = torch.tensor(q_np, dtype=torch.float64, device="cuda:0")
        q_gpu_before = q_gpu.clone()
        dev_res, dev_status = gpu_pes.evaluate_tensor(q_gpu, return_status=True)

        # Input coordinate buffer must never be mutated in place by unit scaling!
        assert torch.equal(q_gpu, q_gpu_before)
        assert torch.all(dev_status == 0).item()
        assert dev_res.virial is None

        for ref_name, ref in (("cpu_v1", host_cpu), ("cuda_host_v1", host_gpu_v1)):
            e_diff = np.max(np.abs(dev_res.energy.cpu().numpy() - ref.energy))
            f_diff = np.max(np.abs(dev_res.forces.cpu().numpy() - ref.forces))
            assert e_diff < 1e-13, f"{ref_name} energy diff {e_diff} >= 1e-13"
            assert f_diff < 1e-13, f"{ref_name} force diff {f_diff} >= 1e-13"


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "cayley-bcocb"])
@pytest.mark.parametrize("layout", ["rfft", "rpmdrate"])
def test_onnx_cuda_torch_dynamics_zero_host_roundtrips_and_trajectory_parity(
    tmp_path, monkeypatch, integrator, layout
):
    torch, onnx_mod = _require_onnx_cuda()
    from pyrpmd.dynamics import Dynamics, State, native_capabilities
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
    from pyrpmd.torch_backend import TorchDynamics, TorchState

    model_path = _make_onnx_anharmonic_pair(
        onnx_mod, tmp_path / "dyn.onnx", length="angstrom", energy="eV"
    )
    masses = [1836.15, 1836.15, 29156.95]
    beads = 8
    dt = 0.08
    beta = 1052.5
    rng = np.random.default_rng(77)
    q0 = rng.normal(0.0, 0.2, size=(2, beads, 3, 3)).astype(np.float64)
    p0 = rng.normal(0.0, 1.5, size=(2, beads, 3, 3)).astype(np.float64)

    with CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cuda:0"}) as gpu_pes:
        bound = make_workflow_torch_pes(gpu_pes, device="cuda:0")
        thermostat = "pile-l" if integrator == "cayley-bcocb" else None
        gamma = 0.002 if thermostat is not None else 0.0
        dyn = TorchDynamics(
            bound,
            masses=masses,
            beta=beta,
            nbeads=beads,
            dt=dt,
            integrator=integrator,
            thermostat=thermostat,
            gamma_centroid=gamma,
            normal_mode_layout=layout,
            seed=99,
        )
        state = TorchState(
            torch.tensor(q0, dtype=torch.float64, device="cuda:0"),
            torch.tensor(p0, dtype=torch.float64, device="cuda:0"),
        )

        # Warm up first evaluation if needed, then enforce zero inner-step host copies.
        orig_item = torch.Tensor.item
        item_calls = []

        def counted_item(tensor_self, *args, **kwargs):
            item_calls.append(tuple(tensor_self.shape))
            return orig_item(tensor_self, *args, **kwargs)

        def forbidden_transfer(*args, **kwargs):
            raise AssertionError("forbidden host-device round-trip inside TorchDynamics.propagate")

        with monkeypatch.context() as mp:
            mp.setattr(gpu_pes, "_evaluate", forbidden_transfer)
            mp.setattr(torch.Tensor, "cpu", forbidden_transfer)
            mp.setattr(torch.Tensor, "numpy", forbidden_transfer)
            mp.setattr(torch.Tensor, "tolist", forbidden_transfer)
            mp.setattr(torch.Tensor, "item", counted_item)
            dyn.propagate(state, 20)

        # TorchDynamics reads exactly one scalar finiteness flag at the end of the block.
        assert len(item_calls) == 1
        assert sum(dyn.force_evaluations.values()) > 0

        # For deterministic NVE integrators, verify multi-step trajectory agreement
        # against the independent CPU C++ Dynamics integrator!
        if thermostat is None and layout == "rfft" and native_capabilities()["available"]:
            q_final = state.q.cpu().numpy()
            p_final = state.p.cpu().numpy()
            with CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cpu"}) as cpu_pes:
                cpu_dyn = Dynamics(cpu_pes, masses, beta, beads, dt, integrator=integrator)
                for r in range(2):
                    s_cpu = State(q0[r].copy(), p0[r].copy())
                    cpu_dyn.propagate_block(s_cpu, 20, backend="native")
                    np.testing.assert_allclose(q_final[r], s_cpu.q, atol=2e-11, rtol=1e-11)
                    np.testing.assert_allclose(p_final[r], s_cpu.p, atol=2e-10, rtol=1e-11)


def test_onnx_cuda_rpc_contraction_and_public_torch_materials_workflow(tmp_path):
    torch, onnx_mod = _require_onnx_cuda()
    from pyrpmd.runner import prepare, resume_run, run
    from pyrpmd.torch_backend import TorchDynamics, TorchRPCTerm, TorchSplitPotential, TorchState

    model_path = _make_onnx_anharmonic_pair(
        onnx_mod, tmp_path / "wf.onnx", length="angstrom", energy="eV"
    )
    masses = [1836.15, 1836.15, 29156.95]

    with CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cuda:0"}) as gpu_pes:
        torch_pes = gpu_pes.as_torch_pes()
        split = TorchSplitPotential([TorchRPCTerm(torch_pes, nbeads=16, contracted_beads=4)])
        dyn = TorchDynamics(split, masses=masses, beta=900.0, nbeads=16, dt=0.06, integrator="cayley-bcb")
        q = torch.zeros((3, 16, 3, 3), dtype=torch.float64, device="cuda:0")
        state = TorchState(q, dyn.sample_momenta(3))
        dyn.propagate(state, 8)
        assert torch.isfinite(dyn.energy(state)).all().item()

    for task in ("pimd", "diffusion"):
        job = {
            "task": task,
            "temperature": "300 K",
            "beads": 4,
            "seed": 31,
            "system": {
                "symbols": ["H", "H", "O"],
                "masses": masses,
                "geometry": [[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]],
                "geometry_unit": "angstrom",
            },
            "potential": {
                "provider": "cabi",
                "library": str(ONNX_LIBRARY),
                "params": {"model": str(model_path), "device": "cuda:0"},
            },
            "dynamics": {"timestep": 0.05, "integrator": "cayley-bcb", "block_steps": 4},
            "sampling": {
                "engine": "torch",
                "device": "cuda:0",
                "normal_mode_layout": "rfft",
                "thermostat": "pile-l",
                "integrator": "bcocb",
                "gamma_centroid": 0.5,
                "equilibration_steps": 4,
                "production_steps": 12,
                "sample_interval": 2,
                "replicas": 4,
            },
            "output": {"checkpoint_steps": 6, "chunk_frames": 3},
        }
        if task == "diffusion":
            job["diffusion"] = {"atom_indices": [0, 1], "axes": [0, 1, 2], "bootstrap": 0, "max_lag": 3}

        prep = prepare(job, numerical=True)
        assert prep["pes_check"]["engine"] == "torch"
        assert prep["pes_check"]["device"] == "cuda:0"
        assert prep["pes_check"]["status"] == [0] * 16

        out_dir = tmp_path / f"run_{task}"
        summary = run(job, output=out_dir)
        assert summary["engine"] == "torch"
        assert summary["tensor_runtime"]["device"] == "cuda:0"
        assert summary["status"] == "complete"
        resumed = resume_run(out_dir)
        assert resumed["status"] == "complete"


def test_onnx_cuda_non_default_stream_ordering_and_multi_stream_safety(tmp_path):
    torch, onnx_mod = _require_onnx_cuda()
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
    from pyrpmd.torch_backend import TorchDynamics, TorchState

    model_path = _make_onnx_anharmonic_pair(
        onnx_mod, tmp_path / "stream.onnx", length="angstrom", energy="eV"
    )
    rng = np.random.default_rng(512)
    q_base_np = rng.normal(0.0, 0.25, size=(64, 3, 3)).astype(np.float64)

    with CABIPotential(ONNX_LIBRARY, {"model": str(model_path), "device": "cuda:0"}) as gpu_pes:
        bound = make_workflow_torch_pes(gpu_pes, device="cuda:0")
        s1 = torch.cuda.Stream(device="cuda:0")
        s2 = torch.cuda.Stream(device="cuda:0")
        bg_stream = torch.cuda.Stream(device="cuda:0")

        q_base = torch.tensor(q_base_np, dtype=torch.float64, device="cuda:0")
        mat_a = torch.randn((1024, 1024), dtype=torch.float32, device="cuda:0")
        mat_b = torch.randn((1024, 1024), dtype=torch.float32, device="cuda:0")
        bg_a = torch.randn((1024, 1024), dtype=torch.float32, device="cuda:0")
        torch.cuda.synchronize()

        outputs_s1 = []
        outputs_s2 = []
        expected_s1 = []
        expected_s2 = []

        # Keep background stream busy while s1 and s2 issue upstream matmuls ->
        # coordinate writes -> ORT IoBinding evaluate_tensor -> downstream ops
        # with ZERO host synchronization inside the loop!
        for step in range(1, 13):
            scale1 = 0.01 * step
            scale2 = -0.008 * step
            expected_s1.append(q_base_np * (1.0 + scale1))
            expected_s2.append(q_base_np * (1.0 + scale2))

            with torch.cuda.stream(bg_stream):
                bg_c = bg_a @ bg_a

            with torch.cuda.stream(s1):
                # Heavy upstream GPU work that determines the coordinate tensor q1:
                work1 = mat_a @ mat_b
                zero1 = (work1[0, 0].to(torch.float64) * 0.0)
                q1 = q_base * (1.0 + scale1) + zero1
                r1 = bound.evaluate_tensor(q1)
                # Downstream consumer immediately reads forces & energy on s1:
                combined1 = r1.forces + r1.energy[:, None, None] * 0.1
                outputs_s1.append((r1.energy, r1.forces, combined1))

            with torch.cuda.stream(s2):
                work2 = mat_b @ mat_a
                zero2 = (work2[0, 0].to(torch.float64) * 0.0)
                q2 = q_base * (1.0 + scale2) + zero2
                r2 = bound.evaluate_tensor(q2)
                combined2 = r2.forces - r2.energy[:, None, None] * 0.1
                outputs_s2.append((r2.energy, r2.forces, combined2))

        s1.synchronize()
        s2.synchronize()
        bg_stream.synchronize()
        assert bg_c.shape == (1024, 1024)

        for q_exp, (e_t, f_t, comb_t) in zip(expected_s1, outputs_s1):
            ref = gpu_pes.evaluate(q_exp)
            e_np = e_t.cpu().numpy()
            f_np = f_t.cpu().numpy()
            c_np = comb_t.cpu().numpy()
            assert np.max(np.abs(e_np - ref.energy)) < 1e-13
            assert np.max(np.abs(f_np - ref.forces)) < 1e-13
            assert np.max(np.abs(c_np - (ref.forces + ref.energy[:, None, None] * 0.1))) < 1e-13

        for q_exp, (e_t, f_t, comb_t) in zip(expected_s2, outputs_s2):
            ref = gpu_pes.evaluate(q_exp)
            e_np = e_t.cpu().numpy()
            f_np = f_t.cpu().numpy()
            c_np = comb_t.cpu().numpy()
            assert np.max(np.abs(e_np - ref.energy)) < 1e-13
            assert np.max(np.abs(f_np - ref.forces)) < 1e-13
            assert np.max(np.abs(c_np - (ref.forces - ref.energy[:, None, None] * 0.1))) < 1e-13

        # Also verify full TorchDynamics propagation on a non-default stream matches
        # default-stream deterministic propagation bit-for-bit / < 1e-14!
        masses = [1836.15, 1836.15, 29156.95]
        q_dyn = torch.tensor(q_base_np[:16].reshape(2, 8, 3, 3), dtype=torch.float64, device="cuda:0")
        p_dyn = torch.ones_like(q_dyn) * 0.3
        dyn_default = TorchDynamics(bound, masses, beta=900.0, nbeads=8, dt=0.05, integrator="cayley-bcb")
        st_default = TorchState(q_dyn.clone(), p_dyn.clone())
        dyn_default.propagate(st_default, 15)

        st_nondefault = TorchState(q_dyn.clone(), p_dyn.clone())
        with torch.cuda.stream(s1):
            upstream = mat_a @ mat_b
            st_nondefault.q.add_(upstream[0, 0].to(torch.float64) * 0.0)
            dyn_default.propagate(st_nondefault, 15)
        s1.synchronize()
        torch.testing.assert_close(st_nondefault.q, st_default.q, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(st_nondefault.p, st_default.p, atol=1e-14, rtol=1e-14)


def test_onnx_cuda_nonfinite_inputs_outputs_and_invalid_pointers(tmp_path):
    torch, onnx_mod = _require_onnx_cuda()
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
    from pyrpmd.torch_backend import TorchDynamics, TorchState

    # 1. Bounded exponential model where +inf input evaluates to finite 0.0 in
    #    pure IEEE arithmetic unless input coordinates are checked on device!
    bounded_path = _make_onnx_harmonic(
        onnx_mod, tmp_path / "bounded.onnx", bounded_gaussian=True
    )
    with CABIPotential(ONNX_LIBRARY, {"model": str(bounded_path), "device": "cuda:0"}) as pes:
        q = torch.full((8, 3, 3), 0.2, dtype=torch.float64, device="cuda:0")
        q[5, 1, 2] = float("inf")
        res, status = pes.evaluate_tensor(q, return_status=True)
        status_cpu = status.cpu().numpy()
        assert status_cpu[5] == 3
        assert np.all(status_cpu[[0, 1, 2, 3, 4, 6, 7]] == 0)
        assert np.isnan(res.energy[5].item())
        assert torch.isnan(res.forces[5]).all().item()
        assert torch.isfinite(res.energy[[0, 1, 2, 3, 4, 6, 7]]).all().item()
        assert torch.isfinite(res.forces[[0, 1, 2, 3, 4, 6, 7]]).all().item()

    # 2. Output NaN model rolls back TorchDynamics state + RNG atomically!
    nan_path = _make_onnx_harmonic(onnx_mod, tmp_path / "nan.onnx", output_nan=True)
    with CABIPotential(ONNX_LIBRARY, {"model": str(nan_path), "device": "cuda:0"}) as nan_pes:
        bound_nan = make_workflow_torch_pes(nan_pes, device="cuda:0")
        dyn = TorchDynamics(
            bound_nan,
            [1836.15, 1836.15, 29156.95],
            beta=1000.0,
            nbeads=4,
            dt=0.05,
            integrator="cayley-bcocb",
            thermostat="pile-l",
            gamma_centroid=0.01,
            seed=123,
        )
        q0 = torch.ones((2, 4, 3, 3), dtype=torch.float64, device="cuda:0") * 0.1
        p0 = dyn.sample_momenta(2)
        state = TorchState(q0.clone(), p0.clone())
        rng_before = dyn.generator.get_state()
        with pytest.raises(FloatingPointError, match="nonfinite"):
            dyn.propagate(state, 5)
        assert torch.equal(state.q, q0)
        assert torch.equal(state.p, p0)
        assert state.step == 0 and state.time == 0.0
        assert torch.equal(dyn.generator.get_state(), rng_before)

    # 3. Raw C ABI pointer validation: host pointer, aliased outputs, wrong device_id,
    #    float32/CPU tensors, and closed context.
    good_path = _make_onnx_harmonic(onnx_mod, tmp_path / "good.onnx")
    with CABIPotential(ONNX_LIBRARY, {"model": str(good_path), "device": "cuda:0"}) as pes:
        q_gpu = torch.zeros((4, 3, 3), dtype=torch.float64, device="cuda:0")
        e_gpu = torch.empty((4,), dtype=torch.float64, device="cuda:0")
        f_gpu = torch.empty((4, 3, 3), dtype=torch.float64, device="cuda:0")

        # Wrong dtype or CPU tensor rejected cleanly in Python without conversion:
        with pytest.raises(PotentialError, match="no implicit conversion"):
            pes.evaluate_tensor(q_gpu.to(torch.float32))
        with pytest.raises(PotentialError, match="no implicit conversion"):
            pes.evaluate_tensor(q_gpu.cpu())

        # Host NumPy pointer passed to raw pyrpmd_pes_evaluate_cuda_v1 -> INVALID_ARGUMENT (1)
        host_arr = np.zeros((4, 3, 3), dtype=np.float64)
        eb = C.create_string_buffer(1024)
        err = cabi.Error(C.addressof(eb), len(eb), 0)
        bad_host_inp = cabi.CudaInputV1(
            1, C.sizeof(cabi.CudaInputV1), 1, 1, 0, 0, 0, 4, 3,
            host_arr.ctypes.data, 36, 0, 0, 0, 0
        )
        good_out = cabi.CudaOutputV1(
            1, C.sizeof(cabi.CudaOutputV1), e_gpu.data_ptr(), 4, f_gpu.data_ptr(), 36, 0, 0, 0, 0
        )
        rc = pes._evaluate_cuda(pes._context, C.byref(bad_host_inp), C.byref(good_out), C.byref(err))
        assert rc == 1 and b"CUDA device memory" in eb.value

        # Aliased output buffers -> INVALID_ARGUMENT (1)
        aliased_out = cabi.CudaOutputV1(
            1, C.sizeof(cabi.CudaOutputV1), f_gpu.data_ptr(), 4, f_gpu.data_ptr(), 36, 0, 0, 0, 0
        )
        good_inp = cabi.CudaInputV1(
            1, C.sizeof(cabi.CudaInputV1), 1, 1, 0, 0, 0, 4, 3,
            q_gpu.data_ptr(), 36, 0, 0, 0, 0
        )
        rc = pes._evaluate_cuda(pes._context, C.byref(good_inp), C.byref(aliased_out), C.byref(err))
        assert rc == 1 and (b"overlap" in eb.value or b"alias" in eb.value)

        # Wrong device_id -> INVALID_ARGUMENT (1)
        wrong_dev_inp = cabi.CudaInputV1(
            1, C.sizeof(cabi.CudaInputV1), 1, 1, 7, 0, 0, 4, 3,
            q_gpu.data_ptr(), 36, 0, 0, 0, 0
        )
        rc = pes._evaluate_cuda(pes._context, C.byref(wrong_dev_inp), C.byref(good_out), C.byref(err))
        assert rc == 1

        # Subsequent valid GPU evaluation still works with clean CUDA state!
        res = pes.evaluate_tensor(q_gpu)
        assert torch.all(res.energy == 0.0).item()
        bound = pes.as_torch_pes()

    with pytest.raises(PotentialError, match="closed"):
        bound.evaluate_tensor(q_gpu)
