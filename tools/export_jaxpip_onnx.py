#!/usr/bin/env python3
"""Export and verify a JaxPIP model for the native PyRPMD ONNX PES plugin.

JAX/Equinox/jax2onnx are used only offline, not in the MD force loop. Input
geometries are an .npy array [M,N,3], in the model's declared length unit.
Run exports in separate processes, not concurrent threads in one interpreter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def _symbolic_pad_metadata_compat():
    """Work around one pinned converter's metadata-only symbolic Pad bug.

    jax2onnx 0.17.1 PadPlugin.lower passes JAX DimExpr directly to ir.Shape,
    which calls int(B) and fails. Only that plugin's local `ir` reference is
    wrapped, only for this export, and restored even if conversion fails.
    The emitted ONNX Pad operation and its inputs are completely unchanged.
    No installed dependency files or global onnx_ir classes are modified.
    """
    if importlib.metadata.version("jax2onnx") != "0.17.1":
        yield []
        return
    from jax.export import is_symbolic_dim
    from jax2onnx.plugins.jax.lax import pad
    original = pad.ir

    class PadIR:
        def __getattr__(self, name):
            return getattr(original, name)

        @staticmethod
        def Shape(dims, *args, **kwargs):
            normalized = [str(d) if is_symbolic_dim(d) else d for d in dims]
            return original.Shape(normalized, *args, **kwargs)

    pad.ir = PadIR()
    try:
        yield ["jax2onnx-0.17.1-symbolic-pad-metadata"]
    finally:
        pad.ir = original


def export_model(*, basis: Path, checkpoint: Path, geometries: Path, output: Path,
                 kind: str, atomic_numbers: list[int], charge: int, multiplicity: int,
                 position_unit: str, energy_unit: str, atol: float = 1e-9,
                 rtol: float = 1e-8, fd_step: float = 1e-5) -> dict:
    """Write the output atomically only after structural/numerical validation."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    from jax.extend.core import jaxpr_as_fun
    from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork
    from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU

    if not atomic_numbers or any(type(z) is not int or not 1 <= z <= 118 for z in atomic_numbers):
        raise ValueError("atomic_numbers must be ordered integers in [1,118]")
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    if position_unit not in ("angstrom", "bohr") or energy_unit not in ("eV", "hartree"):
        raise ValueError("unsupported model units")
    if not all(np.isfinite(v) and v > 0 for v in (atol, rtol, fd_step)):
        raise ValueError("tolerances and fd_step must be finite and positive")
    xyz = np.load(geometries, allow_pickle=False)
    n = len(atomic_numbers)
    if (xyz.ndim != 3 or xyz.shape[0] < 2 or xyz.shape[1:] != (n, 3)
            or xyz.dtype.kind not in "fiu" or not np.isfinite(xyz).all()):
        raise ValueError("validation geometries must be finite numeric [M>=2,N,3]")
    xyz = np.asarray(xyz, dtype=np.float64)
    cls = {"linear": PolynomialLinearModel, "nn": PolynomialNeuralNetwork}[kind]
    model = cls.from_file(basis_file=str(basis), model_file=str(checkpoint))
    if model.descriptor.basis_info.num_atoms != n or jnp.dtype(model.dtype) != jnp.float64:
        raise ValueError("model atom count/float64 precision mismatch")

    @eqx.filter_jit
    def batch_fn(q):
        return jax.vmap(model.get_energy_and_forces)(q)

    @eqx.filter_jit
    def energy_fn(q):
        return model.get_energy(q)

    # Compute references and differentiate BEFORE importing the converter.
    # Its Equinox plugins replace nn.Linear with a lowering primitive which
    # does not have JAX AD rules. Warming filter_jit alone is insufficient:
    # to_onnx can unwrap the jitted callable and retrace the Python model.
    batches = [xyz[np.arange(b) % len(xyz)] for b in (1, 3, 32)]
    reference = []
    for q in batches:
        e, f = jax.device_get(batch_fn(jnp.asarray(q)))
        if e.shape != (len(q),) or f.shape != q.shape or not (np.isfinite(e).all() and np.isfinite(f).all()):
            raise ValueError("JaxPIP returned invalid energies/forces")
        reference.append((np.asarray(e), np.asarray(f)))
    numerical = np.empty_like(xyz[0])
    for a in range(n):
        for c in range(3):
            plus, minus = xyz[0].copy(), xyz[0].copy()
            plus[a, c] += fd_step
            minus[a, c] -= fd_step
            numerical[a, c] = -(float(energy_fn(jnp.asarray(plus))) - float(energy_fn(jnp.asarray(minus)))) / (2*fd_step)
    np.testing.assert_allclose(reference[0][1][0], numerical, atol=1e-5, rtol=1e-4,
                               err_msg="JaxPIP forces fail finite differences in model units")

    # Freeze only ONE structure's differentiated graph. Tracing a batch here
    # would bake B=1 into intermediate abstract values. Instead apply vmap to
    # the primitive graph during conversion so batch provenance stays dynamic.
    # No Equinox layer is called or differentiated inside the converter, and
    # no PIP/NN mathematics is reimplemented in this adapter.
    closed = jax.make_jaxpr(model.get_energy_and_forces)(jnp.asarray(xyz[0]))
    primitives = jaxpr_as_fun(closed)

    def single_fn(q):
        energy, forces = primitives(q)
        return energy, forces

    def export_fn(q):
        return jax.vmap(single_fn)(q)

    import jax2onnx
    import onnx
    import onnxruntime as ort

    with _symbolic_pad_metadata_compat() as workarounds:
        graph = jax2onnx.to_onnx(fn=export_fn, inputs=[("B", n, 3)], enable_double_precision=True)
    if not isinstance(graph, onnx.ModelProto):
        raise TypeError("jax2onnx must return an ONNX ModelProto")
    io = list(graph.graph.input) + list(graph.graph.output)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 2:
        raise ValueError("export must have one input and two ordered outputs")
    for value, expected in zip(io, ((None, n, 3), (None,), (None, n, 3))):
        t = value.type.tensor_type
        dims = t.shape.dim
        if (t.elem_type != onnx.TensorProto.DOUBLE or len(dims) != len(expected)
                or not dims[0].dim_param
                or any(dim.dim_value != size for dim, size in zip(dims[1:], expected[1:]))):
            raise ValueError(f"invalid dynamic float64 I/O contract: {value.name}; type={t}")
    spec = {
        "schema": "pyrpmd.onnx/1", "batch_independent": True,
        "atomic_numbers": atomic_numbers,
        "state": {"charge": charge, "multiplicity": multiplicity},
        "position_unit": position_unit, "energy_unit": energy_unit,
        "force_unit": f"{energy_unit}/{position_unit}", "derivative": "force",
        "constants": {"bohr_angstrom": float(BOHR_ANGSTROM), "ev_to_hartree": float(EV_TO_AU)},
        "source_sha256": {"basis": _digest(basis), "checkpoint": _digest(checkpoint)},
        "export_strategy": "vmap-pretraced-single-energy-forces-jaxpr",
        "export_workarounds": workarounds,
        "export_versions": {name: importlib.metadata.version(name)
                            for name in ("jaxpip", "jax", "equinox", "jax2onnx", "onnx", "onnxruntime", "onnx-ir")},
    }
    props = {p.key: p.value for p in graph.metadata_props}
    props["pyrpmd.onnx"] = json.dumps(spec, sort_keys=True)
    onnx.helper.set_model_props(graph, props)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    max_e = max_f = 0.0
    for q, (expected_e, expected_f) in zip(batches, reference):
        got_e, got_f = session.run(None, {name: q})
        if got_e.dtype != np.float64 or got_f.dtype != np.float64:
            raise ValueError("ONNX Runtime lost float64 precision")
        np.testing.assert_allclose(got_e, expected_e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(got_f, expected_f, atol=atol, rtol=rtol)
        max_e = max(max_e, float(np.max(np.abs(got_e-expected_e))))
        max_f = max(max_f, float(np.max(np.abs(got_f-expected_f))))
        single = [session.run(None, {name: q[i:i+1]}) for i in range(len(q))]
        np.testing.assert_allclose(got_e, np.concatenate([v[0] for v in single]), atol=atol, rtol=rtol)
        np.testing.assert_allclose(got_f, np.concatenate([v[1] for v in single]), atol=atol, rtol=rtol)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name+".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(graph.SerializeToString())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"model": str(output), "sha256": _digest(output), "batches_checked": [1, 3, 32],
            "max_energy_error_model_units": max_e, "max_force_error_model_units": max_f,
            "force_fd_max_error_model_units": float(np.max(np.abs(reference[0][1][0]-numerical))),
            "metadata": spec}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--geometries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=("linear", "nn"), required=True)
    p.add_argument("--atomic-numbers", type=int, nargs="+", required=True)
    p.add_argument("--charge", type=int, required=True)
    p.add_argument("--multiplicity", type=int, required=True)
    p.add_argument("--position-unit", choices=("angstrom", "bohr"), required=True)
    p.add_argument("--energy-unit", choices=("eV", "hartree"), required=True)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--fd-step", type=float, default=1e-5)
    args = vars(p.parse_args())
    report = export_model(**args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
