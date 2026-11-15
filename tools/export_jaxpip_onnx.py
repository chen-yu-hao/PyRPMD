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


def _canonicalize_dynamic_batch_graph(model):
    """Rewrite CPU-bound dynamic batch shape helpers to pure GPU/static forms.

    jax2onnx lowers batched indexing (Gather/ScatterND/Expand/Reshape/Range)
    with runtime int64/bool shape subgraphs rooted at Shape(input). Upstream
    ONNX Runtime's GetCpuPreferredNodes assigns those shape subgraphs and any
    Expand/Reshape/Range/Slice/ConstantOfShape with non-initializer shape inputs
    to CPUExecutionProvider, inserting host-device copies and failing strict
    CUDA sessions (allow_cpu_fallback=false). For single-dynamic-batch [B,N,3]
    graphs every shape expression is affine in B (a*B + c), so B-independent
    index/shape subgraphs fold to static initializers while dynamic Reshape,
    Slice, Range, and Expand rewrite to static initializers or pure tensor ops.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    orig_bytes = model.SerializeToString()
    work = onnx.ModelProto.FromString(orig_bytes)
    graph = work.graph
    inp0 = graph.input[0]
    in_dims = inp0.type.tensor_type.shape.dim
    batch_param = in_dims[0].dim_param
    natoms = int(in_dims[1].dim_value)
    ncart = int(in_dims[2].dim_value)

    inferred = onnx.shape_inference.infer_shapes(
        onnx.ModelProto.FromString(orig_bytes),
        check_type=True, strict_mode=True, data_prop=True,
    )
    seeded_shape: dict[str, tuple[tuple[int, int] | None, ...]] = {}
    seeded_dtype: dict[str, int] = {}
    for vi in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if tt.elem_type:
            seeded_dtype[vi.name] = int(tt.elem_type)
        if tt.HasField("shape"):
            dims_parsed: list[tuple[int, int] | None] = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims_parsed.append((0, int(d.dim_value)))
                elif d.HasField("dim_param") and d.dim_param == batch_param:
                    dims_parsed.append((1, 0))
                else:
                    dims_parsed.append(None)
            seeded_shape[vi.name] = tuple(dims_parsed)

    def add_ac(x, y):
        return (x[0] + y[0], x[1] + y[1])

    def sub_ac(x, y):
        return (x[0] - y[0], x[1] - y[1])

    def mul_ac(x, y):
        if x[0] == 0:
            return (y[0] * x[1], y[1] * x[1])
        if y[0] == 0:
            return (x[0] * y[1], x[1] * y[1])
        return None

    def div_ac(x, y):
        if y[0] != 0 or y[1] == 0:
            return None
        denom = y[1]
        if x[0] == 0:
            return (0, int(np.trunc(x[1] / denom)))
        if x[1] == 0 and denom > 0 and x[0] % denom == 0:
            return (x[0] // denom, 0)
        return None

    def is_const_arr(arr) -> bool:
        return arr is not None and all(v[0] == 0 for v in arr.flat)

    def to_np_const(arr, dtype):
        out = np.empty(arr.shape, dtype=dtype)
        for idx, v in np.ndenumerate(arr):
            out[idx] = v[1]
        return out

    def from_np_const(np_arr):
        out = np.empty(np_arr.shape, dtype=object)
        for idx, v in np.ndenumerate(np_arr):
            out[idx] = (0, int(v))
        return out

    onnx_to_np = {
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.BOOL: np.bool_,
    }

    init_map: dict[str, onnx.TensorProto] = {t.name: t for t in graph.initializer}
    sym_shape: dict[str, tuple[tuple[int, int] | None, ...]] = {
        inp0.name: ((1, 0), (0, natoms), (0, ncart)),
    }
    sym_dtype: dict[str, int] = {inp0.name: int(inp0.type.tensor_type.elem_type)}
    sym_val: dict[str, np.ndarray] = {}

    for name, t in init_map.items():
        sym_shape[name] = tuple((0, int(d)) for d in t.dims)
        sym_dtype[name] = int(t.data_type)
        if t.data_type in onnx_to_np and int(np.prod(t.dims, initial=1)) <= 4096:
            sym_val[name] = from_np_const(numpy_helper.to_array(t))

    used_names = set(init_map) | {inp0.name} | {o.name for o in graph.output}
    for n_proto in graph.node:
        used_names.update(n_proto.input)
        used_names.update(n_proto.output)
        if n_proto.name:
            used_names.add(n_proto.name)

    counter = 0

    def fresh(prefix: str) -> str:
        nonlocal counter
        while True:
            name = f"__pyrpmd_{prefix}_{counter}"
            counter += 1
            if name not in used_names:
                used_names.add(name)
                return name

    def add_init(arr: np.ndarray, prefix: str = "const") -> str:
        name = fresh(prefix)
        tp = numpy_helper.from_array(arr, name=name)
        graph.initializer.append(tp)
        init_map[name] = tp
        sym_shape[name] = tuple((0, int(d)) for d in arr.shape)
        sym_dtype[name] = int(tp.data_type)
        if tp.data_type in onnx_to_np and arr.size <= 4096:
            sym_val[name] = from_np_const(arr)
        return name

    def get_attr(node, attr_name, default=None):
        for a in node.attribute:
            if a.name == attr_name:
                return helper.get_attribute_value(a)
        return default

    b_f64_1d: str | None = None
    b_carrier_cache: dict[tuple[int, int, int, int], str] = {}
    b_range_cache: dict[tuple[int, int], str] = {}
    true_scalar_name: str | None = None
    prelude_nodes: list[onnx.NodeProto] = []
    rewrote = False

    def ensure_true_scalar() -> str:
        nonlocal true_scalar_name
        if true_scalar_name is None:
            true_scalar_name = add_init(np.array(True, dtype=np.bool_), "true")
        return true_scalar_name

    def ensure_b_f64_1d() -> str:
        nonlocal b_f64_1d
        if b_f64_1d is None:
            s_st = add_init(np.array([0, 0, 0], dtype=np.int64), "s_starts")
            s_en = add_init(np.array([1 << 62, 1, 1], dtype=np.int64), "s_ends")
            s_ax = add_init(np.array([0, 1, 2], dtype=np.int64), "s_axes")
            sh_1d = add_init(np.array([-1], dtype=np.int64), "sh_1d")
            sliced = fresh("in_b11")
            b_f64_1d = fresh("in_b")
            prelude_nodes.append(helper.make_node("Slice", [inp0.name, s_st, s_en, s_ax], [sliced]))
            prelude_nodes.append(helper.make_node("Reshape", [sliced, sh_1d], [b_f64_1d]))
            sym_shape[sliced] = ((1, 0), (0, 1), (0, 1))
            sym_dtype[sliced] = TensorProto.DOUBLE
            sym_shape[b_f64_1d] = ((1, 0),)
            sym_dtype[b_f64_1d] = TensorProto.DOUBLE
        return b_f64_1d

    def ensure_kb_carrier(dtype: int, rank: int, dyn_axis: int, scale: int) -> str:
        key = (dtype, rank, dyn_axis, scale)
        if key in b_carrier_cache:
            return b_carrier_cache[key]
        cur = ensure_b_f64_1d()
        if scale > 1:
            sh_b1 = add_init(np.array([-1, 1], dtype=np.int64), "sh_b1")
            b2d = fresh("b2d")
            prelude_nodes.append(helper.make_node("Reshape", [cur, sh_b1], [b2d]))
            z1k = add_init(np.zeros((1, scale), dtype=np.float64), "z1k")
            bk = fresh("bk")
            prelude_nodes.append(helper.make_node("Where", [ensure_true_scalar(), z1k, b2d], [bk]))
            sh_1d = add_init(np.array([-1], dtype=np.int64), "sh_kb")
            cur = fresh("kb_f64")
            prelude_nodes.append(helper.make_node("Reshape", [bk, sh_1d], [cur]))
        if dtype != TensorProto.DOUBLE:
            casted = fresh("kb_cast")
            prelude_nodes.append(helper.make_node("Cast", [cur], [casted], to=dtype))
            cur = casted
        if rank != 1 or dyn_axis != 0:
            dims = [1] * rank
            dims[dyn_axis] = -1
            sh_nd = add_init(np.array(dims, dtype=np.int64), "sh_carrier")
            nd = fresh("kb_nd")
            prelude_nodes.append(helper.make_node("Reshape", [cur, sh_nd], [nd]))
            cur = nd
        b_carrier_cache[key] = cur
        return cur

    def ensure_range_kb(scale: int, out_dtype: int) -> str:
        rkey = (scale, out_dtype)
        if rkey in b_range_cache:
            return b_range_cache[rkey]
        np_dt = np.int64 if out_dtype == TensorProto.INT64 else np.int32
        carrier = ensure_kb_carrier(out_dtype, 1, 0, scale)
        one_c = add_init(np.array([1], dtype=np_dt), "one_vec")
        ones = fresh("ones_kb")
        axis0 = add_init(np.array(0, dtype=np.int64), "axis0")
        rng = fresh("range_kb")
        prelude_nodes.append(helper.make_node("Where", [ensure_true_scalar(), one_c, carrier], [ones]))
        prelude_nodes.append(helper.make_node("CumSum", [ones, axis0], [rng], exclusive=1))
        b_range_cache[rkey] = rng
        return rng

    def bcast_shapes(s1, s2):
        if s1 is None or s2 is None:
            return None
        r = max(len(s1), len(s2))
        p1 = ((0, 1),) * (r - len(s1)) + tuple(s1)
        p2 = ((0, 1),) * (r - len(s2)) + tuple(s2)
        out = []
        for d1, d2 in zip(p1, p2):
            if d1 is None or d2 is None:
                out.append(None)
            elif d1 == d2:
                out.append(d1)
            elif d1 == (0, 1):
                out.append(d2)
            elif d2 == (0, 1):
                out.append(d1)
            else:
                out.append(None)
        return tuple(out)

    unary_math_ops = {
        "Exp", "Log", "Reciprocal", "Sqrt", "Tanh", "Neg", "Abs", "Sign",
        "Sin", "Cos", "Sigmoid", "Softplus", "Relu", "Elu", "Celu", "Selu",
        "Gelu", "Mish", "Erf", "Floor", "Ceil", "ScatterND",
    }
    binary_ops = {
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max",
        "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or",
    }
    bool_out_ops = {"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or"}

    new_nodes: list[onnx.NodeProto] = []

    for node in graph.node:
        op = node.op_type
        ins = list(node.input)
        outs = list(node.output)
        out0 = outs[0] if outs else ""

        if op == "Constant" and out0:
            val = get_attr(node, "value")
            if val is not None:
                arr = numpy_helper.to_array(val)
                sym_shape[out0] = tuple((0, int(d)) for d in arr.shape)
                sym_dtype[out0] = int(val.data_type)
                if val.data_type in onnx_to_np and arr.size <= 4096:
                    sym_val[out0] = from_np_const(arr)
            new_nodes.append(node)
            continue

        s0 = sym_shape.get(ins[0]) if ins else None
        dt0 = sym_dtype.get(ins[0]) if ins else None

        if op == "Identity":
            if s0 is not None:
                sym_shape[out0] = s0
            if dt0 is not None:
                sym_dtype[out0] = dt0
            if ins[0] in sym_val:
                sym_val[out0] = sym_val[ins[0]].copy()

        elif op == "Cast":
            to_dt = int(get_attr(node, "to"))
            if s0 is not None:
                sym_shape[out0] = s0
            sym_dtype[out0] = to_dt
            if ins[0] in sym_val and to_dt in onnx_to_np:
                sym_val[out0] = sym_val[ins[0]].copy()

        elif op == "Not":
            if s0 is not None:
                sym_shape[out0] = s0
            sym_dtype[out0] = TensorProto.BOOL
            if ins[0] in sym_val and is_const_arr(sym_val[ins[0]]):
                sym_val[out0] = from_np_const(np.logical_not(to_np_const(sym_val[ins[0]], np.bool_)))

        elif op == "Shape":
            sym_dtype[out0] = TensorProto.INT64
            if s0 is not None and all(d is not None for d in s0):
                start = int(get_attr(node, "start", 0))
                end = get_attr(node, "end", None)
                sl = s0[start:] if end is None else s0[start:int(end)]
                sym_shape[out0] = ((0, len(sl)),)
                arr = np.empty((len(sl),), dtype=object)
                for i, d in enumerate(sl):
                    arr[i] = d
                sym_val[out0] = arr

        elif op == "Gather":
            axis = int(get_attr(node, "axis", 0))
            s_idx = sym_shape.get(ins[1])
            if dt0 is not None:
                sym_dtype[out0] = dt0
            if s0 is not None and s_idx is not None:
                ax = axis if axis >= 0 else axis + len(s0)
                sym_shape[out0] = s0[:ax] + s_idx + s0[ax + 1:]
            if ins[0] in sym_val and ins[1] in sym_val and is_const_arr(sym_val[ins[1]]):
                idx_np = to_np_const(sym_val[ins[1]], np.int64)
                sym_val[out0] = np.take(sym_val[ins[0]], idx_np, axis=axis)

        elif op == "Slice":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            starts_v = sym_val.get(ins[1])
            ends_v = sym_val.get(ins[2])
            axes_v = sym_val.get(ins[3]) if len(ins) > 3 and ins[3] else None
            steps_v = sym_val.get(ins[4]) if len(ins) > 4 and ins[4] else None
            if ins[0] in sym_val and is_const_arr(starts_v) and is_const_arr(ends_v):
                st = to_np_const(starts_v, np.int64).tolist()
                en = to_np_const(ends_v, np.int64).tolist()
                ax = (to_np_const(axes_v, np.int64).tolist()
                      if axes_v is not None and is_const_arr(axes_v) else list(range(len(st))))
                sp = (to_np_const(steps_v, np.int64).tolist()
                      if steps_v is not None and is_const_arr(steps_v) else [1] * len(st))
                slices = [slice(None)] * sym_val[ins[0]].ndim
                for a, s_i, e_i, p_i in zip(ax, st, en, sp):
                    slices[a] = slice(s_i, e_i, p_i)
                sym_val[out0] = sym_val[ins[0]][tuple(slices)]
                sym_shape[out0] = tuple((0, int(d)) for d in sym_val[out0].shape)
            elif s0 is not None and is_const_arr(starts_v) and ends_v is not None:
                st = to_np_const(starts_v, np.int64).tolist()
                ax = (to_np_const(axes_v, np.int64).tolist()
                      if axes_v is not None and is_const_arr(axes_v) else list(range(len(st))))
                sp = (to_np_const(steps_v, np.int64).tolist()
                      if steps_v is not None and is_const_arr(steps_v) else [1] * len(st))
                out_sh = list(s0)
                new_ends = []
                can_static_ends = True
                for idx_a, (a, s_i, p_i) in enumerate(zip(ax, st, sp)):
                    a_norm = a if a >= 0 else a + len(s0)
                    e_ac = ends_v.flat[idx_a]
                    dim_ac = s0[a_norm]
                    if e_ac[0] == 0:
                        e_val = e_ac[1]
                        new_ends.append(e_val)
                        if dim_ac is not None and dim_ac[0] == 0 and p_i == 1:
                            lo = max(0, s_i if s_i >= 0 else s_i + dim_ac[1])
                            hi = min(dim_ac[1], e_val if e_val >= 0 else e_val + dim_ac[1])
                            out_sh[a_norm] = (0, max(0, hi - lo))
                        else:
                            out_sh[a_norm] = None
                    elif dim_ac is not None and e_ac == dim_ac and s_i == 0 and p_i == 1:
                        new_ends.append(1 << 62)
                        out_sh[a_norm] = dim_ac
                    else:
                        can_static_ends = False
                        out_sh[a_norm] = None
                sym_shape[out0] = tuple(out_sh)
                if can_static_ends and ins[2] not in init_map:
                    ins[2] = add_init(np.array(new_ends, dtype=np.int64), "slice_ends")
                    if ins[1] not in init_map:
                        ins[1] = add_init(np.array(st, dtype=np.int64), "slice_starts")
                    if len(ins) > 3 and ins[3] and ins[3] not in init_map and axes_v is not None and is_const_arr(axes_v):
                        ins[3] = add_init(np.array(ax, dtype=np.int64), "slice_axes")
                    if len(ins) > 4 and ins[4] and ins[4] not in init_map and steps_v is not None and is_const_arr(steps_v):
                        ins[4] = add_init(np.array(sp, dtype=np.int64), "slice_steps")
                    node = helper.make_node("Slice", ins, outs, name=node.name)
                    rewrote = True

        elif op == "Squeeze":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            axes_v = sym_val.get(ins[1]) if len(ins) > 1 and ins[1] else None
            if axes_v is not None and is_const_arr(axes_v):
                axes = tuple(int(x) for x in to_np_const(axes_v, np.int64).reshape(-1))
                if ins[0] in sym_val:
                    sym_val[out0] = np.squeeze(sym_val[ins[0]], axis=axes)
                    sym_shape[out0] = tuple((0, int(d)) for d in sym_val[out0].shape)
                elif s0 is not None:
                    norm = {a if a >= 0 else a + len(s0) for a in axes}
                    sym_shape[out0] = tuple(d for i, d in enumerate(s0) if i not in norm)

        elif op == "Unsqueeze":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            axes_v = sym_val.get(ins[1]) if len(ins) > 1 else None
            if axes_v is not None and is_const_arr(axes_v):
                axes = [int(x) for x in to_np_const(axes_v, np.int64).reshape(-1)]
                if ins[0] in sym_val:
                    cur = sym_val[ins[0]]
                    out_rank = cur.ndim + len(axes)
                    for a in sorted(a if a >= 0 else a + out_rank for a in axes):
                        cur = np.expand_dims(cur, axis=a)
                    sym_val[out0] = cur
                    sym_shape[out0] = tuple((0, int(d)) for d in cur.shape)
                elif s0 is not None:
                    out_rank = len(s0) + len(axes)
                    norm = sorted(a if a >= 0 else a + out_rank for a in axes)
                    sh = list(s0)
                    for a in norm:
                        sh.insert(a, (0, 1))
                    sym_shape[out0] = tuple(sh)

        elif op == "Concat":
            axis = int(get_attr(node, "axis", 0))
            if dt0 is not None:
                sym_dtype[out0] = dt0
            if all(i in sym_val for i in ins):
                sym_val[out0] = np.concatenate([sym_val[i] for i in ins], axis=axis)
                sym_shape[out0] = tuple((0, int(d)) for d in sym_val[out0].shape)
            elif all(sym_shape.get(i) is not None for i in ins):
                shs = [sym_shape[i] for i in ins]
                ax = axis if axis >= 0 else axis + len(shs[0])
                acc = (0, 0)
                for sh in shs:
                    acc = add_ac(acc, sh[ax]) if (acc is not None and sh[ax] is not None) else None
                out_sh = list(shs[0])
                out_sh[ax] = acc
                sym_shape[out0] = tuple(out_sh)

        elif op == "ReduceProd":
            keepdims = int(get_attr(node, "keepdims", 1))
            if dt0 is not None:
                sym_dtype[out0] = dt0
            if ins[0] in sym_val:
                arr = sym_val[ins[0]]
                axes_v = sym_val.get(ins[1]) if len(ins) > 1 and ins[1] else None
                if is_const_arr(arr) and (axes_v is None or is_const_arr(axes_v)):
                    np_arr = to_np_const(arr, onnx_to_np[dt0])
                    ax_arg = (None if axes_v is None
                              else tuple(int(x) for x in to_np_const(axes_v, np.int64).reshape(-1)))
                    prod_arr = np.prod(np_arr, axis=ax_arg, keepdims=bool(keepdims))
                    sym_val[out0] = from_np_const(np.asarray(prod_arr, dtype=onnx_to_np[dt0]))
                    sym_shape[out0] = tuple((0, int(d)) for d in sym_val[out0].shape)
                elif arr.ndim == 1 and (axes_v is None or (
                        is_const_arr(axes_v) and to_np_const(axes_v, np.int64).reshape(-1).tolist() in ([0], [-1]))):
                    acc = (0, 1)
                    for v in arr:
                        acc = mul_ac(acc, v) if acc is not None else None
                    if acc is not None:
                        if keepdims:
                            res = np.empty((1,), dtype=object)
                            res[0] = acc
                        else:
                            res = np.array(None, dtype=object)
                            res[()] = acc
                        sym_val[out0] = res
                        sym_shape[out0] = ((0, 1),) if keepdims else ()

        elif op in binary_ops:
            s1 = sym_shape.get(ins[1])
            sym_shape[out0] = bcast_shapes(s0, s1)
            if op in bool_out_ops:
                sym_dtype[out0] = TensorProto.BOOL
            elif dt0 is not None:
                sym_dtype[out0] = dt0
            if ins[0] in sym_val and ins[1] in sym_val:
                v0, v1 = np.broadcast_arrays(sym_val[ins[0]], sym_val[ins[1]])
                if op in ("Add", "Sub", "Mul", "Div") and dt0 in (TensorProto.INT64, TensorProto.INT32) and v0.size <= 4096:
                    res = np.empty(v0.shape, dtype=object)
                    ok = True
                    for idx in np.ndindex(v0.shape):
                        x, y = v0[idx], v1[idx]
                        if op == "Add":
                            r = add_ac(x, y)
                        elif op == "Sub":
                            r = sub_ac(x, y)
                        elif op == "Mul":
                            r = mul_ac(x, y)
                        else:
                            r = div_ac(x, y)
                        if r is None:
                            ok = False
                            break
                        res[idx] = r
                    if ok:
                        sym_val[out0] = res
                elif op == "Equal" and v0.size <= 4096:
                    res = np.empty(v0.shape, dtype=object)
                    ok = True
                    for idx in np.ndindex(v0.shape):
                        x, y = v0[idx], v1[idx]
                        if x[0] == y[0]:
                            res[idx] = (0, int(x[1] == y[1]))
                        else:
                            ok = False
                            break
                    if ok:
                        sym_val[out0] = res
                elif is_const_arr(sym_val[ins[0]]) and is_const_arr(sym_val[ins[1]]) and sym_dtype.get(ins[0]) in onnx_to_np:
                    dt_in = onnx_to_np[sym_dtype[ins[0]]]
                    a0 = to_np_const(sym_val[ins[0]], dt_in)
                    a1 = to_np_const(sym_val[ins[1]], dt_in)
                    fn_map = {
                        "Min": np.minimum, "Max": np.maximum,
                        "Less": np.less, "LessOrEqual": np.less_equal,
                        "Greater": np.greater, "GreaterOrEqual": np.greater_equal,
                        "And": np.logical_and, "Or": np.logical_or,
                    }
                    if op in fn_map:
                        sym_val[out0] = from_np_const(fn_map[op](a0, a1))

        elif op == "NonZero":
            sym_dtype[out0] = TensorProto.INT64
            if ins[0] in sym_val and is_const_arr(sym_val[ins[0]]):
                c_arr = to_np_const(sym_val[ins[0]], np.bool_)
                nz = np.asarray(np.nonzero(c_arr), dtype=np.int64)
                sym_val[out0] = from_np_const(nz)
                sym_shape[out0] = tuple((0, int(d)) for d in nz.shape)

        elif op == "ConstantOfShape":
            val_tp = get_attr(node, "value")
            fill_dt = int(val_tp.data_type) if val_tp is not None else TensorProto.FLOAT
            sym_dtype[out0] = fill_dt
            sh_v = sym_val.get(ins[0])
            if sh_v is not None:
                sym_shape[out0] = tuple(v for v in sh_v.flat)
                if is_const_arr(sh_v):
                    dims = tuple(int(v[1]) for v in sh_v.flat)
                    if val_tp is not None:
                        fill_arr = numpy_helper.to_array(val_tp)
                        arr = np.full(dims, fill_arr.flat[0], dtype=fill_arr.dtype)
                    else:
                        arr = np.zeros(dims, dtype=np.float32)
                    out_init = add_init(arr, "cos_const")
                    new_nodes.append(helper.make_node("Identity", [out_init], [out0]))
                    rewrote = True
                    continue

        elif op == "Transpose":
            perm = get_attr(node, "perm")
            if dt0 is not None:
                sym_dtype[out0] = dt0
            if s0 is not None:
                p = list(perm) if perm is not None else list(reversed(range(len(s0))))
                sym_shape[out0] = tuple(s0[i] for i in p)
            if ins[0] in sym_val:
                sym_val[out0] = np.transpose(sym_val[ins[0]], axes=perm)

        elif op == "Reshape":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            sh_v = sym_val.get(ins[1])
            if sh_v is not None:
                dims_ac = [v for v in sh_v.flat]
                resolved = []
                for idx_d, d in enumerate(dims_ac):
                    if d == (0, 0) and s0 is not None and idx_d < len(s0):
                        resolved.append(s0[idx_d])
                    else:
                        resolved.append(d)
                dyn_indices = [i for i, d in enumerate(resolved) if d is not None and d[0] != 0]
                if all(d is not None for d in resolved):
                    sym_shape[out0] = tuple(resolved)
                    if ins[0] in sym_val and len(dyn_indices) == 0 and all(d[1] >= 0 for d in resolved):
                        sym_val[out0] = sym_val[ins[0]].reshape(tuple(d[1] for d in resolved))
                    if ins[1] not in init_map:
                        if len(dyn_indices) == 0:
                            static_sh = np.array([d[1] for d in resolved], dtype=np.int64)
                            ins[1] = add_init(static_sh, "reshape_sh")
                            node = helper.make_node("Reshape", ins, outs, name=node.name)
                            rewrote = True
                        elif (len(dyn_indices) == 1 and resolved[dyn_indices[0]][1] == 0
                              and all(resolved[i][1] > 0 for i in range(len(resolved)) if i != dyn_indices[0])):
                            static_sh = np.array(
                                [-1 if i == dyn_indices[0] else resolved[i][1] for i in range(len(resolved))],
                                dtype=np.int64,
                            )
                            ins[1] = add_init(static_sh, "reshape_dyn_sh")
                            node = helper.make_node("Reshape", ins, outs, name=node.name)
                            rewrote = True

        elif op == "Range":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            st_v, lim_v, dlt_v = sym_val.get(ins[0]), sym_val.get(ins[1]), sym_val.get(ins[2])
            if is_const_arr(st_v) and is_const_arr(lim_v) and is_const_arr(dlt_v) and dt0 in onnx_to_np:
                arr = np.arange(st_v.flat[0][1], lim_v.flat[0][1], dlt_v.flat[0][1], dtype=onnx_to_np[dt0])
                c_name = add_init(arr, "range_const")
                sym_shape[out0] = ((0, int(arr.size)),)
                new_nodes.append(helper.make_node("Identity", [c_name], [out0]))
                rewrote = True
                continue
            if (dt0 in (TensorProto.INT64, TensorProto.INT32)
                    and is_const_arr(st_v) and is_const_arr(dlt_v) and lim_v is not None
                    and st_v.flat[0] == (0, 0) and dlt_v.flat[0] == (0, 1)
                    and lim_v.flat[0][0] > 0 and lim_v.flat[0][1] == 0):
                scale = lim_v.flat[0][0]
                rng = ensure_range_kb(scale, dt0)
                sym_shape[out0] = ((scale, 0),)
                new_nodes.append(helper.make_node("Identity", [rng], [out0]))
                rewrote = True
                continue

        elif op == "Expand":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            sh_v = sym_val.get(ins[1])
            if sh_v is not None:
                tgt = tuple(v for v in sh_v.flat)
                out_sh = bcast_shapes(s0, tgt) if s0 is not None else tgt
                sym_shape[out0] = out_sh
                dyn_axes = [i for i, d in enumerate(out_sh) if d is not None and d[0] != 0] if out_sh else []
                if out_sh is not None and all(d is not None for d in out_sh):
                    if len(dyn_axes) == 0:
                        if ins[0] in sym_val and is_const_arr(sym_val[ins[0]]) and dt0 in onnx_to_np:
                            arr = np.broadcast_to(
                                to_np_const(sym_val[ins[0]], onnx_to_np[dt0]),
                                tuple(d[1] for d in out_sh),
                            ).copy()
                            c_name = add_init(arr, "expand_const")
                            sym_val[out0] = from_np_const(arr)
                            new_nodes.append(helper.make_node("Identity", [c_name], [out0]))
                            rewrote = True
                            continue
                        if ins[1] not in init_map:
                            ins[1] = add_init(np.array([d[1] for d in out_sh], dtype=np.int64), "expand_sh")
                            node = helper.make_node("Expand", ins, outs, name=node.name)
                            rewrote = True
                    elif len(dyn_axes) == 1 and out_sh[dyn_axes[0]][1] == 0 and s0 is not None and dt0 is not None:
                        d_ax = dyn_axes[0]
                        scale = out_sh[d_ax][0]
                        rank = len(out_sh)
                        padded_in = ((0, 1),) * (rank - len(s0)) + tuple(s0)
                        cur = ins[0]
                        static_dims = [1 if i == d_ax else out_sh[i][1] for i in range(rank)]
                        if any(padded_in[i] != out_sh[i] for i in range(rank) if i != d_ax) or len(s0) != rank:
                            exp_sh = add_init(np.array(static_dims, dtype=np.int64), "expand_static_sh")
                            tmp = fresh("expand_static")
                            new_nodes.append(helper.make_node("Expand", [cur, exp_sh], [tmp]))
                            cur = tmp
                        if padded_in[d_ax] != (scale, 0):
                            carrier = ensure_kb_carrier(dt0, rank, d_ax, scale)
                            new_nodes.append(helper.make_node("Where", [ensure_true_scalar(), cur, carrier], [out0]))
                        else:
                            new_nodes.append(helper.make_node("Identity", [cur], [out0]))
                        rewrote = True
                        continue

        elif op in unary_math_ops:
            if s0 is not None:
                sym_shape[out0] = s0
            if dt0 is not None:
                sym_dtype[out0] = dt0

        elif op == "Sum":
            sh = s0
            for i_name in ins[1:]:
                sh = bcast_shapes(sh, sym_shape.get(i_name))
            sym_shape[out0] = sh
            if dt0 is not None:
                sym_dtype[out0] = dt0

        elif op == "Where":
            sh = bcast_shapes(sym_shape.get(ins[0]), bcast_shapes(sym_shape.get(ins[1]), sym_shape.get(ins[2])))
            sym_shape[out0] = sh
            if sym_dtype.get(ins[1]) is not None:
                sym_dtype[out0] = sym_dtype[ins[1]]
            if (all(i in sym_val and is_const_arr(sym_val[i]) for i in ins)
                    and sym_dtype.get(ins[1]) in onnx_to_np):
                dt_v = onnx_to_np[sym_dtype[ins[1]]]
                c0 = to_np_const(sym_val[ins[0]], np.bool_)
                c1 = to_np_const(sym_val[ins[1]], dt_v)
                c2 = to_np_const(sym_val[ins[2]], dt_v)
                sym_val[out0] = from_np_const(np.where(c0, c1, c2))

        elif op in ("ReduceSum", "ReduceL2", "ReduceMean", "ReduceMax", "ReduceMin"):
            keepdims = int(get_attr(node, "keepdims", 1))
            if dt0 is not None:
                sym_dtype[out0] = dt0
            axes_v = sym_val.get(ins[1]) if len(ins) > 1 and ins[1] else None
            if s0 is not None and axes_v is not None and is_const_arr(axes_v):
                axes = {int(x) if int(x) >= 0 else int(x) + len(s0)
                        for x in to_np_const(axes_v, np.int64).reshape(-1)}
                if keepdims:
                    sym_shape[out0] = tuple((0, 1) if i in axes else d for i, d in enumerate(s0))
                else:
                    sym_shape[out0] = tuple(d for i, d in enumerate(s0) if i not in axes)

        elif op == "Gemm":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            sa, sb = sym_shape.get(ins[0]), sym_shape.get(ins[1])
            ta, tb = int(get_attr(node, "transA", 0)), int(get_attr(node, "transB", 0))
            if sa is not None and sb is not None:
                sym_shape[out0] = (sa[1] if ta else sa[0], sb[0] if tb else sb[1])

        elif op == "Einsum":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            raw_eq = get_attr(node, "equation")
            eq = raw_eq.decode() if isinstance(raw_eq, bytes) else raw_eq
            if isinstance(eq, str) and "->" in eq:
                lhs, rhs = eq.split("->")
                terms = [t.strip() for t in lhs.split(",")]
                dim_map = {}
                for term, i_name in zip(terms, ins):
                    sh = sym_shape.get(i_name)
                    if sh is not None:
                        for ch, d in zip(term, sh):
                            if d is not None and d != (0, 1):
                                dim_map[ch] = d
                            elif ch not in dim_map and d is not None:
                                dim_map[ch] = d
                rhs = rhs.strip()
                if all(ch in dim_map for ch in rhs):
                    sym_shape[out0] = tuple(dim_map[ch] for ch in rhs)

        elif op == "Pad":
            if dt0 is not None:
                sym_dtype[out0] = dt0
            pads_v = sym_val.get(ins[1]) if len(ins) > 1 else None
            if s0 is not None and pads_v is not None and is_const_arr(pads_v):
                pads = to_np_const(pads_v, np.int64).reshape(-1)
                r = len(s0)
                sym_shape[out0] = tuple(
                    add_ac(s0[i], (0, int(pads[i] + pads[i + r]))) if s0[i] is not None else None
                    for i in range(r)
                )

        if out0:
            if out0 not in sym_dtype and out0 in seeded_dtype:
                sym_dtype[out0] = seeded_dtype[out0]
            if out0 not in sym_shape and out0 in seeded_shape:
                sym_shape[out0] = seeded_shape[out0]
            if (out0 in sym_val and is_const_arr(sym_val[out0])
                    and sym_dtype.get(out0) in onnx_to_np and out0 not in init_map):
                arr = to_np_const(sym_val[out0], onnx_to_np[sym_dtype[out0]])
                c_name = add_init(arr, "folded")
                new_nodes.append(helper.make_node("Identity", [c_name], [out0]))
                rewrote = True
                continue

        new_nodes.append(node)

    if not rewrote:
        return model, False

    all_nodes = prelude_nodes + new_nodes
    graph_outputs = {o.name for o in graph.output}
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        while name in alias:
            name = alias[name]
        return name

    compacted: list[onnx.NodeProto] = []
    for n_proto in all_nodes:
        if n_proto.op_type == "Identity" and n_proto.output[0] not in graph_outputs:
            alias[n_proto.output[0]] = resolve(n_proto.input[0])
            continue
        for idx, inp in enumerate(n_proto.input):
            if inp:
                n_proto.input[idx] = resolve(inp)
        compacted.append(n_proto)

    live = set(graph_outputs)
    kept_rev: list[onnx.NodeProto] = []
    for n_proto in reversed(compacted):
        if any(o in live for o in n_proto.output):
            kept_rev.append(n_proto)
            for inp in n_proto.input:
                if inp:
                    live.add(inp)
    kept = list(reversed(kept_rev))

    del graph.node[:]
    graph.node.extend(kept)
    kept_inits = [t for t in graph.initializer if t.name in live]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    del graph.value_info[:]
    work = onnx.shape_inference.infer_shapes(work, check_type=True, strict_mode=True)
    onnx.checker.check_model(work)
    return work, True


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
    graph, canonicalized = _canonicalize_dynamic_batch_graph(graph)
    if canonicalized:
        workarounds = [*workarounds, "dynamic-batch-shape-canonicalization"]
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
