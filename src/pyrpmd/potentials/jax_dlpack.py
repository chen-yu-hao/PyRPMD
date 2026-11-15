"""First-class zero-copy JAX DLPack potential provider and TorchPES bridge.

Coordinates and forces stay on the selected CPU or CUDA device when consumed
through :meth:`JAXDLPackPotential.as_torch_pes` / ``make_workflow_torch_pes``.
PyTorch and JAX exchange ``float64`` buffers through the Python Array API
DLPack protocol with explicit CUDA stream handoff, avoiding host round-trips,
ONNX export gaps, and non-default stream hazards.
"""
from __future__ import annotations

from collections.abc import Mapping
import functools
import importlib.metadata
from pathlib import Path
import re
import sys
import threading

import numpy as np

from ..geometry import validate_cell_pbc
from .base import (
    BackendUnavailableError,
    PESResult,
    Potential,
    PotentialError,
    coordinates,
    validate_result,
)
from .python import import_object, unit_factors


_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")


def _normalize_device_request(device):
    """Validate and canonicalize ``cpu``, ``cuda``, or ``cuda:<index>``."""
    if isinstance(device, (bool, np.bool_)) or device is None:
        raise ValueError("JAX DLPack device must be cpu, cuda or cuda:<index>")
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and isinstance(device, torch_mod.device):
        if device.type not in ("cpu", "cuda") or (
            device.type == "cpu" and device.index not in (None, 0)
        ):
            raise ValueError("JAX DLPack device must be cpu, cuda or cuda:<index>")
        text = "cpu" if device.type == "cpu" else (
            f"cuda:{torch_mod.cuda.current_device()}"
            if device.index is None and torch_mod.cuda.is_available()
            else ("cuda:0" if device.index is None else f"cuda:{device.index}")
        )
    elif isinstance(device, str):
        if not _DEVICE_PATTERN.fullmatch(device):
            raise ValueError("JAX DLPack device must be cpu, cuda or cuda:<index>")
        if device == "cuda":
            if torch_mod is not None and torch_mod.cuda.is_available():
                text = f"cuda:{torch_mod.cuda.current_device()}"
            else:
                text = "cuda:0"
        else:
            text = device
    else:
        raise ValueError("JAX DLPack device must be cpu, cuda or cuda:<index>")
    return text


def _load_jax_runtime(device, *, backend_label="jax-dlpack"):
    """Import JAX, enforce ``float64``, and resolve the requested JAX device."""
    requested = _normalize_device_request(device)
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise BackendUnavailableError(
            f"{backend_label} requires jax and jaxlib; install a compatible "
            "CPU or CUDA JAX runtime"
        ) from error
    jax.config.update("jax_enable_x64", True)
    if requested == "cpu":
        try:
            cpu_devices = list(jax.devices("cpu"))
        except RuntimeError as error:
            raise BackendUnavailableError("JAX CPU device is unavailable") from error
        if not cpu_devices:
            raise BackendUnavailableError("JAX CPU device is unavailable")
        jax_device = cpu_devices[0]
    else:
        index = int(requested.split(":", 1)[1])
        try:
            gpu_devices = list(jax.devices("gpu"))
        except RuntimeError as error:
            raise BackendUnavailableError(
                f"requested JAX device {device!r} is unavailable; no CPU fallback"
            ) from error
        jax_device = next(
            (
                d for d in gpu_devices
                if getattr(d, "local_hardware_id", d.id) == index or d.id == index
            ),
            None,
        )
        if jax_device is None:
            raise BackendUnavailableError(
                f"requested JAX device {device!r} is unavailable; no CPU fallback"
            )
    return jax, jnp, requested, jax_device


def _check_declared_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("JAX model metadata must be a mapping")
    capabilities = metadata.get("capabilities", {}) or {}
    if not isinstance(capabilities, Mapping):
        raise ValueError("JAX model capabilities must be a mapping")
    if (
        metadata.get("batch_independent") is False
        or metadata.get("trajectory_batch_safe") is False
        or metadata.get("stateful")
        or metadata.get("coupled_frames")
        or metadata.get("ring_polymer_coupled")
        or capabilities.get("stateful")
    ):
        raise PotentialError("JAX model explicitly rejects independent stateless batching")


def _check_pytree_float64(jax, jnp, obj):
    for leaf in jax.tree_util.tree_leaves(obj):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            continue
        try:
            jax_dtype = jnp.dtype(dtype)
        except TypeError:
            continue
        if jnp.issubdtype(jax_dtype, jnp.complexfloating):
            raise ValueError("jax-dlpack does not support complex model parameters")
        if jnp.issubdtype(jax_dtype, jnp.floating) and jax_dtype != jnp.float64:
            raise ValueError(
                f"jax-dlpack requires float64 model parameters, got {jax_dtype}; "
                "no implicit precision conversion"
            )


def _invoke_periodic(fn, q, cell, pbc):
    try:
        return fn(q, cell, pbc)
    except TypeError as first_error:
        try:
            return fn(q, cell=cell, pbc=pbc)
        except TypeError:
            raise first_error


class _TorchDLPackProducer:
    """Array-API DLPack wrapper ensuring Torch -> JAX CUDA stream ordering."""

    __slots__ = ("tensor", "cuda_stream")

    def __init__(self, tensor, cuda_stream):
        self.tensor = tensor
        self.cuda_stream = cuda_stream

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        if self.cuda_stream is not None and stream is None:
            # Fallback if the consumer library omits the DLPack stream argument.
            self.cuda_stream.synchronize()
        try:
            return self.tensor.__dlpack__(stream=stream, **kwargs)
        except TypeError:
            return self.tensor.__dlpack__(stream=stream)


class _JAXDLPackProducer:
    """Array-API DLPack wrapper ensuring JAX -> Torch CUDA stream ordering."""

    __slots__ = ("array", "cuda_stream")

    def __init__(self, array, cuda_stream):
        self.array = array
        self.cuda_stream = cuda_stream

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        if self.cuda_stream is not None and stream is None:
            self.array.block_until_ready()
            try:
                return self.array.__dlpack__(**kwargs)
            except TypeError:
                return self.array.__dlpack__()
        try:
            return self.array.__dlpack__(stream=stream, **kwargs)
        except TypeError:
            try:
                return self.array.__dlpack__(stream=stream)
            except TypeError:
                if self.cuda_stream is not None:
                    self.array.block_until_ready()
                return self.array.__dlpack__()
        except (BufferError, RuntimeError):
            if self.cuda_stream is not None and stream is not None:
                self.array.block_until_ready()
                return self.array.__dlpack__()
            raise


class _JAXDLPackTorchModule:
    """Lazily imported ``torch.nn.Module`` around a compiled JAX batch evaluator."""

    def __new__(
        cls,
        *,
        jax,
        jnp,
        jax_device,
        torch_device_name,
        evaluate_jax_batch,
        periodic=False,
        pbc=(False, False, False),
        max_frames_per_call=None,
        provider_name="jax-dlpack",
        owner=None,
    ):
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailableError(
                f"the {provider_name} Torch bridge requires PyTorch"
            ) from error
        expected_device = torch.device(torch_device_name)

        class Module(torch.nn.Module):
            metadata = {"batch_independent": True, "trajectory_batch_safe": True}

            def __init__(self):
                super().__init__()
                self.jax_device = jax_device
                self.torch_device = expected_device
                self.periodic = bool(periodic)
                self.max_frames_per_call = max_frames_per_call
                self.owner = owner
                self._lock = threading.Lock()
                self._last_stream_obj = None
                self._prev_jax_outputs = None
                with jax.default_device(jax_device):
                    self._pbc_single = jax.device_put(
                        jnp.asarray(tuple(bool(x) for x in pbc), dtype=jnp.bool_),
                        jax_device,
                    )
                self.metadata = {
                    **self.metadata,
                    "execution_device": str(expected_device),
                    "dtype": "float64",
                    "energy_dtype": "float64",
                    "provider": provider_name,
                    "periodic": self.periodic,
                    "supports_pbc": self.periodic,
                }

            def update_pbc(self, new_pbc):
                with jax.default_device(self.jax_device):
                    self._pbc_single = jax.device_put(
                        jnp.asarray(tuple(bool(x) for x in new_pbc), dtype=jnp.bool_),
                        self.jax_device,
                    )

            def reset_state(self):
                with self._lock:
                    self._prev_jax_outputs = None
                    self._last_stream_obj = None

            def _eval_chunk(self, q_chunk, cell_chunk, caller_stream):
                q_contig = q_chunk.detach().contiguous()
                if q_contig.device.type == "cuda" and (q_contig.data_ptr() % 16) != 0:
                    # XLA GPU executables require 16-byte buffer alignment; odd-element
                    # float64 slices (e.g. q[5:10] for N=3) can start at an 8-byte offset.
                    q_contig = q_contig.clone()
                cell_contig = None
                if caller_stream is not None:
                    q_contig.record_stream(caller_stream)
                q_jax = jax.dlpack.from_dlpack(_TorchDLPackProducer(q_contig, caller_stream))
                if self.periodic:
                    if cell_chunk is None:
                        raise ValueError("periodic JAX DLPack bridge requires cell tensor")
                    cell_contig = cell_chunk.detach().contiguous()
                    if cell_contig.device.type == "cuda" and (cell_contig.data_ptr() % 16) != 0:
                        cell_contig = cell_contig.clone()
                    if caller_stream is not None:
                        cell_contig.record_stream(caller_stream)
                    cell_jax = jax.dlpack.from_dlpack(
                        _TorchDLPackProducer(cell_contig, caller_stream)
                    )
                    pbc_jax = jnp.broadcast_to(self._pbc_single, (len(q_chunk), 3))
                    out_jax = evaluate_jax_batch(q_jax, cell_jax, pbc_jax)
                else:
                    cell_jax = None
                    out_jax = evaluate_jax_batch(q_jax)

                # Release the previous call's JAX outputs only after XLA has
                # allocated distinct output buffers for the current invocation.
                self._prev_jax_outputs = out_jax

                if len(out_jax) == 3:
                    energy_jax, force_jax, virial_jax = out_jax
                    energy_t = torch.from_dlpack(_JAXDLPackProducer(energy_jax, caller_stream))
                    force_t = torch.from_dlpack(_JAXDLPackProducer(force_jax, caller_stream))
                    virial_t = torch.from_dlpack(_JAXDLPackProducer(virial_jax, caller_stream))
                else:
                    energy_jax, force_jax = out_jax
                    virial_t = None
                    energy_t = torch.from_dlpack(_JAXDLPackProducer(energy_jax, caller_stream))
                    force_t = torch.from_dlpack(_JAXDLPackProducer(force_jax, caller_stream))

                # Keep DLPack input views alive until consumer stream waits on XLA.
                if caller_stream is not None:
                    q_contig.record_stream(caller_stream)
                    if cell_contig is not None:
                        cell_contig.record_stream(caller_stream)
                del q_jax, cell_jax
                return energy_t, force_t, virial_t

            def forward(self, q, cell=None, pbc=None):
                if self.owner is not None and getattr(self.owner, "closed", False):
                    raise PotentialError(f"{provider_name} potential is closed")
                if not isinstance(q, torch.Tensor) or q.dtype != torch.float64:
                    raise ValueError("JAX DLPack bridge expects a float64 CPU/CUDA tensor")
                if q.device.type not in ("cpu", "cuda"):
                    raise ValueError("JAX DLPack bridge expects a float64 CPU/CUDA tensor")
                if torch.is_autocast_enabled(q.device.type):
                    raise PotentialError(
                        "autocast is unsupported: JAX DLPack evaluation must preserve float64 precision"
                    )
                if q.device != self.torch_device:
                    raise ValueError("Torch/JAX devices differ; no implicit device transfer")
                if self.periodic:
                    if (
                        not isinstance(cell, torch.Tensor)
                        or cell.dtype != torch.float64
                        or cell.device != self.torch_device
                        or cell.shape != (len(q), 3, 3)
                    ):
                        raise ValueError(
                            "periodic JAX DLPack bridge expects float64 cell [frames,3,3] on device"
                        )
                    if (
                        pbc is not None
                        and (
                            not isinstance(pbc, torch.Tensor)
                            or pbc.dtype != torch.bool
                            or pbc.device != self.torch_device
                            or pbc.shape != (len(q), 3)
                        )
                    ):
                        raise ValueError(
                            "periodic JAX DLPack bridge expects bool pbc [frames,3] on device"
                        )

                with self._lock:
                    caller_stream = (
                        torch.cuda.current_stream(q.device)
                        if q.device.type == "cuda"
                        else None
                    )
                    if (
                        caller_stream is not None
                        and self._last_stream_obj is not None
                        and self._last_stream_obj.cuda_stream != caller_stream.cuda_stream
                    ):
                        # Order the new stream after all kernels previously enqueued
                        # on the prior stream (including downstream reads of DLPack outputs).
                        switch_event = torch.cuda.Event()
                        switch_event.record(self._last_stream_obj)
                        caller_stream.wait_event(switch_event)

                    limit = self.max_frames_per_call
                    if limit is None or len(q) <= limit:
                        energy_t, force_t, virial_t = self._eval_chunk(
                            q, cell, caller_stream
                        )
                    else:
                        energies, forces, virials = [], [], []
                        for start in range(0, len(q), limit):
                            stop = start + limit
                            c_slice = None if cell is None else cell[start:stop]
                            e_part, f_part, w_part = self._eval_chunk(
                                q[start:stop], c_slice, caller_stream
                            )
                            energies.append(e_part)
                            forces.append(f_part)
                            if w_part is not None:
                                virials.append(w_part)
                        energy_t = torch.cat(energies, dim=0)
                        force_t = torch.cat(forces, dim=0)
                        virial_t = torch.cat(virials, dim=0) if virials else None

                    if caller_stream is not None:
                        self._last_stream_obj = caller_stream

                if virial_t is not None:
                    return {"energy": energy_t, "forces": force_t, "virial": virial_t}
                return energy_t, force_t

        return Module()


def _check_float64_spec(spec, name):
    if not hasattr(spec, "shape") or not hasattr(spec, "dtype"):
        raise TypeError(f"jax-dlpack {name} must be a JAX array")
    dtype = np.dtype(spec.dtype)
    if dtype != np.float64:
        raise ValueError(
            f"jax-dlpack {name} must have dtype float64, got {dtype}; "
            "no implicit precision conversion"
        )


def _inspect_output_spec(out, *, expected_energy_shapes, expected_force_shape,
                         expected_virial_shape):
    """Validate abstract output structure from ``jax.eval_shape``."""
    if isinstance(out, Mapping):
        energy = out.get("energy", out.get("energies"))
        if energy is None:
            raise TypeError("jax-dlpack mapping output must contain 'energy' or 'energies'")
        _check_float64_spec(energy, "energy")
        if tuple(energy.shape) not in expected_energy_shapes:
            raise TypeError(f"jax-dlpack energy shape {tuple(energy.shape)} is incompatible")
        forces = out.get("forces")
        if forces is not None:
            _check_float64_spec(forces, "forces")
            if tuple(forces.shape) != expected_force_shape:
                raise ValueError(
                    f"jax-dlpack forces shape {tuple(forces.shape)} != {expected_force_shape}"
                )
        virial = out.get("virial")
        stress = out.get("stress")
        if virial is not None and stress is not None:
            raise PotentialError("jax-dlpack callable must return either virial or stress, not both")
        tensor_w = virial if virial is not None else stress
        if tensor_w is not None:
            _check_float64_spec(tensor_w, "virial" if virial is not None else "stress")
            if tuple(tensor_w.shape) != expected_virial_shape:
                raise ValueError(
                    f"jax-dlpack virial/stress shape {tuple(tensor_w.shape)} != {expected_virial_shape}"
                )
        return {
            "kind": "mapping",
            "has_forces": forces is not None,
            "has_virial": virial is not None,
            "has_stress": stress is not None,
        }
    if isinstance(out, (tuple, list)):
        if len(out) != 2:
            raise PotentialError("jax-dlpack tuple output must be (energy, forces)")
        energy, forces = out
        _check_float64_spec(energy, "energy")
        _check_float64_spec(forces, "forces")
        if tuple(energy.shape) not in expected_energy_shapes:
            raise TypeError(f"jax-dlpack energy shape {tuple(energy.shape)} is incompatible")
        if tuple(forces.shape) != expected_force_shape:
            raise ValueError(
                f"jax-dlpack forces shape {tuple(forces.shape)} != {expected_force_shape}"
            )
        return {
            "kind": "tuple",
            "has_forces": True,
            "has_virial": False,
            "has_stress": False,
        }
    _check_float64_spec(out, "energy")
    if tuple(out.shape) not in expected_energy_shapes:
        raise TypeError(f"jax-dlpack energy shape {tuple(out.shape)} is incompatible")
    return {
        "kind": "scalar",
        "has_forces": False,
        "has_virial": False,
        "has_stress": False,
    }


class JAXDLPackPotential(Potential):
    """First-class zero-copy JAX potential provider with NumPy and TorchPES APIs."""

    def __init__(
        self,
        model=None,
        *,
        factory=None,
        function=None,
        params=None,
        device="cpu",
        dtype="float64",
        units=None,
        position_unit=None,
        energy_unit=None,
        derivative="force",
        batched="auto",
        periodic=False,
        virial="auto",
        cell=None,
        pbc=None,
        max_frames_per_call=None,
        atomic_numbers=None,
        charge=0,
        multiplicity=1,
        _provider_name="jax-dlpack",
    ):
        if dtype != "float64":
            raise ValueError("jax-dlpack requires float64 precision; no implicit conversion")
        if not isinstance(periodic, (bool, np.bool_)):
            raise ValueError("jax-dlpack periodic must be a boolean")
        if virial not in ("auto", "autograd", "provided", "none"):
            raise ValueError("jax-dlpack virial must be auto, autograd, provided, or none")
        if derivative not in ("force", "gradient"):
            raise ValueError("jax-dlpack derivative must be force or gradient")
        if not (isinstance(batched, (bool, np.bool_)) or batched == "auto"):
            raise ValueError("jax-dlpack batched must be a boolean or 'auto'")
        if max_frames_per_call is not None and (
            isinstance(max_frames_per_call, (bool, np.bool_))
            or not isinstance(max_frames_per_call, (int, np.integer))
            or max_frames_per_call < 1
        ):
            raise ValueError("max_frames_per_call must be a positive integer")
        if factory is not None and function is not None:
            raise ValueError("Specify either factory or function, not both")
        if model is None and factory is None and function is None:
            raise ValueError("jax-dlpack requires model, factory, or function")
        if units is not None and (position_unit is not None or energy_unit is not None):
            raise ValueError("Specify either units mapping or position_unit/energy_unit, not both")
        if position_unit is not None or energy_unit is not None:
            units = {
                "positions": "bohr" if position_unit is None else position_unit,
                "energy": "hartree" if energy_unit is None else energy_unit,
            }
        self.length_factor, self.energy_factor = unit_factors(units)
        resolved_units = units or {}
        self.position_unit = resolved_units.get(
            "positions", resolved_units.get("length", "bohr")
        )
        self.energy_unit = resolved_units.get("energy", "hartree")
        self.derivative = derivative
        self.sign = 1.0 if derivative == "force" else -1.0
        self.batched = bool(batched) if isinstance(batched, (bool, np.bool_)) else "auto"
        self.periodic = bool(periodic)
        self.virial_mode = virial
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        self._check_cell_contract()
        self.max_frames_per_call = (
            None if max_frames_per_call is None else int(max_frames_per_call)
        )
        self.atomic_numbers = (
            None if atomic_numbers is None else tuple(int(z) for z in atomic_numbers)
        )
        self.charge, self.multiplicity = int(charge), int(multiplicity)
        self._provider_name = str(_provider_name)
        self.closed = False
        self._torch_module = None
        self._compiled_by_atoms = {}

        jax, jnp, canonical_device, jax_device = _load_jax_runtime(
            device, backend_label=self._provider_name
        )
        self._jax = jax
        self._jnp = jnp
        self.device = canonical_device
        self.jax_device = jax_device

        options = dict(params or {})
        candidate = None
        try:
            with jax.default_device(self.jax_device):
                if factory is not None:
                    maker = factory if callable(factory) else import_object(factory)
                    if model is not None:
                        options.setdefault("model", model)
                    candidate = maker(**options)
                elif function is not None:
                    fn_obj = function if callable(function) else import_object(function)
                    candidate = functools.partial(fn_obj, **options) if options else fn_obj
                else:
                    fn_obj = import_object(model) if isinstance(model, str) else model
                    candidate = functools.partial(fn_obj, **options) if options else fn_obj

                self._candidate = candidate
                declared = getattr(candidate, "metadata", None)
                if declared is not None:
                    _check_declared_metadata(declared)

                if callable(candidate):
                    fn = candidate
                elif callable(getattr(candidate, "get_energy_and_forces", None)):
                    fn = candidate.get_energy_and_forces
                elif callable(getattr(candidate, "evaluate", None)):
                    fn = candidate.evaluate
                else:
                    raise TypeError(
                        "jax-dlpack requires a JAX callable or an object implementing "
                        "get_energy_and_forces/evaluate"
                    )
                _check_pytree_float64(jax, jnp, (candidate, fn))
                self._fn = fn
                self._cell_jax = (
                    None
                    if self.cell is None
                    else jax.device_put(
                        jnp.asarray(self.cell / self.length_factor, dtype=jnp.float64),
                        self.jax_device,
                    )
                )
                self._pbc_jax = jax.device_put(
                    jnp.asarray(self.pbc, dtype=jnp.bool_), self.jax_device
                )
                if self.atomic_numbers is not None and len(self.atomic_numbers) >= 1:
                    self._get_compiled(len(self.atomic_numbers))
        except BaseException:
            if factory is not None and callable(getattr(candidate, "close", None)):
                candidate.close()
            raise

        self.metadata = {
            "provider": self._provider_name,
            "batch_independent": True,
            "trajectory_batch_safe": True,
            "native_batch": True,
            "stateful": False,
            "thread_safety": "context_serial",
            "execution_device": self.device,
            "dtype": "float64",
            "energy_dtype": "float64",
            "virial_dtype": "float64",
            "periodic": self.periodic,
            "supports_pbc": self.periodic,
            "position_unit": self.position_unit,
            "energy_unit": self.energy_unit,
            "force_unit": f"{self.energy_unit}/{self.position_unit}",
            "atomic_numbers": (
                None if self.atomic_numbers is None else list(self.atomic_numbers)
            ),
            "state": {"charge": self.charge, "multiplicity": self.multiplicity},
            "jax_version": importlib.metadata.version("jax"),
            "max_frames_per_call": self.max_frames_per_call,
        }

    @classmethod
    def from_spec(cls, spec, system=None):
        spec = dict(spec)
        cell, pbc = None, None
        if system is not None:
            get = (
                system.get
                if isinstance(system, Mapping)
                else lambda key, default=None: getattr(system, key, default)
            )
            cell = get("cell")
            pbc = get("pbc")
            raw_numbers = get("atomic_numbers")
            if raw_numbers is not None:
                actual = tuple(int(z) for z in raw_numbers)
                declared = spec.get("atomic_numbers", actual)
                if tuple(int(z) for z in declared) != actual:
                    raise ValueError("jax-dlpack atomic_numbers do not match System order")
                spec.setdefault("atomic_numbers", list(actual))
            spec.setdefault("charge", get("charge", 0))
            spec.setdefault("multiplicity", get("multiplicity", 1))
        params = dict(spec.get("params", {}) or {})
        dtype = spec.get("dtype", params.pop("dtype", "float64"))
        return cls(
            spec.get("model"),
            factory=spec.get("factory"),
            function=spec.get("function"),
            params=params,
            device=spec.get("device", "cpu"),
            dtype=dtype,
            units=spec.get("units"),
            position_unit=spec.get("position_unit"),
            energy_unit=spec.get("energy_unit"),
            derivative=spec.get("derivative", "force"),
            batched=spec.get("batched", "auto"),
            periodic=spec.get("periodic", False),
            virial=spec.get("virial", "auto"),
            cell=cell,
            pbc=pbc,
            max_frames_per_call=spec.get("max_frames_per_call"),
            atomic_numbers=spec.get("atomic_numbers"),
            charge=spec.get("charge", 0),
            multiplicity=spec.get("multiplicity", 1),
        )

    def _check_cell_contract(self):
        if self.pbc.any() and not self.periodic:
            raise PotentialError(
                "Periodic jax-dlpack models require periodic=True and callable(q, cell, pbc)"
            )
        if self.periodic and self.cell is None:
            raise PotentialError("Periodic jax-dlpack models require a nonsingular cell")

    def _check_open(self):
        if self.closed:
            raise PotentialError(f"{self._provider_name} potential is closed")

    def bind_system(self, system):
        self._check_open()
        get = (
            system.get
            if isinstance(system, Mapping)
            else lambda key, default=None: getattr(system, key, default)
        )
        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        raw_numbers = get("atomic_numbers")
        if raw_numbers is not None and self.atomic_numbers is not None:
            if tuple(int(z) for z in raw_numbers) != self.atomic_numbers:
                raise PotentialError("jax-dlpack system atomic_numbers differ from provider")
        old_cell, old_pbc = self.cell, self.pbc
        self.cell, self.pbc = cell, pbc
        try:
            self._check_cell_contract()
        except BaseException:
            self.cell, self.pbc = old_cell, old_pbc
            raise
        with self._jax.default_device(self.jax_device):
            self._cell_jax = (
                None
                if self.cell is None
                else self._jax.device_put(
                    self._jnp.asarray(self.cell / self.length_factor, dtype=self._jnp.float64),
                    self.jax_device,
                )
            )
            self._pbc_jax = self._jax.device_put(
                self._jnp.asarray(self.pbc, dtype=self._jnp.bool_), self.jax_device
            )
        if self._torch_module is not None:
            self._torch_module.update_pbc(self.pbc)

    def set_cell(self, cell):
        self._check_open()
        new_cell, _ = validate_cell_pbc(cell, self.pbc)
        if new_cell is None or not self.pbc.all():
            raise PotentialError("variable-cell jax-dlpack PES requires full periodicity")
        self.cell = new_cell
        with self._jax.default_device(self.jax_device):
            self._cell_jax = self._jax.device_put(
                self._jnp.asarray(self.cell / self.length_factor, dtype=self._jnp.float64),
                self.jax_device,
            )

    def _probe_contract(self, atoms):
        jax, jnp = self._jax, self._jnp
        fn = self._fn
        periodic = self.periodic

        def probe_single(allow_1d_scalar):
            q1 = jax.ShapeDtypeStruct((atoms, 3), jnp.float64)
            if periodic:
                c1 = jax.ShapeDtypeStruct((3, 3), jnp.float64)
                p1 = jax.ShapeDtypeStruct((3,), jnp.bool_)
                out = jax.eval_shape(lambda q, c, p: _invoke_periodic(fn, q, c, p), q1, c1, p1)
            else:
                out = jax.eval_shape(fn, q1)
            energy_shapes = ((), (1,)) if allow_1d_scalar else ((),)
            return _inspect_output_spec(
                out,
                expected_energy_shapes=energy_shapes,
                expected_force_shape=(atoms, 3),
                expected_virial_shape=(3, 3),
            )

        def probe_batched():
            info = None
            for b in (max(atoms + 2, 4), max(atoms + 3, 5)):
                qb = jax.ShapeDtypeStruct((b, atoms, 3), jnp.float64)
                if periodic:
                    cb = jax.ShapeDtypeStruct((b, 3, 3), jnp.float64)
                    pb = jax.ShapeDtypeStruct((b, 3), jnp.bool_)
                    out = jax.eval_shape(
                        lambda q, c, p: _invoke_periodic(fn, q, c, p), qb, cb, pb
                    )
                else:
                    out = jax.eval_shape(fn, qb)
                info = _inspect_output_spec(
                    out,
                    expected_energy_shapes=((b,), (b, 1)),
                    expected_force_shape=(b, atoms, 3),
                    expected_virial_shape=(b, 3, 3),
                )
            return info

        if self.batched is False:
            return "single", probe_single(allow_1d_scalar=True)
        if self.batched is True:
            return "batched", probe_batched()

        single_dtype_err = None
        try:
            return "single", probe_single(allow_1d_scalar=False)
        except ValueError as err:
            if "dtype float64" in str(err):
                single_dtype_err = err
            else:
                single_err = err
        except Exception as err:
            single_err = err
        try:
            return "batched", probe_batched()
        except Exception as batch_err:
            if single_dtype_err is not None:
                raise single_dtype_err from batch_err
            raise PotentialError(
                f"jax-dlpack callable does not satisfy single-frame ({single_err}) "
                f"or batched ({batch_err}) float64 PES contracts"
            ) from batch_err

    def _build_compiled(self, atoms):
        jax, jnp = self._jax, self._jnp
        fn = self._fn
        periodic = self.periodic
        sign = self.sign
        virial_mode = self.virial_mode

        with jax.default_device(self.jax_device):
            mode, spec = self._probe_contract(atoms)
            has_model_virial = spec["has_virial"] or spec["has_stress"]
            if virial_mode == "provided" and not has_model_virial:
                raise PotentialError("virial='provided' requires model virial or stress")
            derive_virial = virial_mode == "autograd" or (
                virial_mode == "auto" and periodic and not has_model_virial
            )
            emit_virial = virial_mode != "none" and (has_model_virial or derive_virial)

            if mode == "single":
                def call_single(q1, c1=None, p1=None):
                    return _invoke_periodic(fn, q1, c1, p1) if periodic else fn(q1)

                def unpack_single(raw, c1=None):
                    if spec["kind"] == "mapping":
                        e = jnp.reshape(raw.get("energy", raw.get("energies")), ())
                        f = raw.get("forces")
                        if f is not None:
                            f = f * sign
                        w = raw.get("virial")
                        if w is None and raw.get("stress") is not None:
                            if c1 is None:
                                raise PotentialError("jax-dlpack stress requires a periodic cell")
                            w = -raw["stress"] * jnp.abs(jnp.linalg.det(c1))
                        return e, f, w
                    if spec["kind"] == "tuple":
                        e, d = raw
                        return jnp.reshape(e, ()), d * sign, None
                    return jnp.reshape(raw, ()), None, None

                if spec["has_forces"] and not derive_virial:
                    def single_eval(q1, c1=None, p1=None):
                        e, f, w = unpack_single(call_single(q1, c1, p1), c1)
                        return (e, f, w) if emit_virial else (e, f)
                elif spec["has_forces"] and derive_virial:
                    def single_eval(q1, c1=None, p1=None):
                        def strained(eps):
                            t = jnp.eye(3, dtype=jnp.float64) + eps
                            q_def = q1 @ t.T
                            c_def = None if c1 is None else c1 @ t.T
                            e_val, f_val, _ = unpack_single(call_single(q_def, c_def, p1), c_def)
                            return e_val, f_val

                        (e, f), grad_eps = jax.value_and_grad(strained, has_aux=True)(
                            jnp.zeros((3, 3), dtype=jnp.float64)
                        )
                        return e, f, -grad_eps
                elif not derive_virial:
                    def single_eval(q1, c1=None, p1=None):
                        def energy_with_aux(q_var):
                            e_val, _, w_val = unpack_single(call_single(q_var, c1, p1), c1)
                            return e_val, w_val

                        (e, w), grad_q = jax.value_and_grad(energy_with_aux, has_aux=True)(q1)
                        return (e, -grad_q, w) if emit_virial else (e, -grad_q)
                else:
                    def single_eval(q1, c1=None, p1=None):
                        def strained(q_var, eps):
                            t = jnp.eye(3, dtype=jnp.float64) + eps
                            q_def = q_var @ t.T
                            c_def = None if c1 is None else c1 @ t.T
                            e_val, _, _ = unpack_single(call_single(q_def, c_def, p1), c_def)
                            return e_val

                        e, (grad_q, grad_eps) = jax.value_and_grad(
                            strained, argnums=(0, 1)
                        )(q1, jnp.zeros((3, 3), dtype=jnp.float64))
                        return e, -grad_q, -grad_eps

                if periodic:
                    compiled = jax.jit(jax.vmap(single_eval, in_axes=(0, 0, 0)))
                else:
                    compiled = jax.jit(jax.vmap(lambda q1: single_eval(q1)))
                return compiled

            # Batched [F, N, 3] callable contract.
            def call_batch(qb, cb=None, pb=None):
                return _invoke_periodic(fn, qb, cb, pb) if periodic else fn(qb)

            def unpack_batch(raw, frames, cb=None):
                if spec["kind"] == "mapping":
                    e = jnp.reshape(raw.get("energy", raw.get("energies")), (frames,))
                    f = raw.get("forces")
                    if f is not None:
                        f = f * sign
                    w = raw.get("virial")
                    if w is None and raw.get("stress") is not None:
                        if cb is None:
                            raise PotentialError("jax-dlpack stress requires a periodic cell")
                        vol = jnp.abs(jnp.linalg.det(cb))[:, None, None]
                        w = -raw["stress"] * vol
                    return e, f, w
                if spec["kind"] == "tuple":
                    e, d = raw
                    return jnp.reshape(e, (frames,)), d * sign, None
                return jnp.reshape(raw, (frames,)), None, None

            if spec["has_forces"] and not derive_virial:
                def batch_eval(qb, cb=None, pb=None):
                    e, f, w = unpack_batch(call_batch(qb, cb, pb), qb.shape[0], cb)
                    return (e, f, w) if emit_virial else (e, f)
            elif spec["has_forces"] and derive_virial:
                def batch_eval(qb, cb=None, pb=None):
                    frames = qb.shape[0]

                    def strained_sum(eps):
                        t = jnp.eye(3, dtype=jnp.float64)[None] + eps
                        q_def = jnp.einsum("fni,fji->fnj", qb, t)
                        c_def = None if cb is None else jnp.einsum("fai,fji->faj", cb, t)
                        e_vec, f_vec, _ = unpack_batch(call_batch(q_def, c_def, pb), frames, c_def)
                        return jnp.sum(e_vec), (e_vec, f_vec)

                    (_, (e, f)), grad_eps = jax.value_and_grad(strained_sum, has_aux=True)(
                        jnp.zeros((frames, 3, 3), dtype=jnp.float64)
                    )
                    return e, f, -grad_eps
            elif not derive_virial:
                def batch_eval(qb, cb=None, pb=None):
                    frames = qb.shape[0]

                    def energy_sum(q_var):
                        e_vec, _, w_vec = unpack_batch(call_batch(q_var, cb, pb), frames, cb)
                        return jnp.sum(e_vec), (e_vec, w_vec)

                    (_, (e, w)), grad_q = jax.value_and_grad(energy_sum, has_aux=True)(qb)
                    return (e, -grad_q, w) if emit_virial else (e, -grad_q)
            else:
                def batch_eval(qb, cb=None, pb=None):
                    frames = qb.shape[0]

                    def strained_sum(q_var, eps):
                        t = jnp.eye(3, dtype=jnp.float64)[None] + eps
                        q_def = jnp.einsum("fni,fji->fnj", q_var, t)
                        c_def = None if cb is None else jnp.einsum("fai,fji->faj", cb, t)
                        e_vec, _, _ = unpack_batch(call_batch(q_def, c_def, pb), frames, c_def)
                        return jnp.sum(e_vec), e_vec

                    (_, e), (grad_q, grad_eps) = jax.value_and_grad(
                        strained_sum, argnums=(0, 1), has_aux=True
                    )(qb, jnp.zeros((frames, 3, 3), dtype=jnp.float64))
                    return e, -grad_q, -grad_eps

            if periodic:
                return jax.jit(batch_eval)
            return jax.jit(lambda qb: batch_eval(qb))

    def _get_compiled(self, atoms):
        if self.atomic_numbers is not None and atoms != len(self.atomic_numbers):
            raise PotentialError(
                f"{self._provider_name} atom count {atoms} does not match "
                f"declared atomic_numbers ({len(self.atomic_numbers)})"
            )
        compiled = self._compiled_by_atoms.get(atoms)
        if compiled is None:
            compiled = self._build_compiled(atoms)
            self._compiled_by_atoms[atoms] = compiled
        return compiled

    def _evaluate_jax_batch(self, q_jax, cell_jax=None, pbc_jax=None):
        compiled = self._get_compiled(int(q_jax.shape[1]))
        if self.periodic:
            return compiled(q_jax, cell_jax, pbc_jax)
        return compiled(q_jax)

    def _eval_numpy_chunk(self, q_slice):
        jax, jnp = self._jax, self._jnp
        q_model = jax.device_put(
            jnp.asarray(q_slice / self.length_factor, dtype=jnp.float64),
            self.jax_device,
        )
        if self.periodic:
            frames = len(q_slice)
            cell_batch = jnp.broadcast_to(self._cell_jax, (frames, 3, 3))
            pbc_batch = jnp.broadcast_to(self._pbc_jax, (frames, 3))
            out = self._evaluate_jax_batch(q_model, cell_batch, pbc_batch)
        else:
            out = self._evaluate_jax_batch(q_model)
        return jax.device_get(out)

    def evaluate(self, positions):
        self._check_open()
        x = coordinates(positions)
        limit = self.max_frames_per_call
        if limit is None or len(x) <= limit:
            host_out = self._eval_numpy_chunk(x)
            if len(host_out) == 3:
                e_raw, f_raw, w_raw = host_out
            else:
                e_raw, f_raw = host_out
                w_raw = None
        else:
            energies, forces, virials = [], [], []
            for start in range(0, len(x), limit):
                chunk_out = self._eval_numpy_chunk(x[start:start + limit])
                energies.append(np.asarray(chunk_out[0], dtype=np.float64))
                forces.append(np.asarray(chunk_out[1], dtype=np.float64))
                if len(chunk_out) == 3:
                    virials.append(np.asarray(chunk_out[2], dtype=np.float64))
            e_raw = np.concatenate(energies, axis=0)
            f_raw = np.concatenate(forces, axis=0)
            w_raw = np.concatenate(virials, axis=0) if virials else None

        energy = np.asarray(e_raw, dtype=np.float64) * self.energy_factor
        force = np.asarray(f_raw, dtype=np.float64) * (
            self.energy_factor / self.length_factor
        )
        virial = (
            None
            if w_raw is None
            else np.asarray(w_raw, dtype=np.float64) * self.energy_factor
        )
        return validate_result(PESResult(energy, force, virial=virial), x.shape)

    def as_torch_pes(self):
        self._check_open()
        if self._torch_module is None:
            self._torch_module = _JAXDLPackTorchModule(
                jax=self._jax,
                jnp=self._jnp,
                jax_device=self.jax_device,
                torch_device_name=self.device,
                evaluate_jax_batch=self._evaluate_jax_batch,
                periodic=self.periodic,
                pbc=self.pbc,
                max_frames_per_call=self.max_frames_per_call,
                provider_name=self._provider_name,
                owner=self,
            )
        from ..torch_backend import TorchPES

        torch_virial = (
            "none"
            if self.virial_mode == "none"
            else (
                "provided"
                if (self.virial_mode in ("provided", "autograd") or self.periodic)
                else "auto"
            )
        )
        result = TorchPES(
            self._torch_module,
            device=self.device,
            length_factor=self.length_factor,
            energy_factor=self.energy_factor,
            cell=self.cell,
            pbc=self.pbc,
            periodic=self.periodic,
            virial=torch_virial,
        )
        result.metadata.update(self.metadata)
        return result

    def warmup_torch(self, frames, atoms):
        """Compile the selected JAX batch shape before production sampling."""
        self._check_open()
        if isinstance(frames, (bool, np.bool_)) or not isinstance(frames, (int, np.integer)) or frames < 1:
            raise ValueError("warmup frames must be a positive integer")
        if isinstance(atoms, (bool, np.bool_)) or not isinstance(atoms, (int, np.integer)) or atoms < 1:
            raise ValueError("warmup atoms must be a positive integer")
        import torch

        adapter = self.as_torch_pes()
        q = torch.zeros((int(frames), int(atoms), 3), dtype=torch.float64, device=adapter.device)
        adapter.evaluate_tensor(q)
        if adapter.device.type == "cuda":
            torch.cuda.synchronize(adapter.device)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._torch_module is not None:
            self._torch_module.reset_state()
            self._torch_module = None
        self._compiled_by_atoms.clear()
        close_fn = getattr(getattr(self, "_candidate", None), "close", None)
        if callable(close_fn):
            close_fn()


__all__ = ["JAXDLPackPotential"]
