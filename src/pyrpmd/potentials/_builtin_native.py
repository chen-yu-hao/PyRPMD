"""Shape-aware leases for analytic C ABI evaluators.

Process-local capsules live outside provider instances so deepcopy/pickle and
scientific identities retain their original, parameter-only representation.
Every context owns a parameter snapshot; an outstanding lease survives both
provider deletion and replacement of the cached snapshot.
"""

import ctypes
import hashlib
import json
import numbers
import threading
import weakref

import numpy as np

from .base import PESResult, PotentialError, coordinates
from .cabi import DP, Error, I32, InputV2, OutputV2
from .native import NativePESBinding


_CACHE = weakref.WeakKeyDictionary()
_LJ_CACHE = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()
_EVALUATE = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
)
_EVALUATE_V2 = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(InputV2),
    ctypes.POINTER(OutputV2), ctypes.POINTER(Error),
)


def resolve_native_binding(potential, natoms, *, cell=None):
    """Supply shape only to the built-in factory, preserving existing hooks.

    A non-None cell denotes the cell-aware thermal executor. Its molecular
    C ABI v1 route cannot accept cells/virials; decline before creating it.
    Fixed-cell legacy propagation instead uses the cosine context's own cell.
    """
    from .builtin import _BUILTIN_NATIVE_BINDING
    from ..propagation import NativeUnavailableError

    factory = getattr(potential, "native_binding", None)
    if not callable(factory):
        return None
    if getattr(factory, "__func__", None) is _BUILTIN_NATIVE_BINDING:
        if cell is not None:
            raise NativeUnavailableError(
                "built-in C ABI v1 binding cannot supply cell-aware thermal "
                "forces/virials; this periodic operator combination requires Python"
            )
        return factory(natoms)
    return factory()


def _parameters(potential, natoms):
    from .builtin import (
        FreePotential, HarmonicPotential, MorsePairPotential,
        DoubleWellPotential, PeriodicCosinePotential,
    )
    from ..propagation import NativeUnavailableError

    kind = type(potential)
    if kind is FreePotential:
        return "free", {}
    if kind is HarmonicPotential:
        parameters = {}
        for name in ("k", "center"):
            value = np.asarray(getattr(potential, name))
            if value.ndim > 2:
                raise NativeUnavailableError(
                    "native harmonic parameters cannot depend on the frame axis"
                )
            # Match NumPy's [B,N,3] broadcasting, without binding the batch axis.
            parameters[name] = np.broadcast_to(value, (natoms, 3)).ravel().tolist()
        return "harmonic", parameters
    if kind is MorsePairPotential:
        return "morse_pair", {
            "atoms": np.asarray(potential.atoms).tolist(), "depth": potential.depth,
            "alpha": potential.alpha, "equilibrium": potential.equilibrium,
        }
    if kind is DoubleWellPotential:
        return "double_well", {
            name: getattr(potential, name)
            for name in ("height", "width", "atom", "axis", "confinement")
        }
    if kind is PeriodicCosinePotential:
        return "periodic_cosine", {
            "inverse": np.asarray(potential.inverse).ravel().tolist(),
            "height": np.asarray(potential.height).tolist(),
            "repeats": np.asarray(potential.repeats).tolist(),
            "perpendicular": np.asarray(potential.perpendicular).ravel().tolist(),
            "atoms": None if potential.atoms is None else np.asarray(potential.atoms).tolist(),
            "confinement": potential.confinement,
        }
    raise NativeUnavailableError("custom analytic potential requires Python")


def builtin_binding(potential, natoms=None):
    from .builtin import _BUILTIN_EVALUATE
    from ..propagation import NativeUnavailableError

    kind = type(potential)
    if (kind not in _BUILTIN_EVALUATE or "evaluate" in potential.__dict__
            or kind.evaluate is not _BUILTIN_EVALUATE[kind]):
        raise NativeUnavailableError("overridden built-in potential requires Python")
    metadata = potential.metadata
    if (metadata.get("stateful") is True
            or metadata.get("capabilities", {}).get("stateful") is True
            or metadata.get("ring_polymer_coupled") is True
            or metadata.get("coupled_frames") is True):
        raise NativeUnavailableError("native built-in PES requires independent, stateless frames")
    if natoms is None:
        natoms = getattr(potential, "_native_natoms", None)
    if natoms is None:
        raise NativeUnavailableError(
            "built-in native binding requires natoms or a prior evaluate() call"
        )
    if (isinstance(natoms, (bool, np.bool_))
            or not isinstance(natoms, numbers.Integral) or natoms < 1):
        raise ValueError("native built-in atom count must be a positive integer")
    natoms = int(natoms)
    try:
        from .. import _native
    except ImportError as exc:
        raise NativeUnavailableError("compiled built-in PES extension is unavailable") from exc
    if not hasattr(_native, "builtin_pes"):
        raise NativeUnavailableError("installed native extension lacks built-in PES bindings; rebuild it")

    model, parameters = _parameters(potential, natoms)
    # Historical Morse/quartic constructors accepted some nonfinite parameters.
    # Their Python semantics remain available, but no invalid native context is
    # advertised (e.g. an infinite quartic width has a finite Python limit).
    if any(value is not None and not np.isfinite(value).all()
           for value in parameters.values()):
        raise NativeUnavailableError("native built-in PES requires finite parameters")
    key = tuple((name, tuple(value) if isinstance(value, list) else value)
                for name, value in parameters.items())
    with _CACHE_LOCK:
        cached = _CACHE.setdefault(potential, {})
        previous = cached.get(natoms)
        if previous is None or previous[0] != key:
            capsule, address, context = _native.builtin_pes(model, natoms, parameters)
            binding = NativePESBinding(
                "cabi-v1", _EVALUATE(address), natoms,
                (capsule, _native), threading.RLock(), context=context,
            )
            cached[natoms] = (key, binding)
        return cached[natoms][1]


def _lj_binding(potential):
    from ..propagation import NativeUnavailableError

    try:
        from .. import _native
    except ImportError as exc:
        raise NativeUnavailableError(
            "compiled periodic Lennard-Jones extension is unavailable; select backend='numpy' explicitly"
        ) from exc
    if not hasattr(_native, "builtin_lj"):
        raise NativeUnavailableError(
            "installed native extension lacks periodic Lennard-Jones C ABI v2 support; rebuild it"
        )
    key = (potential.epsilon, potential.sigma, potential.cutoff, potential.skin)
    with _CACHE_LOCK:
        previous = _LJ_CACHE.get(potential)
        if previous is None or previous[0] != key:
            capsule, address, context = _native.builtin_lj(*key)
            previous = (
                key,
                _EVALUATE_V2(address),
                context,
                (capsule, _native),
                threading.RLock(),
            )
            _LJ_CACHE[potential] = previous
        return previous


def evaluate_lj(potential, x):
    """Evaluate the built-in periodic LJ context through the physical ABI v2."""
    q = coordinates(x)
    batches, natoms, _ = q.shape
    cells = np.ascontiguousarray(
        np.broadcast_to(np.asarray(potential.cell), (batches, 3, 3)), dtype=np.float64
    )
    pbc = np.ones((batches, 3), dtype=np.int32)
    energy = np.empty(batches, dtype=np.float64)
    forces = np.empty_like(q)
    virial = np.empty((batches, 3, 3), dtype=np.float64)
    status = np.empty(batches, dtype=np.int32)
    message_stride = 192
    messages = ctypes.create_string_buffer(batches * message_stride)
    error_buffer = ctypes.create_string_buffer(512)
    error = Error(ctypes.addressof(error_buffer), len(error_buffer), 0)
    inp = InputV2(
        abi_version=2,
        struct_size=ctypes.sizeof(InputV2),
        layout=1,
        scalar_type=1,
        nstructures=batches,
        natoms=natoms,
        positions=q.ctypes.data_as(DP),
        positions_count=q.size,
        cells=cells.ctypes.data_as(DP),
        cells_count=cells.size,
        pbc=pbc.ctypes.data_as(ctypes.POINTER(I32)),
        pbc_count=pbc.size,
    )
    out = OutputV2(
        abi_version=2,
        struct_size=ctypes.sizeof(OutputV2),
        energies=energy.ctypes.data_as(DP),
        energies_count=energy.size,
        forces=forces.ctypes.data_as(DP),
        forces_count=forces.size,
        structure_status=status.ctypes.data_as(ctypes.POINTER(I32)),
        structure_status_count=status.size,
        structure_messages=ctypes.addressof(messages),
        structure_message_stride=message_stride,
        structure_messages_capacity=len(messages),
        virials=virial.ctypes.data_as(DP),
        virials_count=virial.size,
    )
    _, function, context, owner, lock = _lj_binding(potential)
    del owner  # Kept alive by the cache entry throughout this call.
    with lock:
        rc = function(context, ctypes.byref(inp), ctypes.byref(out), ctypes.byref(error))
    if rc:
        detail = error_buffer.value.decode("utf-8", "replace")
        raise PotentialError(
            f"periodic Lennard-Jones C ABI v2 evaluation failed ({rc})"
            + (f": {detail}" if detail else "")
        )
    decoded = tuple(
        messages.raw[i * message_stride:(i + 1) * message_stride]
        .split(b"\0", 1)[0].decode("utf-8", "replace")
        for i in range(batches)
    )
    return PESResult(energy, forces, status, decoded, virial)


def builtin_identity(potential, natoms):
    """Internal restart identity for known mathematics, never a run option.

    Before bindings existed, an unlabelled built-in could restore a Python
    bath in a fresh Dynamics. Give its new full native executor the same
    portability, while still rejecting different scientific parameters.
    Arbitrary providers retain their explicit identity/runtime-only policy.
    """
    from .builtin import _BUILTIN_EVALUATE, _BUILTIN_NATIVE_BINDING

    kind = type(potential)
    if (kind not in _BUILTIN_EVALUATE or "evaluate" in potential.__dict__
            or kind.evaluate is not _BUILTIN_EVALUATE[kind]
            or getattr(potential.native_binding, "__func__", None) is not _BUILTIN_NATIVE_BINDING):
        return None
    model, parameters = _parameters(potential, natoms)
    payload = json.dumps(
        {"model": model, "natoms": natoms, "parameters": parameters},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return "builtin-v1:" + hashlib.sha256(payload.encode()).hexdigest()
