"""Shape-aware leases for the stateless analytic C ABI v1 evaluators.

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

from .native import NativePESBinding


_CACHE = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()
_EVALUATE = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
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

    # A harmonic binding expands scalar/vector parameters over all atoms.
    # Compare the original arrays first; unchanged values need no expansion.
    from .builtin import HarmonicPotential
    source_parameters = None
    if kind is HarmonicPotential:
        current = tuple(np.asarray(getattr(potential, name)) for name in ("k", "center"))
        with _CACHE_LOCK:
            previous = _CACHE.get(potential, {}).get(natoms)
            if previous is not None and len(previous) == 3 and all(
                    new.dtype == old.dtype and new.shape == old.shape and np.array_equal(new, old)
                    for new, old in zip(current, previous[2])):
                return previous[1]
        source_parameters = tuple(value.copy() for value in current)

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
            cached[natoms] = ((key, binding) if source_parameters is None else
                              (key, binding, source_parameters))
        return cached[natoms][1]


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
