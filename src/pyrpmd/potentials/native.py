"""Opt-in native bindings; a lease pins the library and locks its live context.

Only trusted compiled functions may be bound. ``fortran-batch-v1`` is a BIND(C)
function: void(int32_t B, double q[B,N,3], double E[B], double F[B,N,3],
int32_t status[B]). Arrays use atomic units and physical forces, not gradients.
No Python callbacks, hidden units conversions or stateful evaluators are implied.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import numbers

from .base import PotentialError


@dataclass(frozen=True)
class NativePESBinding:
    kind: str
    function: object
    natoms: int
    owner: object
    lock: object
    context: object = None
    is_open: object = None

    @contextmanager
    def acquire(self):
        """Hold the provider's existing lock until the GIL-free block finishes."""
        with self.lock:
            if getattr(self.owner, "closed", False) or (self.is_open is not None and not self.is_open()):
                raise PotentialError("PES context is closed")
            if self.kind not in ("cabi-v1", "fortran-batch-v1"):
                raise PotentialError("unsupported native PES binding version")
            if not isinstance(self.function, ctypes._CFuncPtr):
                raise PotentialError("native binding requires a compiled ctypes symbol")
            if any(type(value).__name__ == "CThunkObject"
                   for value in (getattr(self.function, "_objects", None) or {}).values()):
                raise PotentialError("native binding does not accept Python callbacks")
            address = ctypes.cast(self.function, ctypes.c_void_p).value
            context = getattr(self.context, "value", self.context) or 0
            if (not address or isinstance(self.natoms, bool)
                    or not isinstance(self.natoms, numbers.Integral) or self.natoms < 1):
                raise PotentialError("invalid native PES address or atom count")
            # self.owner and self.function remain alive throughout the with block.
            yield (self.kind, address, int(context), int(self.natoms))


def descriptor_binding(binding):
    """Adapt the original capsule binding without changing its lifetime lease."""
    if isinstance(binding, NativePESBinding):
        return binding
    from ..propagation import NativePESBinding as CapsuleBinding
    if isinstance(binding, CapsuleBinding):
        kinds = {0: "cabi-v1", 1: "fortran-batch-v1"}
        if binding.kind not in kinds:
            raise PotentialError("unsupported native PES binding version")
        return NativePESBinding(
            kinds[binding.kind], binding.function, binding.natoms,
            binding.owner, binding.lock, binding.context, binding.is_open,
        )
    raise ValueError("invalid native PES binding")
