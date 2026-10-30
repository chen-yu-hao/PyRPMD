"""Opt-in native bindings; a lease pins the library and locks its live context.

Only trusted compiled functions may be bound. ``fortran-batch-v1`` is a BIND(C)
function: void(int32_t B, double q[B,N,3], double E[B], double F[B,N,3],
int32_t status[B]). Arrays use atomic units and physical forces, not gradients.
No Python callbacks, hidden units conversions or stateful evaluators are implied.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes

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
            if self.is_open is not None and not self.is_open():
                raise PotentialError("PES context is closed")
            if self.kind not in ("cabi-v1", "fortran-batch-v1"):
                raise PotentialError("unsupported native PES binding version")
            if not isinstance(self.function, ctypes._CFuncPtr):
                raise PotentialError("native binding requires a compiled ctypes symbol")
            address = ctypes.cast(self.function, ctypes.c_void_p).value
            context = getattr(self.context, "value", self.context) or 0
            if not address or isinstance(self.natoms, bool) or self.natoms < 1:
                raise PotentialError("invalid native PES address or atom count")
            # self.owner and self.function remain alive throughout the with block.
            yield (self.kind, address, int(context), int(self.natoms))
