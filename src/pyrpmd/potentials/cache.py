"""Short-lived reuse of a physical PES evaluation within one accepted step.

No result survives a transaction boundary. Providers, native bindings and
their process-local lifetimes remain owned by the caller. This is intended
for fixed-Hamiltonian workflows whose energy and force consumers evaluate
the same geometry during one integration/observation transaction.
"""

from contextlib import contextmanager

import numpy as np

from .base import PESResult, coordinates, validate_result


class StepEvaluationCache:
    """Forward a provider, reusing its last geometry only inside ``step()``.

    Own both the coordinate key and returned arrays: some external providers
    reuse their output buffers. Exact coordinate comparison avoids stale
    hits when an integrator updates an array in place. A failed step clears
    all cached data, so retries always consult the provider again.
    """

    def __init__(self, provider):
        self._provider = provider
        self._active = False
        self._coordinates = None
        self._result = None

    def __getattr__(self, name):
        return getattr(self._provider, name)

    @contextmanager
    def step(self):
        if self._active:
            raise RuntimeError("PES evaluation transactions cannot be nested")
        self._active = True
        try:
            yield self
        finally:
            self._coordinates = self._result = None
            self._active = False

    def evaluate(self, q):
        if not self._active:
            return self._provider.evaluate(q)
        q = coordinates(q)
        if self._coordinates is not None and np.array_equal(q, self._coordinates):
            return self._result
        result = validate_result(self._provider.evaluate(q), np.shape(q))
        owned = PESResult(
            result.energy.copy(), result.forces.copy(), result.status.copy(),
            result.messages, None if result.virial is None else result.virial.copy(),
        )
        for array in (owned.energy, owned.forces, owned.status, owned.virial):
            if array is not None:
                array.flags.writeable = False
        self._coordinates = np.array(q, copy=True)
        self._result = owned
        return owned
