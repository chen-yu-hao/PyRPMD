"""CPU Python PES adapter for Hermitian diabatic matrices and positive gradients."""

import numpy as np

from .base import Potential, coordinates
from .python import unit_factors


class DiabaticPythonPotential(Potential):
    def __init__(self, obj, *, units=None, derivative="gradient"):
        if derivative != "gradient":
            raise ValueError("diabatic_python requires positive matrix gradients")
        self.obj = obj
        self.fn = obj.evaluate if hasattr(obj, "evaluate") else obj
        if not callable(self.fn):
            raise TypeError("Matrix PES must be callable or have evaluate(q)")
        self.length_factor, self.energy_factor = unit_factors(units)
        self.metadata = {"provider": "diabatic_python", "execution_device": "cpu",
                         "thread_safety": "context_serial", "derivative": "gradient"}

    def evaluate(self, q):
        from ..nonadiabatic import DiabaticResult, validate_diabatic_result

        q = coordinates(q)
        result = validate_diabatic_result(self.fn(q / self.length_factor), q.shape)
        return DiabaticResult(
            result.matrix * self.energy_factor,
            result.gradients * (self.energy_factor / self.length_factor),
        )

    def close(self):
        close = getattr(self.obj, "close", None)
        if close is not None:
            close()


def check_diabatic_potential(potential, q, nstates, epsilon=1e-5):
    """Compare every matrix-element gradient against central finite differences."""
    from ..nonadiabatic import validate_diabatic_result

    q = coordinates(q)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    result = validate_diabatic_result(potential.evaluate(q), q.shape, nstates)
    numerical = np.empty_like(result.gradients)
    for a in range(q.shape[1]):
        for c in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, a, c] += epsilon
            minus[:, a, c] -= epsilon
            vp = validate_diabatic_result(potential.evaluate(plus), q.shape, nstates).matrix
            vm = validate_diabatic_result(potential.evaluate(minus), q.shape, nstates).matrix
            numerical[..., a, c] = (vp - vm) / (2 * epsilon)
    delta = result.gradients - numerical
    return {"max_absolute_gradient_error": float(np.max(np.abs(delta))),
            "rms_gradient_error": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
            "gradient_unit": "hartree/bohr", "finite_difference_step_bohr": epsilon}
