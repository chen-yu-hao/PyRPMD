"""Checked Condon surfaces for the existing two-state diabatic PES contract.

The returned scalar providers borrow the matrix provider. Every evaluation
checks the declared constant real electronic coupling and its zero gradient;
this adapter never replaces a position-dependent coupling with a constant.
All values use the existing atomic-unit energy/force convention.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import math
from pathlib import Path

import numpy as np

from .io import file_sha256
from .nonadiabatic import validate_diabatic_result
from .potentials.base import PESResult, Potential, PotentialError, coordinates
from .validation import finite_real


@dataclass(frozen=True)
class _CondonSettings:
    coupling_hartree: float
    coupling_rtol: float
    coupling_atol_hartree: float
    coupling_gradient_tolerance: float

    @property
    def value_tolerance(self):
        return self.coupling_atol_hartree + self.coupling_rtol * abs(self.coupling_hartree)


def _source_identity(provider):
    sources = {}
    for name in ("evaluate", "fn"):
        method = getattr(provider, name, None)
        if callable(method):
            try:
                filename = inspect.getsourcefile(method)
            except (TypeError, OSError):
                filename = None
            if filename and Path(filename).is_file():
                sources[name] = file_sha256(filename)
    return {"provider_class": f"{type(provider).__module__}:{type(provider).__qualname__}",
            "source_sha256": sources,
            "metadata": deepcopy(getattr(provider, "metadata", {}))}


class _CondonSurface(Potential):
    def __init__(self, provider, state, settings, identity):
        self._provider = provider
        self._state = state
        self._settings = settings
        self.metadata = {
            "provider": "golden_rule_diabatic_diagonal",
            "execution_device": "cpu",
            "thread_safety": "context_serial",
            "diabatic_state": state,
            "matrix_provider": deepcopy(identity),
            "condon": {
                "coupling_hartree": settings.coupling_hartree,
                "coupling_rtol": settings.coupling_rtol,
                "coupling_atol_hartree": settings.coupling_atol_hartree,
                "coupling_gradient_tolerance_hartree_per_bohr": settings.coupling_gradient_tolerance,
                "validation": "every matrix evaluation; exactly two states; real constant coupling",
            },
            "hessian": "central force differences unless separate scalar providers are supplied",
            "ownership": "borrowed matrix provider; caller closes it",
        }

    @property
    def coupling_hartree(self):
        return self._settings.coupling_hartree

    @property
    def diabatic_state(self):
        return self._state

    def evaluate(self, q):
        q = coordinates(q)
        raw = self._provider.evaluate(q)
        value = validate_diabatic_result(raw, q.shape, nstates=2)
        # The common validator permits a small Hermitian roundoff error and
        # returns its symmetric projection. Check both *original* off-diagonal
        # elements so that averaging cannot erase a stricter Condon violation.
        matrix, gradients = (raw[:2] if isinstance(raw, (tuple, list))
                             else (raw.matrix, raw.gradients))
        deviation = np.abs(np.asarray(matrix)[:, (0, 1), (1, 0)] - self.coupling_hartree)
        if np.any(deviation > self._settings.value_tolerance):
            raise PotentialError(
                "golden-rule adapter requires the declared real constant Condon coupling; "
                f"maximum matrix-element deviation is {float(np.max(deviation)):.6g} hartree"
            )
        derivative = np.abs(np.asarray(gradients)[:, (0, 1), (1, 0)])
        if np.any(derivative > self._settings.coupling_gradient_tolerance):
            raise PotentialError(
                "golden-rule adapter requires a zero coupling gradient; "
                f"maximum is {float(np.max(derivative)):.6g} hartree/bohr"
            )
        # The matrix validation above establishes Hermitian, finite diagonals.
        # Own both output buffers even when the matrix provider reuses memory.
        return PESResult(value.matrix[:, self._state, self._state].real.copy(),
                         -value.gradients[:, self._state, self._state].real.copy())


def diabatic_surfaces(matrix_potential, *, coupling_hartree, coupling_rtol=1e-10,
                      coupling_atol_hartree=0.0, coupling_gradient_tolerance=0.0):
    """Return ``(surface0, surface1)`` for a real, constant-coupling 2-state PES.

    ``matrix_potential.evaluate(q)`` follows :class:`DiabaticResult`: matrix
    energies and positive matrix gradients in atomic units. An explicitly
    declared, nonzero real ``coupling_hartree`` is checked at every sampled
    geometry, including finite-difference displacements. Absolute and relative
    coupling tolerances cannot admit a zero or missing electronic coupling.
    The gradient tolerance applies to all Cartesian coordinates, including
    frozen ones. No PES work is performed when constructing the adapters.

    Each returned surface exposes read-only ``coupling_hartree`` and
    ``diabatic_state`` properties. The golden-rule rate API verifies that its
    coupling magnitude agrees with this declaration; a global sign change is
    equivalent. These adapters do not claim an analytic Hessian interface.
    Closing a surface does not close the borrowed matrix provider.
    """
    if not callable(getattr(matrix_potential, "evaluate", None)):
        raise TypeError("matrix_potential must implement evaluate(q)")
    coupling = finite_real(coupling_hartree, "coupling_hartree")
    if coupling == 0:
        raise ValueError("coupling_hartree must be nonzero in a golden-rule calculation")
    rtol = finite_real(coupling_rtol, "coupling_rtol", nonnegative=True)
    atol = finite_real(coupling_atol_hartree, "coupling_atol_hartree", nonnegative=True)
    gtol = finite_real(coupling_gradient_tolerance, "coupling_gradient_tolerance", nonnegative=True)
    settings = _CondonSettings(coupling, rtol, atol, gtol)
    if not math.isfinite(settings.value_tolerance) or settings.value_tolerance >= abs(coupling):
        raise ValueError("coupling tolerances must be smaller than the coupling magnitude")
    identity = _source_identity(matrix_potential)
    return (_CondonSurface(matrix_potential, 0, settings, identity),
            _CondonSurface(matrix_potential, 1, settings, identity))
