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


def make_diabatic_potential(spec, system=None):
    """A diabatic potential energy matrix (DPEM) from a live object or a specification.

    ``spec`` is one of

    * an object with ``evaluate(q[B,N,3])`` returning a ``DiabaticResult``
      (a builtin model, ``NaH2LEPS5G``, ``ChemPotPyDPEM`` or a plugin), used as is;
    * ``{"model": name, "params": {...}}`` for a builtin or published DPEM by the
      names of the nonadiabatic input (``tully1``, ``tully2``, ``tully3``,
      ``lvc``, ``spin_boson``, ``nah2_leps_5g_1992``, ``newns_anderson``);
      ``nah2_leps_5g_1992`` takes its atom order from ``system`` when
      ``params.symbols`` is omitted;
    * ``{"provider": "diabatic_python", "factory" | "function": ..., ...}``,
      the plugin adapter of :func:`pyrpmd.potentials.make_potential`;
    * ``{"provider": "chempotpy", "system": "OH3", "surface": "OH3_PIP_FFW1_2019_DPEM",
      "symbols": [...], "gradient": "auto"}``, a ChemPotPy DPEM through
      :class:`~pyrpmd.potentials.multistate_external.ChemPotPyDPEM`; ``symbols``
      default to the atom order of ``system``.
    """
    from collections.abc import Mapping

    if not isinstance(spec, Mapping):
        if callable(getattr(spec, "evaluate", None)):
            return spec
        raise TypeError("a diabatic potential must be an object with evaluate(q) or a specification table")
    if "model" in spec and "provider" not in spec:
        unknown = set(spec) - {"model", "params"}
        if unknown:
            raise ValueError(f"unknown diabatic model fields {sorted(unknown)}; use model and params")
        params = dict(spec.get("params") or {})
        if spec["model"] == "nah2_leps_5g_1992" and "symbols" not in params and system is not None:
            symbols = system.get("symbols") if isinstance(system, Mapping) else getattr(system, "symbols", None)
            if symbols is not None:
                params["symbols"] = list(symbols)
        from ..nonadiabatic_input import _model

        return _model({"name": spec["model"], "params": params})
    kind = spec.get("provider", spec.get("type"))
    if kind == "diabatic_python":
        from . import make_potential

        return make_potential(dict(spec), system)
    if kind == "chempotpy":
        from .multistate_external import ChemPotPyDPEM

        allowed = {"provider", "type", "system", "surface", "symbols", "gradient", "difference_step_bohr",
                   "nstates", "source_sha256", "gradient_check_tolerance", "check_points",
                   "check_displacement_bohr"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"unknown chempotpy fields {sorted(unknown)}; allowed: {sorted(allowed)}")
        if "system" not in spec or "surface" not in spec:
            raise ValueError("a chempotpy DPEM needs system and surface, e.g. system = 'OH3', "
                             "surface = 'OH3_PIP_FFW1_2019_DPEM'")
        options = {k: v for k, v in spec.items() if k not in ("provider", "type")}
        if "symbols" not in options and system is not None:
            symbols = system.get("symbols") if isinstance(system, Mapping) else getattr(system, "symbols", None)
            if symbols is not None:
                options["symbols"] = list(symbols)
        return ChemPotPyDPEM(**options)
    raise ValueError("a diabatic potential specification needs model = '<name>', provider = 'diabatic_python' "
                     "or provider = 'chempotpy'")


class DiabaticAdiabatPotential(Potential):
    """One adiabatic surface of a DPEM as a single-surface PES.

    ``E_k(q)`` is eigenvalue ``state`` (ascending) of the diabatic matrix and the
    force is the Hellmann--Feynman ``-<k|dV/dq|k>``, exact for a fixed diabatic
    basis wherever ``E_k`` is nondegenerate.  At a conical intersection the
    adiabat has a cusp; frames within ``degeneracy_tolerance`` (hartree) of a
    neighbouring adiabat are reported in ``messages``.  Nonadiabatic couplings
    are not part of this PES: use it for the ground-state RPMD rate and the
    subspace nonadiabatic factor for the coupled-state correction.
    """

    def __init__(self, diabatic, *, state=0, degeneracy_tolerance=1e-8, owns=False):
        if not callable(getattr(diabatic, "evaluate", None)):
            raise TypeError("diabatic must have evaluate(q) returning a diabatic matrix")
        if isinstance(state, bool) or int(state) != state or int(state) < 0:
            raise ValueError("state must be a nonnegative adiabat index")
        tolerance = float(degeneracy_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("degeneracy_tolerance must be finite and nonnegative")
        self.diabatic, self.state, self.degeneracy_tolerance = diabatic, int(state), tolerance
        self._owns = bool(owns)
        inner = dict(getattr(diabatic, "metadata", {}) or {})
        nstates = inner.get("nstates", getattr(diabatic, "nstates", None))
        if nstates is not None and self.state >= int(nstates):
            raise ValueError(f"state {self.state} is outside the {int(nstates)} diabatic states")
        self.metadata = {
            "provider": "diabatic_adiabat", "state": self.state, "derivative": "gradient",
            "execution_device": inner.get("execution_device", "cpu"),
            "thread_safety": inner.get("thread_safety", "context_serial"),
            "periodic": False,
            "diabatic": {k: inner[k] for k in ("provider", "nstates", "natoms", "model_identity", "symbols")
                         if k in inner},
        }
        if "trajectory_batch_safe" in inner:
            self.metadata["trajectory_batch_safe"] = inner["trajectory_batch_safe"]

    def evaluate(self, q):
        from ..nonadiabatic import validate_diabatic_result
        from .base import PESResult

        q = coordinates(q)
        result = validate_diabatic_result(self.diabatic.evaluate(q), q.shape)
        matrix = np.asarray(result.matrix)
        if self.state >= matrix.shape[-1]:
            raise ValueError(f"state {self.state} is outside the {matrix.shape[-1]} diabatic states")
        energies, vectors = np.linalg.eigh(matrix)
        vector = vectors[..., self.state]
        gradient = np.einsum("bi,bijna,bj->bna", vector.conj(), np.asarray(result.gradients), vector).real
        messages = [""] * len(q)
        if matrix.shape[-1] > 1:
            gaps = np.diff(energies, axis=-1)
            near = np.zeros(len(q), dtype=bool)
            if self.state > 0:
                near |= gaps[:, self.state - 1] <= self.degeneracy_tolerance
            if self.state < matrix.shape[-1] - 1:
                near |= gaps[:, self.state] <= self.degeneracy_tolerance
            for b in np.flatnonzero(near):
                messages[b] = "adiabat degenerate with a neighbour: Hellmann-Feynman force is one-sided"
        status = None if result.status is None else np.asarray(result.status)
        return PESResult(np.ascontiguousarray(energies[:, self.state].real), -gradient, status, tuple(messages))

    def close(self):
        if self._owns and callable(getattr(self.diabatic, "close", None)):
            self.diabatic.close()
