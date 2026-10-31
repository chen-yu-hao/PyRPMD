"""Number-dependent PESs and a per-configuration adiabatic electrode surface.

The batch axis always denotes independent nuclear configurations. A shared
electronic number across an entire ring belongs to ``GrandCanonicalSampler``;
minimizing that shared number would couple beads and violate this PES contract.
"""

from dataclasses import dataclass
import numpy as np

from ..geometry import validate_cell_pbc
from ..potentials.base import Potential, PESResult, PotentialError, coordinates, validate_result
from ..potentials.python import import_object
from .reservoir import ElectronReservoir, _finite_real, _electron_bounds


@dataclass
class ElectronicPESResult:
    """E, forces and partial dE/dN at fixed geometry, all in atomic units.

    ``energy`` and ``denergy_dnumber`` have shape [B]; ``forces`` is [B,A,3].
    The electronic derivative is positive dE/dN, while nuclear forces are
    negative dE/dq. N is a continuous model coordinate, not an electron count
    rounded to integers. Provider status/messages/virial reuse the PES contract.
    """

    energy: np.ndarray
    forces: np.ndarray
    denergy_dnumber: np.ndarray
    status: np.ndarray | None = None
    messages: tuple[str, ...] | None = None
    virial: np.ndarray | None = None

    def as_pes_result(self):
        return PESResult(self.energy, self.forces, self.status, self.messages, self.virial)


def evaluate_electronic(model, q, electron_numbers):
    """Evaluate an AU provider with ``evaluate(q, electron_numbers[B])``.

    Scalar numbers are broadcast explicitly, which is useful for shared-N MC.
    Arrays are copied before passing them to providers, so iteration bounds and
    scalar-number state are never exposed as writable provider work buffers.
    """
    q = coordinates(q)
    number = np.asarray(electron_numbers)
    if number.dtype.kind not in "fiu" or number.ndim > 1:
        raise PotentialError("electron numbers must be a real scalar or [batch] array")
    if number.ndim == 0:
        number = np.full(len(q), float(number))
    elif number.shape != (len(q),):
        raise PotentialError("electron number shape must be [batch]")
    if not np.isfinite(number).all():
        raise PotentialError("electron numbers contain NaN/Inf")
    result = model.evaluate(q, np.array(number, dtype=float, copy=True))
    if not isinstance(result, ElectronicPESResult):
        raise PotentialError("number-dependent PES must return ElectronicPESResult")
    pes = validate_result(result.as_pes_result(), q.shape)
    derivative = np.asarray(result.denergy_dnumber)
    if (derivative.dtype.kind not in "fiu" or derivative.shape != (len(q),)
            or not np.isfinite(derivative).all()):
        raise PotentialError("denergy_dnumber must be a finite real [batch] array")
    return ElectronicPESResult(
        pes.energy, pes.forces, np.ascontiguousarray(derivative, dtype=float),
        pes.status, pes.messages, pes.virial,
    )


class _ElectrodeWrapper(Potential):
    def __init__(self, model):
        if not callable(getattr(model, "evaluate", None)):
            raise TypeError("electrode model must implement evaluate(q, electron_numbers)")
        self.model = model
        # Do not advertise an upstream native binding for a different Hamiltonian.
        source = dict(getattr(model, "metadata", {}) or {})
        self.metadata = {
            "thread_safety": source.get("thread_safety", "context_serial"),
            "execution_device": source.get("execution_device", "cpu"),
            "requires_atomic_units": True,
        }
        if source.get("supports_pbc") is True and callable(getattr(model, "bind_system", None)):
            self.metadata["supports_pbc"] = True

    def bind_system(self, system):
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        _, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        if pbc.any() and self.metadata.get("supports_pbc") is not True:
            raise PotentialError("periodic electronic model requires supports_pbc=True and bind_system")
        bind = getattr(self.model, "bind_system", None)
        if callable(bind):
            bind(system)

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


class FixedElectronNumberPotential(_ElectrodeWrapper):
    """Physical E(q,N) at a fixed scalar N; primarily for fixed-N HMC proposals.

    The returned energy does not include the reservoir term. Do not change N
    inside an active PES evaluation cache or a deterministic propagation block.
    """

    def __init__(self, model, electron_number):
        super().__init__(model)
        self.electron_number = electron_number
        self.metadata["electrode_approximation"] = "fixed_electron_number"

    @property
    def electron_number(self):
        return self._electron_number

    @electron_number.setter
    def electron_number(self, value):
        self._electron_number = _finite_real(value, "electron_number")

    def evaluate(self, q):
        return evaluate_electronic(self.model, q, self.electron_number).as_pes_result()


@dataclass(frozen=True)
class AdiabaticElectrodeResult:
    pes: PESResult
    electron_numbers: np.ndarray
    chemical_potentials_au: np.ndarray
    iterations: int


class AdiabaticElectrodePotential(_ElectrodeWrapper):
    """Independent-frame BO surface Omega(q)=min_N[E(q,N)-mu*N].

    E(q,N) must be smooth and strictly convex in N throughout electron_bounds.
    An interior stationary point is required for every configuration. Endpoint
    and root checks detect common failures, but cannot prove global convexity of
    an arbitrary external model. The energy and force belong to the minimized
    grand potential; envelope-theorem forces are the fixed-N nuclear forces at
    the optimum. This model omits finite-temperature electronic-number entropy.

    A ring uses one optimum per bead. It is intentionally distinct from the
    shared-N grand canonical ensemble and from extended-system TP-RPMD.
    """

    def __init__(
        self, model, reservoir, *, electron_bounds, number_tolerance=1e-10,
        chemical_potential_tolerance_au=1e-9, max_iterations=100,
    ):
        super().__init__(model)
        if not isinstance(reservoir, ElectronReservoir):
            raise TypeError("reservoir must be an ElectronReservoir")
        self.reservoir = reservoir
        self.electron_bounds = _electron_bounds(electron_bounds)
        self.number_tolerance = _finite_real(number_tolerance, "number_tolerance", positive=True)
        self.chemical_potential_tolerance_au = _finite_real(
            chemical_potential_tolerance_au, "chemical_potential_tolerance_au", positive=True,
        )
        if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = int(max_iterations)
        self.metadata["electrode_approximation"] = "independent_frame_adiabatic_minimum"

    def solve(self, q):
        q = coordinates(q)
        lo = np.full(len(q), self.electron_bounds[0])
        hi = np.full(len(q), self.electron_bounds[1])
        mu = self.reservoir.chemical_potential_au
        def residual_of(result):
            with np.errstate(over="ignore", invalid="ignore"):
                residual = result.denergy_dnumber - mu
            if not np.isfinite(residual).all():
                raise PotentialError("electronic chemical-potential residual overflow")
            return residual

        rlo = residual_of(evaluate_electronic(self.model, q, lo))
        rhi = residual_of(evaluate_electronic(self.model, q, hi))
        if np.any(rlo >= 0) or np.any(rhi <= 0):
            raise PotentialError(
                "electronic minimum is not bracketed strictly inside electron_bounds; "
                "check the model/calibration/domain (no boundary clamping is applied)"
            )
        number = lo + (hi - lo) * 0.5
        done = np.zeros(len(q), dtype=bool)
        for iteration in range(1, self.max_iterations + 1):
            # Safeguarded secant interpolation retains batching and is exact in
            # one update for the common quadratic charging model. The 10% bound
            # guarantees bracket contraction for strongly nonlinear derivatives.
            scale = np.maximum(np.abs(rlo), np.abs(rhi))
            fraction = np.clip(-(rlo / scale) / (rhi / scale - rlo / scale), 0.1, 0.9)
            number = np.where(done, number, lo + (hi - lo) * fraction)
            result = evaluate_electronic(self.model, q, number)
            residual = residual_of(result)
            if np.any((~done) & ((residual < rlo) | (residual > rhi))):
                raise PotentialError("dE/dN is not monotone in electron_bounds; a convex electronic model is required")
            done |= np.abs(residual) <= self.chemical_potential_tolerance_au
            if np.all(done):
                # Reject a stationary local maximum (or a numerically
                # unresolved/flat electronic branch). Endpoint bracketing
                # alone accepts, for example, the maximum of (N**2-1)**2.
                # These local probes still cannot establish global convexity;
                # that remains a required property of the supplied model.
                lower, upper = self.electron_bounds
                delta = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(number))
                minus = number - np.minimum(delta, 0.25 * (number - lower))
                plus = number + np.minimum(delta, 0.25 * (upper - number))
                minus = np.where(minus < number, minus, np.nextafter(number, lower))
                plus = np.where(plus > number, plus, np.nextafter(number, upper))
                if np.any((minus >= number) | (plus <= number)):
                    raise PotentialError("electronic minimum has no resolvable interior neighborhood")
                # Providers may reuse all result buffers. Snapshot the root
                # before evaluating either displaced electronic number.
                energy = result.energy - mu * number
                chemical_potentials = result.denergy_dnumber.copy()
                pes = validate_result(
                    PESResult(
                        energy, result.forces.copy(), result.status.copy(), result.messages,
                        None if result.virial is None else result.virial.copy(),
                    ), q.shape,
                )
                left = residual_of(evaluate_electronic(self.model, q, minus))
                right = residual_of(evaluate_electronic(self.model, q, plus))
                if np.any((left >= residual) | (right <= residual)):
                    raise PotentialError(
                        "local dE/dN is not strictly increasing or is numerically unresolved; "
                        "a convex electronic model is required"
                    )
                return AdiabaticElectrodeResult(
                    pes, number.copy(), chemical_potentials, iteration,
                )
            below = (~done) & (residual < 0)
            above = (~done) & ~below
            lo = np.where(below, number, lo)
            rlo = np.where(below, residual, rlo)
            hi = np.where(above, number, hi)
            rhi = np.where(above, residual, rhi)
            if np.any((~done) & ((hi - lo) <= self.number_tolerance)):
                raise PotentialError("electronic bracket reached number_tolerance before chemical-potential convergence")
        raise PotentialError("electronic minimization exceeded max_iterations")

    def evaluate(self, q):
        return self.solve(q).pes


def build_adiabatic_electrode(*, model, reservoir, electron_bounds, **solver_options):
    """Importable factory for the existing ``potential.provider='python'`` route.

    ``model={factory: 'module:builder', params: {...}}`` constructs an atomic-unit
    number-dependent provider. ``reservoir`` contains ElectronReservoir fields.
    The *outer* Python potential must also use atomic units and physical forces.
    Existing input-path resolution handles nested file-based factory strings.
    """
    if not isinstance(model, dict) or set(model) - {"factory", "params"} or "factory" not in model:
        raise ValueError("model requires factory and optional params")
    if not isinstance(model.get("params", {}), dict):
        raise ValueError("model.params must be a mapping")
    if not isinstance(reservoir, dict):
        raise ValueError("reservoir must be a mapping of ElectronReservoir fields")
    bath = ElectronReservoir(**reservoir)
    provider = import_object(model["factory"])(**model.get("params", {}))
    try:
        return AdiabaticElectrodePotential(provider, bath, electron_bounds=electron_bounds, **solver_options)
    except BaseException:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        raise
