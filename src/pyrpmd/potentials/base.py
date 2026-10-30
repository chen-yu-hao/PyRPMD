"""Physical PES contract: bohr, hartree and forces=-dV/dq; no bead scaling."""

from dataclasses import dataclass
import numpy as np


class PotentialError(RuntimeError):
    pass


class BackendUnavailableError(PotentialError):
    pass


@dataclass
class PESResult:
    energy: np.ndarray
    forces: np.ndarray
    status: np.ndarray | None = None
    messages: tuple[str, ...] | None = None
    # Optional physical configurational virial, [B,3,3], in hartree. No
    # kinetic/ring-spring terms; not by itself a quantum pressure estimator.
    virial: np.ndarray | None = None

    def __post_init__(self):
        if (
            np.asarray(self.energy).dtype.kind not in "fiu"
            or np.asarray(self.forces).dtype.kind not in "fiu"
        ):
            raise PotentialError("PES energy/forces must be real numerical arrays")
        self.energy = np.asarray(self.energy, dtype=np.float64)
        self.forces = np.asarray(self.forces, dtype=np.float64)
        if self.virial is not None:
            if np.asarray(self.virial).dtype.kind not in "fiu":
                raise PotentialError("PES virial must be a real numerical array")
            self.virial = np.asarray(self.virial, dtype=np.float64)
        n = len(self.energy) if self.energy.ndim else 0
        if self.status is None:
            self.status = np.zeros(n, dtype=np.int32)
        else:
            status = np.asarray(self.status)
            if (
                status.dtype.kind not in "iu"
                or np.any(status < 0)
                or np.any(status > np.iinfo(np.int32).max)
            ):
                raise PotentialError("PES status must be nonnegative int32 values")
            self.status = np.asarray(status, dtype=np.int32)
        if self.messages is None:
            self.messages = ("",) * n
        else:
            self.messages = tuple(self.messages)


def coordinates(value):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape) < 1:
        raise PotentialError(
            "PES coordinates must be a nonempty real [batch,atom,3] array"
        )
    a = np.ascontiguousarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise PotentialError("PES coordinates contain NaN/Inf")
    return a


def validate_result(result, shape, *, require_success=True):
    if isinstance(result, (tuple, list)):
        if len(result) not in (2, 3, 4, 5):
            raise PotentialError(
                "PES must return energy, forces, optional status/messages/virial"
            )
        result = PESResult(*result)
    if not isinstance(result, PESResult):
        if all(hasattr(result, k) for k in ("energy", "forces")):
            result = PESResult(
                result.energy,
                result.forces,
                getattr(result, "status", None),
                getattr(result, "messages", None),
                getattr(result, "virial", None),
            )
        else:
            raise PotentialError(
                "PES result does not implement the energy/forces contract"
            )
    n = shape[0]
    if (
        result.energy.shape != (n,)
        or result.forces.shape != tuple(shape)
        or result.status.shape != (n,)
        or len(result.messages) != n
    ):
        raise PotentialError(
            "PES result shape mismatch: expected energy[B], forces[B,N,3], status[B], messages[B]"
        )
    good = result.status == 0
    finite = np.isfinite(result.energy) & np.isfinite(result.forces).all(axis=(1, 2))
    if result.virial is not None:
        if result.virial.shape != (n, 3, 3):
            raise PotentialError("PES virial shape must be [B,3,3]")
        finite &= np.isfinite(result.virial).all(axis=(1, 2))
        result.virial = np.ascontiguousarray(result.virial)
    if np.any(good & ~finite):
        raise PotentialError("PES reported success with nonfinite energy/forces")
    if require_success and not np.all(good):
        bad = np.flatnonzero(~good)
        raise PotentialError(
            "PES frames failed: "
            + ", ".join(
                f"{i} (status={result.status[i]}: {result.messages[i]})"
                for i in bad[:8]
            )
        )
    result.energy = np.ascontiguousarray(result.energy)
    result.forces = np.ascontiguousarray(result.forces)
    return result


class Potential:
    metadata = {"thread_safety": "context_serial", "execution_device": "cpu"}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def check_potential(potential, q, epsilon=1e-5):
    """Finite difference check in atomic units; small geometry diagnostic only."""
    q = coordinates(q)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    r = validate_result(potential.evaluate(q), q.shape)
    numerical = np.empty_like(q)
    for a in range(q.shape[1]):
        for c in range(3):
            plus = q.copy()
            minus = q.copy()
            plus[:, a, c] += epsilon
            minus[:, a, c] -= epsilon
            ep = validate_result(potential.evaluate(plus), q.shape).energy
            em = validate_result(potential.evaluate(minus), q.shape).energy
            numerical[:, a, c] = -(ep - em) / (2 * epsilon)
    delta = r.forces - numerical
    return {
        "max_absolute_force_error": float(np.max(np.abs(delta))),
        "rms_force_error": float(np.sqrt(np.mean(delta * delta))),
        "force_unit": "hartree/bohr",
        "finite_difference_step_bohr": float(epsilon),
    }
