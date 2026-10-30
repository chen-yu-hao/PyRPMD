"""Real-mass RPMD: sum-beads Hamiltonian sampled at beta/P, q,p[P,N,3].

The production BCB map uses the cyclic Cayley midpoint method from the user's
RPMDrate implementation (commits 19ba2602, 9a44d6b3); reference NumPy arithmetic
evaluates exactly the same map in Fourier modes. No thermostat is applied in
step(). Andersen thermalization has bead momentum variance m*P/beta.
"""

from dataclasses import dataclass
import importlib
import math
import threading
import numpy as np
from .potentials.base import validate_result


@dataclass
class State:
    q: np.ndarray
    p: np.ndarray
    step: int = 0
    time: float = 0.0

    def __post_init__(self):
        self.q = np.array(self.q, dtype=np.float64, order="C", copy=True)
        self.p = np.array(self.p, dtype=np.float64, order="C", copy=True)
        if (
            self.q.ndim != 3
            or self.q.shape[-1] != 3
            or min(self.q.shape) < 1
            or self.q.shape != self.p.shape
        ):
            raise ValueError("q,p must have equal nonempty [P,N,3] shapes")
        if not np.isfinite(self.q).all() or not np.isfinite(self.p).all():
            raise ValueError("q,p must be finite")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, (int, np.integer))
            or self.step < 0
            or not np.isfinite(self.time)
        ):
            raise ValueError("invalid state clock")

    def copy(self):
        return State(self.q, self.p, self.step, self.time)


def native_capabilities():
    try:
        mod = importlib.import_module("pyrpmd._native")
    except ImportError:
        return {"available": False, "openmp": False}
    return {"available": True, "openmp": bool(mod.has_openmp())}


class CayleyIntegrator:
    """Full cyclic ring map; immutable coefficients, independent working state."""

    def __init__(self, masses, beta, nbeads, dt, threads=1, native=True):
        self.masses = np.array(masses, dtype=float, copy=True)
        if (
            self.masses.ndim != 1
            or not len(self.masses)
            or not np.isfinite(self.masses).all()
            or np.any(self.masses <= 0)
        ):
            raise ValueError("invalid masses")
        self.masses.flags.writeable = False
        self.beta = float(beta)
        self.dt = float(dt)
        if not math.isfinite(self.beta) or self.beta <= 0 or not math.isfinite(self.dt):
            raise ValueError("invalid beta/dt")
        if (
            isinstance(nbeads, bool)
            or not isinstance(nbeads, (int, np.integer))
            or nbeads < 1
        ):
            raise ValueError("nbeads must be positive integer")
        if (
            isinstance(threads, bool)
            or not isinstance(threads, (int, np.integer))
            or threads < 1
        ):
            raise ValueError("threads must be positive integer")
        if native not in (True, False, None, "required"):
            raise ValueError("native must be True/False/None/'required'")
        self.nbeads = int(nbeads)
        self.threads = int(threads)
        self.shape = (self.nbeads, len(self.masses), 3)
        self._native = None
        self._context = None
        self._lock = threading.Lock()
        if native is not False:
            try:
                self._native = importlib.import_module("pyrpmd._native")
            except ImportError:
                if native == "required":
                    raise RuntimeError(
                        "Native kernel not installed; build/install PyRPMD or use native=False"
                    )
        if self.threads > 1 and (self._native is None or not self._native.has_openmp()):
            raise RuntimeError("threads>1 requires the compiled OpenMP kernel")
        if self._native:
            self._context = self._native.create(
                self.masses.tolist(), self.beta, self.nbeads, self.dt, self.threads
            )
        k = np.arange(self.nbeads // 2 + 1)
        self._omega = 2 * self.nbeads / self.beta * np.sin(np.pi * k / self.nbeads)
        self.native = self._native is not None

    def _check(self, q, p):
        for a in (q, p):
            if (
                not isinstance(a, np.ndarray)
                or a.dtype != np.float64
                or a.shape != self.shape
                or not a.flags.c_contiguous
                or not a.flags.writeable
            ):
                raise ValueError(f"expected writable C-contiguous float64 {self.shape}")
            if not np.isfinite(a).all():
                raise ValueError("state contains NaN/Inf")
        if np.shares_memory(q, p):
            raise ValueError("q and p must not share memory")

    def free_step(self, q, p):
        self._check(q, p)
        with self._lock:
            if self._native:
                self._native.free_step(self._context, q, p)
            else:
                self._fourier_step(q, p, half=False)

    def _fourier_step(self, q, p, half=False):
        qk = np.fft.rfft(q, axis=0, norm="ortho")
        pk = np.fft.rfft(p, axis=0, norm="ortho")
        z = 0.5 * self.dt * self._omega
        if half:
            den = np.sqrt(1 + z * z)
            a = 1 / den
            b = (0.5 * self.dt) / den
            c = 0.5 * self.dt * self._omega**2 / den
        else:
            den = 1 + z * z
            a = (1 - z * z) / den
            b = self.dt / den
            c = self.dt * self._omega**2 / den
        q[:] = np.fft.irfft(
            a[:, None, None] * qk + b[:, None, None] * pk / self.masses[None, :, None],
            n=self.nbeads,
            axis=0,
            norm="ortho",
        )
        p[:] = np.fft.irfft(
            a[:, None, None] * pk - c[:, None, None] * self.masses[None, :, None] * qk,
            n=self.nbeads,
            axis=0,
            norm="ortho",
        )

    def half_step(self, q, p):
        """Square root C(dt)**(1/2), deliberately not C(dt/2)."""
        self._check(q, p)
        with self._lock:
            self._fourier_step(q, p, half=True)

    def kick(self, p, force, h):
        force = np.ascontiguousarray(force, dtype=np.float64)
        if force.shape != self.shape or not np.isfinite(force).all():
            raise ValueError("invalid physical force")
        if self._native:
            self._native.kick(p, force, float(h), self.threads)
        else:
            p += float(h) * force


class Dynamics:
    def __init__(
        self, potential, masses, beta, nbeads, dt, seed=0, native=True, threads=1,
        loop_backend="auto", block_steps=128
    ):
        if loop_backend not in ("auto", "native", "python"):
            raise ValueError("loop_backend must be auto, native or python")
        if isinstance(block_steps, bool) or not isinstance(block_steps, (int, np.integer)) or not 1 <= block_steps <= 4096:
            raise ValueError("block_steps must be an integer in [1,4096]")
        self.loop_backend = loop_backend
        self.block_steps = int(block_steps)
        self.loop_stats = {"native_steps": 0, "python_steps": 0, "fallback_reasons": {}}
        self.potential = potential
        self.integrator = CayleyIntegrator(
            masses, beta, nbeads, dt, threads=threads, native=native
        )
        self.masses = self.integrator.masses
        self.beta = float(beta)
        self.nbeads = int(nbeads)
        self.dt = float(dt)
        self.rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        self.native = self.integrator.native
        self._cached_normal_modes = None

    def sample_momenta(self):
        return self.rng.normal(size=self.integrator.shape) * np.sqrt(
            self.masses[None, :, None] * self.nbeads / self.beta
        )

    def _force(self, q, bias=None):
        result = validate_result(self.potential.evaluate(q), q.shape)
        force = result.forces.copy()
        if bias is not None:
            _, extra = bias.energy_force(q)
            if np.shape(extra) != q.shape or not np.isfinite(extra).all():
                raise ValueError("bias returned invalid forces")
            force += extra
        return force

    def step(self, state, bias=None):
        return self._propagate(state, bias)

    def propagate_block(self, state, steps, **kwargs):
        """Fused BCB block with native observations or compatible Python fallback."""
        from .propagation import propagate_block
        return propagate_block(self, state, steps, **kwargs)

    def _check_parameters(self):
        if (
            self.dt != self.integrator.dt
            or self.beta != self.integrator.beta
            or self.nbeads != self.integrator.nbeads
        ):
            raise ValueError(
                "Dynamics parameters are immutable; create a new Dynamics to change dt, beta or P"
            )

    def _propagate(self, state, bias=None, fragment_data=()):
        self._check_parameters()
        self.integrator._check(state.q, state.p)
        # Commit the state only after both force calls succeed. Failed PES calls
        # cannot leave a half-stepped state masquerading as a completed step.
        q = state.q.copy()
        p = state.p.copy()
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        self._project_fragments(q, p, fragment_data)
        self.integrator.free_step(q, p)
        self._project_fragments(q, p, fragment_data)
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        self._project_fragments(q, p, fragment_data)
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite integration state")
        state.q[:] = q
        state.p[:] = p
        state.step += 1
        state.time += self.dt
        return state

    def energy(self, state, bias=None):
        q, p = state.q, state.p
        self.integrator._check(q, p)
        e = float(np.sum(validate_result(self.potential.evaluate(q), q.shape).energy))
        e += float(np.sum(p * p / (2 * self.masses[None, :, None])))
        dq = q - np.roll(q, -1, axis=0)
        e += float(
            0.5
            * (self.nbeads / self.beta) ** 2
            * np.sum(self.masses[None, :, None] * dq * dq)
        )
        if bias is not None:
            e += float(bias.energy_force(q)[0])
        return e

    def _fragment_data(self, state, fragments):
        if fragments is None:
            return []
        data = []
        used = set()
        for f in fragments:
            raw = np.asarray(f)
            if raw.dtype.kind not in "iu":
                raise ValueError("fragment indices must be integers")
            ix = raw.astype(int, copy=False)
            if (
                ix.ndim != 1
                or not len(ix)
                or len(set(ix.tolist())) != len(ix)
                or np.any(ix < 0)
                or np.any(ix >= len(self.masses))
                or used.intersection(ix.tolist())
            ):
                raise ValueError("invalid/overlapping fragment indices")
            used.update(ix.tolist())
            m = self.masses[ix]
            total = float(m.sum())
            ref = np.sum(state.q[:, ix, :] * m[None, :, None], axis=(0, 1)) / (
                self.nbeads * total
            )
            data.append((ix, m, total, ref))
        return data

    def _fix_fragments(self, state, data):
        self._project_fragments(state.q, state.p, data)

    def _project_fragments(self, q, p, data):
        for ix, m, total, ref in data:
            com = np.sum(q[:, ix, :] * m[None, :, None], axis=(0, 1)) / (
                self.nbeads * total
            )
            velocity = np.sum(p[:, ix, :], axis=(0, 1)) / (self.nbeads * total)
            q[:, ix, :] -= (com - ref)[None, None, :]
            p[:, ix, :] -= m[None, :, None] * velocity[None, None, :]

    def thermalize(
        self, state, steps, interval=10, fragments=None, bias=None, *, resample=True
    ):
        """Andersen heat-bath propagation; fixed fragment centroid COM only.

        For continuous parent sampling call resample=False after initial burn-in;
        collision times follow state.step, so segmented calls share one clock.
        """
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("steps must be a nonnegative integer")
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, np.integer))
            or interval < 1
        ):
            raise ValueError("interval must be positive integer")
        if self.dt <= 0:
            raise ValueError("thermostatted sampling requires positive dt")
        data = self._fragment_data(state, fragments)
        if resample:
            state.p[:] = self.sample_momenta()
        self._fix_fragments(state, data)
        remaining = int(steps)
        while remaining:
            if state.step % interval == 0:
                state.p[:] = self.sample_momenta()
                self._fix_fragments(state, data)
            count = min(remaining, self.block_steps, interval - state.step % interval)
            self.propagate_block(state, count, bias=bias, fragment_data=data,
                                 observe=np.zeros(count, dtype=np.uint8), record_energy=False)
            remaining -= count
        return state

    def sample_step(self, state, bias=None, *, gamma_centroid=1.0):
        """BCOCB with PILE-L OU friction. gamma_centroid is inverse atomic time."""
        self._check_parameters()
        if not np.isfinite(gamma_centroid) or gamma_centroid <= 0:
            raise ValueError("gamma_centroid must be positive")
        if self.dt <= 0:
            raise ValueError("thermostatted sampling requires positive dt")
        q = state.q.copy()
        p = state.p.copy()
        self.integrator._check(q, p)
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        self.integrator.half_step(q, p)
        # Real orthogonal cyclic normal modes avoid complex-noise covariance
        # errors at the zero and Nyquist modes. Small systems/thermalization.
        P = self.nbeads
        if self._cached_normal_modes is None:
            j = np.arange(P)
            rows = [np.ones(P) / np.sqrt(P)]
            freq = [0.0]
            for k in range(1, (P + 1) // 2):
                rows.extend(
                    [
                        np.sqrt(2 / P) * np.cos(2 * np.pi * k * j / P),
                        np.sqrt(2 / P) * np.sin(2 * np.pi * k * j / P),
                    ]
                )
                freq.extend([2 * P / self.beta * np.sin(np.pi * k / P)] * 2)
            if P % 2 == 0:
                rows.append((-1.0) ** j / np.sqrt(P))
                freq.append(2 * P / self.beta)
            self._cached_normal_modes = (np.asarray(rows), np.asarray(freq))
        C, freq = self._cached_normal_modes
        pk = np.einsum("kb,bac->kac", C, p)
        gamma = 2 * np.asarray(freq)
        gamma[0] = gamma_centroid
        c = np.exp(-gamma * self.dt)
        noise = self.rng.normal(size=p.shape) * np.sqrt(
            self.masses[None, :, None] * P / self.beta
        )
        pk = c[:, None, None] * pk + np.sqrt(1 - c * c)[:, None, None] * noise
        p[:] = np.einsum("kb,kac->bac", C, pk)
        self.integrator.half_step(q, p)
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite integration state")
        state.q[:] = q
        state.p[:] = p
        state.step += 1
        state.time += self.dt
        return state
