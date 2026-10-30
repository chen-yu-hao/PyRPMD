"""Real-mass RPMD: sum-beads Hamiltonian sampled at beta/P, q,p[P,N,3].

Cayley BCB remains the default. Exact normal-mode BAB and full-Hamiltonian
velocity Verlet use the same real masses and physical force convention.
No thermostat is applied in step(). Andersen sampling has bead momentum
variance m*P/beta; BCOCB/BAOAB with PILE-L are explicit sampling-only APIs.
"""

from dataclasses import dataclass
import copy
import importlib
import math
import threading
import numpy as np
from .potentials.base import validate_result


INTEGRATORS = ("cayley-bcb", "normal-mode-bab", "velocity-verlet")


def normalize_integrator(value):
    """Return a canonical, checkpoint-significant real-time integrator name."""
    if not isinstance(value, str):
        raise ValueError("integrator must be cayley-bcb, normal-mode-bab or velocity-verlet")
    key = value.strip().lower().replace("_", "-")
    aliases = {
        "cayley": "cayley-bcb", "bcb": "cayley-bcb", "cayley-bcb": "cayley-bcb",
        "normal-mode": "normal-mode-bab", "normal-mode-bab": "normal-mode-bab",
        "exact-normal-mode": "normal-mode-bab", "bab": "normal-mode-bab",
        "velocity-verlet": "velocity-verlet", "verlet": "velocity-verlet", "vv": "velocity-verlet",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("integrator must be cayley-bcb, normal-mode-bab or velocity-verlet") from exc


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
        return {"available": False, "openmp": False, "integrators": []}
    schemes = getattr(mod, "integrator_schemes", lambda: ("cayley-bcb",))()
    return {"available": True, "openmp": bool(mod.has_openmp()), "integrators": list(schemes)}


class CayleyIntegrator:
    """Free-ring maps with real masses; legacy class name remains compatible.

    ``free_step`` is C for Cayley, exact A for normal modes, or the spring
    kick/drift/kick for Verlet. Surrounding it by physical half-kicks gives
    the requested complete reversible map in every case.
    """

    def __init__(self, masses, beta, nbeads, dt, threads=1, native=True, scheme="cayley-bcb"):
        self._scheme = normalize_integrator(scheme)
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
        k = np.arange(self.nbeads // 2 + 1)
        self._omega = 2 * self.nbeads / self.beta * np.sin(np.pi * k / self.nbeads)
        if not np.isfinite(self._omega).all() or not np.isfinite(self.dt * self._omega).all():
            raise ValueError("free-ring frequencies or phase overflow")
        if self.scheme == "velocity-verlet" and abs(self.dt) * float(self._omega.max()) >= 2:
            raise ValueError("velocity-Verlet ring stability requires abs(dt)*max(omega_k) < 2; reduce timestep")
        self._native = None
        self._context = None
        self._lock = threading.Lock()
        self.native_fallback_reason = None
        if native is not False:
            try:
                self._native = importlib.import_module("pyrpmd._native")
            except ImportError:
                if native == "required":
                    raise RuntimeError(
                        "Native kernel not installed; build/install PyRPMD or use native=False"
                    )
        if self._native is not None and self.scheme != "cayley-bcb":
            available = getattr(self._native, "integrator_schemes", lambda: ("cayley-bcb",))()
            if self.scheme not in available:
                self.native_fallback_reason = f"installed native extension does not support {self.scheme}; rebuild it"
                if native == "required":
                    raise RuntimeError(self.native_fallback_reason)
                # A stale extension must never turn a requested BAB/VV into BCB.
                self._native = None
        if self.threads > 1 and (self._native is None or not self._native.has_openmp()):
            raise RuntimeError("threads>1 requires the compiled OpenMP kernel")
        if self._native:
            args = (self.masses.tolist(), self.beta, self.nbeads, self.dt, self.threads)
            self._context = self._native.create(*args) if self.scheme == "cayley-bcb" else self._native.create(*args, self.scheme)
        self.native = self._native is not None

    @property
    def scheme(self):
        return self._scheme

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
            elif self.scheme == "velocity-verlet":
                p += 0.5 * self.dt * self.spring_force(q)
                q += self.dt * p / self.masses[None, :, None]
                p += 0.5 * self.dt * self.spring_force(q)
            else:
                self._fourier_step(q, p, half=False)

    def spring_force(self, q):
        """Physical spring force of H_P (no 1/P factor or fictitious masses)."""
        return -(self.nbeads / self.beta) ** 2 * self.masses[None, :, None] * (
            2 * q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0)
        )

    def _fourier_step(self, q, p, half=False):
        qk = np.fft.rfft(q, axis=0, norm="ortho")
        pk = np.fft.rfft(p, axis=0, norm="ortho")
        z = 0.5 * self.dt * self._omega
        if self.scheme == "normal-mode-bab":
            h = 0.5 * self.dt if half else self.dt
            theta = h * self._omega
            a = np.cos(theta)
            # sinc gives the exact centroid drift without a zero-frequency divide.
            b = h * np.sinc(theta / np.pi)
            c = self._omega * np.sin(theta)
        elif half:
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
        """Cayley square root C(dt)**(1/2) or exact normal-mode A(dt/2)."""
        if self.scheme == "velocity-verlet":
            raise ValueError("velocity-verlet has no BCOCB/BAOAB half map; use Andersen sampling")
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
        loop_backend="auto", block_steps=256, integrator="cayley-bcb",
        fixed_atoms=(), fixed_positions=None
    ):
        self.potential = potential
        self.integrator = CayleyIntegrator(
            masses, beta, nbeads, dt, threads=threads, native=native, scheme=integrator
        )
        self.integrator_name = self.integrator.scheme
        self.masses = self.integrator.masses
        # Fixed substrate atoms are external coordinates, not particles with
        # infinite mass. Every bead shares the same reference position and has
        # exactly zero momentum. Remaining atoms keep their physical masses.
        raw_fixed = tuple(fixed_atoms)
        if any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < len(self.masses)
            for i in raw_fixed
        ) or len(set(raw_fixed)) != len(raw_fixed):
            raise ValueError("fixed_atoms must contain distinct in-range integer atom indices")
        self.fixed_atoms = tuple(int(i) for i in raw_fixed)
        self._fixed_indices = np.asarray(self.fixed_atoms, dtype=int)
        self.fixed_positions = None
        if fixed_positions is not None:
            positions = np.asarray(fixed_positions)
            if (
                positions.dtype.kind not in "fiu"
                or positions.shape != (len(self.masses), 3)
                or not np.isfinite(positions).all()
            ):
                raise ValueError("fixed_positions must be finite real [N,3] reference coordinates")
            self.fixed_positions = np.array(positions, dtype=float, copy=True)
            self.fixed_positions.flags.writeable = False
        elif self.fixed_atoms:
            raise ValueError("fixed_atoms require fixed_positions with complete [N,3] reference geometry")
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
        self._cached_pile_coefficients = None
        if loop_backend not in ("auto", "native", "python"):
            raise ValueError("loop_backend must be auto/native/python")
        if isinstance(block_steps, bool) or not isinstance(block_steps, (int, np.integer)) or not 1 <= block_steps <= 65536:
            raise ValueError("block_steps must be an integer in [1, 65536]")
        self.loop_backend = loop_backend
        self.block_steps = int(block_steps)
        self.propagation_stats = {"native_steps": 0, "python_steps": 0, "fallback_reasons": {}}

    def propagate(self, state, steps, **options):
        """Advance a block, using native PES/CVs when available, else Python."""
        from .propagation import propagate
        return propagate(self, state, steps, **options)

    def propagate_block(self, state, steps, **options):
        """Advance a block, retaining its valid prefix if a later step fails."""
        from .block_propagation import propagate_block
        return propagate_block(self, state, steps, **options)

    @property
    def loop_stats(self):
        """Compatibility name for the live propagation counters."""
        return self.propagation_stats


    def can_fuse(self, *, bias=None, observe=None):
        from .propagation import native_supported
        return native_supported(self, bias=bias, observe=observe)

    def sample_momenta(self):
        momenta = self.rng.normal(size=self.integrator.shape) * np.sqrt(
            self.masses[None, :, None] * self.nbeads / self.beta
        )
        momenta[:, self._fixed_indices, :] = 0.0
        return momenta

    def _project_fixed(self, q, p):
        """Pin fixed atoms in working arrays; callers commit them transactionally."""
        if self.fixed_atoms:
            q[:, self._fixed_indices, :] = self.fixed_positions[self._fixed_indices]
            p[:, self._fixed_indices, :] = 0.0

    def _check_fixed(self, q, p):
        """Reject inconsistent initial/restarted states instead of changing them."""
        if self.fixed_atoms and (
            not np.array_equal(
                q[:, self._fixed_indices, :],
                np.broadcast_to(
                    self.fixed_positions[self._fixed_indices],
                    (self.nbeads, len(self.fixed_atoms), 3),
                ),
            )
            or np.any(p[:, self._fixed_indices, :] != 0.0)
        ):
            raise ValueError("fixed atoms must have reference positions on every bead and zero momenta")

    def _record_python_steps(self, steps):
        """Record workflow steps, including the fixed-atom compatibility route."""
        self.propagation_stats["python_steps"] += int(steps)
        if self.fixed_atoms and self.loop_backend != "python":
            reason = "fixed atoms require the Python propagation loop"
            reasons = self.propagation_stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + int(steps)

    def _force(self, q, bias=None):
        result = validate_result(self.potential.evaluate(q), q.shape)
        force = result.forces.copy()
        if bias is not None:
            _, extra = bias.energy_force(q)
            if np.shape(extra) != q.shape or not np.isfinite(extra).all():
                raise ValueError("bias returned invalid forces")
            force += extra
        force[:, self._fixed_indices, :] = 0.0
        return force

    def step(self, state, bias=None):
        return self._propagate(state, bias)

    def _check_parameters(self):
        if (
            self.dt != self.integrator.dt
            or self.beta != self.integrator.beta
            or self.nbeads != self.integrator.nbeads
            or self.integrator_name != self.integrator.scheme
        ):
            raise ValueError(
                "Dynamics parameters are immutable; create a new Dynamics to change dt, beta, P or integrator"
            )

    def _propagate(self, state, bias=None, fragment_data=()):
        self._check_parameters()
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        # Commit the state only after both force calls succeed. Failed PES calls
        # cannot leave a half-stepped state masquerading as a completed step.
        q = state.q.copy()
        p = state.p.copy()
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        self._project_fragments(q, p, fragment_data)
        self._project_fixed(q, p)
        self.integrator.free_step(q, p)
        self._project_fragments(q, p, fragment_data)
        self._project_fixed(q, p)
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        self._project_fragments(q, p, fragment_data)
        self._project_fixed(q, p)
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
        self._check_fixed(q, p)
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
            if set(self.fixed_atoms).intersection(ix.tolist()):
                raise ValueError("fragment COM constraints cannot include fixed atoms")
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
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        data = self._fragment_data(state, fragments)
        fused_loop = self.can_fuse(bias=bias)
        if resample:
            state.p[:] = self.sample_momenta()
        self._fix_fragments(state, data)
        if fused_loop:
            remaining = int(steps)
            while remaining:
                if state.step % interval == 0:
                    state.p[:] = self.sample_momenta()
                    self._fix_fragments(state, data)
                count = min(remaining, self.block_steps, interval - state.step % interval)
                self.propagate(state, count, bias=bias, fragment_data=data,
                               record_centroids=False, sample_interval=count)
                remaining -= count
        else:
            for _ in range(steps):
                if state.step % interval == 0:
                    state.p[:] = self.sample_momenta()
                    self._fix_fragments(state, data)
                self._propagate(state, bias, data)
                self._record_python_steps(1)
        return state

    def sample_step(self, state, bias=None, *, gamma_centroid=1.0, method=None):
        """Explicit BCOCB/BAOAB PILE-L sampling; never real-time RPMD.

        BCOCB uses the square root of the full Cayley map, whereas BAOAB uses
        exact half normal-mode rotations. gamma_centroid is inverse atomic time.
        State and RNG are restored together if a force or observation fails.
        Auto uses a native free/OU/free kernel when available; Python still
        owns the two PES calls and the exact real-mode NumPy noise sequence.
        """
        self._check_parameters()
        expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(self.integrator_name)
        if expected is None:
            raise ValueError("velocity-verlet sampling uses Andersen; BCOCB/BAOAB require their matching integrator")
        selected = expected if method is None else str(method).lower()
        if selected not in ("bcocb", "baoab"):
            raise ValueError("sampling method must be bcocb or baoab")
        if selected != expected:
            raise ValueError(f"{selected} sampling requires {'cayley-bcb' if selected == 'bcocb' else 'normal-mode-bab'}; selected integrator is {self.integrator_name}")
        if self.loop_backend == "native":
            raise RuntimeError("BCOCB/BAOAB PILE-L retains Python/NumPy sampling orchestration; choose loop_backend='auto' or 'python'")
        if not np.isfinite(gamma_centroid) or gamma_centroid <= 0:
            raise ValueError("gamma_centroid must be positive")
        if self.dt <= 0:
            raise ValueError("thermostatted sampling requires positive dt")
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        q = state.q.copy()
        p = state.p.copy()
        self.integrator._check(q, p)
        self._check_fixed(q, p)
        self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
        fused_sampler = (
            self.loop_backend == "auto"
            and type(self) is Dynamics and type(self.integrator) is CayleyIntegrator
            and hasattr(self.integrator._native, "sample_free")
            # The dependency-free native DFT is useful only for small rings.
            # Keep the FFT/NumPy route for larger non-power-of-two sizes.
            and (self.nbeads <= 8 or self.nbeads & (self.nbeads - 1) == 0)
            and all(name not in self.integrator.__dict__
                    and getattr(CayleyIntegrator, name, None) is original
                    for name, original in _PILE_INTEGRATOR_METHODS.items())
            and all(name not in self.__dict__ and getattr(Dynamics, name, None) is original
                    for name, original in _PILE_DYNAMICS_METHODS.items())
        )
        if fused_sampler:
            P = self.nbeads
            # Immutable dynamics parameters are checked above. Only the
            # caller-selected centroid friction can change between steps.
            if (self._cached_pile_coefficients is None
                    or self._cached_pile_coefficients[0] != gamma_centroid):
                frequencies = np.empty(P)
                frequencies[0] = 0.
                frequencies[1:1 + 2 * ((P - 1) // 2)] = np.repeat(
                    self.integrator._omega[1:(P + 1) // 2], 2)
                if P % 2 == 0:
                    frequencies[-1] = self.integrator._omega[-1]
                gamma = 2 * frequencies
                gamma[0] = gamma_centroid
                decay = np.exp(-gamma * self.dt)
                amplitude = np.sqrt(1 - decay * decay)
                scale = np.sqrt(self.masses[None, :, None] * P / self.beta)
                fixed = np.zeros(len(self.masses), dtype=np.uint8)
                fixed[self._fixed_indices] = 1
                self._cached_pile_coefficients = (gamma_centroid, decay, amplitude, scale, fixed)
            _, decay, amplitude, scale, fixed = self._cached_pile_coefficients
            try:
                noise = self.rng.normal(size=p.shape) * scale
                noise[:, self._fixed_indices, :] = 0.0
                with self.integrator._lock:
                    self.integrator._native.sample_free(
                        self.integrator._context, q, p, noise, decay, amplitude, fixed)
                self._project_fixed(q, p)
                self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
                self._project_fixed(q, p)
                if not np.isfinite(q).all() or not np.isfinite(p).all():
                    raise FloatingPointError("nonfinite integration state")
                if not np.isfinite(state.time + self.dt):
                    raise FloatingPointError("nonfinite integration clock")
            except BaseException:
                self.rng.bit_generator.state = rng_state
                raise
            state.q[:] = q
            state.p[:] = p
            state.step += 1
            state.time += self.dt
            return state
        self.integrator.half_step(q, p)
        self._project_fixed(q, p)
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
        try:
            noise = self.rng.normal(size=p.shape) * np.sqrt(
                self.masses[None, :, None] * P / self.beta
            )
            noise[:, self._fixed_indices, :] = 0.0
            pk = c[:, None, None] * pk + np.sqrt(1 - c * c)[:, None, None] * noise
            p[:] = np.einsum("kb,kac->bac", C, pk)
            self._project_fixed(q, p)
            self.integrator.half_step(q, p)
            self._project_fixed(q, p)
            self.integrator.kick(p, self._force(q, bias), 0.5 * self.dt)
            self._project_fixed(q, p)
            if not np.isfinite(q).all() or not np.isfinite(p).all():
                raise FloatingPointError("nonfinite integration state")
            if not np.isfinite(state.time + self.dt):
                raise FloatingPointError("nonfinite integration clock")
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        state.q[:] = q
        state.p[:] = p
        state.step += 1
        state.time += self.dt
        return state


# A Python extension/monkeypatch is part of the requested mathematics. Never
# bypass custom half maps or projections just because their base type matches.
_PILE_INTEGRATOR_METHODS = {
    name: getattr(CayleyIntegrator, name) for name in ("_check", "half_step", "_fourier_step")
}
_PILE_DYNAMICS_METHODS = {
    name: getattr(Dynamics, name) for name in ("_project_fixed", "_check_fixed")
}

# Capture these here, before the propagation module can be imported lazily
# after a user has replaced a class method. Native loops bypass these methods.
_NATIVE_INTEGRATOR_METHODS = {
    name: getattr(CayleyIntegrator, name)
    for name in ("free_step", "kick", "_fourier_step", "spring_force")
}
_NATIVE_DYNAMICS_METHODS = {
    name: getattr(Dynamics, name)
    for name in (
        "step", "_propagate", "_force", "energy", "_project_fragments",
        "_fragment_data", "_project_fixed", "_check_fixed",
    )
}
