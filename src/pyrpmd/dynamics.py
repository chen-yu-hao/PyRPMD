"""Real-mass RPMD: sum-beads Hamiltonian sampled at beta/P, q,p[P,N,3].

Cayley BCB remains the default. Exact normal-mode BAB and full-Hamiltonian
velocity Verlet use the same real masses and physical force convention.
step() is unthermostatted by default; an explicit propagation plan composes
its selected bath and pressure operators. Andersen sampling has bead momentum
variance m*P/beta. PILE-L sampling and internal-mode-only TRPMD are explicit
choices; neither changes the physical masses.
"""

from dataclasses import dataclass
import copy
import importlib
import math
import numbers
import threading
import numpy as np
from .potentials.base import evaluate_checked, validate_result  # noqa: F401 (re-export)
from .validation import checked_integer, finite_real


INTEGRATORS = ("cayley-bcb", "normal-mode-bab", "velocity-verlet")


def _state_clock(step, time):
    """Validate a live clock and normalize NumPy scalars without integer wrap."""
    return (
        checked_integer(step, "state clock", minimum=0, message="invalid state clock"),
        finite_real(time, "state clock", message="invalid state clock"),
    )


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
        if any(np.asarray(value).dtype.kind not in "fiu" for value in (self.q, self.p)):
            raise ValueError("q,p must be real numerical arrays")
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
        self.step, self.time = _state_clock(self.step, self.time)

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
        if np.asarray(masses).dtype.kind not in "fiu":
            raise ValueError("masses must be a real numerical array")
        self.masses = np.array(masses, dtype=float, copy=True)
        if (
            self.masses.ndim != 1
            or not len(self.masses)
            or not np.isfinite(self.masses).all()
            or np.any(self.masses <= 0)
        ):
            raise ValueError("invalid masses")
        self.masses.flags.writeable = False
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
               for value in (beta, dt)):
            raise ValueError("beta/dt must be real numerical scalars")
        try:
            self.beta = float(beta)
            self.dt = float(dt)
        except (OverflowError, ValueError) as exc:
            raise ValueError("invalid beta/dt") from exc
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
        self._cached_fourier_coefficients = ((self.dt, self.scheme), {})
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
        # These maps depend only on the context, not the evolving coordinates.
        # Keep at most the full and half maps. Include dt/scheme in the key so
        # direct integrator use cannot reuse coefficients from another map.
        key = (self.dt, self.scheme)
        if self._cached_fourier_coefficients[0] != key:
            self._cached_fourier_coefficients = (key, {})
        coefficients = self._cached_fourier_coefficients[1]
        if half not in coefficients:
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
            coefficients[half] = tuple(value[:, None, None] for value in (a, b, c))
            for value in coefficients[half]:
                value.flags.writeable = False
        a, b, c = coefficients[half]
        q[:] = np.fft.irfft(
            a * qk + b * pk / self.masses[None, :, None],
            n=self.nbeads,
            axis=0,
            norm="ortho",
        )
        p[:] = np.fft.irfft(
            a * pk - c * self.masses[None, :, None] * qk,
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
        fixed_atoms=(), fixed_positions=None, bond_constraints=(),
        constraint_cell=None, constraint_pbc=None, constraint_tolerance=1e-12,
        constraint_max_iterations=1000, *, thermostat=None, barostat=None,
        force_splitting=None, cell=None, pbc=None, potential_identity=None, sampling_integrator=None,
    ):
        from .engine import PropagationPlan, _Composition
        self.plan = PropagationPlan(integrator, thermostat, barostat, force_splitting, sampling_integrator)
        self.force_splitting = self.plan.force_splitting
        self.inner_steps = 1 if force_splitting is None else force_splitting.inner_steps
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, numbers.Real):
            raise ValueError("beta/dt must be real numerical scalars")
        if force_splitting is not None:
            from .potentials.contraction import SplitPotential
            if not isinstance(potential, SplitPotential):
                raise TypeError("force splitting requires an explicit SplitPotential")
            if potential.nbeads != nbeads:
                raise ValueError("Dynamics and SplitPotential bead counts must agree")
        self.potential = potential
        self.integrator = CayleyIntegrator(
            masses, beta, nbeads, dt / self.inner_steps, threads=threads, native=native, scheme=integrator
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
        self._sampling_kernels = {}
        self._native_requested = native
        if loop_backend not in ("auto", "native", "python"):
            raise ValueError("loop_backend must be auto/native/python")
        if isinstance(block_steps, bool) or not isinstance(block_steps, (int, np.integer)) or not 1 <= block_steps <= 65536:
            raise ValueError("block_steps must be an integer in [1, 65536]")
        self.loop_backend = loop_backend
        self.block_steps = int(block_steps)
        self.propagation_stats = {"native_steps": 0, "python_steps": 0, "fallback_reasons": {}}
        self.force_evaluations = {"fast": 0, "slow": 0}
        if not isinstance(self.plan.barostat_spec, dict):
            piston_cell = self.plan.barostat_spec.cell
            if cell is not None and not np.array_equal(np.asarray(cell), piston_cell):
                raise ValueError("barostat and Dynamics initial cells differ")
            if pbc is not None and tuple(pbc) != (True, True, True):
                raise ValueError("BZP requires a fully periodic 3D cell")
            cell, pbc = piston_cell, (True, True, True)
        if cell is not None or pbc is not None:
            from .geometry import validate_cell_pbc
            cell, flags = validate_cell_pbc(cell, pbc)
            pbc = tuple(bool(flag) for flag in flags)
        self.cell = None if cell is None else np.array(cell, dtype=float, copy=True)
        self._initial_cell = None if self.cell is None else self.cell.copy()
        self.pbc = None if pbc is None else tuple(pbc)
        if potential_identity is not None and (not isinstance(potential_identity, str) or not potential_identity.strip()):
            raise ValueError("potential_identity must be a nonempty string")
        self.potential_identity = potential_identity
        import uuid
        self._runtime_potential_identity = potential_identity or "runtime:" + uuid.uuid4().hex
        from .bond_constraints import BondConstraints
        if constraint_cell is None:
            constraint_cell = self.cell
        if constraint_pbc is None:
            constraint_pbc = self.pbc
        self.constraints = BondConstraints(
            self.masses, bond_constraints, cell=constraint_cell, pbc=constraint_pbc,
            tolerance=constraint_tolerance, max_iterations=constraint_max_iterations,
        )
        self.bond_constraints = self.constraints.bonds
        required_bonds = getattr(potential, "required_bond_constraints", None)
        if required_bonds is None:
            required_bonds = getattr(potential, "metadata", {}).get("required_bond_constraints", ())
        if required_bonds:
            from .bond_constraints import normalize_bond_constraints
            declared = {(i, j): length for i, j, length in self.bond_constraints}
            for i, j, length in normalize_bond_constraints(required_bonds, len(self.masses)):
                if (i, j) not in declared or not np.isclose(declared[i, j], length, rtol=1e-10, atol=1e-12):
                    raise ValueError("potential-required bond constraints must be declared with matching lengths in Dynamics")
        if self.bond_constraints:
            if self.cell is not None and (
                not np.array_equal(self.constraints.geometry.cell, self.cell)
                or not np.array_equal(self.constraints.geometry.pbc, self.pbc)
            ):
                raise ValueError("constraint geometry must match the Dynamics cell and pbc")
            if type(self) is not Dynamics:
                raise ValueError("bond constraints are supported by Dynamics, not specialized dynamics subclasses")
            if native == "required":
                raise ValueError("native='required' is not supported by the Python bond-constraint solver")
            if self.integrator_name != "velocity-verlet":
                raise ValueError("bond constraints require integrator='velocity-verlet'")
            if loop_backend == "native":
                raise ValueError("bond constraints require loop_backend='auto' or 'python'")
            if self.dt == 0:
                raise ValueError("bond constraints require a nonzero timestep")
            if any(i in self.fixed_atoms or j in self.fixed_atoms for i, j, _ in self.bond_constraints):
                raise ValueError("bond constraints cannot include fixed atoms")
        self._composition = _Composition(self, self.plan)
        self._sampling_compositions = {}
        self.last_step_backend = "python"
        self._composition.validate_binding()

    def operator_state_dict(self):
        """JSON-serializable bath/RNG/cell state for a physical checkpoint.

        This accompanies q/p/clock and the caller's PES/physics identity; it
        does not replace those records. Execution backend and block size are
        deliberately absent, so changing them never resets a heat bath.
        """
        from .engine import _json
        return _json({"version": 1, "rng": copy.deepcopy(self.rng.bit_generator.state),
                      "composition": self._composition.snapshot(),
                      "sampling": {key: component.snapshot()
                                   for key, component in self._sampling_compositions.items()}})

    def _operator_rollback_token(self):
        """Copy live operators without the public checkpoint's JSON conversion."""
        from .native_ensemble import NativeEnsemble
        components = [self._composition, *self._sampling_compositions.values()]
        saved = []
        for component in components:
            # Retain the original helpers so failed lazy creation or backend
            # switching is undone. Native physical caches are refreshed on restore.
            attributes = vars(component).copy()
            attributes["baths"] = component.baths.copy()
            helpers = [*component.baths.values()]
            if component.native_engine is not None:
                helpers.append(component.native_engine)
            states = [(helper, helper._rollback_token() if isinstance(helper, NativeEnsemble)
                       else helper.state()) for helper in helpers]
            barostat = component.plan.barostat_spec
            piston = None if isinstance(barostat, dict) else (barostat, barostat.snapshot())
            saved.append((component, attributes, states, piston))
        return (copy.deepcopy(self.rng.bit_generator.state),
                self._composition, self._sampling_compositions.copy(),
                None if self.cell is None else self.cell.copy(), saved)

    def _restore_operator_rollback_token(self, token):
        """Restore a local transaction, including lazily created operators."""
        from .native_ensemble import NativeEnsemble
        rng, composition, sampling, cell, saved = token
        for component, attributes, states, piston in saved:
            for helper, state in states:
                if isinstance(helper, NativeEnsemble):
                    helper._restore_rollback_token(state)
                else:
                    helper.restore(state)
            vars(component).update(attributes)
            if component.native_engine is not None and component._native_state is not None:
                # Physical cache arrays may have been reused by an accepted prefix.
                # Rebuild them from the restored executor; no PES/RNG is consumed.
                component._native_state = component.native_engine._physical_state()
            if piston is not None:
                piston[0].restore(piston[1])
        self._composition = composition
        self._sampling_compositions = sampling.copy()
        self.cell = None if cell is None else cell.copy()
        if cell is not None:
            setter = getattr(self.potential, "set_cell", None)
            if callable(setter):
                setter(cell.copy())
        self.rng.bit_generator.state = copy.deepcopy(rng)

    def load_operator_state_dict(self, data):
        """Restore all operators atomically, including lazily created baths."""
        from .engine import PropagationPlan, _Composition
        import json
        if not isinstance(data, dict) or set(data) != {"version", "rng", "composition", "sampling"} or data["version"] != 1:
            raise ValueError("invalid propagation operator checkpoint")
        if not isinstance(data["sampling"], dict):
            raise ValueError("invalid sampling operator checkpoint")
        # Check the generator schema before touching any coupled operator.
        candidate_rng = copy.deepcopy(self.rng.bit_generator)
        candidate_rng.state = copy.deepcopy(data["rng"])
        previous = self._composition.snapshot()
        previous_sampling = {key: helper.snapshot() for key, helper in self._sampling_compositions.items()}
        try:
            component = _Composition(self, self.plan)
            component.baths = dict(self._composition.baths)
            component.native_engine = self._composition.native_engine
            component.restore(data["composition"])
            sampling = {}
            for key, saved in data["sampling"].items():
                spec = json.loads(key)
                method = spec.get("sampling_integrator") if "thermostat" in spec else None
                if "thermostat" in spec:
                    if set(spec) != {"thermostat", "sampling_integrator"}:
                        raise ValueError("invalid sampling operator key")
                    spec = spec["thermostat"]
                plan = PropagationPlan(self.integrator_name, spec, self.plan.barostat_spec,
                                       self.force_splitting, method)
                helper = _Composition(self, plan)
                existing = self._sampling_compositions.get(key)
                if existing is not None:
                    helper.baths = dict(existing.baths)
                    helper.native_engine = existing.native_engine
                helper.restore(saved)
                sampling[key] = helper
            self.rng.bit_generator.state = copy.deepcopy(data["rng"])
        except BaseException:
            self._composition.restore(previous)
            for key, saved in previous_sampling.items():
                self._sampling_compositions[key].restore(saved)
            raise
        self._composition, self._sampling_compositions = component, sampling

    def _sampling_kernel(self, method):
        """Select/cache the actual BCOCB/BAOAB ring map, without replacing NVE."""
        self._check_parameters()
        expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(self.integrator_name)
        if expected is None:
            raise ValueError("BCOCB/BAOAB require a normal-mode or Cayley integrator")
        if method is None or method == expected:
            return self.integrator
        if method not in ("bcocb", "baoab"):
            raise ValueError("sampling method must be bcocb or baoab")
        # Choosing another kernel must not silently bypass extension hooks.
        if type(self.integrator) is not CayleyIntegrator or any(
            name in self.integrator.__dict__ or getattr(CayleyIntegrator, name) is not original
            for name, original in {**_PILE_INTEGRATOR_METHODS, **_BLOCK_INTEGRATOR_METHODS}.items()
        ):
            raise ValueError("mixed sampling cannot replace custom integrator hooks")
        if method not in self._sampling_kernels:
            self._sampling_kernels[method] = CayleyIntegrator(
                self.masses, self.beta, self.nbeads, self.dt / self.inner_steps,
                native=self._native_requested, threads=self.integrator.threads,
                scheme="cayley-bcb" if method == "bcocb" else "normal-mode-bab")
        return self._sampling_kernels[method]

    def _sample_composition(self, spec, method=None):
        from .engine import PropagationPlan, _Composition, normalize_thermostat
        import json
        normalized = normalize_thermostat(spec)
        key = json.dumps(normalized if method is None else {
            "thermostat": normalized, "sampling_integrator": method}, sort_keys=True)
        if key not in self._sampling_compositions:
            self._sampling_compositions[key] = _Composition(
                self, PropagationPlan(self.integrator_name, normalized,
                                      self.plan.barostat_spec, self.force_splitting, method))
        return self._sampling_compositions[key]

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

    def sample_momenta(self, q=None):
        if self.bond_constraints and q is None:
            raise ValueError("constrained momenta require the current bead positions q")
        if self.bond_constraints:
            self.constraints.check(q)
        rng_state = copy.deepcopy(self.rng.bit_generator.state) if self.bond_constraints else None
        momenta = self.rng.normal(size=self.integrator.shape) * np.sqrt(
            self.masses[None, :, None] * self.nbeads / self.beta
        )
        momenta[:, self._fixed_indices, :] = 0.0
        if self.bond_constraints:
            try:
                self.constraints.project_momenta(q, momenta)
            except BaseException:
                self.rng.bit_generator.state = rng_state
                raise
        return momenta

    def initialize_constraints(self, state):
        """Explicitly project a starting geometry and its momenta, atomically.

        Restarted states are checked, never repaired implicitly. This method
        is for initial conditions; calling it during dynamics changes the map.
        """
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        q, p = state.q.copy(), state.p.copy()
        self.constraints.project_positions(q)
        self.constraints.project_momenta(q, p)
        self.constraints.check(q, p)
        state.q[:], state.p[:] = q, p
        return state

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
        if self.bond_constraints and self.loop_backend != "python":
            reason = "bond constraints require the Python RATTLE propagation loop"
            reasons = self.propagation_stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + int(steps)

    def _force(self, q, bias=None):
        """Return an independent, writable force for Python extension hooks."""
        result = evaluate_checked(self.potential, q)
        force = result.forces.copy()
        if bias is not None:
            _, extra = bias.energy_force(q)
            if np.shape(extra) != q.shape or not np.isfinite(extra).all():
                raise ValueError("bias returned invalid forces")
            force += extra
        if self.fixed_atoms:
            force[:, self._fixed_indices, :] = 0.0
        return force

    def _kick(self, q, p, bias=None):
        """Apply a physical half-kick, borrowing only for built-in consumers."""
        if (
            bias is None and not self.fixed_atoms
            and type(self) is Dynamics and type(self.integrator) is CayleyIntegrator
            and "_force" not in self.__dict__
            and Dynamics._force is _BLOCK_DYNAMICS_METHODS["_force"]
            and "kick" not in self.integrator.__dict__
            and CayleyIntegrator.kick is _BLOCK_INTEGRATOR_METHODS["kick"]
        ):
            # Both built-in kick implementations read force synchronously and
            # never retain or mutate it. No other PES call precedes consumption.
            force = evaluate_checked(self.potential, q).forces
        else:
            force = self._force(q, bias)
        self.integrator.kick(p, force, 0.5 * self.dt)

    def _pile_coefficients(self, kernel, gamma_centroid, friction_scale):
        """Return cached real-mode PILE/TRPMD coefficients.

        The cache key and arithmetic are deliberately kept identical to the
        historical inline calculation in :meth:`sample_step`.  Native
        sampling blocks use this helper so that coefficient construction is a
        shared, auditable part of the sampling contract.
        """
        P = self.nbeads
        coefficient_key = (gamma_centroid, friction_scale)
        if (self._cached_pile_coefficients is None
                or self._cached_pile_coefficients[0] != coefficient_key):
            frequencies = np.empty(P)
            frequencies[0] = 0.
            frequencies[1:1 + 2 * ((P - 1) // 2)] = np.repeat(
                kernel._omega[1:(P + 1) // 2], 2)
            if P % 2 == 0:
                frequencies[-1] = kernel._omega[-1]
            gamma = (2 * friction_scale) * frequencies
            gamma[0] = gamma_centroid
            decay = np.exp(-gamma * self.dt)
            amplitude = np.sqrt(1 - decay * decay)
            scale = np.sqrt(self.masses[None, :, None] * P / self.beta)
            fixed = np.zeros(len(self.masses), dtype=np.uint8)
            fixed[self._fixed_indices] = 1
            self._cached_pile_coefficients = (coefficient_key, decay, amplitude, scale, fixed)
        _, decay, amplitude, scale, fixed = self._cached_pile_coefficients
        return decay, amplitude, scale, fixed

    def step(self, state, bias=None):
        if self.plan.active:
            return self._composition.step(state, bias)
        return self._propagate(state, bias)

    def _check_parameters(self):
        if (
            self.dt / self.inner_steps != self.integrator.dt
            or self.beta != self.integrator.beta
            or self.nbeads != self.integrator.nbeads
            or self.integrator_name != self.integrator.scheme
            or self.inner_steps != (1 if self.force_splitting is None else self.force_splitting.inner_steps)
        ):
            raise ValueError(
                "Dynamics parameters are immutable; create a new Dynamics to change dt, beta, P or integrator"
            )

    def _propagate(self, state, bias=None, fragment_data=()):
        self._check_parameters()
        step, time = _state_clock(state.step, state.time)
        next_time = time + self.dt
        if not math.isfinite(next_time):
            raise FloatingPointError("nonfinite integration clock")
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        if self.bond_constraints:
            return self._constrained_step(state, bias, fragment_data, step, next_time)
        # Commit the state only after both force calls succeed. Failed PES calls
        # cannot leave a half-stepped state masquerading as a completed step.
        q = state.q.copy()
        p = state.p.copy()
        from .engine import deterministic_step
        def project(q, p):
            self._project_fragments(q, p, fragment_data)
            self._project_fixed(q, p)
        deterministic_step(q, p, kick=lambda q, p: self._kick(q, p, bias),
                           free=self.integrator.free_step, project=project)
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite integration state")
        state.q[:] = q
        state.p[:] = p
        state.step = step + 1
        state.time = next_time
        return state

    def _constrained_step(self, state, bias, fragment_data, step, next_time):
        if fragment_data:
            raise ValueError("bond constraints cannot be combined with fragment COM projections")
        self.constraints.check(state.q, state.p)
        q, p = state.q.copy(), state.p.copy()
        # RATTLE for the complete H_P: physical + ring-spring half kicks,
        # a constrained drift, and a final mass-metric tangent projection.
        p += 0.5 * self.dt * (self._force(q, bias) + self.integrator.spring_force(q))
        self._project_fixed(q, p)
        q += self.dt * p / self.masses[None, :, None]
        self.constraints.project_positions(q, previous=state.q, momenta=p, dt=self.dt)
        self._project_fixed(q, p)
        p += 0.5 * self.dt * (self._force(q, bias) + self.integrator.spring_force(q))
        self.constraints.project_momenta(q, p)
        self._project_fixed(q, p)
        self.constraints.check(q, p)
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite constrained integration state")
        state.q[:], state.p[:] = q, p
        state.step, state.time = step + 1, next_time
        return state

    def energy(self, state, bias=None):
        q, p = state.q, state.p
        self.integrator._check(q, p)
        self._check_fixed(q, p)
        if self.bond_constraints:
            self.constraints.check(q, p)
        e = float(np.sum(evaluate_checked(self.potential, q).energy))
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
        self, state, steps, interval=10, fragments=None, bias=None, *, resample=True,
        thermostat="andersen", method=None, gamma_centroid=1.0, fragment_reference=None
    ):
        """Andersen or PILE-L sampling with fixed fragment centroid COM.

        Fragment constraints are linear centroid translations, not rigid bonds.
        PILE projects forces and OU increments with the mass metric. Supplying
        fragment_reference[N,3] keeps the same anchors across blocks/restarts.
        With resample=False, completed steps and the bath stream are continuous.
        Each PILE step is atomic; a failed block retains its completed prefix.
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
        if thermostat not in ("andersen", "pile-l"):
            raise ValueError("thermalize supports Andersen or fragment-constrained PILE-L")
        if self.plan.thermostat_spec["mode"] != "none":
            raise ValueError("thermalize owns its bath; use propagate for a configured thermostat")
        if thermostat == "pile-l":
            self._sampling_kernel(method)
            if self.bond_constraints:
                raise ValueError("fragment PILE does not support nonlinear bond constraints")
            if self.loop_backend == "native":
                raise ValueError("fragment PILE requires loop_backend='auto' or 'python'")
            if (isinstance(gamma_centroid, (bool, np.bool_))
                    or not isinstance(gamma_centroid, numbers.Real)
                    or not np.isfinite(gamma_centroid) or gamma_centroid <= 0):
                raise ValueError("gamma_centroid must be positive and finite")
        elif method is not None:
            raise ValueError("thermalize method applies only to PILE-L")
        self._check_parameters()
        _, time = _state_clock(state.step, state.time)
        try:
            final_time = time + int(steps) * self.dt
        except OverflowError as exc:
            raise FloatingPointError("nonfinite integration clock") from exc
        if not math.isfinite(final_time):
            raise FloatingPointError("nonfinite integration clock")
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)
        data = self._fragment_data(state, fragments)
        if fragment_reference is not None:
            reference = np.asarray(fragment_reference)
            if (reference.shape != (len(self.masses), 3) or reference.dtype.kind not in "fiu"
                    or not np.isfinite(reference).all() or not data):
                raise ValueError("fragment_reference requires finite [N,3] coordinates and fragments")
            data = [(ix, m, total, np.sum(reference[ix] * m[:, None], axis=0) / total)
                    for ix, m, total, _ in data]
        if self.bond_constraints and data:
            raise ValueError("bond constraints cannot be combined with fragment COM projections")
        fused_loop = self.can_fuse(bias=bias)
        if resample:
            state.p[:] = self.sample_momenta(state.q) if self.bond_constraints else self.sample_momenta()
        if thermostat == "pile-l":
            for _ in range(int(steps)):
                self.sample_step(state, bias, method=method, gamma_centroid=gamma_centroid,
                                 fragment_data=data)
                self._record_python_steps(1)
            return state
        self._fix_fragments(state, data)
        if fused_loop:
            remaining = int(steps)
            while remaining:
                if state.step % interval == 0:
                    state.p[:] = self.sample_momenta(state.q) if self.bond_constraints else self.sample_momenta()
                    self._fix_fragments(state, data)
                count = min(remaining, self.block_steps, interval - state.step % interval)
                self.propagate(state, count, bias=bias, fragment_data=data,
                               record_centroids=False, sample_interval=count)
                remaining -= count
        else:
            for _ in range(steps):
                if state.step % interval == 0:
                    state.p[:] = self.sample_momenta(state.q) if self.bond_constraints else self.sample_momenta()
                    self._fix_fragments(state, data)
                if self.plan.active:
                    self._composition.step(state, bias, data)
                else:
                    self._propagate(state, bias, data)
                self._record_python_steps(1)
        return state

    def trpmd_step(self, state, bias=None, *, friction_scale=1.0, method=None):
        """Advance TRPMD with gamma_k=2*friction_scale*omega_k and gamma_0=0.

        Only internal ring modes are thermostatted. Physical bead masses are
        retained. This is an approximate real-time method, distinct from PILE-L
        equilibrium sampling and from mass-scaled (PA-)CMD. A failed step leaves
        the state, clock, and RNG unchanged, just as ``sample_step`` does.
        """
        return self.sample_step(
            state, bias, gamma_centroid=0.0, method=method,
            thermostat="trpmd", friction_scale=friction_scale,
        )

    def sample_step(
        self, state, bias=None, *, gamma_centroid=None, method=None,
        thermostat="pile-l", friction_scale=1.0, thermostat_params=None, fragment_data=(),
    ):
        """BCOCB/BAOAB with explicit PILE-L sampling or TRPMD dynamics.

        BCOCB uses the square root of the full Cayley map, whereas BAOAB uses
        exact half normal-mode rotations. gamma_centroid is inverse atomic time.
        Its default is 1 for PILE-L and 0 for TRPMD; a nonzero centroid friction
        is forbidden for TRPMD. ``friction_scale`` applies only to TRPMD internal
        modes. Existing PILE-L calls retain their coefficients and noise order.
        State and RNG are restored together if a force or observation fails.
        Auto uses a native free/OU/free kernel when available; Python still
        owns the two PES calls and the exact real-mode NumPy noise sequence.
        """
        if self.bond_constraints:
            raise ValueError("bond constraints use velocity-verlet RATTLE with legacy Andersen sampling; sample_step is unsupported")
        from .engine import normalize_thermostat
        import json
        if (isinstance(friction_scale, (bool, np.bool_))
                or not isinstance(friction_scale, numbers.Real)
                or not np.isfinite(friction_scale) or friction_scale < 0):
            raise ValueError("friction_scale must be a finite nonnegative number")
        spec = dict(thermostat) if isinstance(thermostat, dict) else {"mode": thermostat}
        if thermostat_params is not None:
            if not isinstance(thermostat_params, dict) or set(spec).intersection(thermostat_params):
                raise ValueError("thermostat_params must be a nonduplicating parameter mapping")
            spec.update(thermostat_params)
        if gamma_centroid is not None:
            if "gamma_centroid" in spec and spec["gamma_centroid"] != gamma_centroid:
                raise ValueError("conflicting centroid friction parameters")
            spec["gamma_centroid"] = gamma_centroid
        normalized = normalize_thermostat(spec)
        if normalized["mode"] == "trpmd":
            if ("pile_scale" in normalized and friction_scale != 1.0
                    and normalized["pile_scale"] != friction_scale):
                raise ValueError("conflicting TRPMD friction_scale and pile_scale")
            normalized.setdefault("pile_scale", friction_scale)
        elif friction_scale != 1.0:
            raise ValueError("friction_scale is a TRPMD option")
        kernel = self._sampling_kernel(method) if method is not None else self.integrator
        expected_method = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(self.integrator_name)
        # Historical paired sample_step calls used the bare thermostat key.
        # Preserve it on restart. An explicit MTS middle choice is different
        # from the historical outer-OU split, even when its free map matches.
        composition_method = (method if self.force_splitting is not None or method != expected_method else None)
        key = json.dumps(normalized if composition_method is None else {
            "thermostat": normalized, "sampling_integrator": composition_method}, sort_keys=True)
        barostat = self.plan.barostat_spec
        if not isinstance(barostat, dict) or barostat.get("mode", "none") != "none":
            if normalized != self.plan.thermostat_spec or (method is not None and kernel.scheme != self.plan.ring_integrator):
                raise ValueError("use the configured pressure-plan thermostat through step(); do not attach a second piston")
            return self._composition.step(state, bias, fragment_data)
        if (isinstance(thermostat, dict) or thermostat_params is not None
                or thermostat not in ("pile-l", "trpmd") or self.loop_backend == "native"
                or self.force_splitting is not None or key in self._sampling_compositions):
            existed = key in self._sampling_compositions
            try:
                return self._sample_composition(normalized, composition_method).step(state, bias, fragment_data)
            except BaseException:
                if not existed:
                    self._sampling_compositions.pop(key, None)
                raise
        self._check_parameters()
        step, time = _state_clock(state.step, state.time)
        if thermostat not in ("pile-l", "trpmd"):
            raise ValueError("thermostat must be pile-l or trpmd")
        if (isinstance(friction_scale, (bool, np.bool_))
                or not isinstance(friction_scale, numbers.Real)
                or not np.isfinite(friction_scale) or friction_scale < 0):
            raise ValueError("friction_scale must be a finite nonnegative number")
        if thermostat == "pile-l" and friction_scale != 1.0:
            raise ValueError("friction_scale is a TRPMD option; PILE-L retains gamma_k=2*omega_k")
        if gamma_centroid is None:
            gamma_centroid = 0.0 if thermostat == "trpmd" else 1.0
        if (isinstance(gamma_centroid, (bool, np.bool_))
                or not isinstance(gamma_centroid, numbers.Real)
                or not np.isfinite(gamma_centroid)):
            raise ValueError("gamma_centroid must be a finite real number")
        if thermostat == "trpmd":
            if gamma_centroid != 0.0:
                raise ValueError("TRPMD requires gamma_centroid=0")
        elif gamma_centroid <= 0:
            raise ValueError("gamma_centroid must be positive")
        if self.dt <= 0:
            raise ValueError("thermostatted sampling requires positive dt")
        next_time = time + self.dt
        if not math.isfinite(next_time):
            raise FloatingPointError("nonfinite integration clock")
        expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(self.integrator_name)
        if expected is None:
            raise ValueError("velocity-verlet sampling uses Andersen; BCOCB/BAOAB require their matching integrator")
        selected = expected if method is None else str(method).lower()
        if selected not in ("bcocb", "baoab"):
            raise ValueError("sampling method must be bcocb or baoab")
        if thermostat == "trpmd" and (self.nbeads == 1 or friction_scale == 0):
            # No stochastic degrees of freedom: use the identical RPMD map and
            # do not consume random numbers for a thermostat with zero strength.
            if kernel is self.integrator and not fragment_data:
                return self.step(state, bias)
            return self._sample_composition(normalized, selected).step(state, bias, fragment_data)
        if self.loop_backend == "native":
            raise RuntimeError("BCOCB/BAOAB PILE-L/TRPMD retains Python/NumPy sampling orchestration; choose loop_backend='auto' or 'python'")
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        q = state.q.copy()
        p = state.p.copy()
        kernel._check(q, p)
        self._check_fixed(q, p)
        def project():
            self._project_fragments(q, p, fragment_data)
            self._project_fixed(q, p)
        if fragment_data:
            project()
        self._kick(q, p, bias)
        if fragment_data:
            project()
        fused_sampler = (
            not fragment_data and self.loop_backend == "auto"
            and type(self) is Dynamics and type(kernel) is CayleyIntegrator
            and hasattr(kernel._native, "sample_free")
            # The dependency-free native DFT is useful only for small rings.
            # Keep the FFT/NumPy route for larger non-power-of-two sizes.
            and (self.nbeads <= 8 or self.nbeads & (self.nbeads - 1) == 0)
            and all(name not in kernel.__dict__
                    and getattr(CayleyIntegrator, name, None) is original
                    for name, original in _PILE_INTEGRATOR_METHODS.items())
            and all(name not in self.__dict__ and getattr(Dynamics, name, None) is original
                    for name, original in _PILE_DYNAMICS_METHODS.items())
        )
        if fused_sampler:
            decay, amplitude, scale, fixed = self._pile_coefficients(
                kernel, gamma_centroid, friction_scale)
            try:
                noise = self.rng.normal(size=p.shape) * scale
                noise[:, self._fixed_indices, :] = 0.0
                with kernel._lock:
                    kernel._native.sample_free(
                        kernel._context, q, p, noise, decay, amplitude, fixed)
                project()
                self._kick(q, p, bias)
                project()
                if not np.isfinite(q).all() or not np.isfinite(p).all():
                    raise FloatingPointError("nonfinite integration state")
            except BaseException:
                self.rng.bit_generator.state = rng_state
                raise
            state.q[:] = q
            state.p[:] = p
            state.step = step + 1
            state.time = next_time
            return state
        kernel.half_step(q, p)
        project()
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
        gamma = (2 * friction_scale) * np.asarray(freq)
        gamma[0] = gamma_centroid
        c = np.exp(-gamma * self.dt)
        try:
            noise = self.rng.normal(size=p.shape) * np.sqrt(
                self.masses[None, :, None] * P / self.beta
            )
            noise[:, self._fixed_indices, :] = 0.0
            pk = c[:, None, None] * pk + np.sqrt(1 - c * c)[:, None, None] * noise
            p[:] = np.einsum("kb,kac->bac", C, pk)
            project()
            kernel.half_step(q, p)
            project()
            self._kick(q, p, bias)
            project()
            if not np.isfinite(q).all() or not np.isfinite(p).all():
                raise FloatingPointError("nonfinite integration state")
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        state.q[:] = q
        state.p[:] = p
        state.step = step + 1
        state.time = next_time
        return state


# A Python extension/monkeypatch is part of the requested mathematics. Never
# bypass custom half maps or projections just because their base type matches.
_PILE_INTEGRATOR_METHODS = {
    name: getattr(CayleyIntegrator, name) for name in ("_check", "half_step", "_fourier_step")
}
_PILE_DYNAMICS_METHODS = {
    name: getattr(Dynamics, name)
    for name in ("_project_fixed", "_check_fixed", "_pile_coefficients")
}

# Capture these in the defining module, before a caller can patch a class and
# lazily import either propagation API. Native blocks bypass these methods.
_BLOCK_DYNAMICS_METHODS = {
    name: getattr(Dynamics, name)
    for name in ("step", "_propagate", "_force", "_kick", "energy", "_project_fragments",
                 "_fragment_data", "_project_fixed", "_check_fixed")
}
_BLOCK_INTEGRATOR_METHODS = {
    name: getattr(CayleyIntegrator, name)
    for name in ("free_step", "kick", "_fourier_step", "spring_force", "_check")
}
