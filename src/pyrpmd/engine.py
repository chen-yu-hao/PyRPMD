"""Composable propagation operators shared by the public dynamics adapters.

``Dynamics`` owns the physical state contract.  This module owns operator
ordering, force splitting and the state of attached baths; native kernels are
executors of that ordering, not a second choice of physical ensemble.  Legacy
unthermostatted and NumPy PILE calls keep their existing random streams.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import numbers

import numpy as np


def deterministic_step(q, p, *, kick, free, project):
    """Apply B/2--A--B/2 to working arrays (scalar or trajectory batched).

    Projection belongs after every operator, including each kick.  The caller
    owns transaction boundaries and endpoint-force caching.
    """
    kick(q, p)
    project(q, p)
    free(q, p)
    project(q, p)
    kick(q, p)
    project(q, p)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def normalize_thermostat(spec=None):
    """Validate and detach an atomic-unit thermostat specification.

    GLE matrix dimensions/covariances are checked when binding to P, beta and
    the actual bath duration.  No user-supplied matrix is changed in place.
    """
    if spec is None:
        spec = "none"
    if isinstance(spec, str):
        result = {"mode": spec}
    elif isinstance(spec, dict):
        result = copy.deepcopy(spec)
    else:
        raise ValueError("thermostat must be a name or a parameter mapping")
    mode = result.get("mode", "pile-l")
    if not isinstance(mode, str):
        raise ValueError("thermostat mode must be a string")
    mode = mode.lower().replace("_", "-")
    mode = {"nose-hoover-chain": "nhc", "nose-hoover": "nhc"}.get(mode, mode)
    result["mode"] = mode
    if mode == "nhc" and "tdamp" in result:
        if "tau" in result:
            raise ValueError("specify only one of NHC tau and tdamp")
        result["tau"] = result.pop("tdamp")
    fields = {
        "none": set(), "andersen": {"interval"}, "langevin": {"gamma"},
        "pile-l": {"gamma_centroid", "pile_scale"}, "pile-g": {"tau", "pile_scale"},
        "svr": {"tau"}, "gle": {"gle"}, "nm-gle": {"gle"},
        "nm-gle-g": {"gle", "tau"}, "piglet": {"gle"},
        "trpmd": {"pile_scale", "gamma_centroid"}, "ffl": {"gamma"},
        "nhc": {"tau", "chain_length", "substeps"},
    }
    if mode not in fields:
        raise ValueError(f"unsupported thermostat {mode!r}")
    unknown = set(result) - fields[mode] - {"mode"}
    if unknown:
        raise ValueError(f"unknown {mode} thermostat parameters: {', '.join(sorted(unknown))}")
    for name in ("gamma", "gamma_centroid", "pile_scale", "tau"):
        if name in result:
            result[name] = _real(result[name], name, zero=name != "tau")
    for name in ("interval", "chain_length", "substeps"):
        if name in result:
            result[name] = _integer(result[name], name)
    if mode in ("gle", "nm-gle", "nm-gle-g", "piglet"):
        if not isinstance(result.get("gle"), dict):
            raise ValueError("GLE requires an explicit drift-matrix mapping")
    if mode == "trpmd":
        if result.get("gamma_centroid", 0) != 0:
            raise ValueError("TRPMD requires gamma_centroid=0")
        result["gamma_centroid"] = 0.0
    return _json(result)


def normalize_barostat(spec=None):
    """Validate and detach an atomic-unit pressure-operator specification.

    System-dependent requirements (mobile atoms, periodicity and physical
    virials) are checked when binding the plan to a Dynamics instance.
    """
    if spec is None:
        spec = "none"
    if isinstance(spec, str):
        result = {"mode": spec}
    elif isinstance(spec, dict):
        result = copy.deepcopy(spec)
    else:
        raise ValueError("barostat must be a name or a parameter mapping")
    mode = result.get("mode", "none")
    if not isinstance(mode, str):
        raise ValueError("barostat mode must be a string")
    mode = mode.lower().replace("_", "-")
    mode = {"isotropic": "bzp", "flexible": "mttk", "mtk": "mttk",
            "anisotropic": "rgb", "stochastic-rescaling": "scr"}.get(mode, mode)
    if mode in ("sc-isotropic", "suzuki-chin"):
        raise NotImplementedError("Suzuki-Chin NPT requires the high-order action and its force/virial derivatives")
    fields = {
        "none": set(), "bzp": {"pressure", "tau", "gamma"},
        "mttk": {"pressure", "tau", "gamma", "cell_mask"},
        "rgb": {"stress", "reference_cell", "tau", "gamma", "cell_mask"},
        "scr": {"pressure", "tau", "compressibility"},
    }
    if mode not in fields:
        raise ValueError(f"unsupported barostat {mode!r}")
    unknown = set(result) - fields[mode] - {"mode"}
    if unknown:
        raise ValueError(f"unknown {mode} barostat parameters: {', '.join(sorted(map(str, unknown)))}")
    result["mode"] = mode
    if mode not in ("none", "rgb") and "pressure" not in result:
        raise ValueError("NPT requires an explicit external pressure")
    if mode == "rgb" and ("stress" not in result or "reference_cell" not in result):
        raise ValueError("RGB requires explicit stress and reference_cell")
    if "pressure" in result:
        value = result["pressure"]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not math.isfinite(value)):
            raise ValueError("pressure must be a finite real number")
        result["pressure"] = float(value)
    for name in ("tau", "gamma", "compressibility"):
        if name in result:
            result[name] = _real(result[name], f"barostat {name}", zero=name == "gamma")
    if mode == "scr" and "compressibility" not in result:
        raise ValueError("SCR requires an explicit positive compressibility")
    for name in ("stress", "cell_mask"):
        if name not in result:
            continue
        value = np.asarray(result[name])
        kinds = "biuf" if name == "cell_mask" else "iuf"
        if value.shape != (3, 3) or value.dtype.kind not in kinds or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite real 3x3 matrix")
        value = np.array(value, dtype=float, copy=True)
        if name == "stress" and not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
            raise ValueError("external stress must be symmetric")
        if name == "cell_mask":
            if not np.all((value == 0) | (value == 1)):
                raise ValueError("cell_mask must contain only zero/one flags")
            if np.any(value[np.tril_indices(3, -1)]) or not value.any():
                raise ValueError("cell_mask must be upper triangular and nonempty")
        result[name] = value.tolist()
    if "reference_cell" in result:
        from .geometry import validate_cell_pbc
        reference, _ = validate_cell_pbc(result["reference_cell"], [True, True, True])
        if np.linalg.det(reference) <= 0:
            raise ValueError("reference_cell must have positive volume")
        result["reference_cell"] = reference.tolist()
    return result


@dataclass(frozen=True)
class ForceSplitting:
    """Two-level reversible RESPA; dt counts outer steps, springs use dt/M."""

    inner_steps: int = 1

    def __post_init__(self):
        object.__setattr__(self, "inner_steps", _integer(self.inner_steps, "inner_steps"))


@dataclass(frozen=True)
class PropagationPlan:
    """Immutable scientific choices, independent of block size and backend."""

    integrator: str = "cayley-bcb"
    thermostat: object = None
    barostat: object = None
    force_splitting: ForceSplitting | None = None
    sampling_integrator: str | None = None

    def __post_init__(self):
        from .dynamics import normalize_integrator
        object.__setattr__(self, "integrator", normalize_integrator(self.integrator))
        # Store serialized configuration, so mutating the caller's dictionaries
        # cannot silently alter a bound trajectory or its checkpoint identity.
        object.__setattr__(self, "_thermostat_json", json.dumps(normalize_thermostat(self.thermostat), sort_keys=True))
        if self.force_splitting is not None and not isinstance(self.force_splitting, ForceSplitting):
            raise TypeError("force_splitting must be ForceSplitting(inner_steps=...)")
        if self.sampling_integrator is not None:
            if self.sampling_integrator not in ("bcocb", "baoab"):
                raise ValueError("sampling_integrator must be bcocb or baoab")
            if self.integrator == "velocity-verlet":
                raise ValueError("middle normal-mode sampling requires Cayley or normal-mode dynamics")
            if self.thermostat_spec["mode"] not in ("pile-l", "trpmd"):
                raise ValueError("explicit sampling_integrator requires PILE-L or TRPMD")
        barostat = self.barostat
        if barostat is None or isinstance(barostat, (str, dict)):
            normalized_barostat = normalize_barostat(barostat)
            if (self.sampling_integrator is not None and normalized_barostat["mode"] != "none"
                    and (normalized_barostat["mode"] != "bzp" or self.thermostat_spec["mode"] != "pile-l")):
                raise ValueError("explicit middle NPT requires isotropic BZP and PILE-L")
            object.__setattr__(self, "_barostat_json", json.dumps(normalized_barostat, sort_keys=True, allow_nan=False))
        else:
            from .barostat import BZPBarostat
            if not isinstance(barostat, BZPBarostat):
                raise TypeError("barostat must be a specification or a BZPBarostat")
            object.__setattr__(self, "_barostat_json", None)
            object.__setattr__(self, "_external_barostat_json", json.dumps({
                "mode": "external-bzp", "pressure": barostat.pressure_au,
                "tau": barostat.tau_au, "gamma": barostat.friction_au,
                "piston_mass": barostat.mass,
                "reference_cell": barostat._reference_cell.tolist(),
            }, sort_keys=True, allow_nan=False))

    @property
    def thermostat_spec(self):
        return json.loads(self._thermostat_json)

    @property
    def barostat_spec(self):
        return json.loads(self._barostat_json) if self._barostat_json is not None else self.barostat

    @property
    def ring_integrator(self):
        """Actual free-ring kernel, independent of the requested NVE kernel."""
        return {"bcocb": "cayley-bcb", "baoab": "normal-mode-bab"}.get(
            self.sampling_integrator, self.integrator)

    @property
    def active(self):
        return (self.thermostat_spec["mode"] != "none" or self.force_splitting is not None
                or self._barostat_json != '{"mode": "none"}')

    def state_dict(self):
        barostat = self.barostat_spec
        result = {"integrator": self.integrator, "thermostat": self.thermostat_spec,
                "barostat": barostat if isinstance(barostat, dict) else json.loads(self._external_barostat_json),
                "inner_steps": None if self.force_splitting is None else self.force_splitting.inner_steps}
        if self.sampling_integrator is not None:
            result["sampling_integrator"] = self.sampling_integrator
        return result


class _Composition:
    """Bound operators for one Dynamics; all physical commits stay in State."""

    def __init__(self, dynamics, plan):
        self.dynamics, self.plan = dynamics, plan
        method = plan.sampling_integrator
        self.integrator = dynamics._sampling_kernel(method) if method else dynamics.integrator
        self.baths = {}
        self.native_engine = None
        self._native_state = None
        self._native_seed = None
        self._native_provider_key = None
        self._candidate_provider_key = None
        self._last_backend = "python"

    def validate_binding(self):
        """Check physical compatibility before any force calls or RNG draws."""
        d = self.dynamics
        if getattr(d, "bond_constraints", ()) and self.plan.active:
            raise ValueError(
                "bond constraints require velocity-verlet RATTLE and legacy Andersen sampling; "
                "composed thermostat, barostat and force-splitting operators are unsupported"
            )
        barostat = self.plan.barostat_spec
        external = not isinstance(barostat, dict)
        mode = "bzp" if external else barostat["mode"]
        if mode == "none":
            return
        if self.plan.force_splitting is not None:
            raise ValueError("pressure control with force splitting requires a validated split/RPC physical virial")
        if external and (d.potential is not barostat.potential or d.beta != barostat.beta
                         or d.nbeads != barostat.nbeads
                         or not np.array_equal(d.masses, barostat.masses)):
            raise ValueError("BZP and Dynamics must share the potential, masses, beta and bead count")
        if external:
            actual = {"mode": "external-bzp", "pressure": barostat.pressure_au,
                      "tau": barostat.tau_au, "gamma": barostat.friction_au,
                      "piston_mass": barostat.mass,
                      "reference_cell": barostat._reference_cell.tolist()}
            if actual != self.plan.state_dict()["barostat"]:
                raise ValueError("external barostat parameters changed; construct a new Dynamics")
        if d.fixed_atoms:
            raise ValueError("pressure control requires all atoms mobile")
        _real(d.dt, "pressure-control timestep")
        if self.plan.thermostat_spec["mode"] in ("piglet", "trpmd"):
            raise ValueError("PIGLET/TRPMD NPT is not validated; use canonical PIMD")
        from .geometry import validate_cell_pbc
        cell, flags = validate_cell_pbc(barostat.cell if external else d.cell, d.pbc)
        if cell is None or not flags.all():
            raise ValueError("pressure control requires a fully periodic 3D cell")
        if external:
            if d.cell is not None and not np.array_equal(d.cell, cell):
                raise ValueError("BZP and Dynamics must share the current cell")
        elif np.linalg.det(cell) <= 0:
            raise ValueError("native pressure control requires a cell with positive volume")
        if mode in ("mttk", "rgb") and (np.any(cell[np.triu_indices(3, 1)])
                                         or np.any(np.diag(cell) <= 0)):
            raise ValueError("flexible cell requires lower-triangular row lattice vectors with positive diagonal")
        if mode == "scr" and d.nbeads != 1:
            raise ValueError("SCR requires P=1 and positive compressibility")
        from .native_ensemble import NativePeriodicPES
        if not isinstance(d.potential, NativePeriodicPES) and not callable(getattr(d.potential, "set_cell", None)):
            raise ValueError("pressure control requires explicit potential.set_cell(cell) support and physical virials")

    def _identity(self):
        d = self.dynamics
        identity = {"plan": self.plan.state_dict(), "masses": d.masses.tolist(),
                    "beta": d.beta, "nbeads": d.nbeads, "dt": d.dt,
                    "fixed_atoms": list(d.fixed_atoms),
                    "fixed_positions": None if d.fixed_positions is None else d.fixed_positions.tolist(),
                    "cell": None if d._initial_cell is None else d._initial_cell.tolist(),
                    "pbc": None if d.pbc is None else list(d.pbc),
                    "potential_identity": d.potential_identity}
        if getattr(d, "bond_constraints", ()):
            constraints = d.constraints
            identity["bond_constraints"] = {
                "bonds": [list(bond) for bond in d.bond_constraints],
                "cell": None if constraints.geometry.cell is None else constraints.geometry.cell.tolist(),
                "pbc": constraints.geometry.pbc.tolist(),
                "tolerance": constraints.tolerance,
                "max_iterations": constraints.max_iterations,
            }
        return identity

    def snapshot(self):
        barostat = self.plan.barostat_spec
        return {"identity": self._identity(),
                "baths": {key: helper.state() for key, helper in self.baths.items()},
                "native": None if self.native_engine is None else self.native_engine.state(),
                "native_seed": self._native_seed,
                "cell": None if self.dynamics.cell is None else self.dynamics.cell.copy(),
                "piston": None if isinstance(barostat, dict) else barostat.snapshot()}

    def restore(self, data):
        if not isinstance(data, dict) or data.get("identity") != self._identity():
            raise ValueError("propagation operator checkpoint parameters differ")
        if set(data) != {"identity", "baths", "native", "native_seed", "piston", "cell"}:
            raise ValueError("invalid propagation operator checkpoint")
        if not isinstance(data["baths"], dict):
            raise ValueError("invalid thermostat checkpoint")
        # The public cell and its owning piston/executor are one state. Check
        # their agreement before restoring any helper, provider or piston.
        from .geometry import validate_cell_pbc
        d = self.dynamics
        saved_cell, _ = validate_cell_pbc(data["cell"], d.pbc)
        if (saved_cell is None) != (d._initial_cell is None):
            raise ValueError("checkpoint cell presence differs from the propagation plan")
        barostat = self.plan.barostat_spec
        pressure = not isinstance(barostat, dict) or barostat["mode"] != "none"
        if not isinstance(barostat, dict):
            from .barostat import validate_bzp_snapshot
            saved_piston = validate_bzp_snapshot(data["piston"], barostat._reference_cell)
            if not np.array_equal(saved_cell, saved_piston["cell"]):
                raise ValueError("checkpoint cell differs from the external piston cell")
        elif data["piston"] is not None:
            raise ValueError("unexpected external barostat checkpoint")
        if data["native"] is not None:
            if not isinstance(data["native"], dict):
                raise ValueError("invalid native executor checkpoint")
            native_cell, _ = validate_cell_pbc(data["native"].get("cell"))
            if saved_cell is not None and not np.array_equal(saved_cell, native_cell):
                raise ValueError("checkpoint cell differs from the native executor cell")
        elif isinstance(barostat, dict) and pressure:
            # Before the first step the plan has no persistent executor yet.
            # An evolving cell without that executor is incomplete state.
            if not np.array_equal(saved_cell, d._initial_cell):
                raise ValueError("changed checkpoint cell requires a pressure executor")
        if not pressure and not np.array_equal(saved_cell, d._initial_cell):
            raise ValueError("fixed-cell checkpoint differs from the initial cell")
        # Construct/validate every helper before replacing existing operators.
        restored = {}
        for key, state in data["baths"].items():
            duration = float.fromhex(key)
            if not isinstance(state, dict):
                raise ValueError("invalid thermostat helper checkpoint")
            helper = self.baths.get(key)
            if helper is None or helper.state().get("format") != state.get("format"):
                helper = self._new_bath(np.asarray(state["q"]), duration, seed=0,
                                        saved_format=state.get("format"))
            helper.restore(state)
            restored[key] = helper
        native = None
        if data["native"] is not None:
            from .dynamics import State
            saved = data["native"]
            physical = State(saved["q"], saved["p"], saved["step"])
            native = self.native_engine
            if native is None:
                native_binding = self._has_native_binding()
                native = self._new_native(physical, seed=data["native_seed"], allow_python=not native_binding)
            native.restore(saved)
        if not isinstance(barostat, dict):
            barostat.restore(data["piston"])
        self.baths = restored
        self.native_engine = native
        self._native_state = None if native is None else native.state()
        self._native_seed = data["native_seed"]
        self._native_provider_key = None
        if native is not None:
            self._has_native_binding()
            self._native_provider_key = self._candidate_provider_key
        self.dynamics.cell = saved_cell
        if native is None and self.dynamics.cell is not None:
            setter = getattr(self.dynamics.potential, "set_cell", None)
            if callable(setter):
                setter(self.dynamics.cell.copy())

    def _new_bath(self, q, duration, *, seed, saved_format=None):
        from importlib import import_module
        from .native_ensemble import NativeEnsemble, _FORMAT
        from .thermostats import PythonNHCBath
        d = self.dynamics
        spec = self.plan.thermostat_spec
        if saved_format == PythonNHCBath.checkpoint_format:
            return PythonNHCBath(d, q, duration, spec)
        if saved_format not in (None, _FORMAT):
            raise ValueError("unsupported thermostat helper checkpoint format")
        if saved_format is None and spec["mode"] == "nhc":
            try:
                import_module("pyrpmd._thermal")
            except ImportError:
                return PythonNHCBath(d, q, duration, spec)
        return NativeEnsemble(q, d.masses, d.beta, duration, p=np.zeros_like(q),
                              thermostat=spec, potential=None,
                              integrator="normal-mode-bab", fixed_atoms=d.fixed_atoms, seed=seed)

    def _bath(self, q, p, duration, step):
        if self.native_engine is not None:
            # Changing execution backend must not start another random stream
            # or reset colored-noise/chain auxiliaries. The full executor can
            # expose its own bath as an elementary operator.
            self.native_engine.apply_thermostat(q, p, duration, step=step)
            self._native_state = None
            return
        key = float(duration).hex()
        if key not in self.baths:
            seed = self._seed()
            self.baths[key] = self._new_bath(q, duration, seed=seed)
        self.baths[key].apply_thermostat(q, p, duration, step=step)

    def _seed(self):
        mode = self.plan.thermostat_spec["mode"]
        if mode in ("none", "nhc") or (mode == "trpmd" and (
                self.dynamics.nbeads == 1 or self.plan.thermostat_spec.get("pile_scale", 1) == 0)):
            return 0
        return int(self.dynamics.rng.integers(0, 2**63))

    def _new_native(self, state, *, seed, allow_python=False):
        from .native_ensemble import NativeEnsemble
        d = self.dynamics
        return NativeEnsemble(state.q, d.masses, d.beta, d.dt, p=state.p,
                              potential=d.potential,
                              potential_identity=d._runtime_potential_identity,
                              thermostat=self.plan.thermostat_spec, barostat=self.plan.barostat_spec,
                              integrator=self.plan.integrator, fixed_atoms=d.fixed_atoms,
                              sampling_integrator=self.plan.sampling_integrator,
                              cell=d._initial_cell, pbc=d.pbc, seed=seed, **({"allow_python": True} if allow_python else {}))

    def _can_native(self, bias, fragments):
        d = self.dynamics
        if (getattr(d, "bond_constraints", ()) or bias is not None or fragments
                or self.plan.force_splitting is not None
                or d.integrator_name == "velocity-verlet"
                or not isinstance(self.plan.barostat_spec, dict)):
            return False
        from .dynamics import (Dynamics, CayleyIntegrator, _BLOCK_DYNAMICS_METHODS,
                               _BLOCK_INTEGRATOR_METHODS, _PILE_INTEGRATOR_METHODS)
        from .propagation import _original_methods
        thermostat = self.plan.thermostat_spec
        mode = thermostat["mode"]
        # Middle-splitting baths invoke two free half maps. A caller's
        # override is part of that map, even when every full-step hook is
        # unchanged. NHC and bath-free TRPMD use the full free map instead.
        half_maps = mode not in ("none", "nhc") and not (
            mode == "trpmd" and (d.nbeads == 1 or thermostat.get("pile_scale", 1) == 0)
        )
        return (type(d) is Dynamics and type(d.integrator) is CayleyIntegrator
                and _original_methods(d, _BLOCK_DYNAMICS_METHODS)
                and _original_methods(d.integrator, _BLOCK_INTEGRATOR_METHODS)
                and (not half_maps or _original_methods(d.integrator, _PILE_INTEGRATOR_METHODS)))

    def _has_native_binding(self):
        from .native_ensemble import NativePeriodicPES
        from .propagation import NativeUnavailableError
        from .potentials.native import descriptor_binding
        potential = self.dynamics.potential
        self._candidate_provider_key = None
        if isinstance(potential, NativePeriodicPES):
            self._candidate_provider_key = (id(potential), potential.address, potential.context, id(potential.owner))
            return True
        factory = getattr(potential, "native_binding", None)
        if not callable(factory):
            return False
        try:
            binding = descriptor_binding(factory())
            with binding.acquire() as descriptor:
                self._candidate_provider_key = (id(potential), *descriptor, id(binding.owner))
        except NativeUnavailableError:
            return False
        return True

    def _prepare_native(self, state, bias=None, fragments=(), backend=None):
        self.validate_binding()
        d = self.dynamics
        backend = d.loop_backend if backend is None else backend
        b = self.plan.barostat_spec
        pressure = isinstance(b, dict) and b.get("mode", "none") != "none"
        if self.native_engine is not None and self.native_engine._potential is not d.potential:
            raise ValueError("a bound native provider is immutable; create a new Dynamics to change the PES")
        if backend == "python" and not pressure:
            return None
        if not d.native and not pressure:
            if backend == "native":
                from .propagation import NativeUnavailableError
                raise NativeUnavailableError("native loop requested with native=False")
            return None
        eligible = self._can_native(bias, fragments)
        has_binding = self._has_native_binding()
        key = self._candidate_provider_key
        if self.native_engine is not None and (
                self.native_engine._potential is not d.potential or self._native_provider_key != key):
            raise ValueError("a bound native provider is immutable; create a new Dynamics to change the PES")
        if not eligible or (not has_binding and not pressure):
            if backend == "native" or pressure:
                from .propagation import NativeUnavailableError
                raise NativeUnavailableError("requested operator combination requires Python force/operator scheduling")
            return None
        if backend == "native" and not has_binding:
            from .propagation import NativeUnavailableError
            raise NativeUnavailableError("native loop requires a native PES binding")
        from .native_ensemble import NativeEnsemble
        if any(not isinstance(helper, NativeEnsemble) for helper in self.baths.values()):
            if backend == "native":
                from .propagation import NativeUnavailableError
                raise NativeUnavailableError("saved Python bath requires Python scheduling; use auto/python")
            return None
        if self.native_engine is None:
            # A transfer reuses the existing stream; it must not consume an
            # extra NumPy seed merely because execution tuning changed.
            seed = 0 if self.baths else self._seed()
            self.native_engine = self._new_native(state, seed=seed, allow_python=not has_binding)
            if self.baths:
                if len(self.baths) != 1:
                    raise ValueError("cannot combine distinct thermostat-duration streams")
                self.native_engine.copy_thermostat_from(next(iter(self.baths.values())), step=state.step)
                self.baths.clear()
            self._native_seed = seed
            self._native_provider_key = key
            self._native_state = self.native_engine.state()
        cached = self._native_state
        if (cached is None or cached["step"] != state.step
                or not np.array_equal(cached["q"], state.q) or not np.array_equal(cached["p"], state.p)):
            self.native_engine.set_state(state.q, state.p, step=state.step)
        return self.native_engine

    def _split_step(self, q, p, bias, project, cache, *, middle=False, step=0):
        d = self.dynamics
        from .potentials.base import validate_result
        def force(kind):
            d.force_evaluations[kind] += 1
            fn = d.potential.evaluate_fast if kind == "fast" else d.potential.evaluate_slow
            result = validate_result(fn(q), q.shape).forces.copy()
            result[:, d._fixed_indices, :] = 0.0
            return result
        for kind in ("fast", "slow"):
            if kind not in cache:
                cache[kind] = force(kind)
        def slow_kick():
            f = cache["slow"]
            if bias is not None:
                _, extra = bias.energy_force(q)
                extra = np.asarray(extra)
                if extra.shape != q.shape or not np.isfinite(extra).all():
                    raise ValueError("bias returned invalid forces")
                f = f + extra
            self.integrator.kick(p, f, 0.5 * d.dt)
            project(q, p)
        slow_kick()
        h = self.integrator.dt
        for _ in range(self.plan.force_splitting.inner_steps):
            endpoint = False
            def kick(_q, _p):
                nonlocal endpoint
                if endpoint:
                    cache["fast"] = force("fast")
                self.integrator.kick(_p, cache["fast"], 0.5 * h)
                endpoint = True
            if middle:
                kick(q, p)
                project(q, p)
                self.integrator.half_step(q, p)
                project(q, p)
                self._bath(q, p, h, step)
                project(q, p)
                self.integrator.half_step(q, p)
                project(q, p)
                kick(q, p)
                project(q, p)
            else:
                deterministic_step(q, p, kick=kick, free=self.integrator.free_step, project=project)
        cache["slow"] = force("slow")
        slow_kick()

    def _python_map(self, state, bias, fragments, cache):
        d = self.dynamics
        if getattr(d, "bond_constraints", ()):
            # The inactive plan can still be reached through the common
            # executor; retain the complete RATTLE map on that path too.
            return d._propagate(state, bias, fragments)
        mode = self.plan.thermostat_spec["mode"]
        def project(q, p):
            d._project_fragments(q, p, fragments)
            d._project_fixed(q, p)
        def deterministic():
            if self.plan.force_splitting is None:
                deterministic_step(state.q, state.p, kick=lambda q, p: d._kick(q, p, bias),
                                   free=self.integrator.free_step, project=project)
            else:
                self._split_step(state.q, state.p, bias, project, cache)
        if mode == "none" or (mode == "trpmd" and (
                d.nbeads == 1 or self.plan.thermostat_spec.get("pile_scale", 1) == 0)):
            deterministic()
        elif self.plan.force_splitting is not None and self.plan.sampling_integrator is not None:
            # One OU of duration h per inner step, not dt (which over-damps M-fold).
            self._split_step(state.q, state.p, bias, project, cache, middle=True, step=state.step)
        elif mode == "andersen" and (self.plan.force_splitting is not None or d.integrator_name == "velocity-verlet"):
            if state.step % self.plan.thermostat_spec.get("interval", 10) == 0:
                self._bath(state.q, state.p, d.dt, state.step)
                project(state.q, state.p)
            deterministic()
        elif mode == "nhc" or self.plan.force_splitting is not None or d.integrator_name == "velocity-verlet":
            self._bath(state.q, state.p, 0.5 * d.dt, state.step)
            project(state.q, state.p)
            deterministic()
            self._bath(state.q, state.p, 0.5 * d.dt, state.step)
            project(state.q, state.p)
        else:
            d._kick(state.q, state.p, bias)
            project(state.q, state.p)
            self.integrator.half_step(state.q, state.p)
            project(state.q, state.p)
            self._bath(state.q, state.p, d.dt, state.step)
            project(state.q, state.p)
            self.integrator.half_step(state.q, state.p)
            project(state.q, state.p)
            d._kick(state.q, state.p, bias)
            project(state.q, state.p)

    def step(self, state, bias=None, fragments=(), *, cache=None, backend=None):
        from .dynamics import _state_clock
        d = self.dynamics
        d._check_parameters()
        self.validate_binding()
        step, time = _state_clock(state.step, state.time)
        next_time = time + d.dt
        if not math.isfinite(next_time):
            raise FloatingPointError("nonfinite integration clock")
        if self.plan.thermostat_spec["mode"] != "none" and d.dt <= 0:
            raise ValueError("thermostatted sampling requires positive dt")
        if fragments and self.plan.thermostat_spec["mode"] in ("nhc", "svr", "pile-g", "nm-gle-g"):
            raise ValueError("global kinetic baths with fragment constraints require a projected degree-of-freedom contract")
        d.integrator._check(state.q, state.p)
        d._check_fixed(state.q, state.p)
        # Validate before creating random streams, then transact all operators.
        before, rng = self.snapshot(), copy.deepcopy(d.rng.bit_generator.state)
        previous_stats = copy.deepcopy(d.propagation_stats)
        trial = state.copy()
        try:
            engine = self._prepare_native(trial, bias, fragments, backend)
            if engine is not None:
                # A direct step observes no thermal estimators. Match the
                # scalar contract by validating only the propagated state and
                # forces; observation APIs own their requested energy checks.
                result = engine.run(1, sample_start=step + 1)
                updated = engine.state()
                trial.q[:], trial.p[:] = updated["q"], updated["p"]
                if d._initial_cell is not None:
                    d.cell = updated["cell"].copy()
                self._native_state = updated
                self._last_backend = "native" if result.get("python_step_callbacks", 0) == 0 else "python"
            else:
                barostat = self.plan.barostat_spec
                if not isinstance(barostat, dict):
                    if bias is not None or fragments:
                        raise ValueError("barostat with bias/fragment constraints requires an explicit virial")
                    # The existing piston exposes its elementary reversible
                    # operators, so all particle baths use the same wrapper.
                    barostat._check_state(trial)
                    d.cell = barostat.cell
                    barostat._thermostat(d.rng, d.dt / 2)
                    barostat._piston_kick(trial, d.dt / 2)
                    barostat._dilate(trial, d.dt / 2)
                self._python_map(trial, bias, fragments, {} if cache is None else cache)
                if not isinstance(barostat, dict):
                    barostat._dilate(trial, d.dt / 2)
                    barostat._piston_kick(trial, d.dt / 2)
                    barostat._thermostat(d.rng, d.dt / 2)
                    barostat._check_state(trial)
                    d.cell = barostat.cell
                self._last_backend = "python"
            if not np.isfinite(trial.q).all() or not np.isfinite(trial.p).all():
                raise FloatingPointError("nonfinite integration state")
        except BaseException:
            self.restore(before)
            d.rng.bit_generator.state = rng
            d.propagation_stats = previous_stats
            raise
        state.q[:], state.p[:] = trial.q, trial.p
        state.step, state.time = step + 1, next_time
        d.last_step_backend = self._last_backend
        return state


__all__ = ["ForceSplitting", "PropagationPlan", "normalize_thermostat", "normalize_barostat"]
