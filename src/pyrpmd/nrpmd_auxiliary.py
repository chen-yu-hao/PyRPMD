"""Auxiliary-operator adapter for the legacy MMST NRPMD public API."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from .nonadiabatic import (
    NRPMDState,
    _mint,
    validate_diabatic_result,
)
from .auxiliary import (
    AuxiliaryCompatibility,
    AuxiliaryLayout,
    AuxiliaryPropagationCore,
    AuxiliarySplitPoint,
)
from .dynamics import CayleyIntegrator, Dynamics, State
from .potentials.base import PESResult, Potential, coordinates, validate_result


NRPMD_COMPATIBILITY = AuxiliaryCompatibility(
    thermostats=frozenset({"none"}),
    preparation_thermostats=frozenset({"none", "andersen", "pile-l"}),
    barostats=frozenset({"none"}),
    force_splitting=frozenset({"none", "scalar", "split", "rpc", "rpc-mts"}),
    allow_pbc=True,
    allow_constraints=False,
    allow_bias=False,
    rationale=(
        "NRPMD is validated here for fixed-cell NVE propagation and for a scalar "
        "identity contribution split by RESPA/RPC. Nuclear Andersen/PILE-L is a "
        "separate preparation map with frozen mapping variables. NPT, constraints "
        "and bias coupling require method-specific derivations."
    ),
)


def _json(value):
    from .engine import _json as normalize
    return normalize(value)


def _provider_identity(provider, explicit=None):
    if provider is None:
        return None
    if explicit is not None:
        value = _json(explicit)
        try:
            json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("potential identity must be finite JSON data") from exc
        return {"explicit": value}
    method = getattr(provider, "to_dict", None)
    if callable(method):
        try:
            value = method()
        except TypeError:
            value = None
        if value is not None:
            return _json(value)
    from .potentials.contraction import DifferencePotential, SplitPotential
    if isinstance(provider, SplitPotential):
        return {
            "schema": "pyrpmd.split-potential", "version": 1,
            "nbeads": provider.nbeads,
            "contracted_beads": provider.contracted_beads,
            "fast": _provider_identity(provider.fast),
            "slow": _provider_identity(provider.slow),
        }
    if isinstance(provider, DifferencePotential):
        return {
            "schema": "pyrpmd.difference-potential", "version": 1,
            "target": _provider_identity(provider.target),
            "reference": _provider_identity(provider.reference),
        }
    from .potentials.builtin import FreePotential, HarmonicPotential
    if isinstance(provider, FreePotential):
        return {"schema": "pyrpmd.free-potential", "version": 1}
    if isinstance(provider, HarmonicPotential):
        return {
            "schema": "pyrpmd.harmonic-potential", "version": 1,
            "k": _json(provider.k), "center": _json(provider.center),
        }
    metadata = getattr(provider, "metadata", {}) or {}
    return {
        "class": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "metadata": _json(metadata),
        "identity_completeness": "class-and-metadata-only",
    }


class NRPMDAuxiliaryOperator:
    """One MMST mapping vector per bead under the auxiliary protocol."""

    layout = AuxiliaryLayout.PER_BEAD
    split_points = frozenset({AuxiliarySplitPoint.SYMMETRIC_AROUND_FREE})
    compatibility = NRPMD_COMPATIBILITY
    checkpoint_version = 1

    def __init__(self, dynamics, state):
        self.dynamics, self.state = dynamics, state
        self.nstates = dynamics.nstates

    @property
    def state_arrays(self):
        return {"mapping_q": self.state.mapping_q, "mapping_p": self.state.mapping_p}

    def propagate(self, dt, q):
        duration = float(dt)
        if not math.isfinite(duration):
            raise ValueError("auxiliary propagation duration must be finite")
        result = self.dynamics._evaluate(q)
        if self.dynamics.electronic_backend == "numpy":
            _mint(self.state, result, duration)
        else:
            from .electronic_backends import electronic_mapping_flow
            z, change = electronic_mapping_flow(
                self.state.mapping_q + 1j * self.state.mapping_p,
                result.matrix, result.gradients, duration,
                backend=self.dynamics.electronic_backend,
                device=self.dynamics.electronic_device,
                threads=self.dynamics.integrator.threads,
            )
            self.state.p += change
            self.state.mapping_q[:], self.state.mapping_p[:] = z.real, z.imag

    def nuclear_force(self, q):
        result = self.dynamics._evaluate(q)
        z = self.state.mapping_q + 1j * self.state.mapping_p
        force = -0.5 * np.einsum(
            "bi,bijac,bj->bac", z.conj(), result.gradients, z
        ).real
        force += 0.5 * np.einsum("biiac->bac", result.gradients).real
        return np.ascontiguousarray(force, dtype=np.float64)

    def energy(self, q):
        result = self.dynamics._evaluate(q)
        z = self.state.mapping_q + 1j * self.state.mapping_p
        return float(0.5 * (
            np.einsum("bi,bij,bj->", z.conj(), result.matrix, z)
            - np.trace(result.matrix, axis1=1, axis2=2).sum()
        ).real)

    def state_dict(self):
        return {
            "version": self.checkpoint_version,
            "layout": self.layout.value,
            "nstates": self.nstates,
            "mapping_q": self.state.mapping_q.tolist(),
            "mapping_p": self.state.mapping_p.tolist(),
        }

    def restore_state_dict(self, data):
        if (not isinstance(data, dict)
                or set(data) != {"version", "layout", "nstates", "mapping_q", "mapping_p"}
                or data["version"] != self.checkpoint_version
                or data["layout"] != self.layout.value
                or data["nstates"] != self.nstates):
            raise ValueError("invalid NRPMD auxiliary checkpoint")
        q, p = np.asarray(data["mapping_q"]), np.asarray(data["mapping_p"])
        expected = (self.dynamics.nbeads, self.nstates)
        if (q.shape != expected or p.shape != expected
                or q.dtype.kind not in "fiu" or p.dtype.kind not in "fiu"
                or not np.isfinite(q).all() or not np.isfinite(p).all()):
            raise ValueError("invalid NRPMD auxiliary checkpoint arrays")
        self.state.mapping_q[:], self.state.mapping_p[:] = q, p

    def identity(self):
        return {
            "schema": "pyrpmd.nrpmd-mmst-auxiliary", "version": 1,
            "layout": self.layout.value, "nstates": self.nstates,
            "split_points": sorted(point.value for point in self.split_points),
            "electronic_backend": self.dynamics.electronic_backend,
            "electronic_device": self.dynamics.electronic_device,
            "potential": copy.deepcopy(self.dynamics._electronic_identity),
        }


class _FrozenMappingPotential(Potential):
    def __init__(self, dynamics, mapping_q, mapping_p):
        self.dynamics = dynamics
        self.mapping_q = np.array(mapping_q, copy=True)
        self.mapping_p = np.array(mapping_p, copy=True)
        self.metadata = {"provider": "frozen-nrpmd-mapping", "batch_independent": False}

    def evaluate(self, q):
        q = coordinates(q)
        result = self.dynamics._evaluate(q)
        z = self.mapping_q + 1j * self.mapping_p
        energy = 0.5 * (
            np.einsum("bi,bij,bj->b", z.conj(), result.matrix, z)
            - np.trace(result.matrix, axis1=1, axis2=2)
        ).real
        force = -0.5 * np.einsum(
            "bi,bijac,bj->bac", z.conj(), result.gradients, z
        ).real + 0.5 * np.einsum("biiac->bac", result.gradients).real
        if self.dynamics.nuclear_potential is not None:
            scalar = self.dynamics._nuclear_result(q)
            energy, force = energy + scalar.energy, force + scalar.forces
        return PESResult(np.asarray(energy, dtype=float), np.asarray(force, dtype=float))

    def close(self):
        pass


class _NRPMDAdapterMixin:
    """Implementation mixed into the public class in ``nonadiabatic.py``."""

    checkpoint_version = 2

    def __init__(
        self, potential, masses, beta, nbeads, dt, nstates, threads=1,
        native=True, integrator="cayley-bcb", electronic_backend="numpy",
        electronic_device="cpu", *, cell=None, pbc=None, nuclear_potential=None,
        force_splitting=None, thermostat=None, barostat=None,
        bond_constraints=(), bias=None, seed=0, potential_identity=None,
        nuclear_potential_identity=None,
    ):
        from .engine import ForceSplitting, normalize_barostat, normalize_thermostat
        from .geometry import validate_cell_pbc
        from .potentials.contraction import SplitPotential

        thermostat_spec = normalize_thermostat(thermostat)
        barostat_spec = normalize_barostat(barostat)
        cell_value, flags = validate_cell_pbc(cell, pbc)
        split_kind = "none"
        if nuclear_potential is not None:
            split_kind = "scalar"
            if isinstance(nuclear_potential, SplitPotential):
                if nuclear_potential.nbeads != nbeads:
                    raise ValueError("NRPMD and SplitPotential bead counts must agree")
                split_kind = "rpc" if nuclear_potential.contracted_beads != nbeads else "split"
        if force_splitting is not None:
            if not isinstance(force_splitting, ForceSplitting):
                raise TypeError("force_splitting must be ForceSplitting(inner_steps=...)")
            if not isinstance(nuclear_potential, SplitPotential):
                raise TypeError("NRPMD force splitting requires a scalar nuclear SplitPotential")
            split_kind = "rpc-mts"
        NRPMD_COMPATIBILITY.validate(
            thermostat=thermostat_spec["mode"], barostat=barostat_spec["mode"],
            force_splitting=split_kind, pbc=bool(flags.any()),
            constraints=bool(tuple(bond_constraints)), bias=bias is not None,
            label="NRPMD MMST auxiliary operator",
        )
        if not callable(getattr(potential, "evaluate", None)):
            raise TypeError("NRPMD potential must implement evaluate(q)")
        if nuclear_potential is not None and not callable(getattr(nuclear_potential, "evaluate", None)):
            raise TypeError("nuclear_potential must implement evaluate(q)")
        electronic_identity = _provider_identity(potential, potential_identity)
        scalar_identity = _provider_identity(nuclear_potential, nuclear_potential_identity)
        if (nuclear_potential is not None
                and scalar_identity.get("identity_completeness") == "class-and-metadata-only"):
            raise ValueError("external nuclear_potential requires nuclear_potential_identity")

        super().__init__(
            potential, masses, beta, nbeads, dt, nstates, threads=threads,
            native=native, integrator=integrator,
            electronic_backend=electronic_backend,
            electronic_device=electronic_device,
        )
        self.nuclear_potential = nuclear_potential
        self.force_splitting = force_splitting
        self.inner_steps = 1 if force_splitting is None else force_splitting.inner_steps
        if self.inner_steps != 1:
            self.integrator = CayleyIntegrator(
                masses, beta, nbeads, float(dt) / self.inner_steps,
                threads=threads, native=native, scheme=integrator,
            )
            self.masses = self.integrator.masses
        self.beta, self.dt, self.nbeads = float(beta), float(dt), int(nbeads)
        self.integrator_name, self.native = self.integrator.scheme, self.integrator.native
        self._native_requested = native
        self.cell = None if cell_value is None else np.array(cell_value, copy=True)
        self.pbc = tuple(bool(value) for value in flags)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.potential_identity = potential_identity
        self.nuclear_potential_identity = nuclear_potential_identity
        self._electronic_identity, self._nuclear_identity = electronic_identity, scalar_identity
        self._split_kind = split_kind
        self._core = AuxiliaryPropagationCore(
            dt=self.dt, nbeads=self.nbeads, operator_factory=self._operator,
            free_step=self.integrator.free_step, validate_state=self._check,
        )
        if self.cell is not None:
            for provider in (self.potential, self.nuclear_potential):
                setter = getattr(provider, "set_cell", None)
                if callable(setter):
                    setter(self.cell.copy())

    def _operator(self, state):
        return NRPMDAuxiliaryOperator(self, state)

    def _check(self, state):
        if (self.electronic_backend, self.electronic_device) != self._electronic_parameters:
            raise ValueError("NRPMDDynamics electronic backend parameters are immutable")
        if (self.integrator.dt != self.dt / self.inner_steps
                or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads
                or self.integrator_name != self.integrator.scheme):
            raise ValueError("NRPMDDynamics parameters are immutable")
        if not isinstance(state, NRPMDState):
            raise ValueError("NRPMDDynamics requires NRPMDState")
        self.integrator._check(state.q, state.p)
        if (isinstance(state.step, bool) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")
        for value in (state.mapping_q, state.mapping_p):
            if (value.shape != (self.nbeads, self.nstates) or value.dtype != np.float64
                    or not value.flags.writeable or not value.flags.c_contiguous
                    or not np.isfinite(value).all()):
                raise ValueError("invalid mapping variables")
        arrays = (state.q, state.p, state.mapping_q, state.mapping_p)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
            raise ValueError("state arrays must not share memory")

    def _coordinates(self, q):
        q = coordinates(q)
        if self.cell is None or not any(self.pbc):
            return q
        from .geometry import wrap_positions
        return wrap_positions(q, self.cell, self.pbc)

    def _evaluate(self, q):
        value = self._coordinates(q)
        return validate_diabatic_result(self.potential.evaluate(value), value.shape, self.nstates)

    def _nuclear_result(self, q, component=None):
        if self.nuclear_potential is None:
            raise ValueError("NRPMD has no scalar nuclear potential")
        value = self._coordinates(q)
        evaluator = self.nuclear_potential.evaluate if component is None else getattr(
            self.nuclear_potential, f"evaluate_{component}"
        )
        return validate_result(evaluator(value), value.shape)

    def _kick_nuclear(self, state, duration, component=None):
        state.p += float(duration) * self._nuclear_result(state.q, component).forces

    def _extended_step(self, state):
        self._check(state)
        trial, h = state.copy(), self.dt / self.inner_steps
        operator = self._operator(trial)
        if self.force_splitting is not None:
            self._kick_nuclear(trial, 0.5 * self.dt, "slow")
            for _ in range(self.inner_steps):
                self._kick_nuclear(trial, 0.5 * h, "fast")
                operator.propagate(0.5 * h, trial.q)
                self.integrator.free_step(trial.q, trial.p)
                operator.propagate(0.5 * h, trial.q)
                self._kick_nuclear(trial, 0.5 * h, "fast")
            self._kick_nuclear(trial, 0.5 * self.dt, "slow")
        else:
            self._kick_nuclear(trial, 0.5 * self.dt)
            operator.propagate(0.5 * self.dt, trial.q)
            self.integrator.free_step(trial.q, trial.p)
            operator.propagate(0.5 * self.dt, trial.q)
            self._kick_nuclear(trial, 0.5 * self.dt)
        trial.step, trial.time = int(trial.step) + 1, float(trial.time) + self.dt
        if not math.isfinite(trial.time):
            raise FloatingPointError("nonfinite NRPMD integration clock")
        self._check(trial)
        for name in ("q", "p", "mapping_q", "mapping_p"):
            getattr(state, name)[:] = getattr(trial, name)
        state.step, state.time = trial.step, trial.time
        return state

    def step(self, state):
        return self._core.step(state) if self.nuclear_potential is None else self._extended_step(state)

    def energy(self, state):
        self._check(state)
        electronic = self._operator(state).energy(state.q)
        scalar = 0.0 if self.nuclear_potential is None else float(
            np.sum(self._nuclear_result(state.q).energy)
        )
        kinetic = np.sum(state.p**2 / (2 * self.masses[None, :, None]))
        dq = state.q - np.roll(state.q, -1, axis=0)
        spring = 0.5 * (self.nbeads / self.beta) ** 2 * np.sum(
            self.masses[None, :, None] * dq**2
        )
        return float(electronic + scalar + kinetic + spring)

    def thermalize(self, state, steps, interval=10, *, thermostat="andersen",
                   method=None, gamma_centroid=1.0, resample=True):
        from .engine import normalize_thermostat
        spec = normalize_thermostat({
            "mode": thermostat,
            **({} if thermostat != "pile-l" else {"gamma_centroid": gamma_centroid}),
        })
        NRPMD_COMPATIBILITY.validate(
            preparation_thermostat=spec["mode"], force_splitting=self._split_kind,
            pbc=any(self.pbc), label="NRPMD MMST auxiliary operator",
        )
        self._check(state)
        mapping_q, mapping_p = state.mapping_q.copy(), state.mapping_p.copy()
        sampler = Dynamics(
            _FrozenMappingPotential(self, mapping_q, mapping_p), self.masses,
            self.beta, self.nbeads, self.dt, seed=self.rng,
            native=self._native_requested, threads=self.integrator.threads,
            loop_backend="python", integrator=self.integrator_name,
            cell=self.cell, pbc=self.pbc,
        )
        nuclear = State(state.q, state.p, state.step, state.time)
        sampler.thermalize(
            nuclear, steps, interval=interval, resample=resample,
            thermostat=spec["mode"], method=method, gamma_centroid=gamma_centroid,
        )
        if (not np.array_equal(state.mapping_q, mapping_q)
                or not np.array_equal(state.mapping_p, mapping_p)):
            raise RuntimeError("frozen NRPMD preparation changed mapping variables")
        state.q[:], state.p[:] = nuclear.q, nuclear.p
        state.step, state.time = nuclear.step, nuclear.time
        self._check(state)
        return state

    def identity(self):
        return {
            "schema": "pyrpmd.nrpmd-dynamics", "version": self.checkpoint_version,
            "masses": self.masses.tolist(), "beta": self.beta,
            "nbeads": self.nbeads, "dt": self.dt,
            "integrator": self.integrator_name,
            "cell": None if self.cell is None else self.cell.tolist(),
            "pbc": list(self.pbc),
            "force_splitting": None if self.force_splitting is None else {
                "inner_steps": self.force_splitting.inner_steps
            },
            "nuclear_potential": copy.deepcopy(self._nuclear_identity),
            "potential_identity": self.potential_identity,
            "auxiliary": self._operator_identity(),
        }

    def _operator_identity(self):
        return {
            "schema": "pyrpmd.nrpmd-mmst-auxiliary", "version": 1,
            "layout": AuxiliaryLayout.PER_BEAD.value, "nstates": self.nstates,
            "electronic_backend": self.electronic_backend,
            "electronic_device": self.electronic_device,
            "potential": copy.deepcopy(self._electronic_identity),
        }

    def operator_state_dict(self, state=None):
        if state is not None:
            self._check(state)
        return {
            "version": self.checkpoint_version, "identity": self.identity(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "auxiliary": None if state is None else self._operator(state).state_dict(),
        }

    def load_operator_state_dict(self, data, state=None):
        legacy = False
        if isinstance(data, dict) and "bit_generator" in data and "state" in data:
            rng_data, auxiliary, legacy = data, None, True
        elif (isinstance(data, dict) and data.get("version") == 1
              and set(data) == {"version", "rng"}):
            rng_data, auxiliary, legacy = data["rng"], None, True
        elif (isinstance(data, dict)
              and set(data) == {"version", "identity", "rng", "auxiliary"}
              and data["version"] == self.checkpoint_version
              and data["identity"] == self.identity()):
            rng_data, auxiliary = data["rng"], data["auxiliary"]
        else:
            raise ValueError("invalid NRPMD operator checkpoint")
        if auxiliary is not None and state is None:
            raise ValueError("NRPMD auxiliary checkpoint requires trajectory state")
        candidate = copy.deepcopy(self.rng.bit_generator)
        try:
            candidate.state = copy.deepcopy(rng_data)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("invalid NRPMD random generator checkpoint") from exc
        old_rng = copy.deepcopy(self.rng.bit_generator.state)
        old_mapping = None if state is None else (state.mapping_q.copy(), state.mapping_p.copy())
        try:
            if auxiliary is not None:
                self._operator(state).restore_state_dict(auxiliary)
                self._check(state)
            self.rng.bit_generator.state = copy.deepcopy(rng_data)
        except BaseException:
            self.rng.bit_generator.state = old_rng
            if old_mapping is not None:
                state.mapping_q[:], state.mapping_p[:] = old_mapping
            raise
        return {"legacy": legacy, "version": 1 if legacy else self.checkpoint_version}
