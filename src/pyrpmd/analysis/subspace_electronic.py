"""Coupled electronic states along the 1D reaction path of the subspace pipeline.

Research-level.  The pipeline builds its single-surface 1D model ``V_S^(0)(s)``
on the arc length of the assembled reaction path.  This module adds the
electronic states that meet along that path (avoided crossings near a conical
intersection, intersystem or charge-transfer crossings):

1. electronic geometries are placed evenly in arc length along the same
   mass-weighted path spline as the 1D model (``points``, or every model grid
   point with ``points = "grid"``);
2. a source supplies adiabatic energies and adjacent-frame overlaps
   ``O_n[i, j] = <phi_i(s_n)|phi_j(s_{n+1})>`` (or diabatic matrices directly):
   a diabatic potential energy matrix (DPEM, ``source = "diabatic"``, no
   diabatization needed), a Python callable, an
   :class:`~pyrpmd.potentials.excited_state.ExcitedStateBackend`, or a
   precomputed ``.npz`` file; results are cached next to the report;
3. :func:`~pyrpmd.analysis.multistate_subspace.propagation_diabatization`
   (anchored at the reactant end) gives ``V_d(s)``, which is splined onto the
   model grid;
4. every grid point is shifted by ``V_S^(0)(s) - E_k(s)``, ``E_k`` the adiabat
   ``state`` of ``V_d``, so adiabat ``state`` of the coupled model *is* the
   single-surface model of ``C_S``.  The electronic structure supplies only the
   gaps and the state mixing; the transverse (zero-point) term of ``V_S^(0)`` is
   shared by every electronic state;
5. ``eta_NA = k_QM(H_S^multi) / k_QM(H_S^(0))`` multiplies ``C_S``.

The factorization assumes that the electronic coupling acts along the path
and that the transverse modes follow adiabat ``state``.  Derivative couplings
perpendicular to the path, geometric-phase effects of encircling a conical
intersection, and state-dependent transverse frequencies are not represented
by a 1D model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from .multistate_subspace import (
    MultistateSubspaceModel,
    nonadiabatic_subspace_factor,
    propagation_diabatization,
)
from .subspace_model import _arc_length_spline, _positive_vector
from .subspace_quantum import _json_safe

ELECTRONIC_DEFAULTS = {
    "enabled": True,
    "source": None,
    "object": None,
    "arguments": {},
    "file": None,
    "potential": None,
    "nstates": None,
    "state": 0,
    "points": 64,
    "channels": "adiabatic",
    "min_singular_value": 0.9,
    "max_interpolation_error_kt": 0.1,
    "max_ground_state_mismatch_kt": 1.0,
    "mismatch_window_kt": 30.0,
    "leakage": "reject",
    "resolution": "reject",
    "ground_state_mismatch": "allow",
}
ELECTRONIC_FIELDS = frozenset(ELECTRONIC_DEFAULTS)
_SOURCES = ("diabatic", "python", "backend", "file")
_POLICIES = ("leakage", "resolution", "ground_state_mismatch")
CACHE_SCHEMA = "pyrpmd.subspace_electronic"


class ElectronicDataRequired(RuntimeError):
    """A ``file`` source has no data for the requested electronic geometries yet."""

    def __init__(self, request: Path, reason: str):
        self.request = Path(request)
        super().__init__(f"{reason}; electronic geometries were written to {self.request}: compute "
                         "'energies' [N, S] and 'overlaps' [N-1, S, S] (or 'diabatic' [N, S, S]) for them "
                         "and save them with the same 'geometries' in electronic.file")


def electronic_options(raw: Mapping[str, Any] | None) -> dict | None:
    """Complete and check an ``[electronic]`` table; ``None`` or ``enabled = false`` disables it."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("electronic must be a table")
    unknown = set(raw) - ELECTRONIC_FIELDS
    if unknown:
        raise ValueError(f"unknown electronic fields {sorted(unknown)}; allowed: {sorted(ELECTRONIC_FIELDS)}")
    options = dict(ELECTRONIC_DEFAULTS)
    options.update({k: v for k, v in raw.items() if v is not None or k in ("object", "file", "potential")})
    if options["source"] == "diabatic" and "points" not in raw:
        # A DPEM is evaluated directly: use every model grid point, no spline.
        options["points"] = "grid"
    if not isinstance(options["enabled"], bool):
        raise ValueError("electronic.enabled must be boolean")
    if not options["enabled"]:
        return None
    if options["source"] not in _SOURCES:
        raise ValueError(f"electronic.source must be one of {_SOURCES}")
    if options["source"] in ("python", "backend") and options["object"] is None:
        raise ValueError(f"electronic.source = '{options['source']}' needs electronic.object")
    if options["source"] == "file" and not options["file"]:
        raise ValueError("electronic.source = 'file' needs electronic.file")
    if options["source"] == "diabatic":
        if (options["potential"] is None) == (options["object"] is None):
            raise ValueError("electronic.source = 'diabatic' needs exactly one of electronic.potential "
                             "(a DPEM table such as {model = 'nah2_leps_5g_1992'}) and electronic.object")
        if options["potential"] is not None and not isinstance(options["potential"], Mapping) and not callable(
                getattr(options["potential"], "evaluate", None)):
            raise ValueError("electronic.potential must be a table or a DPEM object with evaluate(q)")
    if not isinstance(options["arguments"], Mapping):
        raise ValueError("electronic.arguments must be a table")
    options["arguments"] = dict(options["arguments"])
    nstates = options["nstates"]
    if isinstance(nstates, bool) or not isinstance(nstates, (int, np.integer)) or not 2 <= int(nstates) <= 32:
        raise ValueError("electronic.nstates must be an integer from 2 to 32")
    options["nstates"] = int(nstates)
    state = options["state"]
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= int(state) < nstates:
        raise ValueError("electronic.state must index one of the electronic states")
    options["state"] = int(state)
    points = options["points"]
    if not (points == "grid" or (isinstance(points, (int, np.integer)) and not isinstance(points, bool)
                                 and int(points) >= 8)):
        raise ValueError("electronic.points must be 'grid' or an integer >= 8")
    if points != "grid":
        options["points"] = int(points)
    if options["channels"] not in ("adiabatic", "diabatic"):
        raise ValueError("electronic.channels must be 'adiabatic' or 'diabatic'")
    value = float(options["min_singular_value"])
    if not 0 < value <= 1:
        raise ValueError("electronic.min_singular_value must lie in (0, 1]")
    options["min_singular_value"] = value
    for key in ("max_interpolation_error_kt", "max_ground_state_mismatch_kt", "mismatch_window_kt"):
        value = float(options[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"electronic.{key} must be finite and positive")
        options[key] = value
    for key in _POLICIES:
        if options[key] not in ("reject", "allow"):
            raise ValueError(f"electronic.{key} must be 'reject' or 'allow'")
    return options


def _object_name(value) -> str:
    if isinstance(value, str):
        return value
    import inspect

    # A function or a factory class is named itself; an instance (a live
    # backend or a callable object) by its class, never by its address.
    named = isinstance(value, type) or inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value)
    target = value if named else type(value)
    return f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', repr(target))}"


def public_options(options: Mapping[str, Any] | None) -> dict | None:
    """JSON-safe copy of electronic options (live objects replaced by their import names)."""
    if options is None:
        return None
    out = dict(options)
    if out.get("object") is not None:
        out["object"] = _object_name(out["object"])
    out["arguments"] = _json_safe(dict(out.get("arguments") or {}))
    if out.get("file") is not None:
        out["file"] = str(out["file"])
    potential = out.get("potential")
    if potential is not None:
        out["potential"] = _json_safe(dict(potential)) if isinstance(potential, Mapping) else _object_name(potential)
    return out


def path_geometries(path: Any, masses: Any, arcs: Any) -> np.ndarray:
    """Cartesian configurations at mass-weighted arc lengths ``arcs`` of ``path``.

    Uses the same arc-length spline as :func:`~pyrpmd.analysis.subspace_model.project_path_subspace`,
    so the geometries are those at which the 1D model ``V_S^(0)`` was evaluated.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("path must be a finite [points, coordinates] array with at least four points")
    sqrt_m = np.sqrt(_positive_vector(masses, points.shape[1], "masses"))
    spline, fine, arc = _arc_length_spline(points * sqrt_m[None, :])
    arcs = np.asarray(arcs, dtype=float)
    if arcs.ndim != 1 or arcs.min() < -1e-9 * arc[-1] or arcs.max() > arc[-1] * (1 + 1e-9):
        raise ValueError(f"arcs must lie within the path arc length [0, {arc[-1]:.6g}]")
    return spline(np.interp(arcs, arc, fine)) / sqrt_m[None, :]


@dataclass(frozen=True)
class ElectronicLine:
    """Electronic data at ordered geometries along the path."""

    arcs: np.ndarray
    geometries: np.ndarray
    energies: np.ndarray
    overlaps: np.ndarray | None
    diabatic: np.ndarray | None
    identity: Mapping[str, Any]
    info: Mapping[str, Any] = field(default_factory=dict)

    def save(self, target: Path) -> None:
        payload = {"arcs": self.arcs, "geometries": self.geometries, "energies": self.energies,
                   "identity": np.array(json.dumps(_json_safe(self.identity), sort_keys=True)),
                   "schema": np.array(CACHE_SCHEMA)}
        if self.overlaps is not None:
            payload["overlaps"] = self.overlaps
        if self.diabatic is not None:
            payload["diabatic"] = self.diabatic
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez(temporary, **payload)
        temporary.replace(target)


def _check_frame(energies, nstates, label):
    energies = np.asarray(energies, dtype=float).reshape(-1)
    if energies.shape != (nstates,) or not np.isfinite(energies).all():
        raise ValueError(f"{label}: expected {nstates} finite state energies, got shape {energies.shape}")
    return energies


def _python_line(function: Callable, arguments, geometries, nstates):
    energies, overlaps, diabatic, previous = [], [], [], None
    kinds = set()
    for k, geometry in enumerate(geometries):
        value = function(np.array(geometry, copy=True), **arguments)
        if isinstance(value, tuple) and len(value) == 2:
            value = {"energies": value[0], "vectors": value[1]}
        if not isinstance(value, Mapping):
            raise TypeError("an electronic python source must return a mapping with 'energies' and "
                            "'vectors' or 'diabatic', or a tuple (energies, vectors)")
        if "diabatic" in value:
            kinds.add("diabatic")
            matrix = np.asarray(value["diabatic"])
            if matrix.shape != (nstates, nstates) or not np.isfinite(matrix).all():
                raise ValueError(f"frame {k}: diabatic matrix must be finite [{nstates}, {nstates}]")
            diabatic.append(matrix)
            energies.append(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)))
            continue
        kinds.add("vectors")
        energies.append(_check_frame(value["energies"], nstates, f"frame {k}"))
        vectors = np.asarray(value["vectors"])
        if vectors.ndim != 2 or vectors.shape[1] != nstates or not np.isfinite(vectors).all():
            raise ValueError(f"frame {k}: vectors must be finite [basis, {nstates}] state coefficients")
        if previous is not None:
            if previous.shape != vectors.shape:
                raise ValueError(f"frame {k}: the state-vector basis changed size")
            overlaps.append(previous.conj().T @ vectors)
        previous = vectors
    if len(kinds) != 1:
        raise ValueError("an electronic python source must return the same kind of data at every frame")
    if "diabatic" in kinds:
        return np.asarray(energies), None, np.asarray(diabatic)
    return np.asarray(energies), np.asarray(overlaps), None


def _backend_line(backend, geometries, nstates, scratch_root: Path):
    energies, overlaps, previous = [], [], None
    for k, geometry in enumerate(geometries):
        scratch = scratch_root / f"frame_{k:04d}"
        scratch.mkdir(parents=True, exist_ok=True)
        result = backend.calculate(np.asarray(geometry, dtype=float).reshape(-1, 3), scratch, previous)
        energies.append(_check_frame(result.energies, nstates, f"frame {k}"))
        if previous is not None:
            overlap = result.overlap
            if overlap is None:
                overlap = backend.overlap(previous, result.payload, scratch)
            if overlap is None:
                raise ValueError(f"backend {getattr(backend, 'name', backend)!r} returned no overlap for frame {k}; "
                                 "propagation diabatization needs adjacent-frame overlaps")
            overlap = np.asarray(overlap)
            if overlap.shape != (nstates, nstates) or not np.isfinite(overlap).all():
                raise ValueError(f"frame {k}: overlap must be finite [{nstates}, {nstates}]")
            overlaps.append(overlap)
        previous = result.payload
    return np.asarray(energies), np.asarray(overlaps), None


def _resolve_plugin_table(table, base):
    """Resolve relative ``file.py:object`` plugins of a DPEM table against ``base``."""
    import copy

    table = copy.deepcopy(dict(table))
    if base is not None:
        from ..input_paths import _plugin_paths

        _plugin_paths(table, Path(base))
    return table


def _resolve_plugin(target, base):
    """A relative ``file.py:object`` plugin is resolved against the specification directory."""
    if isinstance(target, str) and base is not None and ":" in target:
        module, name = target.rsplit(":", 1)
        if module.endswith((".py", ".so")) and not Path(module).is_absolute():
            return f"{Path(base) / module}:{name}"
    return target


def _diabatic_line(dpem, geometries, nstates, chunk=512):
    """Diabatic matrices of a DPEM at Cartesian ``geometries[K, 3N]`` (bohr)."""
    from ..nonadiabatic import validate_diabatic_result

    geometries = np.asarray(geometries, dtype=float)
    if geometries.shape[1] % 3:
        raise ValueError("a diabatic potential needs Cartesian geometries with three coordinates per atom")
    frames = geometries.reshape(len(geometries), -1, 3)
    matrices = []
    # Only the matrices are needed: a DPEM with matrices() (e.g. ChemPotPy) skips its derivatives.
    only = getattr(dpem, "matrices", None)
    for start in range(0, len(frames), chunk):
        block = frames[start:start + chunk]
        if callable(only):
            values = np.asarray(only(block))
            if values.shape != (len(block), nstates, nstates) or not np.isfinite(values).all():
                raise ValueError(f"the diabatic potential must return finite [{len(block)}, {nstates}, {nstates}] "
                                 "matrices")
            matrices.append(values)
            continue
        result = validate_diabatic_result(dpem.evaluate(block), block.shape, nstates)
        status = getattr(result, "status", None)
        if status is not None and np.any(np.asarray(status) != 0):
            bad = start + int(np.flatnonzero(np.asarray(status) != 0)[0])
            raise ValueError(f"the diabatic potential failed at electronic frame {bad}")
        matrices.append(np.asarray(result.matrix))
    diabatic = np.concatenate(matrices)
    diabatic = 0.5 * (diabatic + np.conj(np.swapaxes(diabatic, -1, -2)))
    if np.iscomplexobj(diabatic) and not np.any(diabatic.imag):
        diabatic = diabatic.real
    return np.linalg.eigvalsh(diabatic), None, diabatic


def _load(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _identity(options) -> dict:
    identity = {"source": options["source"],
                "object": None if options["object"] is None else _object_name(options["object"]),
                "arguments": _json_safe(dict(options["arguments"])), "nstates": options["nstates"]}
    if options.get("potential") is not None:
        identity["potential"] = public_options({"potential": options["potential"]})["potential"]
    return identity


def electronic_line(options: Mapping[str, Any], path, masses, arcs, out_dir: Path, *, base: Path | None = None,
                    reuse: bool = False, system: Any = None) -> ElectronicLine:
    """Electronic energies and overlaps at ``arcs`` along ``path`` (cached in ``out_dir``)."""
    out_dir = Path(out_dir)
    cache = out_dir / "electronic_1d.npz"
    geometries = path_geometries(path, masses, arcs)
    nstates = options["nstates"]
    identity = _identity(options)
    if options["source"] == "file":
        source = Path(options["file"])
        if base is not None and not source.is_absolute():
            source = base / source
        request = out_dir / "electronic_1d_request.npz"
        if not source.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(request, arcs=np.asarray(arcs, dtype=float), geometries=geometries)
            raise ElectronicDataRequired(request, f"electronic.file {source} does not exist")
        data = _load(source)
        stored = np.asarray(data.get("geometries"))
        if stored.shape != geometries.shape or not np.allclose(stored, geometries, rtol=0, atol=1e-8):
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(request, arcs=np.asarray(arcs, dtype=float), geometries=geometries)
            raise ElectronicDataRequired(request, f"electronic.file {source} was computed at other geometries")
        energies = np.asarray(data["energies"], dtype=float)
        if energies.shape != (len(geometries), nstates):
            raise ValueError(f"electronic.file energies must have shape {(len(geometries), nstates)}")
        overlaps = data.get("overlaps")
        diabatic = data.get("diabatic")
        if (overlaps is None) == (diabatic is None):
            raise ValueError("electronic.file must hold exactly one of 'overlaps' and 'diabatic'")
        identity = dict(identity, file=str(source))
        return ElectronicLine(np.asarray(arcs, dtype=float), geometries, energies, overlaps, diabatic, identity,
                              {"cached": False, "file": str(source)})
    if reuse and cache.is_file():
        try:
            data = _load(cache)
            saved_identity = json.loads(str(data["identity"]))
            if (saved_identity == json.loads(json.dumps(_json_safe(identity), sort_keys=True))
                    and data["geometries"].shape == geometries.shape
                    and np.allclose(data["geometries"], geometries, rtol=0, atol=1e-10)):
                return ElectronicLine(np.asarray(arcs, dtype=float), geometries, data["energies"],
                                      data.get("overlaps"), data.get("diabatic"), identity,
                                      {"cached": True, "file": str(cache)})
        except (OSError, ValueError, KeyError):
            pass  # A damaged or foreign cache is recomputed.
    started = perf_counter()
    target = _resolve_plugin(options["object"], base)
    if options["source"] == "diabatic":
        from ..potentials.diabatic import make_diabatic_potential

        spec = options["potential"]
        if isinstance(spec, Mapping):
            spec = _resolve_plugin_table(spec, base)
        elif spec is None:
            spec = target
            if isinstance(spec, str):
                from ..potentials.python import import_object

                spec = import_object(spec)
                if not callable(getattr(spec, "evaluate", None)) or isinstance(spec, type):
                    spec = spec(**options["arguments"])
        dpem = make_diabatic_potential(spec, system)
        model_identity = dict(getattr(dpem, "metadata", {}) or {}).get("model_identity")
        if model_identity is not None:
            identity = dict(identity, model_identity=_json_safe(model_identity))
        try:
            energies, overlaps, diabatic = _diabatic_line(dpem, geometries, nstates)
        finally:
            if dpem is not options["potential"] and dpem is not options["object"] \
                    and callable(getattr(dpem, "close", None)):
                dpem.close()
        owned = None
    elif options["source"] == "python":
        if isinstance(target, str):
            from ..potentials.python import import_object

            target = import_object(target)
        if not callable(target):
            raise TypeError("electronic.object must name a callable for source = 'python'")
        energies, overlaps, diabatic = _python_line(target, options["arguments"], geometries, nstates)
        owned = None
    else:
        if isinstance(target, str):
            from ..potentials.python import import_object

            target = import_object(target)
        # A backend class or factory is called with the arguments; an instance is used as is.
        live = hasattr(target, "calculate") and not isinstance(target, type)
        backend = target if live else target(**options["arguments"])
        if not hasattr(backend, "calculate"):
            raise TypeError("electronic.object must be an ExcitedStateBackend or a factory returning one")
        owned = backend if backend is not options["object"] else None
        try:
            scientific = getattr(backend, "scientific_identity", None)
            if callable(scientific):
                identity = dict(identity, backend=_json_safe(dict(scientific())))
            energies, overlaps, diabatic = _backend_line(backend, geometries, nstates, out_dir / "electronic_scratch")
        finally:
            if owned is not None and callable(getattr(owned, "close", None)):
                owned.close()
    line = ElectronicLine(np.asarray(arcs, dtype=float), geometries, energies, overlaps, diabatic, identity,
                          {"cached": False, "file": str(cache), "seconds": perf_counter() - started})
    line.save(cache)
    return line


def _spline_matrices(arcs, matrices, axis):
    from scipy.interpolate import CubicSpline

    if np.iscomplexobj(matrices):
        return (CubicSpline(arcs, matrices.real, axis=0)(axis)
                + 1j * CubicSpline(arcs, matrices.imag, axis=0)(axis))
    return CubicSpline(arcs, matrices, axis=0)(axis)


def _interpolation_error(arcs, diabatic, beta):
    """Half-grid cubic-spline error of the diabatic elements at the omitted points (in kT)."""
    if len(arcs) < 9:
        return None
    half = _spline_matrices(arcs[::2], diabatic[::2], arcs[1::2])
    return float(beta * np.max(np.abs(half - diabatic[1::2])))


def multistate_path_model(model, line: ElectronicLine, options: Mapping[str, Any], *, beta: float,
                          bare_energies: np.ndarray | None = None):
    """Coupled-state 1D model on the axis of the single-surface path ``model``.

    Returns ``(MultistateSubspaceModel, diagnostics)``.  ``bare_energies`` (the
    rate run's PES at the electronic geometries) feeds the ground-state
    consistency diagnostic.
    """
    state = options["state"]
    arcs = np.asarray(line.arcs, dtype=float)
    info: dict[str, Any] = {"points": int(len(arcs)), "nstates": int(options["nstates"]), "state": int(state),
                            "source": options["source"], "cached": bool(line.info.get("cached", False))}
    if line.diabatic is not None:
        diabatic = np.asarray(line.diabatic)
        diabatic = 0.5 * (diabatic + np.conj(np.swapaxes(diabatic, -1, -2)))
        info["diabatization"] = "supplied"
    else:
        # Only a numerically singular overlap stops here; the leakage gate and its
        # policy decide on min_singular_value.
        result = propagation_diabatization(line.energies, line.overlaps, anchor=0,
                                           min_singular_value=min(options["min_singular_value"], 1e-3))
        diabatic = result.diabatic
        info["diabatization"] = "propagation, anchored at the reactant end"
        info["singular_values"] = {"minimum": float(result.singular_values.min()),
                                   "per_step_minimum": result.singular_values.min(axis=1).tolist()}
        info.update({k: v for k, v in result.diagnostics.items() if k in ("coupling_at_ends", "maximum_coupling")})
    axis = model.axes[0]
    same_grid = len(arcs) == len(axis) and np.allclose(arcs, axis, rtol=0, atol=1e-12)
    gridded = diabatic if same_grid else _spline_matrices(arcs, diabatic, axis)
    adiabats = np.linalg.eigvalsh(gridded)
    shift = np.asarray(model.values, dtype=float) - adiabats[:, state]
    identity = np.eye(options["nstates"])
    values = gridded + shift[:, None, None] * identity

    def function(points, _arcs=arcs, _diabatic=diabatic, _model=model, _state=state, _identity=identity):
        s = np.asarray(points, dtype=float)[..., 0]
        matrices = _spline_matrices(_arcs, _diabatic, s.ravel()).reshape(s.shape + _identity.shape)
        offset = np.asarray(_model.energy(points), dtype=float) - np.linalg.eigvalsh(matrices)[..., _state]
        return matrices + offset[..., None, None] * _identity

    labels = tuple(f"diabat{i}" for i in range(options["nstates"]))
    provenance = {"kind": "electronic_path", "state": int(state), "points": int(len(arcs)),
                  "construction": "V_d(s) + (V_S^(0)(s) - E_state(s)) 1", "parent": dict(model.provenance)}
    multistate = MultistateSubspaceModel(model.axes, values, model.masses, model.dividing_surface, model.reactant,
                                         model.reactant_reference, labels, provenance, function)
    info["interpolation_error_kt"] = 0.0 if same_grid else _interpolation_error(arcs, diabatic, beta)
    gaps = np.diff(np.linalg.eigvalsh(diabatic), axis=-1)
    info["minimum_adiabatic_gap"] = float(gaps.min())
    info["minimum_gap_arc"] = float(arcs[int(np.argmin(gaps.min(axis=-1)))])
    info["shift_range"] = [float(shift.min()), float(shift.max())]
    if bare_energies is not None:
        electronic = np.linalg.eigvalsh(diabatic)[:, state]
        bare = np.asarray(bare_energies, dtype=float)
        finite = np.isfinite(bare)
        window = finite & (bare - np.min(bare[finite]) <= options["mismatch_window_kt"] / beta)
        difference = electronic - bare
        if np.any(window):
            centred = difference[window] - np.median(difference[window])
            info["ground_state_mismatch_kt"] = float(beta * np.max(np.abs(centred)))
            info["ground_state_offset"] = float(np.median(difference[window]))
            info["ground_state_mismatch_points"] = int(np.count_nonzero(window))
    return multistate, info


def _gates(info, options):
    from .subspace_correction import _gate

    gates = []
    singular = info.get("singular_values")
    if singular is not None:
        value = singular["minimum"]
        gates.append(_gate("electronic_leakage", value >= options["min_singular_value"], options["leakage"],
                           f"adjacent-frame overlap singular value {value:.3g} < {options['min_singular_value']}: "
                           "states leak out of the retained subspace (retain more states or add points)",
                           {"minimum_singular_value": value}))
    error = info.get("interpolation_error_kt")
    if error is not None:
        limit = options["max_interpolation_error_kt"]
        gates.append(_gate("electronic_resolution", error <= limit, options["resolution"],
                           f"half-grid spline error of the diabatic matrix is {error:.3g} kT > {limit} kT "
                           "(raise electronic.points)", {"interpolation_error_kt": error, "limit_kt": limit}))
    mismatch = info.get("ground_state_mismatch_kt")
    if mismatch is not None:
        limit = options["max_ground_state_mismatch_kt"]
        gates.append(_gate("ground_state_mismatch", mismatch <= limit, options["ground_state_mismatch"],
                           f"electronic adiabat {options['state']} differs in shape from the rate PES by "
                           f"{mismatch:.3g} kT > {limit} kT along the thermally accessible path",
                           {"mismatch_kt": mismatch, "limit_kt": limit,
                            "offset": info.get("ground_state_offset")}))
    failures = [g["reason"] for g in gates if not g["passed"] and not g["overridden"]]
    return {"valid": not failures, "failures": failures,
            "overrides": [g["gate"] for g in gates if g["overridden"]], "gates": gates}


def nonadiabatic_path_factor(run, path, model, quantum: Mapping[str, Any], options: Mapping[str, Any],
                             out_dir: Path, *, base: Path | None = None, reuse: bool = False,
                             single_surface_rate: float | None = None) -> dict:
    """``eta_NA`` of the 1D path model, with its electronic data, gates and coupled-state rate.

    Never raises for a scientific failure: the record carries ``valid = False``
    and the reason (``status`` is ``"complete"``, ``"invalid"``,
    ``"electronic_data_required"`` or ``"failed"``).
    """
    record: dict[str, Any] = {"dimension": 1, "valid": False, "factor": None, "options": public_options(options)}
    out_dir = Path(out_dir)
    try:
        axis = model.axes[0]
        arcs = axis.copy() if options["points"] == "grid" else np.linspace(axis[0], axis[-1], options["points"])
        line = electronic_line(options, path, run.cartesian_masses, arcs, out_dir, base=base, reuse=reuse,
                               system=getattr(run, "system", None))
        bare = np.array([run.energy(g) for g in line.geometries])
        multistate, info = multistate_path_model(model, line, options, beta=run.beta, bare_energies=bare)
        np.savez(out_dir / "multistate_1d.npz", axis=axis, diabatic=multistate.diabatic,
                 electronic_arcs=line.arcs)
        validity = _gates(info, options)
        allowed = {"cap_width", "cap_strength", "cap_power", "energy_floor", "dense_limit", "memory_limit_mb"}
        rate_options = {k: v for k, v in dict(quantum).items() if k in allowed}
        result = nonadiabatic_subspace_factor(multistate, run.beta, state=options["state"],
                                              channels=options["channels"], **rate_options)
        factor = float(result["factor"])
        coupled, single = result["multistate"], result["born_oppenheimer"]
        record.update({
            "status": "complete" if validity["valid"] else "invalid",
            "valid": bool(validity["valid"] and np.isfinite(factor) and factor > 0),
            "factor": factor if validity["valid"] else None,
            "diagnostic_factor": factor,
            "multistate_rate": coupled.rate,
            "born_oppenheimer_rate": single.rate,
            "single_surface_quantum_rate": single_surface_rate,
            "branching": coupled.branching,
            "channel_basis": coupled.channel_basis,
            "electronic": info,
            "validity": validity,
            "multistate_diagnostics": {k: coupled.diagnostics.get(k) for k in (
                "channel_sum_residual", "flux_side_t0", "energy_shift", "electronic", "partition", "solver_plan",
                "minimum_adiabatic_gap_in_product_absorber", "channel_resolution_ambiguous")},
            "files": {"electronic": line.info.get("file"), "model": str(out_dir / "multistate_1d.npz")},
            "approximation": "1D coupled-state path model sharing the transverse term of V_S^(0); "
                             "couplings perpendicular to the path and geometric phase are not represented",
        })
        if single_surface_rate is not None and single_surface_rate > 0:
            record["born_oppenheimer_consistency"] = float(single.rate / single_surface_rate - 1.0)
    except ElectronicDataRequired as error:
        record.update(status="electronic_data_required", reason=str(error), request=str(error.request))
    except Exception as error:  # recorded, never discards the single-surface result
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return _json_safe(record)


__all__ = ["ELECTRONIC_DEFAULTS", "ElectronicDataRequired", "ElectronicLine", "electronic_line",
           "electronic_options", "multistate_path_model", "nonadiabatic_path_factor", "path_geometries",
           "public_options"]
