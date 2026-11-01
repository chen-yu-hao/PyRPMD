"""Strict input resolution for unweighted model nonadiabatic trajectories.

This task chooses a fixed diabatic model and an explicit nonequilibrium
initial ensemble.  ``point`` repeats the supplied nuclear phase point on every
bead.  ``harmonic_ring`` samples an independent harmonic reference Hamiltonian
only in the listed Cartesian coordinates, with all other initial q and p zero.
Neither choice samples the coupled electronic/nuclear thermal distribution.

Model parameters, positions, momenta and harmonic-reference arrays use atomic
units without mass weighting.  Temperature, beta, time and atom masses also
accept the package's explicit quantity syntax.  Resolution is read-only: no
files, trajectory states, random draws or native libraries are created here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .constants import KB_AU
from .dynamics import normalize_integrator
from .potentials.nonadiabatic_models import (
    TullyModelOne, TullyModelTwo, TullyModelThree,
    LinearVibronicCoupling, DiscreteSpinBoson,
)
from .system import System
from .units import parse_quantity


_ROOT = set("schema_version task method beta temperature beads seed model system "
            "initial ensemble dynamics parallel output native description".split())
_MODELS = {
    "tully1": (TullyModelOne, {"a", "b", "c", "d", "natoms", "coordinate"}),
    "tully2": (TullyModelTwo, {"a", "b", "c", "d", "e0", "natoms", "coordinate"}),
    "tully3": (TullyModelThree, {"a", "b", "c", "natoms", "coordinate"}),
    "lvc": (LinearVibronicCoupling,
            {"h0", "linear", "harmonic", "reference", "coordinates", "natoms"}),
    "spin_boson": (DiscreteSpinBoson,
                   {"frequencies", "couplings", "masses", "bias", "tunnelling",
                    "coordinates", "natoms", "reference"}),
}
_RPSH = set("electronic_substeps rescaling frustrated_hops gap_tolerance "
            "decoherence decoherence_constant".split())
_MISSING = object()


def _error(message):
    # JobSpec imports this resolver at dispatch time.  Keep this dependency
    # local so importing either module first cannot create an import cycle.
    from .input import InputError
    return InputError(message)


def _table(value, name, allowed):
    if not isinstance(value, Mapping):
        raise _error(f"{name} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise _error(f"Unknown {name} fields: {', '.join(sorted(map(str, unknown)))}")
    return dict(value)


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise _error(f"{name} must be an integer >= {minimum}")
    return int(value)


def _no_booleans(value, name):
    """Reject bools before heterogeneous Python arrays can coerce them to 0/1."""
    if isinstance(value, (bool, np.bool_)):
        raise _error(f"{name} must contain numerical values, not booleans")
    if isinstance(value, Mapping):
        for item in value.values():
            _no_booleans(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_booleans(item, name)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise _error(f"{name} must contain numerical values, not booleans")
        if value.dtype.kind == "O":
            for item in value.flat:
                _no_booleans(item, name)


def _real_array(value, name, shape=None, positive=False):
    _no_booleans(value, name)
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or shape is not None and raw.shape != shape:
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=float, copy=True)
        if not np.isfinite(result).all() or positive and np.any(result <= 0):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        qualifier = "positive finite" if positive else "finite real"
        expected = "" if shape is None else f" with shape {shape}"
        raise _error(f"{name} must be a {qualifier} numeric array{expected}") from exc
    return result


def _quantity(value, name, dimension):
    _no_booleans(value, name)
    try:
        result = float(parse_quantity(value, dimension))
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(f"{name}: {exc}") from exc
    if not np.isfinite(result) or result <= 0:
        raise _error(f"{name} must be positive and finite")
    return result


def _choice(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise _error(f"{name} must be one of {', '.join(sorted(allowed))}")
    return value


def _native(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or isinstance(value, str) and value == "required":
        return value
    raise _error(f"{name} must be a boolean, None or 'required'")


def _complex_array(value, name):
    """TOML can encode complex model parameters as {real=..., imag=...}."""
    if not isinstance(value, Mapping):
        return value
    if set(value) != {"real", "imag"}:
        raise _error(f"{name} complex encoding must contain exactly real and imag")
    real = _real_array(value["real"], name + ".real")
    imag = _real_array(value["imag"], name + ".imag", shape=real.shape)
    return real + 1j * imag if np.any(imag != 0) else real


def _model(raw):
    spec = _table(raw, "model", {"name", "params"})
    name = _choice(spec.get("name"), "model.name", _MODELS)
    factory, allowed = _MODELS[name]
    params = _table(spec.get("params", {}), "model.params", allowed)
    _no_booleans(params, "model.params")
    complex_fields = ("h0", "linear") if name == "lvc" else ("tunnelling",) if name == "spin_boson" else ()
    for key in complex_fields:
        if key in params:
            params[key] = _complex_array(params[key], "model.params." + key)
    try:
        return factory(**params)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(f"model.params ({name}): {exc}") from exc


def _masses(raw, model):
    if raw is _MISSING:
        if isinstance(model, DiscreteSpinBoson):
            return np.array(model.nuclear_masses, copy=True)
        raise _error("system.masses is required, with one positive atom mass per model atom")
    if (not isinstance(raw, (list, tuple, np.ndarray))
            or isinstance(raw, np.ndarray) and raw.ndim != 1
            or len(raw) != model.natoms):
        raise _error(f"system.masses must contain {model.natoms} atom masses")
    result = np.array([_quantity(value, f"system.masses[{i}]", "mass")
                       for i, value in enumerate(raw)])
    if (isinstance(model, DiscreteSpinBoson)
            and not np.allclose(result, model.nuclear_masses, rtol=1e-12, atol=0)):
        raise _error("system.masses must match the spin_boson nuclear_masses; "
                     "its force constants already contain the model mode masses")
    return result


def _coordinates(value, count, natoms):
    _no_booleans(value, "initial.coordinates")
    try:
        raw = np.asarray(value)
        if (raw.shape != (count,) or raw.dtype.kind not in "iu"
                or np.any(raw < 0) or np.any(raw >= 3*natoms)
                or len(np.unique(raw)) != count):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error("initial.coordinates must be distinct integer Cartesian indices "
                     "in [0, 3*natoms), one per frequency") from exc
    return [int(index) for index in raw]


def _initial(raw, method, model, system_spec):
    electronic_key = "initial_state" if method == "nrpmd" else "active_surface"
    common = {"kind", electronic_key}
    if not isinstance(raw, Mapping):
        raise _error("initial must be a table")
    kind = _choice(raw.get("kind", "point"), "initial.kind", {"point", "harmonic_ring"})
    fields = {"momenta"} if kind == "point" else {
        "frequencies", "coordinates", "centers", "centroid_momenta"}
    spec = _table(raw, "initial", common | fields)
    electronic = _integer(spec.get(electronic_key, 0), f"initial.{electronic_key}", 0)
    if electronic >= model.nstates:
        raise _error(f"initial.{electronic_key} must be < model.nstates ({model.nstates})")
    initial = {"kind": kind, electronic_key: electronic}
    shape = (model.natoms, 3)
    if kind == "point":
        geometry = _real_array(system_spec.get("positions", np.zeros(shape)),
                               "system.positions", shape)
        initial["momenta"] = _real_array(spec.get("momenta", np.zeros(shape)),
                                          "initial.momenta", shape).tolist()
        warning = ("point initialization repeats one nuclear phase point on every bead; "
                   "it is not a thermal nuclear ensemble")
    else:
        if "frequencies" not in spec:
            raise _error("initial.frequencies is required for harmonic_ring")
        frequencies = _real_array(spec["frequencies"], "initial.frequencies", positive=True)
        if frequencies.ndim != 1 or not len(frequencies):
            raise _error("initial.frequencies must be a nonempty vector of positive angular frequencies")
        count = len(frequencies)
        indices = _coordinates(spec.get("coordinates", model.coordinates), count, model.natoms)
        centers = _real_array(spec.get("centers", np.zeros(count)), "initial.centers", (count,))
        momenta = _real_array(spec.get("centroid_momenta", np.zeros(count)),
                              "initial.centroid_momenta", (count,))
        geometry = np.zeros(shape)
        geometry.reshape(-1)[indices] = centers
        if "positions" in system_spec:
            supplied = _real_array(system_spec["positions"], "system.positions", shape)
            if not np.array_equal(supplied, geometry):
                raise _error("system.positions must equal the harmonic_ring centers in initial.coordinates "
                             "and zero elsewhere; omit positions to derive this reference geometry")
        initial.update(frequencies=frequencies.tolist(), coordinates=indices,
                       centers=centers.tolist(), centroid_momenta=momenta.tolist())
        warning = ("harmonic_ring samples a finite-bead harmonic reference distribution in initial.coordinates, "
                   "with other initial q/p zero; it is not the coupled electronic/nuclear thermal distribution. "
                   "centroid_momenta shifts the thermal momentum mean and does not fix the realized centroid")
    return initial, geometry, warning


def _dynamics(raw, method, root):
    allowed = {"dt", "integrator", "native", "loop_backend", "block_steps"}
    spec = _table(raw, "dynamics", allowed | (_RPSH if method == "rpsh" else set()))
    native = _native(root.get("native", True), "native")
    if "native" in spec:
        chosen = _native(spec["native"], "dynamics.native")
        if "native" in root and chosen != native:
            raise _error("native and dynamics.native conflict; supply only one or the same value")
        native = chosen
    try:
        integrator = normalize_integrator(spec.get("integrator", "cayley-bcb"))
    except ValueError as exc:
        raise _error(f"dynamics.integrator: {exc}") from exc
    dt = _quantity(spec.get("dt", 0.1), "dynamics.dt", "time")
    options = {
        "integrator": integrator, "native": native,
        "loop_backend": _choice(spec.get("loop_backend", "auto"), "dynamics.loop_backend", {"auto", "python"}),
        "block_steps": _integer(spec.get("block_steps", 128), "dynamics.block_steps"),
    }
    if method == "rpsh":
        decoherence = _choice(spec.get("decoherence", "none"), "dynamics.decoherence", {"none", "edc"})
        options.update(
            electronic_substeps=_integer(spec.get("electronic_substeps", 8), "dynamics.electronic_substeps"),
            rescaling=_choice(spec.get("rescaling", "centroid"), "dynamics.rescaling", {"centroid", "ring"}),
            frustrated_hops=_choice(spec.get("frustrated_hops", "keep"), "dynamics.frustrated_hops", {"keep", "reverse"}),
            gap_tolerance=_quantity(spec.get("gap_tolerance", 1e-10), "dynamics.gap_tolerance", "energy"),
            decoherence=decoherence,
        )
        if decoherence == "edc":
            options["decoherence_constant"] = _quantity(
                spec.get("decoherence_constant", 0.1), "dynamics.decoherence_constant", "energy")
        elif "decoherence_constant" in spec:
            raise _error("dynamics.decoherence_constant requires decoherence='edc'")
    return options, dt, native


def _parallel(raw):
    spec = _table(raw, "parallel", {"backend", "mode", "trajectory_workers", "threads",
                                    "threads_per_rank", "bead_workers"})
    workers = _integer(spec.get("trajectory_workers", 1), "parallel.trajectory_workers")
    default_backend = "process" if workers > 1 else "serial"
    backend = _choice(spec.get("backend", default_backend), "parallel.backend", {"serial", "process"})
    if backend == "serial" and workers != 1:
        raise _error("parallel.backend='serial' requires trajectory_workers=1; use backend='process'")
    _choice(spec.get("mode", "serial"), "parallel.mode", {"serial"})
    if _integer(spec.get("bead_workers", 1), "parallel.bead_workers") != 1:
        raise _error("nonadiabatic trajectories require parallel.bead_workers=1")
    threads = _integer(spec.get("threads_per_rank", spec.get("threads", 1)), "parallel.threads_per_rank")
    if "threads" in spec:
        alias = _integer(spec["threads"], "parallel.threads")
        if alias != threads:
            raise _error("parallel.threads and threads_per_rank conflict")
    return {"backend": backend, "mode": "serial", "trajectory_workers": workers,
            "bead_workers": 1, "threads_per_rank": threads}


def _check_reference(model, geometry, method, options):
    from .potentials.base import PotentialError

    try:
        result = model.evaluate(geometry[None])
        if method != "rpsh":
            return
        if np.any(np.imag(result.matrix) != 0) or np.any(np.imag(result.gradients) != 0):
            raise ValueError("RPSH requires a real symmetric matrix and real gradients; "
                             "complex Hermitian models require method='nrpmd'")
        energies = np.linalg.eigvalsh(result.matrix.real)[0]
        if not np.isfinite(energies).all():
            raise ValueError("RPSH reference energies are not finite")
        if np.any(np.diff(energies) <= options["gap_tolerance"]):
            raise ValueError("RPSH adiabatic states at the initial/reference geometry are degenerate "
                             "or below dynamics.gap_tolerance")
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, PotentialError) as exc:
        raise _error(f"model at initial/reference geometry: {exc}") from exc


def resolve_nonadiabatic_job(raw, *, base_directory=".", input_path=None):
    """Return ``(canonical_config, System)`` for task='nonadiabatic'.

    Exactly one of positive ``beta`` (hartree^-1) and ``temperature`` (kelvin)
    is required.  ``method`` is explicitly nrpmd or rpsh, and ``ensemble.steps``
    is required (zero permits initialization-only output).  Defaults are one
    bead, 32 trajectories, stride 1, seed 0, dt=0.1 atomic time and block 128.

    RPSH ``active_surface`` is a zero-based adiabatic index. NRPMD
    ``initial_state`` is a zero-based fixed diabatic index. The opposite key,
    unused method options and all unknown input fields are rejected.

    ``model`` becomes its complete versioned ``to_dict()`` identity.  Numeric
    arrays in the returned configuration are owned, JSON-safe lists. System
    masses are atom masses and must agree with spin-boson mode-derived masses.
    ``System.geometry`` is the point position or harmonic reference centers;
    harmonic trajectories fluctuate about this geometry. RPSH gap validation
    here tests that reference; every propagated/sample geometry is checked by
    its dynamics separately.
    """
    root = _table(raw, "top-level", _ROOT)
    if _integer(root.get("schema_version", 1), "schema_version") != 1:
        raise _error("Supported schema_version is 1")
    _choice(root.get("task"), "task", {"nonadiabatic"})
    method = _choice(root.get("method"), "method", {"nrpmd", "rpsh"})
    if ("beta" in root) == ("temperature" in root):
        raise _error("Specify exactly one of beta or temperature for nonadiabatic dynamics; "
                     "remove beta before applying a temperature override")
    if "beta" in root:
        beta = _quantity(root["beta"], "beta", (0, -1, 0, 0, 0, 0))
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            temperature = float(np.reciprocal(np.float64(KB_AU) * beta))
    else:
        temperature = _quantity(root["temperature"], "temperature", "temperature")
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            beta = float(np.reciprocal(np.float64(KB_AU) * temperature))
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(temperature) or temperature <= 0:
        raise _error("beta and its corresponding temperature must be representable positive finite floats")
    beads = _integer(root.get("beads", 1), "beads")
    seed = _integer(root.get("seed", 0), "seed", 0)
    model = _model(root.get("model", {}))
    system_spec = _table(root.get("system", {}), "system", {"masses", "positions"})
    masses = _masses(system_spec.get("masses", _MISSING), model)
    initial, geometry, warning = _initial(root.get("initial", {}), method, model, system_spec)
    system = System(symbols=("X",)*model.natoms, masses=masses, geometry=geometry)
    ensemble = _table(root.get("ensemble", {}), "ensemble", {"trajectories", "steps", "sample_stride"})
    if "steps" not in ensemble:
        raise _error("ensemble.steps is required (a nonnegative integer)")
    ensemble = {
        "trajectories": _integer(ensemble.get("trajectories", 32), "ensemble.trajectories"),
        "steps": _integer(ensemble["steps"], "ensemble.steps", 0),
        "sample_stride": _integer(ensemble.get("sample_stride", 1), "ensemble.sample_stride"),
    }
    options, dt, native = _dynamics(root.get("dynamics", {}), method, root)
    parallel = _parallel(root.get("parallel", {}))
    output = _table(root.get("output", {}), "output", {"directory"})
    directory = output.get("directory", "runs/nonadiabatic")
    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise _error("output.directory must be a nonempty path")
    try:
        base = Path(base_directory).expanduser().resolve()
        output_path = Path(directory).expanduser()
        output_path = (base / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
        origin = None if input_path is None else str((base / Path(input_path)).resolve())
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise _error(f"Invalid input/output path: {exc}") from exc
    _check_reference(model, geometry, method, options)
    notes = [warning]
    if beads == 1:
        notes.append("beads=1 contains no nuclear ring-polymer delocalization")
    resolved = {
        "schema_version": 1, "task": "nonadiabatic", "method": method,
        "model": model.to_dict(), "system": system.to_dict(),
        "beta": beta, "temperature_K": temperature, "beads": beads, "seed": seed,
        "initial": initial, "ensemble": ensemble, "dt_au": dt,
        "native": native, "dynamics": options, "parallel": parallel,
        "output": {"directory": str(output_path)},
        "_warnings": notes, "_base_directory": str(base), "_input_path": origin,
    }
    if "description" in root:
        if not isinstance(root["description"], str):
            raise _error("description must be a string")
        resolved["description"] = root["description"]
    return resolved, system
