"""Physical PES providers and their uniform batch interface."""

from .base import (
    PESResult,
    Potential,
    PotentialError,
    BackendUnavailableError,
    validate_result,
    check_potential,
)
from .builtin import (
    FreePotential,
    HarmonicPotential,
    MorsePairPotential,
    DoubleWellPotential,
    PeriodicCosinePotential,
)
from .python import PythonPotential, RPMDratePotential, import_object
from .cabi import CABIPotential, build_object
from ..geometry import validate_cell_pbc
import numpy as np


def validate_periodic_spec(config, system=None):
    """Reject statically incompatible cell/provider choices without loading PESs.

    A Python factory's declared capability still needs provider construction;
    this preflight never imports factories, opens sockets, or loads model data.
    """
    if not isinstance(config, dict):
        return
    get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
    _, pbc = validate_cell_pbc(get("cell"), get("pbc"))
    periodic = bool(pbc.any())
    kind = config.get("provider", config.get("type", "builtin"))
    if periodic and kind in ("rpmdrate", "f2py", "cabi", "native", "torch"):
        raise PotentialError(
            f"Provider {kind!r} has no cell/PBC contract; use ASE, i-PI, or an explicitly periodic Python provider"
        )
    if kind in ("builtin", "free", "harmonic", "morse_pair", "double_well", "periodic_cosine"):
        model = config.get("name", config.get("model", kind if kind != "builtin" else "harmonic"))
        if periodic and model not in ("free", "periodic_cosine"):
            raise PotentialError(f"Builtin {model!r} is nonperiodic; periodic cells require a cell-aware PES")
        if model == "periodic_cosine" and not periodic:
            raise PotentialError("periodic_cosine requires a periodic cell")


def make_potential(config, system=None):
    """Create one process-local provider from a serializable specification.

    An already constructed provider/callable is accepted for serial Python jobs.
    MPI/process workers should use importable factories or a native manifest.
    """
    def get(key, default=None):
        return (
            system.get(key, default)
            if isinstance(system, dict)
            else getattr(system, key, default)
        )

    cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
    periodic = bool(pbc.any())
    if hasattr(config, "evaluate"):
        if not periodic:
            return config
        configured_cell = getattr(config, "cell", None)
        configured_pbc = getattr(config, "pbc", None)
        if configured_cell is not None and configured_pbc is not None:
            if not np.array_equal(configured_cell, cell) or not np.array_equal(configured_pbc, pbc):
                raise PotentialError("PES cell/PBC do not match System")
            if not getattr(config, "metadata", {}).get("periodic", False):
                raise PotentialError("Configured PES does not declare periodic capability")
            return config
        return PythonPotential(config, cell=cell, pbc=pbc)
    if callable(config):
        return PythonPotential(config, cell=cell, pbc=pbc)
    if not isinstance(config, dict):
        raise ValueError("potential must be a provider, callable, or mapping")
    spec = dict(config)
    kind = spec.get("provider", spec.get("type", "builtin"))
    params = dict(spec.get("params", {}))
    validate_periodic_spec(spec, system)

    if kind in ("builtin", "free", "harmonic", "morse_pair", "double_well", "periodic_cosine"):
        model = spec.get(
            "name", spec.get("model", kind if kind != "builtin" else "harmonic")
        )
        if model == "free":
            return FreePotential()
        if model == "periodic_cosine":
            if "cell" in params or "pbc" in params:
                raise ValueError("periodic_cosine obtains cell/PBC from System")
            return PeriodicCosinePotential(cell=cell, pbc=pbc, **params)
        if model == "harmonic":
            for key in ("k", "center"):
                if key in spec:
                    params.setdefault(key, spec[key])
            return HarmonicPotential(**params)
        if model == "morse_pair":
            return MorsePairPotential(**params)
        if model == "double_well":
            return DoubleWellPotential(**params)
        raise ValueError(f"unknown builtin PES {model}")
    if kind == "python":
        if "factory" in spec:
            obj = import_object(spec["factory"])(**params)
        elif "function" in spec:
            obj = import_object(spec["function"])
        else:
            raise ValueError("Python PES requires factory or function")
        return PythonPotential(
            obj, units=spec.get("units"), derivative=spec.get("derivative", "force"),
            cell=cell, pbc=pbc,
        )
    if kind in ("rpmdrate", "f2py"):
        return RPMDratePotential(
            spec.get("module", spec.get("library")),
            function=spec.get("function", "get_potential"),
            initializer=spec.get("initializer", "initialize_potential"),
            with_dimensions=spec.get("with_dimensions", False),
            kwargs=params,
            initializer_kwargs=spec.get("initializer_params"),
            units=spec.get("units"),
        )
    if kind in ("cabi", "native"):
        return CABIPotential(
            spec["library"], config=params, system=system, manifest=spec.get("manifest")
        )
    if kind == "ase":
        from .optional import ASEPotential

        return ASEPotential(
            get("symbols"), import_object(spec["factory"])(**params), cell=cell, pbc=pbc
        )
    if kind == "torch":
        from .optional import TorchPotential

        return TorchPotential(
            spec.get("model"),
            factory=spec.get("factory"),
            params=params,
            device=spec.get("device", "cuda"),
            units=spec.get("units"),
            dtype=spec.get("dtype", params.get("dtype", "float64")),
        )
    if kind in ("ipi", "socket"):
        from .socket import IPIPotential

        if "cell" in params or "pbc" in params:
            raise ValueError("i-PI cell/PBC are supplied by System, not potential.params")
        return IPIPotential(**params, cell=cell, pbc=pbc)
    raise ValueError(f"unknown PES provider {kind!r}")


__all__ = [
    "PESResult",
    "Potential",
    "PotentialError",
    "BackendUnavailableError",
    "validate_result",
    "check_potential",
    "make_potential",
    "FreePotential",
    "HarmonicPotential",
    "MorsePairPotential",
    "DoubleWellPotential",
    "PeriodicCosinePotential",
    "PythonPotential",
    "RPMDratePotential",
    "CABIPotential",
    "build_object",
]
