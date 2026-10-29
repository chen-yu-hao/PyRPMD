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
)
from .python import PythonPotential, RPMDratePotential, import_object
from .cabi import CABIPotential, build_object


def make_potential(config, system=None):
    """Create one process-local provider from a serializable specification.

    An already constructed provider/callable is accepted for serial Python jobs.
    MPI/process workers should use importable factories or a native manifest.
    """
    if hasattr(config, "evaluate"):
        return config
    if callable(config):
        return PythonPotential(config)
    if not isinstance(config, dict):
        raise ValueError("potential must be a provider, callable, or mapping")
    spec = dict(config)
    kind = spec.get("provider", spec.get("type", "builtin"))
    params = dict(spec.get("params", {}))

    def get(key, default=None):
        return (
            system.get(key, default)
            if isinstance(system, dict)
            else getattr(system, key, default)
        )

    if kind in ("builtin", "free", "harmonic", "morse_pair", "double_well"):
        model = spec.get(
            "name", spec.get("model", kind if kind != "builtin" else "harmonic")
        )
        if model == "free":
            return FreePotential()
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
            obj, units=spec.get("units"), derivative=spec.get("derivative", "force")
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

        return ASEPotential(get("symbols"), import_object(spec["factory"])(**params))
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

        return IPIPotential(**params)
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
    "PythonPotential",
    "RPMDratePotential",
    "CABIPotential",
    "build_object",
]
