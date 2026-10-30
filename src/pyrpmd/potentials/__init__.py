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
    """Create a provider and explicitly bind fixed-cell boundary conditions.

    Periodic custom providers must declare ``metadata['supports_pbc']=True``
    and implement ``bind_system(system)``. That callback always receives the
    System in canonical atomic units, including when its coordinate adapter
    uses other units. Unsupported providers fail before acquiring resources.
    """
    periodic = any(
        system.get("pbc", (False, False, False))
        if isinstance(system, dict)
        else getattr(system, "pbc", (False, False, False))
    )
    if periodic and isinstance(config, dict):
        kind = config.get("provider", config.get("type", "builtin"))
        if kind not in ("ase", "python"):
            raise PotentialError(
                f"PES provider {kind!r} does not support System cell/PBC; "
                "use ASE or an explicitly periodic Python provider"
            )
    provider = _make_potential(config, system)
    try:
        bind = getattr(provider, "bind_system", None)
        metadata = getattr(provider, "metadata", {})
        if periodic and (
            not isinstance(metadata, dict)
            or metadata.get("supports_pbc") is not True
            or not callable(bind)
        ):
            raise PotentialError(
                "Periodic PES must declare metadata['supports_pbc']=True "
                "and implement bind_system(system); cell/PBC cannot be ignored"
            )
        if callable(bind) and system is not None:
            bind(system)
    except BaseException:
        if provider is not config and callable(getattr(provider, "close", None)):
            provider.close()
        raise
    return provider


def _make_potential(config, system=None):
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

        return ASEPotential(
            get("symbols"), import_object(spec["factory"])(**params),
            cell=get("cell"), pbc=get("pbc", (False, False, False)),
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
