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
    if kind in ("cabi", "native") and config.get("abi_version") is not None:
        version = config["abi_version"]
        if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (1, 2):
            raise PotentialError("C ABI abi_version must be 1 or 2")
    if kind == "torch":
        if not isinstance(config.get("periodic", False), (bool, np.bool_)):
            raise PotentialError("Torch periodic must be a boolean")
        if config.get("virial", "auto") not in ("auto", "autograd", "provided", "none"):
            raise PotentialError("Torch virial must be auto, autograd, provided, or none")
    if periodic and kind in ("rpmdrate", "f2py", "diabatic_python", "pyscf"):
        raise PotentialError(
            f"Provider {kind!r} has no periodic cell/PBC contract"
        )
    if periodic and kind in ("cabi", "native") and config.get("abi_version") == 1:
        raise PotentialError("C ABI v1 is nonperiodic; use a cell-aware C ABI v2 library")
    if periodic and kind == "torch" and config.get("periodic") is not True:
        raise PotentialError("Periodic Torch models require potential.periodic=true and a cell/PBC contract")
    if periodic and kind == "deepmd" and not pbc.all():
        raise PotentialError("DeePMD requires all or no periodic axes; partial PBC is unsupported")
    if kind in ("builtin", "free", "harmonic", "morse_pair", "double_well", "periodic_cosine"):
        model = config.get("name", config.get("model", kind if kind != "builtin" else "harmonic"))
        if periodic and model not in ("free", "periodic_cosine"):
            raise PotentialError(f"Builtin {model!r} is nonperiodic; periodic cells require a cell-aware PES")
        if model == "periodic_cosine" and not periodic:
            raise PotentialError("periodic_cosine requires a periodic cell")


def make_potential(config, system=None):
    """Construct and bind a provider with an explicit fixed-cell contract.

    Python plugins may use provider-unit ``evaluate(q, *, cell, pbc)`` with
    ``metadata['periodic']=True``, or a canonical-AU ``bind_system(system)``
    callback with ``metadata['supports_pbc']=True``. Builtin periodic models,
    ASE and i-PI preserve their own validated cell-aware interfaces.
    """
    validate_periodic_spec(config, system)
    provider = _make_potential(config, system)
    try:
        bind = getattr(provider, "bind_system", None)
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
    def get(key, default=None):
        return (
            system.get(key, default)
            if isinstance(system, dict)
            else getattr(system, key, default)
        )

    cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
    periodic = bool(pbc.any())
    if hasattr(config, "evaluate"):
        if not periodic or isinstance(config, PythonPotential):
            return config
        metadata = getattr(config, "metadata", {})
        if metadata.get("supports_pbc") is True:
            if not callable(getattr(config, "bind_system", None)):
                raise PotentialError("Periodic PES must implement bind_system(system)")
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
    if kind == "native_periodic":
        from ..native_ensemble import NativePeriodicPES
        if params:
            raise ValueError("native_periodic input does not accept opaque context parameters")
        return NativePeriodicPES(spec["library"], spec.get("function", "evaluate"), len(get("masses")))
    if kind == "diabatic_python":
        from .diabatic import DiabaticPythonPotential

        obj = import_object(spec["factory"])(**params) if "factory" in spec else import_object(spec["function"])
        return DiabaticPythonPotential(obj, units=spec.get("units"), derivative=spec.get("derivative", "gradient"))
    if kind == "python":
        if "factory" in spec:
            obj = import_object(spec["factory"])(**params)
        elif "function" in spec:
            obj = import_object(spec["function"])
        else:
            raise ValueError("Python PES requires factory or function")
        try:
            return PythonPotential(
                obj, units=spec.get("units"), derivative=spec.get("derivative", "force"),
                cell=cell, pbc=pbc,
            )
        except BaseException:
            # A constructed model may own a socket, accelerator or calculator.
            # Constructor-level contract errors happen before make_potential's
            # binding cleanup, so release factory-owned providers here too.
            if "factory" in spec and callable(getattr(obj, "close", None)):
                obj.close()
            raise
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
            spec["library"], config=params, system=system, manifest=spec.get("manifest"),
            abi_version=spec.get("abi_version"),
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
            cell=cell, pbc=pbc, periodic=spec.get("periodic", False),
            virial=spec.get("virial", "auto"),
        )
    if kind in ("mace", "nequip", "deepmd", "fairchem", "tblite", "xtb", "pyscf"):
        from .adapters import make_named_potential

        if "units" in spec:
            if kind != "deepmd":
                raise ValueError(f"{kind} has fixed backend units; potential.units is unsupported")
            if "units" in params:
                raise ValueError("Specify DeePMD units in potential.units or params.units, not both")
            params["units"] = spec["units"]
        return make_named_potential(
            kind, symbols=get("symbols"), model=spec.get("model"),
            device=spec.get("device", "cpu"), params=params, cell=cell, pbc=pbc,
            charge=get("charge", 0), multiplicity=get("multiplicity", 1),
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
