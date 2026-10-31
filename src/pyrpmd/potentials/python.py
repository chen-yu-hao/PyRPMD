"""Importable Python and RPMDrate/f2py potential adapters."""

import importlib
import importlib.util
import hashlib
from pathlib import Path
import sys
import numpy as np
from .base import Potential, PESResult, PotentialError, coordinates, validate_result
from ..geometry import validate_cell_pbc
from ..constants import BOHR_ANGSTROM, EV_TO_AU as EV_HARTREE


def import_object(name):
    if not isinstance(name, str) or ":" not in name:
        raise ValueError("factory must be module:object")
    module, obj = name.rsplit(":", 1)
    if module.endswith(".py") or module.endswith(".so"):
        path = Path(module).resolve()
        if not path.is_file():
            raise ValueError(f"plugin file does not exist: {path}")
        module_name = (
            path.name.split(".")[0]
            if path.suffix == ".so"
            else "_pyrpmd_user_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
        )
        mod = sys.modules.get(module_name)
        if mod is not None and Path(getattr(mod, "__file__", "")).resolve() != path:
            raise ValueError(
                f"plugin module name collision for {module_name}; use distinct extension module names"
            )
        if mod is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError(f"cannot load plugin {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
    else:
        mod = importlib.import_module(module)
    value = mod
    for part in obj.split("."):
        value = getattr(value, part)
    return value


def unit_factors(units):
    units = units or {}
    length = units.get("positions", units.get("length", "bohr"))
    energy = units.get("energy", "hartree")
    lf = {"bohr": 1.0, "angstrom": 1 / BOHR_ANGSTROM}.get(length)
    ef = {
        "hartree": 1.0,
        "eV": EV_HARTREE,
        "ev": EV_HARTREE,
        "kcal/mol": 1 / 627.5094740631,
        "kJ/mol": 1 / 2625.4996394799,
    }.get(energy)
    if lf is None or ef is None:
        raise ValueError(
            "unsupported provider units; use bohr/angstrom and hartree/eV/kcal/mol/kJ/mol"
        )
    return lf, ef


class PythonPotential(Potential):
    def __init__(self, obj, *, units=None, derivative="force", cell=None, pbc=None):
        if not (callable(obj) or callable(getattr(obj, "evaluate", None))):
            raise TypeError("Python factory must return callable or PES provider")
        if derivative not in ("force", "gradient"):
            raise ValueError("derivative must be force or gradient")
        self.obj = obj
        self.fn = obj.evaluate if hasattr(obj, "evaluate") else obj
        self.length_factor, self.energy_factor = unit_factors(units)
        self.sign = -1 if derivative == "gradient" else 1
        self.metadata = dict(getattr(obj, "metadata", {}) or {})
        if self.metadata.get("requires_atomic_units") is True and (
            self.length_factor != 1 or self.energy_factor != 1 or self.sign != 1
        ):
            raise PotentialError("This Python provider requires atomic units and derivative='force'")
        self.metadata.setdefault("thread_safety", "context_serial")
        self.metadata.setdefault("execution_device", "cpu")
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        self._periodic_kwargs = self._check_periodic_contract(self.pbc)

    def _check_periodic_contract(self, pbc):
        """Accept either explicit periodic contract, never infer one from a cell."""
        if not pbc.any():
            return False
        if self.metadata.get("supports_pbc") is True:
            if not callable(getattr(self.obj, "bind_system", None)):
                raise PotentialError("Periodic Python PES must implement bind_system(system)")
            return False
        elif (self.metadata.get("periodic") is True
              or self.metadata.get("boundary_conditions") == "periodic"):
            return True
        else:
            raise PotentialError(
                "Periodic Python PES must declare metadata['periodic']=True with "
                "evaluate(q, *, cell, pbc), or metadata['supports_pbc']=True with bind_system(system)"
            )

    def bind_system(self, system):
        """Bind either provider-unit evaluation or a canonical-AU callback."""
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        periodic_kwargs = self._check_periodic_contract(pbc)
        bind = getattr(self.obj, "bind_system", None)
        if callable(bind):
            bind(system)
        # Commit adapter state only after validation and the callback succeed.
        # User callbacks remain responsible for their own failure side effects.
        self.cell, self.pbc = cell, pbc
        self._periodic_kwargs = periodic_kwargs

    def set_cell(self, cell):
        """Update a fully periodic cell without reinitializing the provider.

        The keyword evaluation contract receives the new provider-unit cell
        on its next call. An object bound through ``supports_pbc`` must also
        implement an explicit canonical-AU ``set_cell(cell_bohr)`` callback;
        rebinding the full System each step could reset stateful models.
        """
        new_cell, _ = validate_cell_pbc(cell, self.pbc)
        if new_cell is None or not self.pbc.all():
            raise PotentialError("variable-cell Python PES requires full periodicity")
        if not self._periodic_kwargs:
            setter = getattr(self.obj, "set_cell", None)
            if not callable(setter):
                raise PotentialError("bound Python PES must implement set_cell(cell_bohr) for NPT")
            setter(new_cell.copy())
        self.cell = new_cell

    def native_binding(self):
        from ..propagation import NativeUnavailableError
        if self.pbc.any():
            raise NativeUnavailableError("periodic Python PES requires the cell-aware Python route")
        if type(self) is not PythonPotential or self.sign != 1 or self.length_factor != 1 or self.energy_factor != 1:
            raise NativeUnavailableError("custom or unit-converting Python PES requires Python")
        if ("evaluate" in self.__dict__ or PythonPotential.evaluate is not _PYTHON_EVALUATE
                or self.fn != (self.obj.evaluate if hasattr(self.obj, "evaluate") else self.obj)):
            raise NativeUnavailableError("overridden Python PES function requires Python")
        factory = getattr(self.obj, "native_binding", None)
        if factory is None:
            raise NativeUnavailableError("Python PES has no explicit native binding")
        return factory()

    def evaluate(self, x):
        x = coordinates(x)
        if self.pbc.any() and self._periodic_kwargs:
            result = self.fn(
                x / self.length_factor,
                cell=self.cell.copy() / self.length_factor,
                pbc=self.pbc.copy(),
            )
        else:
            result = self.fn(x / self.length_factor)
        result = validate_result(result, x.shape, require_success=False)
        result = PESResult(
            result.energy * self.energy_factor,
            result.forces * (self.sign * self.energy_factor / self.length_factor),
            result.status,
            result.messages,
            None if result.virial is None else result.virial * self.energy_factor,
        )
        if self.length_factor == self.energy_factor == self.sign == 1:
            # Multiplication only copied the validated arrays. Keep those
            # independent output buffers without scanning their values again.
            return result
        return validate_result(result, x.shape, require_success=False)

    def close(self):
        close = getattr(self.obj, "close", None)
        if close:
            close()


# Capture before any later lazy import of the native propagation dispatcher.
_PYTHON_EVALUATE = PythonPotential.evaluate


class RPMDratePotential(Potential):
    """Legacy f2py: q(3,N,P), energy(P), positive gradient(3,N,P), info.

    The adapter is process_serial: Fortran COMMON/module state is not made
    reentrant by instantiating multiple Python wrappers.
    """

    metadata = {
        "provider": "rpmdrate",
        "thread_safety": "process_serial",
        "execution_device": "cpu",
    }

    def __init__(
        self,
        module,
        function="get_potential",
        initializer="initialize_potential",
        with_dimensions=False,
        kwargs=None,
        initializer_kwargs=None,
        units=None,
    ):
        self.fn = import_object(module + ":" + function)
        if initializer:
            init = import_object(module + ":" + initializer)
            status = init(**(initializer_kwargs or {}))
            if status is not None and (
                np.ndim(status) != 0 or not np.isfinite(status) or status != 0
            ):
                raise PotentialError(f"PES initialization failed: {status}")
        self.kwargs = kwargs or {}
        self.with_dimensions = bool(with_dimensions)
        self.length_factor, self.energy_factor = unit_factors(units)

    def evaluate(self, x):
        x = coordinates(x)
        q = np.asfortranarray((x / self.length_factor).transpose(2, 1, 0))
        result = (
            self.fn(q, q.shape[1], q.shape[2], **self.kwargs)
            if self.with_dimensions
            else self.fn(q, **self.kwargs)
        )
        if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
            raise PotentialError("legacy PES must return (V,dVdq[,info])")
        e, g = result[:2]
        status = 0 if len(result) == 2 else result[2]
        if np.ndim(status) != 0 or not np.isfinite(status) or status != 0:
            raise PotentialError(f"legacy PES failed with info={status}")
        if np.shape(g) != q.shape:
            raise PotentialError("legacy gradient shape mismatch")
        return validate_result(
            PESResult(
                np.asarray(e) * self.energy_factor,
                -np.asarray(g).transpose(2, 1, 0)
                * self.energy_factor
                / self.length_factor,
            ),
            x.shape,
        )
