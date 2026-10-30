"""Checked, system-free inputs for the local electrochemical PCET model.

This input describes one localized oxidized/reduced interfacial complex and
one transferring proton. It is not an atomistic ``System`` or a constant-
potential trajectory specification. Quantities are resolved to atomic units,
except explicitly named electrode potentials in volts and temperatures in K.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import numbers
from pathlib import Path

import numpy as np

from .units import parse_quantity, unit_factor
from .pcet import ISOTOPE_MASSES_AU


_ROOT_FIELDS = {
    "schema_version", "task", "preset", "description", "temperature", "pcet", "output",
}
_PCET_FIELDS = {
    "method", "normalization", "isotopes", "driving_force", "reorganization_energy",
    "electronic_coupling", "electrode", "proton", "numerics",
}
_ELECTRODE_FIELDS = {
    "model", "density_of_states", "potential_reference", "reference_potential_V", "potentials_V",
}
_PROTON_FIELDS = {
    "grid_min", "grid_max", "grid_points", "states", "oxidized", "reduced",
}
_NUMERICS_FIELDS = {
    "quadrature_rtol", "quadrature_atol", "convergence_check", "relative_tolerance", "absolute_tolerance_s",
}


def _error(message):
    from .input import InputError

    return InputError(message)


def _table(value, name, allowed):
    if not isinstance(value, Mapping):
        raise _error(f"{name} must be a table")
    result = deepcopy(dict(value))
    unknown = set(result) - allowed
    if unknown:
        raise _error(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    return result


def _required(table, name, field):
    if field not in table:
        raise _error(f"{name}.{field} is required")
    return table[field]


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise _error(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise _error(f"{name} must be {bounds}")
    return value


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise _error(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        constraint = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise _error(f"{name} must be finite and {constraint}")
    return value


def _quantity(value, name, dimension, *, positive=False, nonnegative=False):
    try:
        parsed = parse_quantity(value, dimension)
    except (ValueError, TypeError) as exc:
        raise _error(f"{name}: {exc}") from exc
    return _real(parsed, name, positive=positive, nonnegative=nonnegative)


def _scan(value, name, parser):
    values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    if not len(values):
        raise _error(f"{name} must contain at least one value")
    result = [parser(item, f"{name}[{i}]") for i, item in enumerate(values)]
    if len(set(result)) != len(result):
        raise _error(f"{name} must not contain duplicate values")
    return result


@dataclass(frozen=True)
class PCETModelIdentity:
    """A serializable one-coordinate model; deliberately not an atomistic System."""

    coordinate_bohr: np.ndarray
    oxidized_hartree: np.ndarray
    reduced_hartree: np.ndarray
    oxidized_energy_origin_au: float = 0.0
    reduced_energy_origin_au: float = 0.0

    def __post_init__(self):
        arrays = []
        for key in ("coordinate_bohr", "oxidized_hartree", "reduced_hartree"):
            raw = np.asarray(getattr(self, key))
            if raw.dtype.kind not in "fiu" or raw.ndim != 1 or not np.isfinite(raw).all():
                raise _error(f"PCET model {key} must be a finite real vector")
            value = np.array(raw, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, key, value)
            arrays.append(value)
        x, oxidized, reduced = arrays
        if len(x) < 5 or oxidized.shape != x.shape or reduced.shape != x.shape:
            raise _error("PCET model requires matching proton-grid and potential vectors, with >= 5 points")
        spacing = np.diff(x)
        if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=0):
            raise _error("PCET proton coordinate must be a strictly increasing uniform grid")
        for key in ("oxidized_energy_origin_au", "reduced_energy_origin_au"):
            object.__setattr__(self, key, _real(getattr(self, key), key))

    def to_dict(self):
        return {
            "schema": "pyrpmd.pcet-model", "schema_version": 1,
            "representation": "one_proton_coordinate_two_diabatic_surfaces",
            "coordinate_bohr": self.coordinate_bohr.tolist(),
            "oxidized_hartree": self.oxidized_hartree.tolist(),
            "reduced_hartree": self.reduced_hartree.tolist(),
            "oxidized_energy_origin_au": self.oxidized_energy_origin_au,
            "reduced_energy_origin_au": self.reduced_energy_origin_au,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise _error("PCET model identity must be a table")
        expected = {
            "schema", "schema_version", "representation", "coordinate_bohr", "oxidized_hartree", "reduced_hartree",
            "oxidized_energy_origin_au", "reduced_energy_origin_au",
        }
        if (set(value) != expected or value.get("schema") != "pyrpmd.pcet-model"
                or type(value.get("schema_version")) is not int or value["schema_version"] != 1
                or value.get("representation") != "one_proton_coordinate_two_diabatic_surfaces"):
            raise _error("Unknown or invalid PCET model identity schema")
        return cls(
            value["coordinate_bohr"], value["oxidized_hartree"], value["reduced_hartree"],
            value["oxidized_energy_origin_au"], value["reduced_energy_origin_au"],
        )


def sample_proton_potential(spec, coordinate_bohr, *, base_directory=None, name="proton potential"):
    """Resolve a harmonic or explicitly unit-labelled tabulated 1D potential."""
    if not isinstance(spec, Mapping):
        raise _error(f"{name} must be a table")
    kind = spec.get("kind")
    x = np.asarray(coordinate_bohr, dtype=float)
    if kind == "harmonic":
        result = _table(spec, name, {"kind", "center", "force_constant"})
        center = _quantity(_required(result, name, "center"), f"{name}.center", "length")
        spring = _quantity(_required(result, name, "force_constant"), f"{name}.force_constant", "spring", positive=True)
        if not x[0] < center < x[-1]:
            raise _error(f"{name}.center must lie strictly inside the proton grid")
        return {"kind": kind, "center": center, "force_constant": spring}, 0.5 * spring * (x - center) ** 2, 0.0
    if kind == "tabulated":
        result = _table(spec, name, {"kind", "file", "coordinate_unit", "energy_unit"})
        source = _required(result, name, "file")
        if not isinstance(source, (str, Path)) or not str(source):
            raise _error(f"{name}.file must be a file path")
        path = (Path(base_directory or ".") / source).resolve()
        length_unit = _required(result, name, "coordinate_unit")
        energy_unit = _required(result, name, "energy_unit")
        if not isinstance(length_unit, str) or not isinstance(energy_unit, str):
            raise _error(f"{name} coordinate_unit and energy_unit must be unit strings")
        try:
            length_factor = unit_factor(length_unit, "length")
            energy_factor = unit_factor(energy_unit, "energy")
            data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None, ndmin=2)
        except (ValueError, TypeError, OSError) as exc:
            raise _error(f"{name}: cannot read the declared two-column potential: {exc}") from exc
        if data.ndim != 2 or data.shape[1] != 2 or len(data) < 5 or not np.isfinite(data).all():
            raise _error(f"{name}.file requires >= 5 finite rows and exactly two columns (coordinate, energy)")
        tab_x, tab_v = data[:, 0] * length_factor, data[:, 1] * energy_factor
        if not np.all(np.diff(tab_x) > 0):
            raise _error(f"{name}.file coordinates must be strictly increasing")
        if tab_x[0] > x[0] or tab_x[-1] < x[-1]:
            raise _error(f"{name}.file must cover the entire proton grid; extrapolation is not supported")
        minimum_coordinate = tab_x[int(np.argmin(tab_v))]
        if not x[0] < minimum_coordinate < x[-1]:
            raise _error(f"{name}.file potential minimum must lie inside the chosen proton grid")
        values = np.interp(x, tab_x, tab_v)
        if np.argmin(values) in (0, len(values) - 1):
            raise _error(f"{name} minimum lies on the grid boundary; expand the grid or correct the potential")
        return {
            "kind": kind, "file": str(path), "coordinate_unit": length_unit, "energy_unit": energy_unit,
        }, values, float(np.min(tab_v))
    raise _error(f"{name}.kind must be harmonic or tabulated")


def resolve_pcet_job(mapping, *, base_directory=None, input_path=None):
    """Resolve an explicit local PCET task without manufacturing a molecule."""
    raw = _table(mapping, "PCET top-level", _ROOT_FIELDS)
    if raw.get("task") != "pcet":
        raise _error("resolve_pcet_job requires task='pcet'")
    if _integer(raw.get("schema_version", 1), "schema_version") != 1:
        raise _error("Supported input schema_version is 1")
    preset = raw.get("preset", "pilot")
    if preset not in ("pilot", "production"):
        raise _error("preset must be pilot or production")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise _error("description must be text")
    temperatures = _scan(_required(raw, "input", "temperature"), "temperature", lambda v, n: _quantity(v, n, "temperature", positive=True))
    pcet = _table(_required(raw, "input", "pcet"), "pcet", _PCET_FIELDS)
    if pcet.get("method", "golden_rule") != "golden_rule":
        raise _error("PCET currently supports method='golden_rule' only")
    if pcet.get("normalization", "local_first_order") != "local_first_order":
        raise _error("PCET currently supports normalization='local_first_order' only")
    isotopes = pcet.get("isotopes", ["H", "D"])
    if (not isinstance(isotopes, (list, tuple)) or not isotopes
            or any(not isinstance(item, str) or item not in ISOTOPE_MASSES_AU for item in isotopes)
            or len(set(isotopes)) != len(isotopes)):
        raise _error("pcet.isotopes must be a nonempty, duplicate-free list containing H, D and/or T")
    driving = _quantity(_required(pcet, "pcet", "driving_force"), "pcet.driving_force", "energy")
    reorganization = _quantity(_required(pcet, "pcet", "reorganization_energy"), "pcet.reorganization_energy", "energy", positive=True)
    coupling = _quantity(_required(pcet, "pcet", "electronic_coupling"), "pcet.electronic_coupling", "energy", nonnegative=True)
    electrode = _table(_required(pcet, "pcet", "electrode"), "pcet.electrode", _ELECTRODE_FIELDS)
    if electrode.get("model", "wide_band") != "wide_band":
        raise _error("pcet.electrode.model must be wide_band")
    reference = _required(electrode, "pcet.electrode", "potential_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise _error("pcet.electrode.potential_reference must explicitly name the potential reference")
    density = _quantity(_required(electrode, "pcet.electrode", "density_of_states"), "pcet.electrode.density_of_states", (0, -1, 0, 0, 0, 0), positive=True)
    reference_voltage = _real(_required(electrode, "pcet.electrode", "reference_potential_V"), "pcet.electrode.reference_potential_V")
    voltages = _scan(_required(electrode, "pcet.electrode", "potentials_V"), "pcet.electrode.potentials_V", _real)
    proton = _table(_required(pcet, "pcet", "proton"), "pcet.proton", _PROTON_FIELDS)
    minimum = _quantity(_required(proton, "pcet.proton", "grid_min"), "pcet.proton.grid_min", "length")
    maximum = _quantity(_required(proton, "pcet.proton", "grid_max"), "pcet.proton.grid_max", "length")
    if minimum >= maximum:
        raise _error("pcet.proton.grid_min must be less than grid_max")
    points = _integer(proton.get("grid_points", 401), "pcet.proton.grid_points", 5, 20001)
    states = _integer(proton.get("states", 16), "pcet.proton.states", 1, points - 2)
    coordinate = np.linspace(minimum, maximum, points)
    base = Path(base_directory or ".").resolve()
    oxidized, ox_values, ox_origin = sample_proton_potential(_required(proton, "pcet.proton", "oxidized"), coordinate, base_directory=base, name="pcet.proton.oxidized")
    reduced, red_values, red_origin = sample_proton_potential(_required(proton, "pcet.proton", "reduced"), coordinate, base_directory=base, name="pcet.proton.reduced")
    numerics = _table(pcet.get("numerics", {}), "pcet.numerics", _NUMERICS_FIELDS)
    quadrature_rtol = _real(numerics.get("quadrature_rtol", 1e-9), "pcet.numerics.quadrature_rtol", positive=True)
    quadrature_atol = _real(numerics.get("quadrature_atol", 1e-11), "pcet.numerics.quadrature_atol", positive=True)
    if quadrature_rtol >= 1 or quadrature_atol >= 1:
        raise _error("pcet.numerics quadrature tolerances must be smaller than one")
    convergence = numerics.get("convergence_check", True)
    if type(convergence) is not bool:
        raise _error("pcet.numerics.convergence_check must be a boolean")
    relative = _real(numerics.get("relative_tolerance", 0.01), "pcet.numerics.relative_tolerance", positive=True)
    if relative >= 1:
        raise _error("pcet.numerics.relative_tolerance must be smaller than one")
    absolute = _real(numerics.get("absolute_tolerance_s", 1e-12), "pcet.numerics.absolute_tolerance_s", nonnegative=True)
    output = _table(raw.get("output", {}), "output", {"directory"})
    directory = output.get("directory", "pcet-output")
    if not isinstance(directory, (str, Path)) or not str(directory):
        raise _error("output.directory must be a path")
    model = PCETModelIdentity(coordinate, ox_values, red_values, ox_origin, red_origin)
    resolved = {
        "schema_version": 1, "task": "pcet", "preset": preset, "description": description,
        "temperature_K": temperatures,
        "pcet": {
            "method": "golden_rule", "normalization": "local_first_order", "isotopes": list(isotopes),
            "driving_force_au": driving, "reorganization_au": reorganization, "coupling_au": coupling,
            "electrode": {
                "model": "wide_band", "density_per_hartree": density,
                "potential_reference": reference.strip(), "reference_potential_V": reference_voltage, "potentials_V": voltages,
            },
            "proton": {
                "grid_min": minimum, "grid_max": maximum, "grid_points": points, "states": states,
                "oxidized": oxidized, "reduced": reduced,
            },
            "numerics": {
                "quadrature_rtol": quadrature_rtol, "quadrature_atol": quadrature_atol,
                "convergence_check": convergence,
                "relative_tolerance": relative, "absolute_tolerance_s": absolute,
            },
        },
        "output": {"directory": str((base / directory).resolve())},
        "parallel": {"backend": "serial", "mode": "serial", "trajectory_workers": 1, "trajectory_groups": 1},
        "_input_path": str(Path(input_path).resolve()) if input_path else None,
        "_base_directory": str(base),
        "_warnings": [
            "PCET rates apply to one localized oxidized/reduced complex with one transferring proton, weak vibronic coupling, a classical Gaussian environment, and a wide-band electrode.",
            "The electrode reference is a label: no SHE/RHE conversion, proton-activity correction, constant-potential sampling, or macroscopic current normalization is applied.",
        ],
    }
    return resolved, model
