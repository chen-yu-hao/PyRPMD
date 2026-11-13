"""Explain dimensional input numbers and electronic-state declarations."""
from collections.abc import Mapping
import numbers

from .units import parse_quantity


# Only fields whose input contracts accept unit-bearing quantities belong here.
# Integer budgets, dimensionless CVs and opaque provider parameters are excluded.
_FIELDS = {
    "": {"temperature": "temperature"},
    "dynamics": {"timestep": "time"},
    "sampling": {"timestep": "time", "equilibration": "time", "production": "time",
                 "interval": "time", "save_interval": "time", "sample_every": "time",
                 "gamma_centroid": "frequency"},
    "thermostat": {"gamma_centroid": "frequency", "refresh_every": "time",
                   "mean_refresh_interval": "time"},
    "thermostat_params": {"gamma_centroid": "frequency", "gamma": "frequency", "tau": "time"},
    "system": {"separation": "length", "masses": "mass"},
    "reaction": {"rinf": "length", "Rinf": "length"},
    "collision": {"bmax": "length", "max_time": "time", "energy": "energy"},
    "outcomes": {"exit_distance": "length", "persistence": "time"},
    "output": {"checkpoint_interval": "time"},
    "recrossing": {"equilibration": "time", "parent_interval": "time", "child_time": "time"},
    "parent": {"equilibration": "time", "sample_every": "time", "draw_every": "time"},
    "children": {"duration": "time", "time": "time"},
    "metadynamics": {"height": "energy"}, "opes": {"barrier": "energy"},
    "diffusion": {"fit_range": "time"},
    "tempering": {"temperatures": "temperature", "temperature_min": "temperature", "temperature_max": "temperature"},
    "barostat": {"pressure": "pressure", "time_constant": "time", "tau": "time",
                 "friction": "frequency", "gamma": "frequency", "compressibility": "inverse_pressure"},
    "electronic_friction": {"gamma": "frequency", "tensor": "frequency"},
    "nonadiabatic": {"asymptotic_coupling_tolerance": "energy"},
    "path_integral": {"fd_step": "length", "projection_time": "time"},
    "active_learning": {"force_threshold": "force", "energy_threshold": "energy"},
    "surface": {"reactant_basin": "length", "dividing_surface": "length"},
}
_UNITS = {
    "time": ("autime", "fs"), "temperature": ("K", "K"), "length": ("bohr", "angstrom"),
    "energy": ("hartree", "eV"), "mass": ("me", "amu"), "frequency": ("autime^-1", "fs^-1"),
    "pressure": ("hartree/bohr^3", "bar"), "inverse_pressure": ("bohr^3/hartree", "bar^-1"),
    "force": ("hartree/bohr", "eV/angstrom"), "angle": ("rad", "deg"),
}
_CV_DIMENSIONS = {"cartesian": "length", "distance": "length", "com_distance": "length",
                  "bond_difference": "length", "angle": "angle", "dihedral": "angle",
                  "potential_energy": "energy"}


def _bare_values(value):
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[1], str):
            try:
                parse_quantity(value)
            except ValueError:
                pass  # a mixed array, e.g. [300, "400 K"], is not one quantity
            else:
                return []
        return [number for item in value for number in _bare_values(item)]
    return []


def _warning(path, value, dimension):
    bare = _bare_values(value)
    if not bare:
        return None
    atomic, human = _UNITS[dimension]
    converted = bare[0] / parse_quantity("1 " + human, dimension)
    example = f"{bare[0]:g} {atomic} ({converted:.5g} {human})"
    subject = f"{path} = {value!r}" if isinstance(value, numbers.Real) else f"{path} contains bare numbers"
    return f'{subject} is interpreted in {atomic}; {example}. Write an explicit unit, e.g. "{bare[0]:g} {human}", to choose that unit.'


def input_diagnostics(raw, resolved, system):
    notes = []

    def visit(table, section="", prefix=""):
        if not isinstance(table, Mapping):
            return
        for key, value in table.items():
            path = prefix + key
            dimension = _FIELDS.get(section, {}).get(key)
            if section == "bond_cutoffs":
                dimension = "length"
            if section == "umbrella" and key in ("force_constant", "force_matrix", "kappa", "stiffness") and _bare_values(value):
                convention = "k/T in canonical stiffness units per kelvin; multiplied by temperature_K" if key in ("force_constant", "force_matrix") else "canonical physical stiffness units (hartree per product of CV units)"
                notes.append(f"{path} contains bare numbers interpreted as {convention}; use an explicit unit-bearing stiffness for the selected CV.")
            if dimension:
                warning = _warning(path, value, dimension)
                if warning:
                    notes.append(warning)
            if isinstance(value, Mapping):
                visit(value, key, path + ".")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Mapping):
                        visit(item, section if key in ("regions", "windows") else key, f"{path}[{i}].")
    visit(raw)
    spec = raw.get("system", {})
    if isinstance(spec, Mapping):
        for field, selector in (("geometry", "geometry_unit"), ("cell", "cell_unit")):
            if field in spec and selector not in spec and not (field == "cell" and "geometry_unit" in spec):
                if _bare_values(spec[field]):
                    inline = "symbols" in spec and "geometry" in spec and not isinstance(spec["geometry"], str)
                    unit = "bohr" if inline else "angstrom"
                    notes.append(f"system.{field} contains bare numbers interpreted in {unit}; set system.{selector} explicitly to choose the coordinate unit.")
    # CV-dependent quantities must use the selected CV's units, never guessed
    # from a numeric value. Explicit CV unit declarations already resolve these.
    cvs = resolved.get("cvs", [])
    if isinstance(cvs, Mapping):
        cvs = [dict(value, name=name) for name, value in cvs.items()]
    selected = [cv for cv in cvs if isinstance(cv, Mapping)]
    if len(selected) == 1 and not selected[0].get("unit", selected[0].get("units")):
        dimension = _CV_DIMENSIONS.get(selected[0].get("kind"))
        if dimension:
            for section, fields in {
                "umbrella": ("centers", "start", "stop"), "analysis": ("ranges",),
                "metadynamics": ("sigma",), "opes": ("sigma", "sigma_min"),
                "site_rate": ("basin", "dividing_surface"),
            }.items():
                for field in fields:
                    warning = _warning(section + "." + field, raw.get(section, {}).get(field), dimension)
                    if warning:
                        notes.append(warning)
    if hasattr(system, "atomic_numbers"):
        electrons = int(system.atomic_numbers.sum()) - system.charge
        spin = system.multiplicity - 1
        if electrons < 0 or spin > electrons or (electrons - spin) % 2:
            declaration = "explicit" if isinstance(spec, Mapping) and "multiplicity" in spec else "default"
            notes.append(f"system has {electrons} electrons (charge={system.charge}) but {declaration} multiplicity={system.multiplicity} is incompatible with electron/spin parity. Set system.multiplicity for the intended electronic state; analytic teaching models may not represent an electronic state.")
    return notes
