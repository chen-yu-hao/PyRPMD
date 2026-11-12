"""Import trusted RPMDrate rate scripts without loading their PES modules.

The recorded input API follows chen-yu-hao/RPMDrate at UPSTREAM_COMMIT.
This is a settings converter, not a Python sandbox or a trajectory converter.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
import inspect
import json
from numbers import Integral
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from .input import InputError, JobSpec
from .units import parse_quantity, time_steps, unit_factor


UPSTREAM_COMMIT = "710670c6f753b1159c0147c817e82998c2cc8486"
_REQUIRED = inspect.Parameter.empty
_CALLS = {
    "reactants": ("atoms reactant1Atoms reactant2Atoms Rinf", {}),
    "transitionState": ("geometry formingBonds breakingBonds", {}),
    "equivalentTransitionState": ("formingBonds breakingBonds", {}),
    "thermostat": ("type", {"samplingTime": None, "A": None, "C": None}),
    "generateUmbrellaConfigurations": ("dt evolutionTime xi_list kforce", {}),
    "Window": ("", dict.fromkeys(("xi", "kforce", "trajectories", "equilibrationTime",
                                   "evolutionTime", "xi_range"))),
    "conductUmbrellaSampling": ("dt windows", {"saveTrajectories": False}),
    "computePotentialOfMeanForce": ("", {"windows": None, "xi_min": None, "xi_max": None, "bins": 5000}),
    "computeRecrossingFactor": (
        "dt equilibrationTime childTrajectories childSamplingTime childrenPerSampling childEvolutionTime",
        {"xi_current": None, "saveParentTrajectory": False, "saveChildTrajectories": False}),
    "computeRateCoefficient": ("", {}),
}


def _plain(value):
    if isinstance(value, SimpleNamespace):
        return _plain(vars(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise InputError(f"Cannot record RPMDrate parameter of type {type(value).__name__}")


@contextmanager
def _restore_imports():
    """Restore module bindings even after a failed trusted script or preflight.

No PES stub is installed in sys.modules. Snapshotting also removes modules
loaded by permitted helper imports and restores bindings changed by the input.
Like execution of a legacy input, conversion must run without concurrent imports.
"""
    registry, modules, path = sys.modules, sys.modules.copy(), sys.path[:]
    try:
        yield
    finally:
        sys.modules = registry
        for name in set(sys.modules) - modules.keys():
            del sys.modules[name]
        sys.modules.update(modules)
        sys.path[:] = path


class _PESReference:
    def __init__(self, module, symbol):
        self.module, self.symbol = module, symbol

    def __call__(self, *args, **kwargs):
        raise InputError(f"PES call forbidden during conversion: {self.module}.{self.symbol}")


class _PESModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        for symbol in ("get_potential", "initialize_potential"):
            setattr(self, symbol, _PESReference(name, symbol))


def _execute(path, temperature, beads):
    records, imports = [], []

    def recorder(name):
        required, defaults = _CALLS[name]
        signature = inspect.Signature([
            inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=defaults.get(key, _REQUIRED))
            for key in [*required.split(), *defaults]
        ])

        def record(*args, **kwargs):
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as error:
                raise InputError(f"{name}: {error}") from error
            bound.apply_defaults()
            params = dict(bound.arguments)
            records.append({"id": len(records), "call": name,
                            "parameters": _plain(params), "targets": {}, "conversions": []})
            if name == "Window":
                return SimpleNamespace(**params)
        return record

    def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        fromlist = fromlist or ()
        root = name.split(".")[0]
        if level or root == "rpmdrate":
            raise InputError("Relative imports and importing the rpmdrate package are unsupported")
        if root in ("importlib", "runpy") or "*" in fromlist:
            raise InputError("Dynamic and wildcard imports are unsupported; import PES entry points explicitly")
        if root == "numpy" or root in sys.stdlib_module_names:
            return builtins.__import__(name, globals, locals, fromlist, level)
        imports.append({"module": name, "symbols": list(fromlist)})
        module = _PESModule(name)
        if fromlist:
            if set(fromlist) - {"get_potential", "initialize_potential"}:
                raise InputError(f"Unsupported import from {name}: only PES entry points are recorded")
            return module
        parts = name.split(".")
        for index in range(len(parts) - 1, 0, -1):
            parent = ModuleType(".".join(parts[:index]))
            setattr(parent, parts[index], module)
            module = parent
        return module

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    if any((isinstance(node, ast.Name) and "drpmd" in node.id.lower()) or
           (isinstance(node, ast.Constant) and node.value == "drpmd") for node in ast.walk(tree)):
        raise InputError("dRPMD inputs are not supported; this command imports rate inputs only")
    namespace = {name: recorder(name) for name in _CALLS}
    namespace.update(__file__=str(path), __name__="__rpmdrate_input__",
                     __builtins__={**vars(builtins), "__import__": import_hook},
                     numpy=np, T=temperature, Nbeads=beads)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, records, imports


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    try:
        result = parse_quantity(value, "dimensionless")
    except (TypeError, ValueError) as error:
        raise InputError(f"{name}: {error}") from error
    if positive and result <= 0:
        raise InputError(f"{name} must be positive")
    return result


def _quantity(value, dimension, name, *, allow_zero=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2 or not isinstance(value[1], str):
        raise InputError(f"{name} requires a (value, unit) pair")
    result = parse_quantity(value, dimension)
    if result < 0 or (not allow_zero and result == 0):
        raise InputError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return result


def _bonds(value, natoms, name):
    result = []
    for pair in value:
        if len(pair) != 2:
            raise InputError(f"{name} requires atom pairs")
        row = [_integer(atom, name) - 1 for atom in pair]
        if max(row) >= natoms or row[0] == row[1]:
            raise InputError(f"{name}: out-of-range index or self bond")
        if tuple(sorted(row)) in {tuple(sorted(old)) for old in result}:
            raise InputError(f"{name}: duplicate bond")
        result.append(row)
    if not result:
        raise InputError(f"{name} cannot be empty")
    return result


def _permutation(base, equivalent, symbols):
    mapping = {}
    for name in ("formingBonds", "breakingBonds"):
        pairs = _bonds(equivalent[name], len(symbols), name)
        if len(pairs) != len(base[name]):
            raise InputError("Equivalent transition state must preserve the number of bonds")
        for old, new in zip(base[name], pairs):
            for left, right in zip(old, new):
                for a, b in ((left, right), (right, left)):
                    if a in mapping and mapping[a] != b:
                        raise InputError("Inconsistent equivalent transition-state atom mapping")
                    mapping[a] = b
    permutation = [mapping.get(i, i) for i in range(len(symbols))]
    if sorted(permutation) != list(range(len(symbols))) or any(
            symbols[i] != symbols[j] for i, j in enumerate(permutation)):
        raise InputError("Equivalent transition-state permutation must preserve atomic identities")
    return permutation


def _convert(path, temperature, beads, pes_module):
    namespace, calls, imports = _execute(path, temperature, beads)
    report = {"format": "PyRPMD.rpmdrate-import.v1", "source": str(path),
              "upstream_commit": UPSTREAM_COMMIT, "temperature_K": temperature, "beads": beads,
              "calls": calls, "pes_imports": imports, "unmapped": [], "defaults": {}}

    def unmapped(record, setting, reason, replacement):
        report["unmapped"].append({"call_id": record["id"], "setting": setting,
                                   "reason": reason, "replacement": replacement})

    def one(name, required=True):
        found = [record for record in calls if record["call"] == name]
        if len(found) > 1:
            raise InputError(f"Duplicate {name} call; only one stage is supported")
        if not found and required:
            raise InputError(f"Missing {name} call in rate input")
        return found[0] if found else None

    reactants, transition = one("reactants"), one("transitionState")
    r, ts = reactants["parameters"], transition["parameters"]
    symbols = r["atoms"]
    if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
        raise InputError("reactants.atoms must be a nonempty list of atomic symbols")
    natoms = len(symbols)
    fragments = [[_integer(atom, "reactant atom index") - 1 for atom in r[key]]
                 for key in ("reactant1Atoms", "reactant2Atoms")]
    if not all(fragments) or sorted(sum(fragments, [])) != list(range(natoms)):
        raise InputError("Reactant fragments must partition all atoms without duplicate or out-of-range indices")
    rinf = _quantity(r["Rinf"], "length", "Rinf")
    geometry, unit = ts["geometry"]
    geometry = np.asarray(geometry, dtype=float) * unit_factor(unit, "length")
    if geometry.shape != (natoms, 3) or not np.isfinite(geometry).all():
        raise InputError("transitionState.geometry must contain one finite xyz row per atom")
    base = {key: _bonds(ts[key], natoms, key) for key in ("formingBonds", "breakingBonds")}
    if len(base["formingBonds"]) != len(base["breakingBonds"]):
        raise InputError("Forming and breaking bonds must be paired")
    paths = []
    for record in calls:
        if record["call"] == "equivalentTransitionState":
            perm = _permutation(base, record["parameters"], symbols)
            if perm == list(range(natoms)) or perm in paths:
                raise InputError("Duplicate equivalent transition state")
            paths.append(perm)
            record["targets"] = {f"reaction.equivalent_paths[{len(paths)-1}]": perm}
            record["conversions"] = ["Ordered bond correspondences define a 0-based atom permutation; symmetry_factor=1"]
    if paths and namespace.get("symmetry_factor", 1) != 1:
        raise InputError("equivalent_paths already count exchange paths; symmetry_factor must be 1")
    potential = namespace.get("get_potential")
    initializer = namespace.get("initialize_potential")
    if not isinstance(potential, _PESReference):
        raise InputError("get_potential must be an imported PES entry point; local PES functions are unsupported")
    if initializer is not None and (not isinstance(initializer, _PESReference) or initializer.module != potential.module):
        raise InputError("initialize_potential must be imported from the same module as get_potential")
    module = str(Path(pes_module).expanduser().resolve()) if pes_module else potential.module
    if not pes_module:
        stem = path.parent.joinpath(*module.split("."))
        candidates = [p for p in [stem.with_suffix(".py"), *stem.parent.glob(stem.name + "*.so"),
                                  *stem.parent.glob(stem.name + "*.pyd")] if p.is_file()]
        if len(candidates) == 1:
            module = str(candidates[0])
        elif len(candidates) > 1:
            raise InputError("Multiple PES modules found; select one with --pes-module")
    data = {"schema_version": 2, "task": "rate", "temperature": f"{temperature:.17g} K", "beads": beads,
            "seed": _integer(namespace.get("random_seed", 0), "random_seed", 0),
            "system": {"symbols": symbols, "geometry": geometry.tolist(), "geometry_unit": "bohr"},
            "potential": {"provider": "rpmdrate", "module": module, "function": potential.symbol,
                          "initializer": initializer.symbol if initializer else ""},
            "reaction": {"definition": "explicit", "reactant_fragments": fragments,
                         "forming_bonds": base["formingBonds"], "breaking_bonds": base["breakingBonds"],
                         "equivalent_paths": paths, "symmetry_factor": 1.0, "rinf": f"{rinf:.17g} bohr"},
            "preparation": {"ts_action": "validate", "check_channel": False},
            "dynamics": {"integrator": "cayley-bcb"}, "rate": {}, "umbrella": {"coordinate": "xi"}}
    report["potential"] = data["potential"].copy()
    report["pes_entry_points"] = {"get_potential": {"module": potential.module, "symbol": potential.symbol},
                                  "initialize_potential": ({"module": initializer.module, "symbol": initializer.symbol}
                                                           if initializer else None)}
    report["defaults"] = {"seed": data["seed"], "dynamics.integrator": "cayley-bcb",
                          "preparation": data["preparation"], "reaction.symmetry_factor": 1.0,
                          "system.charge": 0, "system.multiplicity": 1}
    reactants["targets"] = {"system.symbols": symbols, "reaction.reactant_fragments": fragments,
                           "reaction.rinf": data["reaction"]["rinf"]}
    reactants["conversions"] = ["1-based atom indices -> 0-based", f"Rinf -> {rinf:.17g} bohr"]
    transition["targets"] = {"system.transition_state": "ts.xyz", "reaction.forming_bonds": base["formingBonds"],
                             "reaction.breaking_bonds": base["breakingBonds"]}
    transition["conversions"] = [f"Geometry {unit} -> bohr; atom order and orientation retained", "1-based bonds -> 0-based"]

    sampling = one("conductUmbrellaSampling")
    sp = sampling["parameters"]
    dt = _quantity(sp["dt"], "time", "conductUmbrellaSampling.dt")
    data["dynamics"]["timestep"] = f"{dt:.17g} autime"
    sampling["targets"] = {"dynamics.timestep": data["dynamics"]["timestep"], "umbrella.windows": "explicit records"}
    if sp["saveTrajectories"]:
        unmapped(sampling, "saveTrajectories", "RPMDrate XYZ trajectory layout is not imported", "PyRPMD default sample/checkpoint output")
    if not sp["windows"]:
        raise InputError("conductUmbrellaSampling.windows cannot be empty")
    windows = []
    for index, window in enumerate(sp["windows"]):
        xi = _number(window["xi"], "Window.xi")
        kforce = _number(window["kforce"], "Window.kforce", positive=True)
        count = _integer(window["trajectories"], "Window.trajectories")
        eq = _quantity(window["equilibrationTime"], "time", "Window.equilibrationTime", allow_zero=True)
        prod = _quantity(window["evolutionTime"], "time", "Window.evolutionTime")
        budget = {"equilibration": {"steps": time_steps(eq, dt, allow_zero=True)},
                  "production": {"steps": time_steps(prod, dt)}, "sample_every": {"steps": 1},
                  "chains_per_window": count}
        windows.append({"center": xi, "force_constant": kforce / temperature, "sampling": budget})
        if window.get("xi_range") is not None:
            unmapped(sampling, f"windows[{index}].xi_range", "No equivalent trajectory rejection boundary", "No xi_range rejection")
        report.setdefault("windows", []).append({"index": index, "xi": xi, "kforce_hartree": kforce,
                                                 "coefficient_hartree_per_K": kforce / temperature, **budget})
    data["umbrella"]["windows"] = windows
    for record in calls:
        if record["call"] == "Window":
            indices = [i for i, w in enumerate(sp["windows"]) if w == record["parameters"]]
            record["targets"] = {f"umbrella.windows[{i}]": windows[i] for i in indices}
            record["conversions"] = ["force_constant = kforce / T; resolved K = force_constant * T hartree; no bead factor",
                                     "trajectories -> chains_per_window; time / sampling dt -> exact steps"]
    if any(w["sampling"]["chains_per_window"] > 1 for w in windows):
        unmapped(sampling, "windows.trajectories", "RPMDrate reuses each previous trajectory's final geometry; PyRPMD uses independent chains",
                 "Preserve trajectory count and per-trajectory budgets as chains_per_window")
    generator = one("generateUmbrellaConfigurations", required=False)
    if generator:
        gp = generator["parameters"]
        _quantity(gp["dt"], "time", "generateUmbrellaConfigurations.dt")
        _quantity(gp["evolutionTime"], "time", "generateUmbrellaConfigurations.evolutionTime")
        unmapped(generator, "generateUmbrellaConfigurations", "No schema-2 stage for RPMDrate's sequential single-bead window seeding",
                 "Each PyRPMD window starts from the supplied TS and uses its own declared equilibration")

    thermostat = one("thermostat")
    th = thermostat["parameters"]
    clock = {"type": "andersen", "scheme": "global_refresh", "schedule": "fixed", "refresh_every": {"steps": 10}}
    if str(th["type"]).lower() != "andersen":
        unmapped(thermostat, "thermostat", "Rate umbrella and constrained-parent sampling only support Andersen; GLE matrices cannot be used",
                 {"umbrella.sampling.thermostat": clock, "recrossing.parent.thermostat": clock})
    elif th["samplingTime"] is None or _quantity(th["samplingTime"], "time", "thermostat.samplingTime", allow_zero=True) == 0:
        unmapped(thermostat, "samplingTime", "RPMDrate defaults to floor(sqrt(stage steps)) separately in equilibration, production and parent segments",
                 {"umbrella.sampling.thermostat": clock, "recrossing.parent.thermostat": clock})
    else:
        clock["refresh_every"] = {"steps": time_steps(_quantity(th["samplingTime"], "time", "thermostat.samplingTime"), dt)}
        if any(w["sampling"]["equilibration"]["steps"] % clock["refresh_every"]["steps"] for w in windows):
            unmapped(thermostat, "samplingTime.phase", "RPMDrate restarts the thermostat clock at each stage boundary", "PyRPMD clock continues through equilibration and production")
    if str(th["type"]).lower() == "andersen" and (th["A"] is not None or th["C"] is not None):
        unmapped(thermostat, "A/C", "GLE matrices are inactive with Andersen", "Matrices omitted")
    data["umbrella"]["sampling"] = {"thermostat": clock}
    thermostat["targets"] = {"umbrella.sampling.thermostat": clock}

    pmf = one("computePotentialOfMeanForce")
    pp = pmf["parameters"]
    if pp["windows"] is not None and pp["windows"] != sp["windows"]:
        unmapped(pmf, "windows", "Separate analysis subsets are unsupported", "Analyze all sampled windows")
    bins = _integer(pp["bins"], "computePotentialOfMeanForce.bins", 20)
    lower = windows[0]["center"] if pp["xi_min"] is None else _number(pp["xi_min"], "xi_min")
    upper = windows[-1]["center"] if pp["xi_max"] is None else _number(pp["xi_max"], "xi_max")
    if lower >= upper:
        raise InputError("computePotentialOfMeanForce requires xi_min < xi_max")
    data["umbrella"]["analysis"] = {"estimator": "ui", "integration_points": bins, "bins": bins,
                                    "ranges": [[lower, upper]]}
    pmf["targets"] = {"umbrella.analysis": data["umbrella"]["analysis"]}
    pmf["conversions"] = ["RPMDrate bins is the UI integration-grid point count; ranges also describes the PyRPMD histogram"]
    grid_max = max(w["center"] for w in windows)
    if lower != 0 or upper != grid_max:
        unmapped(pmf, "xi_min/xi_max", "PyRPMD UI integrates from xi=0 to the largest window center; histogram ranges do not change this grid",
                 {"ui_range": [0.0, grid_max], "histogram_range": [lower, upper]})

    recrossing = one("computeRecrossingFactor", required=False)
    if recrossing:
        rp = recrossing["parameters"]
        rdt = _quantity(rp["dt"], "time", "computeRecrossingFactor.dt")
        if not np.isclose(rdt, dt, rtol=1e-12, atol=0):
            unmapped(recrossing, "dt", "This importer emits one dynamics timestep", {"dynamics.timestep": data["dynamics"]["timestep"]})
        total = _integer(rp["childTrajectories"], "childTrajectories")
        children = _integer(rp["childrenPerSampling"], "childrenPerSampling", 2)
        if children % 2 or total % children:
            raise InputError("childrenPerSampling must be even and divide childTrajectories exactly")
        eq = _quantity(rp["equilibrationTime"], "time", "recrossing.equilibrationTime", allow_zero=True)
        draw = _quantity(rp["childSamplingTime"], "time", "childSamplingTime")
        duration = _quantity(rp["childEvolutionTime"], "time", "childEvolutionTime")
        data["recrossing"] = {
            "parent": {"chains": 1, "configurations_per_chain": total // children,
                       "equilibration": f"{eq:.17g} autime", "draw_every": f"{draw:.17g} autime", "thermostat": clock},
            "children": {"method": "rpmd", "pairs_per_configuration": children // 2, "duration": f"{duration:.17g} autime"}}
        recrossing["targets"] = {"recrossing": data["recrossing"]}
        recrossing["conversions"] = ["parent configurations = childTrajectories / childrenPerSampling; pairs = childrenPerSampling / 2"]
        report["defaults"]["recrossing.first_draw"] = "PyRPMD draws after one draw_every interval following equilibration; RPMDrate draws immediately"
        if rp["xi_current"] is not None:
            data["rate"]["dividing_surface"] = _number(rp["xi_current"], "xi_current")
            recrossing["targets"]["rate.dividing_surface"] = data["rate"]["dividing_surface"]
        else:
            recrossing["conversions"].append("xi_current=None -> select the PMF maximum")
        for key in ("saveParentTrajectory", "saveChildTrajectories"):
            if rp[key]:
                unmapped(recrossing, key, "RPMDrate XYZ trajectory layout is not imported", "PyRPMD default recrossing output")
    else:
        data["rate"]["recrossing"] = False
    rate = one("computeRateCoefficient", required=False)
    if rate:
        rate["targets"] = {"task": "rate"}
    else:
        unmapped(pmf, "computeRateCoefficient", "No rate-coefficient call was requested", "task='rate' also computes the rate coefficient")
    for record in calls:
        record["status"] = "unmapped" if any(item["call_id"] == record["id"] for item in report["unmapped"]) else "mapped"
    # Resolve through the real schema before publishing anything. This does not
    # load/evaluate the PES, and uses inline geometry until ts.xyz is published.
    config, system = JobSpec.from_mapping(data, base_directory=path.parent).resolve()
    from .runner import prepare
    preview = prepare(JobSpec.from_mapping(data, base_directory=path.parent), numerical=False)
    report["resolved_reaction"] = _plain(config["reaction"])
    report["geometry_bohr"] = system.geometry.tolist()
    report["defaults"]["system.masses_au"] = system.masses.tolist()
    report["resolved_windows"] = _plain(preview["umbrella_preparation"]["window_plan"])
    return data, report


def _toml(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml(item)}" for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def import_rpmdrate(input_path, *, temperature, beads, output, pes_module=None, allow_unmapped=False):
    """Convert a trusted script, validating all settings before writing files.

On unsupported settings, the ValueError includes every unmapped item and the
destination is untouched. With allow_unmapped, each replacement is recorded in
both import_report.json and rate.inp. This API does not run dynamics or a PES.
"""
    path, destination = Path(input_path).resolve(), Path(output).resolve()
    temperature = _number(temperature, "temperature", positive=True)
    beads = _integer(beads, "beads")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise InputError("Output directory must be empty or nonexistent")
    with _restore_imports():
        data, report = _convert(path, temperature, beads, pes_module)
        if report["unmapped"] and not allow_unmapped:
            details = "\n".join(f"- {item['setting']}: {item['reason']}; replacement: {json.dumps(item['replacement'])}"
                                for item in report["unmapped"])
            raise InputError("Unmapped RPMDrate settings (no files written; review before using --allow-unmapped):\n" + details)
        header = ["# Converted from trusted RPMDrate input; see import_report.json."]
        for item in report["unmapped"]:
            header.append("# UNMAPPED: " + json.dumps(item, ensure_ascii=True))
        geometry = data["system"].pop("geometry")
        symbols = data["system"].pop("symbols")
        data["system"]["transition_state"] = "ts.xyz"
        lines = header + [f"{key} = {_toml(value)}" for key, value in data.items() if not isinstance(value, dict)]
        for name, table in data.items():
            if isinstance(table, dict):
                lines += ["", f"[{name}]", *[f"{key} = {_toml(value)}" for key, value in table.items()]]
        xyz = [str(len(symbols)), "RPMDrate transition state; coordinates in bohr (declared in rate.inp)"]
        xyz += [symbol + " " + " ".join(format(x, ".17g") for x in row) for symbol, row in zip(symbols, geometry)]
        report.update(allow_unmapped=bool(allow_unmapped), all_settings_mapped=not report["unmapped"],
                      output=str(destination))
        contents = {"rate.inp": "\n".join(lines) + "\n", "ts.xyz": "\n".join(xyz) + "\n",
                    "import_report.json": json.dumps(report, indent=2, allow_nan=False) + "\n"}
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            (destination / name).write_text(content, encoding="utf-8")
    return report
