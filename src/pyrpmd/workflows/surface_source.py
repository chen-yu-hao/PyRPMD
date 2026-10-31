"""Saved multidimensional FES -> consistent, conditional surface reactive flux.

The source defines the CV mapping, grid domain and static-wall ensemble.
Neither MetaD hills nor umbrella springs act on parent/child dynamics. Parents
sample the constrained wall ensemble; snapshots are retained only inside the
same integration domain used by the conditional PMF. Children are unthermostatted
and use that same wall Hamiltonian. The observable is one selected flux-side
exit, not a sum over all chemical channels.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez as _atomic_npz
from ..checkpoint import load_checkpoint
from ..cv import CVStack, make_cvs
from ..io import file_sha256, fingerprint, jsonable, write_json
from ..potentials.base import PESResult, validate_result
from ..system import System
from ..validation import checked_integer


def source_paths(raw, base=None):
    if not isinstance(raw, dict) or set(raw) - {"directory", "axis", "selection", "data_file", "config_file", "system_file", "summary_file", "manifest_file"}:
        raise ValueError("surface.free_energy accepts directory, axis and optional selection (.npy)")
    if not isinstance(raw.get("directory"), str):
        raise ValueError("surface.free_energy.directory is required")
    root = Path(base or ".")
    directory = (root / raw["directory"]).resolve()
    result = {"directory": str(directory), "axis": checked_integer(raw.get("axis"), "surface.free_energy.axis", 0)}
    for name, relative in {"data_file": "fes.npz", "config_file": "input/resolved.json",
                           "system_file": "input/system.json", "summary_file": "summary.json", "manifest_file": "manifest.json"}.items():
        result[name] = str(directory / relative)
        if name in raw and Path(raw[name]).resolve() != Path(result[name]):
            raise ValueError(f"surface.free_energy.{name} must belong to directory")
        if not Path(result[name]).is_file():
            raise ValueError(f"Missing source FES file: {result[name]}")
    if "selection" in raw:
        result["selection"] = str((root / raw["selection"]).resolve())
        if not Path(result["selection"]).is_file():
            raise ValueError("Missing surface.free_energy.selection file")
    return result


def source_context(config, system, *, verify=False):
    paths = source_paths(config["surface"]["free_energy"])
    source = json.loads(Path(paths["config_file"]).read_text())
    source_system = System.from_dict(json.loads(Path(paths["system_file"]).read_text()))
    if source["task"] not in ("fes", "metadynamics") or source.get("analysis", {}).get("measure", "cv_density") != "cv_density":
        raise ValueError("surface source must be an ordinary CV-density FES or metadynamics run")
    specs = source.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(v, name=k) for k, v in specs.items()]
    table = source.get("metadynamics" if source["task"] == "metadynamics" else "umbrella", {})
    names = table.get("cvs")
    if isinstance(names, str):
        names = [names]
    cv = make_cvs(specs, system, names)
    if any(c.dimension != 1 for c in cv.cvs) or paths["axis"] >= cv.dimension:
        raise ValueError("surface source axis must select one scalar CV component")
    if config.get("umbrella"):
        raise ValueError("surface.free_energy replaces umbrella sampling; omit umbrella")
    if config.get("cvs") and jsonable(config["cvs"]) != jsonable(source["cvs"]):
        raise ValueError("surface.free_energy uses source CV definitions; omit cvs or supply the identical definitions")
    if verify:
        if config["beads"] != source["beads"] or not np.isclose(config["beta"], source["beta"], rtol=1e-13, atol=0):
            raise ValueError("surface FES source temperature or bead count differs")
        if jsonable(config["potential"]) != jsonable(source["potential"]):
            raise ValueError("surface FES source PES specification differs")
        if (tuple(system.symbols) != tuple(source_system.symbols) or system.pbc != source_system.pbc
                or system.fixed_atoms != source_system.fixed_atoms
                or not np.allclose(system.masses, source_system.masses, rtol=1e-13, atol=0)
                or (system.cell is None) != (source_system.cell is None)
                or (system.cell is not None and not np.allclose(system.cell, source_system.cell, rtol=1e-13, atol=1e-13))
                or not np.allclose(system.geometry[list(system.fixed_atoms)], source_system.geometry[list(system.fixed_atoms)], rtol=0, atol=1e-12)):
            raise ValueError("surface FES source cell, species, masses or fixed-atom geometry differs")
        manifest = json.loads(Path(paths["manifest_file"]).read_text())
        from ..propagation import scientific_config
        if manifest.get("physics_fingerprint") not in (fingerprint(source, source_system), fingerprint(scientific_config(source), source_system)):
            raise ValueError("source FES saved configuration does not match its manifest")
        for name, digest in manifest.get("files", {}).items():
            if not Path(name).is_file() or file_sha256(name) != digest:
                raise ValueError(f"source FES input/model file changed or is missing: {name}")
    result = {"paths": paths, "source": source, "cv": cv,
              "coordinate": CVStack([cv.cvs[paths["axis"]]])}
    if verify:
        source_domain(result)
    return result


@lru_cache(maxsize=8)
def _domain(data_file, selection_file, data_mtime, selection_mtime):
    # The run manifest hashes these files. Cache only small edges and a Boolean
    # domain per worker, not every replica's large free-energy arrays.
    with np.load(data_file, allow_pickle=False) as data:
        edges = [data[f"edges_{i}"].copy() for i in range(3) if f"edges_{i}" in data]
    shape = tuple(len(e) - 1 for e in edges)
    selection = np.ones(shape, bool) if selection_file is None else np.load(selection_file, allow_pickle=False)
    if not isinstance(selection, np.ndarray):
        if hasattr(selection, "close"):
            selection.close()
        raise ValueError("surface channel selection must be a Boolean .npy array, not an archive")
    if selection.shape != shape or selection.dtype.kind != "b":
        raise ValueError("surface channel selection must be Boolean with the complete source FES shape")
    if not selection.any():
        raise ValueError("surface channel selection is empty")
    return edges, selection


def source_domain(context):
    p = context["paths"]
    selected = p.get("selection")
    edges, selection = _domain(p["data_file"], selected, Path(p["data_file"]).stat().st_mtime_ns,
                              None if selected is None else Path(selected).stat().st_mtime_ns)
    if len(edges) != context["cv"].dimension:
        raise ValueError("source FES grid and CV dimensions differ")
    return edges, selection


def parent_selector(config, system):
    """Accept constrained configurations in exactly the PMF integration domain."""
    if config.get("task") != "surface_rate" or not config.get("surface", {}).get("free_energy"):
        return lambda q: True
    context = source_context(config, system)
    edges, selection = source_domain(context)

    def selected(q):
        values = context["cv"].value(q)
        indices = []
        for value, edge, period in zip(values, edges, context["cv"].periods):
            if period is not None:
                value = (value - edge[0]) % period + edge[0]
            if value < edge[0] or value > edge[-1]:
                return False
            indices.append(min(len(edge) - 2, int(np.searchsorted(edge, value, side="right") - 1)))
        return bool(selection[tuple(indices)])
    return selected


class _WallHamiltonian:
    """Add a full-ring static bias after the physical bead-parallel PES call."""
    def __init__(self, potential, walls):
        self.potential, self.walls = potential, walls

    def evaluate(self, q):
        physical = validate_result(self.potential.evaluate(q), q.shape)
        energy, force = self.walls.energy_force(q)
        return PESResult(physical.energy + energy / len(q), physical.forces + force)


def source_walls(config, system):
    """Static per-ring bias shared by source PMF, parents and children."""
    if config.get("task") != "surface_rate" or not config.get("surface", {}).get("free_energy"):
        return None
    from .metadynamics import resolve_walls
    context = source_context(config, system)
    return resolve_walls(context["source"], system)


def source_hamiltonian(provider, config, system):
    walls = source_walls(config, system)
    return provider if walls is None else _WallHamiltonian(provider, walls)


def source_pmf_and_parents(config, system, output, options):
    from ..analysis.projection import marginalize_fes
    from ..analysis.surface import _basin_statistics, _UnsupportedPMF
    from .surface import surface_definition

    definition = surface_definition(config, system)
    context = source_context(config, system, verify=True)
    edges, domain = source_domain(context)
    paths, axis = context["paths"], context["paths"]["axis"]
    with load_checkpoint(paths["data_file"]) as data:
        projected = marginalize_fes(data["free_energy"], edges, config["beta"], axes=axis,
                                   mask=data["mask"], selection=domain)
    grid = (edges[axis][1:] + edges[axis][:-1]) / 2
    norm, reason = None, None
    try:
        norm = _basin_statistics(grid, projected["free_energy"], config["beta"], definition["mass_metric_au"],
                                 definition["reactant_basin_bohr"], definition["dividing_surface_bohr"], projected["mask"])
    except _UnsupportedPMF as error:
        reason = str(error)
    pmf = {"method": "projected_surface_density", "measure": "cv_density", "ready": reason is None,
           "status": "ready" if reason is None else reason, "failure_reason": reason,
           "xi_star": definition["dividing_surface_bohr"],
           "k_qtst_s_inv": None if norm is None else norm["k_qtst_s_inv"],
           "k_qtst_au": None if norm is None else norm["k_qtst_au"],
           "normalization": norm, "surface": {k: v for k, v in definition.items() if k != "coordinate"},
           "source": paths, "source_sha256": {k: file_sha256(v) for k, v in paths.items() if k.endswith("_file") or k == "selection"},
           "pmf_bootstrap_qtst": [],
           "bootstrap": {"block_plan": {"status": "unavailable"}},
           "warnings": ["Projected finite-time FES; no calibrated PMF uncertainty is available.",
                        "Parent snapshots are conditioned on the same source-grid/channel domain as the PMF.",
                        "Static source walls act on parents and NVE children; MetaD hills and umbrella springs do not.",
                        "This is a selected flux-side exit; multiple chemical exits are not automatically added."]}
    out = Path(output)
    _atomic_npz(out / "surface_pmf.npz", xi=grid, free_energy=projected["free_energy"],
                mask=projected["mask"], coverage=projected["coverage"], bootstrap_qtst_s_inv=np.array([]))
    np.savetxt(out / "surface_pmf.csv", np.column_stack([grid, projected["free_energy"], projected["mask"]]),
               delimiter=",", comments="", header="coordinate_bohr,free_energy_hartree,masked")
    pmf["format_version"] = 2
    write_json(out / "surface_pmf.json", pmf)
    tasks = [] if not pmf["ready"] else [
        {"config": config, "system": system, "options": options, "chain": chain,
         "xi_star": pmf["xi_star"], "initial_q": np.repeat(system.geometry[None], config["beads"], axis=0), "output": str(out)}
        for chain in range(options["parent_replicas"])]
    return pmf, tasks
