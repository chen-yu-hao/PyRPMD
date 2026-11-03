"""Task validation extended for distributed multiple-walker FES modes."""
from __future__ import annotations

from . import _input_tasks_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_base_validate_task_sections = _base.validate_task_sections
_WALKER_MODES = ("serial", "mpi-sync", "async-files")


def _validate_walker_mode(data, section, method):
    table = data.get(section, {})
    if "walkers_mode" not in table:
        return
    mode = table["walkers_mode"]
    if not isinstance(mode, str) or mode not in _WALKER_MODES:
        raise InputError(
            f"{section}.walkers_mode must be serial, mpi-sync or async-files")
    if mode == "serial":
        return
    if data.get("task") not in ("fes", section) or method != section:
        raise InputError(
            f"{section}.walkers_mode='{mode}' requires the {section} FES workflow")
    replicas = data.get("sampling", {}).get("replicas", 1)
    if _integer(replicas, "sampling.replicas") < 2:
        raise InputError(
            f"{section}.walkers_mode='{mode}' requires sampling.replicas >= 2")
    if section == "metadynamics":
        if table.get("shared_walkers") is not True:
            raise InputError(
                "distributed metadynamics requires metadynamics.shared_walkers=true")
        if table.get("multiple_walkers", False):
            raise InputError(
                "distributed shared_walkers cannot combine metadynamics.multiple_walkers")
        if table.get("grid") is not None:
            raise InputError("distributed shared metadynamics does not support a bias grid")
    else:
        if table.get("adaptive_sigma", False):
            raise InputError("distributed OPES requires fixed sigma; adaptive_sigma is unsupported")
        if "freeze_after_steps" in table or "freeze_equilibration_steps" in table:
            raise InputError("distributed OPES does not support frozen-bias production")
        analysis = data.get("analysis", {})
        if analysis.get("method", "kde") != "kde":
            raise InputError("distributed OPES currently requires analysis.method='kde'")
        if analysis.get("measure", "cv_density") != "cv_density":
            raise InputError("distributed OPES currently requires analysis.measure='cv_density'")


def validate_task_sections(d):
    result = _base_validate_task_sections(d)
    task = d["task"]
    method = fes_method(d) if task in ("fes", "metadynamics", "opes") else None
    _validate_walker_mode(d, "metadynamics", method)
    _validate_walker_mode(d, "opes", method)
    return result
