"""Serial and distributed shared-metadynamics workflow dispatcher."""
from __future__ import annotations

from pathlib import Path

from . import _shared_metadynamics_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from . import _distributed_meta_async as _async
from . import _distributed_meta_async_io as _async_io
from . import _distributed_meta_mpi as _mpi

MPI_KIND = _mpi.MPI_KIND
ASYNC_KIND = _async.ASYNC_KIND


def _mode(config):
    return config.get("metadynamics", {}).get("walkers_mode", "serial")


def _sync_legacy_globals():
    for name in ("Dynamics", "State", "make_potential"):
        value = globals()[name]
        setattr(_serial, name, value)
        setattr(_mpi, name, value)
        setattr(_async, name, value)
        setattr(_async_io, name, value)


def run_shared_metadynamics(config, system, output, *, return_data=False):
    selected = _mode(config)
    _sync_legacy_globals()
    if selected == "serial":
        return _serial.run_shared_metadynamics(
            config, system, output, return_data=return_data)
    if selected == "mpi-sync":
        return _mpi._run_mpi(config, system, output, return_data=return_data)
    if selected == "async-files":
        return _async._run_async(config, system, output, return_data=return_data)
    raise ValueError(
        "metadynamics.walkers_mode must be serial, mpi-sync or async-files")


def reanalyze_shared_metadynamics(config, system, output, *, return_data=False):
    selected = _mode(config)
    _sync_legacy_globals()
    if selected == "serial":
        return _serial.reanalyze_shared_metadynamics(
            config, system, output, return_data=return_data)
    cv, _, options = resolve_metadynamics(config, system)
    if selected == "async-files":
        return _async_io._collect_async(
            config, system, output, cv, options, return_data=return_data)
    if selected == "mpi-sync":
        return _mpi.reanalyze_mpi(
            config, system, output, cv, options, return_data=return_data)
    raise ValueError(
        "metadynamics.walkers_mode must be serial, mpi-sync or async-files")
