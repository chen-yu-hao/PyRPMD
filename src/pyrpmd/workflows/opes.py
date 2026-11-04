"""Serial and distributed OPES workflow dispatcher."""
from __future__ import annotations

from . import _opes_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from . import _distributed_opes_async as _async
from . import _distributed_opes_async_io as _async_io
from . import _distributed_opes_common as _common
from . import _distributed_opes_mpi as _mpi
from ..checkpoint import atomic_savez as _atomic_savez

# The split helpers resolve checkpoint writes at call time. Inject the shared
# atomic writer here so import-only modules stay side-effect free.
_async.atomic_savez = _atomic_savez
_mpi.atomic_savez = _atomic_savez


class _DistributedFESSampler:
    """Preserve the serial history-dependent Andersen refresh clock."""

    def __init__(self, delegate, interval):
        self._delegate = delegate
        self._interval = interval

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def step(self, dynamics, state, bias):
        if (self._delegate.thermostat == "andersen"
                and state.step % self._interval == 0):
            state.p[:] = dynamics.sample_momenta()
        return self._delegate.step(dynamics, state, bias)


_original_fes_sampler = fes_sampler


def _distributed_fes_sampler(config):
    return _DistributedFESSampler(
        _original_fes_sampler(config),
        config.get("sampling", {}).get("andersen_interval", 10),
    )


_async.fes_sampler = _distributed_fes_sampler
_mpi.fes_sampler = _distributed_fes_sampler

OPES_FIELDS.add("walkers_mode")
_original_resolve_opes = resolve_opes


def resolve_opes(config, system):
    result = _original_resolve_opes(config, system)
    _, _, options = result
    mode = config.get("opes", {}).get("walkers_mode", "serial")
    if mode not in ("serial", "mpi-sync", "async-files"):
        raise ValueError("opes.walkers_mode must be serial, mpi-sync or async-files")
    if mode != "serial":
        _common._validate_shared_options(config, options)
    return result


def validate_opes(config, system):
    resolve_opes(config, system)


def _mode(config):
    return config.get("opes", {}).get("walkers_mode", "serial")


def _sync_legacy_globals():
    for name in ("Dynamics", "State", "make_task_potential", "_new_bias"):
        value = globals()[name]
        setattr(_serial, name, value)
        setattr(_mpi, name, value)
        setattr(_async, name, value)
        setattr(_async_io, name, value)
        setattr(_common, name, value)


def _sample_replica(task):
    """Preserve the historical monkeypatch surface for direct worker tests."""
    _sync_legacy_globals()
    return _serial._sample_replica(task)


def run_opes(config, system, output, executor=None, return_data=False):
    selected = _mode(config)
    _sync_legacy_globals()
    if selected == "serial":
        return _serial.run_opes(
            config, system, output, executor=executor, return_data=return_data)
    if executor is not None:
        raise ValueError(
            "distributed OPES owns walker scheduling and does not accept an executor")
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "opes")
    validate_opes(config, system)
    if selected == "mpi-sync":
        return _mpi._run_mpi(config, system, output, return_data=return_data)
    if selected == "async-files":
        return _async._run_async(config, system, output, return_data=return_data)
    raise ValueError("opes.walkers_mode must be serial, mpi-sync or async-files")


def reanalyze_opes(config, system, output, return_data=False):
    selected = _mode(config)
    _sync_legacy_globals()
    if selected == "serial":
        return _serial.reanalyze_opes(
            config, system, output, return_data=return_data)
    cv, _, options = resolve_opes(config, system)
    if selected == "async-files":
        return _async_io._collect_async(
            config, system, output, cv, options, return_data=return_data)
    if selected == "mpi-sync":
        return _mpi.reanalyze_mpi(
            config, system, output, cv, options, return_data=return_data)
    raise ValueError("opes.walkers_mode must be serial, mpi-sync or async-files")
