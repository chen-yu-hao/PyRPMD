"""Metadynamics workflow with opt-in distributed-walker registration.

The implementation remains byte-for-byte in ``_metadynamics_base``.  This
module extends its strict field allow-list and installs read-only validation
for the new explicit modes before exposing the same module object. Historical
monkeypatch points and serial checkpoint behavior therefore remain unchanged.
"""
from __future__ import annotations

import sys

from . import _metadynamics_base as _base

_base.METADYNAMICS_FIELDS.add("walkers_mode")
_original_resolve_metadynamics = _base.resolve_metadynamics


def _resolve_metadynamics(config, system):
    result = _original_resolve_metadynamics(config, system)
    table = config.get("metadynamics", {})
    mode = table.get("walkers_mode", "serial")
    if mode not in ("serial", "mpi-sync", "async-files"):
        raise ValueError(
            "metadynamics.walkers_mode must be serial, mpi-sync or async-files")
    if mode != "serial":
        if table.get("shared_walkers") is not True:
            raise ValueError(
                "distributed metadynamics requires shared_walkers=true")
        if table.get("multiple_walkers", False):
            raise ValueError(
                "distributed shared_walkers cannot combine multiple_walkers")
        if table.get("grid") is not None:
            raise ValueError(
                "distributed shared metadynamics does not support a bias grid")
        replicas = config.get("sampling", {}).get("replicas", 1)
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 2:
            raise ValueError(
                "distributed metadynamics requires sampling.replicas >= 2")
    return result


_base.resolve_metadynamics = _resolve_metadynamics
sys.modules[__name__] = _base
