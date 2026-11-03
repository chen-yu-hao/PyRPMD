"""Input schema with opt-in distributed multiple-walker fields."""
from __future__ import annotations

from . import _input_schema_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# Append-only registration: absent inputs retain the exact historical schema
# and identity, while an explicit mode participates naturally in the config
# fingerprint.
_SECTIONS["metadynamics"].add("walkers_mode")
_SECTIONS["opes"].add("walkers_mode")
