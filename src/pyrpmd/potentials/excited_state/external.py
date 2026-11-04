"""Strict file-based excited-state backend infrastructure."""

from ._external_common import (
    ExternalWavefunction, ParsedExternalResult, parse_gradient_blocks,
    parse_transition_properties,
)
from ._external_config import _FileProgramConfigurationMixin
from ._external_runtime import _FileProgramRuntimeMixin
from ._types import ExcitedStateBackend


class FileProgramBackend(
    _FileProgramRuntimeMixin, _FileProgramConfigurationMixin, ExcitedStateBackend
):
    """Base class for strict ORCA/OpenMolcas/Q-Chem file interfaces."""


__all__ = [
    "ExternalWavefunction", "ParsedExternalResult", "parse_gradient_blocks",
    "parse_transition_properties", "FileProgramBackend",
]
