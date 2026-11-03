"""Optional PySCF TDA/TDDFT online excited-state backend."""

from ._pyscf_config import _PySCFConfigurationMixin
from ._pyscf_overlap import (
    PySCFTDWavefunction, _align_displaced_overlap,
    _normalized_auxiliary_amplitudes, auxiliary_wavefunction_overlap,
)
from ._pyscf_runtime import _PySCFRuntimeMixin
from ._pyscf_solve import _PySCFSolveMixin
from ._types import ExcitedStateBackend


class PySCFTDDFTBackend(
    _PySCFRuntimeMixin, _PySCFSolveMixin, _PySCFConfigurationMixin,
    ExcitedStateBackend,
):
    """Restricted closed-shell PySCF TDA/TDDFT backend."""


__all__ = [
    "PySCFTDWavefunction", "PySCFTDDFTBackend",
    "auxiliary_wavefunction_overlap",
]
