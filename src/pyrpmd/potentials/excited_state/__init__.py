"""Online excited-state electronic-structure providers.

All backends follow the :mod:`pyrpmd.adiabatic` convention: coordinates in
bohr, total energies in hartree, positive gradients in hartree/bohr,
``d_ij=<i|grad j>`` in bohr**-1, and overlaps
``O_ij=<old_i|new_j>``.
"""

from .base import (
    BackendFailure,
    BackendResult,
    ExcitedStateBackend,
    ExcitedStateFrameContext,
    ExcitedStateProvider,
    ExcitedStateStatus,
    ProviderStatistics,
)
from .external import (
    ExternalWavefunction,
    FileProgramBackend,
    ParsedExternalResult,
)
from .openmolcas import OpenMolcasExcitedStateBackend
from .orca import ORCAExcitedStateBackend
from .pyscf_tddft import (
    PySCFTDDFTBackend,
    PySCFTDWavefunction,
    auxiliary_wavefunction_overlap,
)
from .qchem import QChemExcitedStateBackend
from .toy import AvoidedCrossingBackend, ToyWavefunction
from .tracking import (
    StateTrackingError,
    TrackingDiagnostics,
    align_reference_overlap,
    align_states,
    integrated_connection,
    path_derivative_couplings,
    polar_unitary,
    time_derivative_coupling,
)

__all__ = [
    "BackendFailure",
    "BackendResult",
    "ExcitedStateBackend",
    "ExcitedStateFrameContext",
    "ExcitedStateProvider",
    "ExcitedStateStatus",
    "ProviderStatistics",
    "ExternalWavefunction",
    "FileProgramBackend",
    "ParsedExternalResult",
    "OpenMolcasExcitedStateBackend",
    "ORCAExcitedStateBackend",
    "PySCFTDDFTBackend",
    "PySCFTDWavefunction",
    "auxiliary_wavefunction_overlap",
    "QChemExcitedStateBackend",
    "AvoidedCrossingBackend",
    "ToyWavefunction",
    "StateTrackingError",
    "TrackingDiagnostics",
    "align_reference_overlap",
    "align_states",
    "integrated_connection",
    "path_derivative_couplings",
    "polar_unitary",
    "time_derivative_coupling",
]
