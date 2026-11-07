"""Public name for the online joint RPMD/quantum workflow.

See :mod:`pyrpmd.analysis.online_quantum` for the implementation and its
explicit mixed-dynamics approximation boundaries.
"""

from .online_quantum import (
    JointPropagationResult,
    JointRPMDQuantumPropagator,
    OnlineQuantumRPMD,
)

__all__ = ["JointPropagationResult", "OnlineQuantumRPMD", "JointRPMDQuantumPropagator"]
