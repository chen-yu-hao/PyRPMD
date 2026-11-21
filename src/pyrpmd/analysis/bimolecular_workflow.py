"""Fail-closed assembly of molecular active-space and matched-rate calculations.

This module does not pretend an arbitrary selected linear subspace automatically
has the correct quantum kinetic operator or exact bath influence functional.
The caller supplies a quantum reference with explicit scope/convergence evidence.
"""
from __future__ import annotations
import numpy as np
from .bimolecular_scattering import matched_bimolecular_rate_defect


def assemble_correction(selection,full_rp,active_rp,active_q,*,sampling_covariance,
                        quantum_reference_gates):
    """Apply a matched rate defect only after reference and selection checks.

    sampling_covariance has order (full_RP, active_RP, active_Q). Paired estimates
    may supply nonzero cross covariance; it must come from whole independent
    replica/block analysis. The resulting SE is sampling-only. No fitted mixing
    coefficient, positivity clipping or post-hoc renormalization is performed.
    """
    if selection.status!='candidate':
        raise ValueError('Active subspace is not a reduced-space candidate: '+selection.status)
    required=('active_hamiltonian_metric','reactant_normalization','same_observable',
              'quantum_grid','quantum_time_or_scattering','angular_momentum_sum')
    if any(not isinstance(quantum_reference_gates.get(k),dict) or
           quantum_reference_gates[k].get('passed') is not True or
           not quantum_reference_gates[k].get('evidence') for k in required):
        raise ValueError('Quantum reference gates are missing; do not output a corrected gas-phase rate')
    C=np.asarray(sampling_covariance,float)
    if C.shape!=(3,3) or not np.isfinite(C).all() or not np.allclose(C,C.T):
        raise ValueError('Symmetric finite 3x3 covariance required')
    scale=max(float(np.max(abs(C))),1e-300)
    if np.linalg.eigvalsh(C).min() < -1e-10*scale:
        raise ValueError('Sampling covariance is not positive semidefinite')
    result=matched_bimolecular_rate_defect(full_rp,active_rp,active_q)
    v=np.array([1.,-1.,1.]);variance=float(v@C@v)
    result.update({'sampling_standard_error':float(np.sqrt(max(variance,0))),
        'uncertainty_scope':'provided whole-replica sampling covariance only; excludes transfer-model bias',
        'active_dimension':selection.selected_dimension,'selection_status':selection.status,
        'reference_gates':quantum_reference_gates,'physical_converged':False,
        'transferability_proven':False})
    return result
