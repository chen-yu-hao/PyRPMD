"""Finite-reference wavefunction propagation with the curvilinear FE measure.

This propagates a closed full-chart Hamiltonian in its retained eigenbasis.
It is NOT a bath-elimination approximation or an online quantum/RPMD coupling.
"""
import numpy as np


def propagate_curvilinear(reference,initial,time,*,hbar,max_projection_loss=1e-8):
    """Return psi(t) with norm psi^dagger M psi, not Cartesian Euclidean norm.

    hbar is mandatory and must match the value used to construct the reference.
    State truncation loss is measured and rejected above max_projection_loss.
    No renormalization hides the missing component. Stored states are bounded.
    """
    psi=np.asarray(initial,dtype=complex);t=np.asarray(time)
    if t.dtype.kind not in 'fiu' or not np.isfinite(t).all() or t.ndim!=1 or not len(t) or np.any(np.diff(t)<=0):
        raise ValueError('Need a finite increasing real time grid')
    E=np.asarray(reference.energies);U=np.asarray(reference.vectors);M=reference.mass_matrix
    if psi.shape!=(U.shape[0],) or not np.isfinite(psi).all():raise ValueError('Invalid initial wavefunction')
    if not np.isfinite([hbar,max_projection_loss]).all() or hbar<=0 or not 0<=max_projection_loss<1:
        raise ValueError('Invalid hbar or projection-loss tolerance')
    if len(t)*len(psi)>10_000_000:raise ValueError('Wavefunction output memory budget exceeded')
    norm=float(np.vdot(psi,M@psi).real)
    if abs(norm-1)>1e-8:raise ValueError('Initial wavefunction must be normalized in the FE mass measure')
    c=U.conj().T@(M@psi);captured=float(np.vdot(c,c).real);loss=max(0.,norm-captured)
    if loss>max_projection_loss:raise ValueError('Retained eigenbasis loses too much initial probability')
    states=(np.exp(-1j*np.outer(t,E)/hbar)*c[None,:])@U.T
    norms=np.einsum('ni,in->n',states.conj(),M@states.T).real
    return {'time':t.astype(float),'wavefunctions':states,'mass_norm':norms,
            'projection_loss':loss,'norm_drift':float(np.max(np.abs(norms-captured))),
            'hbar':float(hbar),'bath_coupling_implemented':False}
