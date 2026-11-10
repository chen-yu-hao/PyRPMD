"""Small fixed-grid same-Hamiltonian channel validation, not a physical rate study."""
import json
import numpy as np
from pyrpmd.analysis.subspace_channels import channels_from_model
from pyrpmd.analysis.subspace_model import model_from_function
from pyrpmd.analysis.subspace_quantum import _GridOperators, exact_quantum_rate


def potential(x):
    s,q=np.asarray(x)[...,0],np.asarray(x)[...,1]
    return .08/np.cosh(s/.8)**2+.5*.22**2*(q-.35*np.tanh(s/1.2))**2


def absorber(x):
    x=np.asarray(x)
    return .15*np.maximum(abs(x[:,0])-1.,0.)**2*(1+.04*x[:,1]**2)


model=model_from_function(potential,[np.linspace(-3,3,11),np.linspace(-5,5,9)],
                          masses=[1.7,1.],dividing_surface=0.,reactant_reference=-1.5)
beta=5.;rows=[]
for cutoff in [None,.35]:
    opts=dict(absorber=absorber,normalization='slice',energy_floor=.1)
    full=exact_quantum_rate(model,beta,method='sylvester',truncation_energy=cutoff,**opts)
    full_h=_GridOperators(model).dense_hamiltonian()
    for count in [1,2,4,6,9]:
        channels=channels_from_model(model,n_channels=count,truncation_energy=cutoff)
        rate=channels.exact_rate(beta,**opts)
        ns,nq,nc=channels.states.shape
        u=np.zeros((ns*nq,ns*nc),complex)
        for i in range(ns):u[i*nq:(i+1)*nq,i*nc:(i+1)*nc]=channels.states[i]
        u=u[:,channels.active_channels.ravel()]
        error=float(np.max(np.abs(channels.hamiltonian-u.conj().T@full_h@u)))
        factor=np.exp(beta*(np.min(channels.energies[channels.active_channels])-model.values.min()))
        row=dict(truncation=cutoff,nchannels=count,retained_channels=channels.size,
                 full_dvr_points=full.diagnostics['retained_points'],matrix_projection_error=error,
                 rate=rate.rate,reference_rate=full.rate,relative_rate_error=rate.rate/full.rate-1,
                 flux_at_common_zero=rate.flux/factor,full_flux=full.flux,
                 partition_at_common_zero=rate.reactant_partition/factor,full_partition=full.reactant_partition,
                 basis_complete=rate.diagnostics['channel_basis_complete'])
        assert error<1e-12
        if row['basis_complete']:
            assert abs(row['relative_rate_error'])<1e-10
            assert np.isclose(row['flux_at_common_zero'],full.flux,rtol=1e-10,atol=1e-14)
            assert np.isclose(row['partition_at_common_zero'],full.reactant_partition,rtol=1e-10,atol=1e-14)
        rows.append(row)
print(json.dumps(dict(scope='fixed-grid Cartesian Galerkin convergence; no physical convergence claim',rows=rows),indent=2))
