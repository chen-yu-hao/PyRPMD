"""Independent 3D nonharmonic quantum-vs-RPMD response benchmarks.

IMPORTANT: This file does NOT claim to compute a thermal reaction rate. It
compares causal Kubo x-response spectra in finite time window, and static
centroid-virial energy (for Morse) with a 3D quantum spectral reference.
"""
import json,time
from pathlib import Path
import numpy as np
from models import Morse3D,Fermi3D
from quantum_ref import morse_spectrum,fermi_spectrum,morse1_spectrum,fermi2_spectrum
from ringpolymer import pimd_mh,propagate_centroid_correlation,centroid_virial
from scipy.special import logsumexp
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(exist_ok=True)

def calc_response(E,A,beta,t):
    p=np.exp(-beta*(E-E[0]));p/=p.sum()
    ii,jj=np.triu_indices(len(E),1);d=E[jj]-E[ii]
    coeff=2*(p[ii]-p[jj])*np.abs(A[ii,jj])**2
    keep=coeff>1e-14
    d=d[keep];coeff=coeff[keep]
    R=np.sin(np.outer(t,d))@coeff
    C0=float(np.sum(coeff/(beta*d))+np.sum(p*np.diag(A)**2)-np.sum(p*np.diag(A))**2)
    return R,C0,d,coeff

def transform(response,t,w,eta=.12):
    # chi(omega) = integral_0^T exp(-(eta+i*omega)*t) response(t) dt.
    kernel=np.exp(-(eta+1j*w[:,None])*t[None,:])
    return np.trapezoid(kernel*response[None,:],t,axis=1)

def model_reduced(m,kind):
    if kind==1:
        class Sub:
            def V(self,q):
                q=np.asarray(q);return 8*(1-np.exp(-.65*q[...,0]))**2
            def force(self,q):
                q=np.asarray(q);e=np.exp(-.65*q[...,0]);return (-2*8*.65*e*(1-e))[...,None]
        return Sub()
    else:
        class Sub:
            def V(self,q):
                q=np.asarray(q);x=q[...,0];y=q[...,1];c4=.35**2/8+.005
                return 2*x*x+.5*y*y+.35*x*y*y+c4*y**4
            def force(self,q):
                q=np.asarray(q);x=q[...,0];y=q[...,1];c4=.35**2/8+.005
                return np.stack([-4*x-.35*y*y,-y-.7*x*y-4*c4*y**3],axis=-1)
        return Sub()

def run_one(name,P=32,nb=24,ns=64,thin=16,burn=600,dt=.02,tmax=36,eta=.12,seed=321):
    tstart=time.time()
    model=Morse3D() if name=='morse' else Fermi3D()
    beta=model.beta;kind=0 if name=='morse' else 2;subkind=1 if name=='morse' else 3
    timegrid=np.arange(round(tmax/dt)//2+1)*2*dt
    omega=np.linspace(.1,4.,320)
    if name=='morse':
        E,U,A,full_meta=morse_spectrum(model,n=(21,21,21),nstate=32)
        Es,Us,As,sub_meta=morse1_spectrum(model)
        Ksub=model.harmonic()[:1,:1]
        qshift=np.array([.18,.20,.22]);Kref=model.harmonic()*.8
        subshift=np.array([.18]);Ksubref=Ksub*.8
    else:
        E,U,A,full_meta=fermi_spectrum(model,n=(12,18,10),states=56)
        Es,Us,As,sub_meta=fermi2_spectrum(model,n=(12,18),states=56)
        Ksub=model.harmonic()[:2,:2]
        qshift=np.array([-.06,0,0]);Kref=model.harmonic()
        subshift=np.array([-.06,0]);Ksubref=Ksub
    qr,c0q,de,cq=calc_response(E,A,beta,timegrid)
    qs,c0qs,des,cqs=calc_response(Es,As,beta,timegrid)
    q,p,meta=pimd_mh(model,Kref,kind,beta,P,nb,ns,thin,burn,seed,shift=qshift)
    rp_block=propagate_centroid_correlation(q,p,kind,beta,dt,round(tmax/dt),2,nb,float(q[:,:,0].mean()))
    sub=model_reduced(model,subkind)
    qsub,psub,sub_sampling=pimd_mh(sub,Ksubref,subkind,beta,P,nb,ns,thin,burn,seed+103,shift=subshift)
    rps_block=propagate_centroid_correlation(qsub,psub,subkind,beta,dt,round(tmax/dt),2,nb,float(qsub[:,:,0].mean()))
    rf=rp_block.mean(0);rs=rps_block.mean(0)
    from scipy.signal import savgol_filter
    # Consistent causal response from finite-time Kubo autocorrelation.
    full_r=-beta*np.gradient(rf,timegrid,edge_order=2)
    sub_r=-beta*np.gradient(rs,timegrid,edge_order=2)
    qsub_r=calc_response(Es,As,beta,timegrid)[0]
    qfull_r=calc_response(E,A,beta,timegrid)[0]
    # Derivatives of exact Kubo C are exact spectral response, unlike noisy finite differences.
    def spectral_R(E,A):
        pop=np.exp(-beta*(E-E[0]));pop/=pop.sum()
        ii,jj=np.triu_indices(len(E),1);gap=E[jj]-E[ii]
        weights=2*(pop[ii]-pop[jj])*np.abs(A[ii,jj])**2
        keep=weights>1e-14
        return np.sin(np.outer(timegrid,gap[keep]))@weights[keep]
    Q=transform(spectral_R(Es,As),timegrid,omega,eta)
    T=transform(spectral_R(E,A),timegrid,omega,eta)
    F=transform(full_r,timegrid,omega,eta);R=transform(sub_r,timegrid,omega,eta)
    # Scalar inverse response defect; algebraically written as products to reduce overflows.
    with np.errstate(divide='ignore',invalid='ignore'):
        corrected=F*R*Q/(R*Q+F*(R-Q))
        add=F+Q-R
    try:
        # In PyRPMD this is the production public correction API; the
        # stand-alone numerical archive uses the algebraically identical
        # scalar formula when the PyRPMD package is not on PYTHONPATH.
        from pyrpmd.analysis import hybrid_quantum_correction
    except ImportError:
        correction_backend='standalone_scalar_inverse_response'
    else:
        work=hybrid_quantum_correction(omega,F,R,Q,inverse_ridge=0.,output_ridge=0.)
        discrepancy=float(np.max(np.abs(work.response-corrected)))
        if discrepancy>1e-8:
            raise AssertionError(f'PyRPMD correction integration mismatch: {discrepancy}')
        corrected=np.asarray(work.response)
        correction_backend='pyrpmd.analysis.hybrid_quantum_correction'
    def err(z):return float(np.linalg.norm(z-T)/np.linalg.norm(T))
    results={'model':name,'beta':beta,'P':P,'n_independent_paths':nb*ns,'nblocks':nb,
             'dt':dt,'tmax':tmax,'eta':eta,'Morse3D_ZPE_exact':float(E[0]) if name=='morse' else None,
             'ground_energy':float(E[0]),'full_quantum_meta':full_meta,'sub_quantum_meta':sub_meta,
             'full_sampling':meta,'sub_sampling':sub_sampling,
             'errors':{'full_RPMD':err(F),'inverse_defect':err(corrected),'direct_difference':err(add)},
             'C0_quantum_full':c0q,'C0_quantum_sub':c0qs,
             'max_abs_full_subspace_identity':float(abs((F*T*F/(T*F+F*(F-T)))-T).max()),
             'elapsed_sec':time.time()-tstart,'correction_backend':correction_backend}
    if name=='morse':
        energies=centroid_virial(q,model,kind,beta)
        energies_blocks=energies.reshape(nb,ns).mean(1)
        results['morse_centroid_virial_energy']={'estimate':float(energies_blocks.mean()),'block_SE':float(energies_blocks.std(ddof=1)/np.sqrt(nb)),
                    'exact_thermal_approx':float(np.dot(E,np.exp(-beta*(E-E[0])))/np.sum(np.exp(-beta*(E-E[0]))))}
    np.savez_compressed(OUT/f'{name}_P{P}.npz',t=timegrid,omega=omega,quantum_full=T,quantum_sub=Q,
       rpmd_full=F,rpmd_sub=R,corrected=corrected,direct=add,quantum_response=qr,rpmd_blocks=rp_block,
       rpmd_sub_blocks=rps_block,exact_transitions=de,exact_weights=cq)
    (OUT/f'{name}_P{P}.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'name':name,'P':P,'err':results['errors'],'sampling_acceptance':[meta['acceptance_rate'],sub_sampling['acceptance_rate']],
           'energy':results.get('morse_centroid_virial_energy'),'seconds':round(results['elapsed_sec'],1)},indent=2),flush=True)
    return results

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['morse','fermi'],required=True)
 ap.add_argument('--P',type=int,default=32);ap.add_argument('--nb',type=int,default=24);ap.add_argument('--ns',type=int,default=64)
 ap.add_argument('--thin',type=int,default=16);ap.add_argument('--tmax',type=float,default=36);ap.add_argument('--dt',type=float,default=.02)
 ap.add_argument('--eta',type=float,default=.12);ap.add_argument('--seed',type=int,default=321)
 a=ap.parse_args();run_one(a.model,P=a.P,nb=a.nb,ns=a.ns,thin=a.thin,tmax=a.tmax,dt=a.dt,eta=a.eta,seed=a.seed)