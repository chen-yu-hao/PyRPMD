"""Reproducible non-harmonic validation; no fitted exact-reference parameters.
Use --stage pilot|morse|fermi|eckart_quantum|eckart_rpmd|all. Results are raw JSON/NPZ.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import numba
from scipy.signal import find_peaks
from models import Model
from sampling import hmc_samples,fermi_samples
from dynamics import response_blocks,transform,flux_side
from quantum import morse_quantum,fermi_quantum,spectrum,eckart_scattering,thermal_rate
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/pyrpmd/analysis'))
from adaptive_quantum import select_active_subspace,passive_embed,spectral_diagnostics
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
numba.set_num_threads(min(4,numba.get_num_threads()))
def savejson(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print(name,json.dumps(data,ensure_ascii=False),flush=True)
def error(s,q):return float(np.linalg.norm(s.imag-q.imag)/np.linalg.norm(q.imag))
def pilot():
    results={}
    for code,beta in [('morse',8.),('fermi',6.),('eckart',8.)]:
        m=Model(code,beta=beta)
        if code=='fermi':q,_,meta=fermi_samples(m,32,2048,8111)
        else:
            samples,meta=hmc_samples(m,32,256,600,4,10,8111,0. if code=='eckart' else None)
            q=samples.reshape(-1,32,3)
        z=q[:,::4].reshape(-1,3);f=m.force(z);h=m.hessian(z)
        a=np.ones(3)/np.sqrt(3) if code=='morse' else np.array([1.,0,0])
        R=np.linalg.qr(np.random.default_rng(9821).normal(size=(3,3)))[0]
        sel=select_active_subspace(z@R.T,f@R.T,np.einsum('ia,nab,jb->nij',R,h,R),R@a,nonlinear_fraction=.999)
        basis=R.T@sel.basis
        results[code]={'dimension':sel.selected_dimension,'status':sel.status,'nonlinear_rank':sel.nonlinear_rank,
            'nonlinear_eigenvalues':sel.nonlinear_eigenvalues.tolist(),'residual_loading':sel.unresolved_score,
            'basis_in_physical_coordinates':basis.tolist(),'projector':(basis@basis.T).tolist(),'sampling':meta,
            'uses_quantum_answer':False,'threshold':.05}
        np.savez_compressed(OUT/f'pilot_{code}.npz',q=q,forces=f,hessians=h)
    savejson('selection',results)

def run_morse(P=32,n=2048,tag='main'):
    start=time.perf_counter();m=Model('morse',beta=8.)
    samples,meta=hmc_samples(m,P,n//4,800,4,20,9201)
    q=np.ascontiguousarray(samples.reshape(n,P,3));p=np.random.default_rng(9202).normal(size=q.shape)*np.sqrt(P/m.beta)
    meanq=q.mean(1);V=m.potential(q);F=m.force(q)
    energy=1.5/m.beta+V.mean(1)-.5*np.mean(np.sum((q-meanq[:,None])*F,axis=-1),axis=1)
    eb=energy.reshape(32,-1).mean(1);means=q.mean((0,1))
    rb=response_blocks(q,p,0,.15,m.beta,.01,9600,4);t=np.arange(rb.shape[1])*.04;w=np.linspace(.2,3.0,421);eta=.06
    specblocks=transform(rb,t,w,eta);rp=specblocks.mean(0);lines,qr=morse_quantum(m.beta)
    Q=np.column_stack([spectrum(d,c,w,eta) for d,c in lines]);fullQ=Q.mean(1);fullRP=rp.mean(1)
    # In this separable model the response embedding is block diagonal. Select
    # actual physical coordinates; never transfer a different observable.
    curves={'quantum':fullQ,'rpmd':fullRP}
    for d in (1,2,3):curves[f'corrected_d{d}']=(Q[:,:d].sum(1)+rp[:,d:].sum(1))/3
    np.savez_compressed(OUT/f'morse_{tag}_P{P}.npz',t=t,omega=w,response_blocks=rb,spectrum_blocks=specblocks,**curves)
    savejson(f'morse_{tag}_P{P}',{'model':'sum of 3 anharmonic Morse stretches','beta':8.,'P':P,'n':n,'sampling':meta,
        'observable':'(q1+q2+q3)/sqrt(3)','energy_quantum':sum(r['thermal_energy'] for r in qr),'energy_pimd':float(eb.mean()),
        'energy_pimd_block_se':float(eb.std(ddof=1)/np.sqrt(32)),'classical_lowT_energy':3/8.,'means':means.tolist(),
        'quantum':qr,'errors':{k:error(v,fullQ) for k,v in curves.items()},'eta':eta,'seconds':time.perf_counter()-start,
        'dimensional_reduction':False,'physical_converged':False})

def run_fermi(P=32,n=2048,tag='main'):
    start=time.perf_counter();w=np.linspace(1.65,2.4,501);eta=.04;rp={};meta={};blocks={};qs={}
    for d in (2,3):
        m=Model('fermi',dim=d,beta=6.,coupling=.15);q,p,sm=fermi_samples(m,P,n,10101+d)
        rb=response_blocks(q,p,1,m.coupling,m.beta,.02 if P==32 else .01,int(192/(.02 if P==32 else .01)),2 if P==32 else 4)
        t=np.arange(rb.shape[1])*.04;sb=transform(rb,t,w,eta)[:,:,0];blocks[d]=sb;rp[d]=sb.mean(0)
        de,coef,qm=fermi_quantum(6.,d);qs[d]=spectrum(de,coef,w,eta);meta[str(d)]={'sampling':sm,'quantum':qm}
    inv=1/(1/rp[3]+1/qs[2]-1/rp[2]);emb=passive_embed(qs[2],w,eta,np.array([[1.69]]),np.array([[-1.69*.15]]))
    # The one-coordinate core is harmonic 2*x^2, deliberately missing xy^2.
    one=1/(4+(eta-1j*w)**2);emb1=passive_embed(one,w,eta,np.array([[1.69]]),np.array([[-1.69*.15]]))
    curves={'quantum':qs[3],'rpmd':rp[3],'inverse_d2':inv,'embedded_d2':emb,'embedded_d1':emb1}
    metric={k:{'relative_absorption_L2':error(v,qs[3]),**spectral_diagnostics(v,w)} for k,v in curves.items()}
    for k,v in curves.items():
        pk,_=find_peaks(v.imag,prominence=.05*qs[3].imag.max());metric[k]['peak_frequencies']=w[pk].tolist()
    np.savez_compressed(OUT/f'fermi_{tag}_P{P}.npz',omega=w,full_blocks=blocks[3],active_blocks=blocks[2],quantum_active=qs[2],**curves)
    # Independent replica block bootstrap: uncertainty of RPMD-dependent methods.
    rng=np.random.default_rng(273);berr={k:[] for k in ('rpmd','inverse_d2')}
    for i in range(256):
        F=blocks[3][rng.integers(32,size=32)].mean(0);R=blocks[2][rng.integers(32,size=32)].mean(0)
        berr['rpmd'].append(error(F,qs[3]));berr['inverse_d2'].append(error(1/(1/F+1/qs[2]-1/R),qs[3]))
    savejson(f'fermi_{tag}_P{P}',{'model':'xy^2 Fermi core + harmonic spectator','beta':6.,'P':P,'n':n,'observable':'x in every calculation',
        'eta':eta,'tmax':192.,'metrics':metric,'references':meta,'bootstrap95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in berr.items()},
        'seconds':time.perf_counter()-start,'physical_converged':False})

def run_eckart_quantum(tag='main',dq=.10,nx=10,nz=4):
    start=time.perf_counter();energy=np.arange(.02,4.001,.04);data={}
    for d in (1,2,3):
        N,T,meta=eckart_scattering(energy,d,nx,nz,dq)
        rates={str(beta):thermal_rate(energy,N,beta,meta['relative_lead_energies']) for beta in (4.,8.)}
        data[str(d)]={'meta':meta,'rates':rates,'T_at_E_0.5':float(np.interp(.5,energy,T))}
        np.savez_compressed(OUT/f'eckart_{tag}_d{d}.npz',energy=energy,crp=N,T0=T)
        print('scattering',tag,d,rates,flush=True)
    data['seconds']=time.perf_counter()-start;savejson('eckart_quantum_'+tag,data)

def run_eckart_rpmd(beta=8.,P=32,nchains=192,tag='main',dimensions=(1,2,3),quadrature_nodes=16,warmup=700,seed_offset=0):
    """Bennett-Chandler centroid-TST PMF and unthermostatted transmission.
    Rates have units length/time and a matched asymptotic reactant normalization.
    PMF quadrature and chain equilibration are separately checked in validation.
    """
    from scipy.special import roots_legendre
    start=time.perf_counter();nodes,weights=roots_legendre(quadrature_nodes);centers=3*(nodes-1);weights=3*weights;result={}
    for d in dimensions:
        m=Model('eckart',dim=d,beta=beta);mean_forces=[];force_se=[];sample_records=[]
        # independent chains at each fixed centroid, no trajectory reuse with pilot
        for i,c in enumerate(centers):
            q,sm=hmc_samples(m,P,nchains,warmup,6,12,14001+i*100+seed_offset,centroid=float(c))
            F=m.force(q)[...,0].mean((1,2));mean_forces.append(F.mean());force_se.append(F.std(ddof=1)/np.sqrt(nchains));sample_records.append(sm)
            print('TI',beta,P,d,i,flush=True)
        barrier=-float(np.dot(weights,mean_forces));bse=float(np.sqrt(np.dot(weights*weights,np.array(force_se)**2)))
        ts,sm=hmc_samples(m,P,512,max(1000,warmup),4,20,16011+seed_offset,centroid=0.);q=np.ascontiguousarray(ts.reshape(-1,P,d));n=len(q)
        rng=np.random.default_rng(16012);p=rng.normal(size=q.shape)*np.sqrt(P/beta)
        p[:,:,0]-=p[:,:,0].mean(1)[:,None];v=np.sqrt(-2*np.log(rng.random(n))/beta);p[:,:,0]+=v[:,None]
        kb=flux_side(q,p,beta,.01,1600,10);t=np.arange(kb.shape[1])*.1;plateau=t>=10
        kappa=kb[:,plateau].mean(1);ktst=np.exp(-beta*barrier)/np.sqrt(2*np.pi*beta)
        rate=ktst*kappa.mean()
        np.savez_compressed(OUT/f'eckart_rp_{tag}_b{beta:g}_P{P}_d{d}.npz',t=t,kappa_blocks=kb,centers=centers,mean_forces=mean_forces,force_se=force_se,weights=weights)
        result[str(d)]={'PMF_barrier':barrier,'PMF_SE':bse,'kappa':float(kappa.mean()),'kappa_SE':float(kappa.std(ddof=1)/np.sqrt(32)),
            'plateau_early':float(kb[:,(t>=8)&(t<12)].mean()),'plateau_late':float(kb[:,t>=12].mean()),'ktst':float(ktst),'rate':float(rate),
            'rate_relative_sampling_SE':float(np.sqrt((beta*bse)**2+(kappa.std(ddof=1)/np.sqrt(32)/kappa.mean())**2)),
            'quadrature_nodes':quadrature_nodes,'centroid_interval':[-6,0],'sampling':sample_records,'TS_sampling':sm}
        savejson(f'eckart_rp_{tag}_b{beta:g}_P{P}_d{d}',result[str(d)])
        print('RP rate',beta,P,d,rate,'barrier',barrier,'kappa',kappa.mean(),flush=True)
    result['seconds']=time.perf_counter()-start;savejson(f'eckart_rp_{tag}_b{beta:g}_P{P}',result)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','morse','fermi','eckart_quantum','eckart_rpmd','all'],default='all');ap.add_argument('--P',type=int,default=32);ap.add_argument('--n',type=int,default=2048);ap.add_argument('--tag',default='main');ap.add_argument('--beta',type=float,default=8.);ap.add_argument('--chains',type=int,default=192);ap.add_argument('--warmup',type=int,default=700);ap.add_argument('--seed-offset',type=int,default=0);ap.add_argument('--quadrature-nodes',type=int,default=16);ap.add_argument('--dimension',type=int,default=0);ap.add_argument('--dq',type=float,default=.10);ap.add_argument('--nx',type=int,default=10);ap.add_argument('--nz',type=int,default=4);args=ap.parse_args()
    for s in (['pilot','morse','fermi','eckart_quantum','eckart_rpmd'] if args.stage=='all' else [args.stage]):
        if s=='pilot':pilot()
        elif s=='morse':run_morse(args.P,args.n,args.tag)
        elif s=='fermi':run_fermi(args.P,args.n,args.tag)
        elif s=='eckart_quantum':run_eckart_quantum(args.tag,args.dq,args.nx,args.nz)
        else:run_eckart_rpmd(args.beta,args.P,nchains=args.chains,tag=args.tag,dimensions=(args.dimension,) if args.dimension else (1,2,3),quadrature_nodes=args.quadrature_nodes,warmup=args.warmup,seed_offset=args.seed_offset)
