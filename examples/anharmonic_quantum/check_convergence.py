"""Independent parameter checks, not parameter fitting to a target spectrum."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
import numpy as np,json
from quantum import morse_quantum,fermi_quantum,spectrum,eckart_scattering,thermal_rate
from models import Model
from sampling import hmc_samples,fermi_samples
from dynamics import response_blocks,transform
from run_benchmarks import OUT,savejson
import numba
numba.set_num_threads(2)
def main():
    checks={};w=np.linspace(1.65,2.4,501)
    for d in (2,3):
        a,c,m=fermi_quantum(6,d);b,e,n=fermi_quantum(6,d,basis=(20,28,14),states=140)
        q=spectrum(a,c,w,.04);qq=spectrum(b,e,w,.04)
        checks[f'fermi_d{d}_basis']={'relative_spectrum_change':float(np.linalg.norm(q-qq)/np.linalg.norm(qq)),
            'sumrule':n['response_sum_rule'],'larger_basis':n['basis'],'states':140}
    l1,m1=morse_quantum(n=300);l2,m2=morse_quantum(n=420,bounds=(-3.5,12.))
    checks['morse_reference']={'ground_energy_max_change':max(abs(a['E0']-b['E0']) for a,b in zip(m1,m2)),
        'thermal_energy_change':sum(a['thermal_energy']-b['thermal_energy'] for a,b in zip(m1,m2))}
    E=np.arange(.01,4.001,.02)
    for d in (1,2,3):
        N,T,m=eckart_scattering(E,d,nx=14,nz=6,dq=.05,L=8.)
        rates={str(beta):thermal_rate(E,N,beta,m['relative_lead_energies']) for beta in (4.,8.)}
        checks[f'eckart_d{d}']={'rates':rates,'meta':m}
        np.savez_compressed(OUT/f'eckart_converged_d{d}.npz',energy=E,crp=N,T0=T)
        if d==1:
            exact=np.sinh(np.pi*np.sqrt(2*E))**2/(np.sinh(np.pi*np.sqrt(2*E))**2+np.cosh(np.pi*np.sqrt(1.75))**2)
            checks['analytic_eckart_max_probability_error']=float(abs(T-exact).max())
        print('reference check',d,rates,flush=True)
    # Same initial paths and momenta separate timestep from sampling variation.
    m=Model('fermi',dim=3);q,p,_=fermi_samples(m,32,256,4482)
    r1=response_blocks(q,p,1,.15,6.,.02,2400,2);r2=response_blocks(q,p,1,.15,6.,.01,4800,4)
    checks['fermi_paired_dt']={'npaths':256,'tmax':48.,'max_response_difference':float(abs(r1.mean(0)-r2.mean(0)).max())}
    savejson('convergence',checks)
if __name__=='__main__':main()
