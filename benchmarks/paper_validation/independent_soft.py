#!/usr/bin/env python3
"""Independent Fourier SOFT quantum reproduction of Liu et al. (JCP 161, 024102).

Tully simple avoided crossing, p=10, Gaussian initial wavepacket:
psi(x,0)=(pi)^(-1/4) exp(-(x+9)^2/2 + 10i(x+9)) |0>.
No PyRPMD propagator is used by this script. Atomic units.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
HERE=Path(__file__).resolve().parent
def run(points,dt,halfwidth=15.):
    if 2800/dt!=round(2800/dt) or any((t/dt)!=round(t/dt) for t in range(0,2801,40)):
        raise ValueError("dt must evenly divide 40 atomic time units")
    x=np.linspace(-halfwidth,halfwidth,points,endpoint=False)
    dx=x[1]-x[0]; m=2000.
    z=.01*np.sign(x)*(-np.expm1(-1.6*np.abs(x)))
    c=.005*np.exp(-x*x)
    V=np.zeros((points,2,2),float)
    V[:,0,0]=z;V[:,1,1]=-z;V[:,0,1]=V[:,1,0]=c
    e,u=eigh(V)
    Uhalf=np.einsum("nai,ni,nbi->nab",u,np.exp(-.5j*dt*e),u)
    k=2*np.pi*np.fft.fftfreq(points,d=dx)
    Tfull=np.exp(-.5j*dt*k*k/m)
    psi=np.zeros((points,2),complex)
    psi[:,0]=np.pi**(-.25)*np.exp(-.5*(x+9)**2+1j*10*(x+9))
    psi/=np.sqrt(dx*np.sum(np.abs(psi)**2))
    result={}
    for i in range(round(2800/dt)+1):
        t=i*dt
        if abs(t/40-round(t/40))<1e-8:
            result[int(round(t))]=float(dx*np.sum(np.abs(psi[:,1])**2))
        if i==round(2800/dt): break
        psi=np.einsum("nab,nb->na",Uhalf,psi)
        psi=np.fft.ifft(np.fft.fft(psi,axis=0)*Tfull[:,None],axis=0)
        psi=np.einsum("nab,nb->na",Uhalf,psi)
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=HERE/"results"/"quantum_soft_convergence.json")
    a=p.parse_args()
    with (HERE/"sac_p10_liu2024_reference.csv").open() as f:
        rows=list(csv.DictReader(line for line in f if not line.startswith("#")))
    reference={int(float(row["time"])):float(row["sigma_1_1_real"]) for row in rows}
    outputs={f"points{n}_dt{dt:g}":run(n,dt) for n,dt in ((1024,2.),(1024,1.),(2048,2.),(2048,1.))}
    stats={}
    for name,r in outputs.items():
        diff=np.array([r[t]-reference[t] for t in sorted(reference)])
        stats[name]={"rmse":float(np.sqrt(np.mean(diff**2))),"max_abs_error":float(np.max(np.abs(diff))),
                     "population1_at_2400":r[2400]}
    verdict={"reference_reproduced_max_abs_lt_1e-5":all(z["max_abs_error"]<1e-5 for z in stats.values()),
             "physical_method_validation":False}
    record={"reference_paper":"Liu et al. JCP 161, 024102 (2024)",
            "reference_url":"https://github.com/xiangsunlab/benchmark-data/blob/3951a0a19008fb545ad17f0d2961d934633f7422/Tully/Tully_SAC_p10.csv",
            "reference_sha256":hashlib.sha256((HERE/"sac_p10_liu2024_reference.csv").read_bytes()).hexdigest(),
            "spatial_domain_bohr":[-15,15],"initial":"Gaussian x=-9, p=10, var_x=0.5, mass=2000, diabatic state 0",
            "quantum_results":outputs,"metrics":stats,"verdict":verdict}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"metrics":stats,"verdict":verdict},indent=2),flush=True)
    if not verdict["reference_reproduced_max_abs_lt_1e-5"]:raise SystemExit(1)
if __name__=="__main__":main()
