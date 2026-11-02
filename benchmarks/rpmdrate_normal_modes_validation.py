"""Repeat the BCOCB observables using RPMDrate-compatible packed normal modes.

PYTHONPATH=src python benchmarks/rpmdrate_normal_modes_validation.py --output nm.json
Each comparison uses the same bead Gaussian field and the same model. All
observations are explicit CPU transfers; this is not a GPU timing benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState
from pyrpmd.torch_normal_modes import RPMDrateNormalModes


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v = load("bcocb_reference_for_modes", "torch_bcocb_validation.py")
ref = load("rpmdrate_reference_for_modes", "rpmdrate_normal_modes.py")


def factory(P, beta, dt, layout, *, device="cpu", quartic=False, thermostat="pile-l", seed=0):
    model = v.obs.Oscillator() if quartic else v.obs.Oscillator(1., 0.)
    return TorchDynamics(TorchPES(model, device=device), [1.], beta, P, dt,
                         integrator="cayley-bcocb", thermostat=thermostat, seed=seed,
                         normal_mode_layout=layout)


def algebra():
    rows = []
    rng = np.random.default_rng(20261101)
    for P in (1, 2, 3, 4, 5, 8, 16, 31, 32, 128, 256):
        modes = RPMDrateNormalModes(P)
        x = rng.normal(size=(3, 2, P)); y = modes.forward(torch.tensor(x)).numpy()
        fftw = ref.fftw_transform(x)
        F, weights, _ = ref.real_transform(P)
        rows.append({"beads": P, "max_fftw_error": float(np.max(abs(y-fftw))),
                     "max_explicit_basis_error": float(np.max(abs(y-x@F.T))),
                     "max_roundtrip_error": float(np.max(abs(modes.inverse(torch.tensor(y)).numpy()-x))),
                     "max_metric_error": float(np.max(abs(np.sum(y*y*weights, -1)-np.sum(x*x, -1))))})
    return rows


def harmonic(device):
    n, beta, dt, steps = 1024, 8., .5, 1600
    rows = []
    for P in (8, 32, 128, 256):
        states = []; stats = []; rng_states = []
        omega = v.obs.frequencies(P, beta)
        variances = P/beta/(1+omega**2)
        exact = {"q2": float(variances.mean()),
                 "kinetic_primitive": float(P/(2*beta)-np.sum(omega**2*variances)/(2*P)),
                 "kinetic_virial": float(.5/beta+np.sum(variances[1:])/(2*P))}
        for layout in ("rfft", "rpmdrate"):
            d = factory(P,beta,dt,layout,device=device,seed=5000+P)
            zero = np.zeros((n,P));s=v.obs.state_1d(zero,zero,device)
            d.propagate(s,steps)
            q,p=s.q[:,:,0,0].cpu().numpy(),s.p[:,:,0,0].cpu().numpy()
            values = {"q2": (q*q).mean(1),
                      "kinetic_primitive": P/(2*beta)-.5*(P/beta)**2*np.mean((q-np.roll(q,1,axis=1))**2,axis=1),
                      "kinetic_virial": .5/beta+.5*np.mean((q-q.mean(1,keepdims=True))*q,axis=1),
                      "p2": np.mean(p*p,axis=1)}
            stats.append({key:v.obs.summary(value) for key,value in values.items()})
            states.append((q,p));rng_states.append(d.generator.get_state())
        rows.append({"beads":P,"seed":5000+P,"finite_P_exact":exact,
                     "rfft":stats[0],"rpmdrate":stats[1],
                     "max_paired_q_error":float(np.max(abs(states[0][0]-states[1][0]))),
                     "max_paired_p_error":float(np.max(abs(states[0][1]-states[1][1]))),
                     "identical_rng_state":bool(torch.equal(*rng_states))})
    return {"trajectories":n,"beta":beta,"dt":dt,"steps":steps,"cold_start":True,"rows":rows}


def quartic(device):
    P,beta,n,equil,production,interval=16,1.,256,20.,100.,.2
    reference=v.obs.transfer_quartic(P,points=512)
    rows=[]
    for dt in (.02,.1):
        stats=[];states=[]
        for layout in ("rfft","rpmdrate"):
            d=factory(P,beta,dt,layout,device=device,quartic=True,seed=717)
            zero=np.zeros((n,P));s=v.obs.state_1d(zero,zero,device)
            d.propagate(s,round(equil/dt))
            sums={key:np.zeros(n) for key in v.obs.observables(zero,beta)}
            blocks=round(production/interval)
            for _ in range(blocks):
                d.propagate(s,round(interval/dt))
                for key,value in v.obs.observables(s.q[:,:,0,0].cpu().numpy(),beta).items():sums[key]+=value
            stats.append({key:v.obs.summary(value/blocks) for key,value in sums.items()})
            states.append((s.q.cpu().numpy(),s.p.cpu().numpy()))
        rows.append({"dt":dt,"rfft":stats[0],"rpmdrate":stats[1],
                     "max_paired_q_error":float(np.max(abs(states[0][0]-states[1][0]))),
                     "max_paired_p_error":float(np.max(abs(states[0][1]-states[1][1])))})
    return {"beads":P,"beta":beta,"trajectories":n,"seed":717,"equil_time":equil,
            "production_time":production,"observation_interval":interval,"reference":reference,
            "uncertainty":"Student-t across independent trajectory means","rows":rows}


def correlation(device):
    P,beta,n,seed,interval,duration=54,1.,2048,20261101,.1,10.
    q,p,acceptance=v.obs.quartic_iid(n,P,beta,np.random.default_rng(seed))
    q0=q.mean(1);C0=float(np.mean(q0*q0));rows=[]
    for dt in (.01,.05,.1):
        products=[];states=[]
        for layout in ("rfft","rpmdrate"):
            d=factory(P,beta,dt,layout,device=device,quartic=True,thermostat=None)
            s=v.obs.state_1d(q,p,device);product=[q0*q0]
            for _ in range(round(duration/interval)):
                d.propagate(s,round(interval/dt))
                product.append(s.q[:,:,0,0].mean(1).cpu().numpy()*q0)
            products.append(np.array(product).T);states.append((s.q.cpu().numpy(),s.p.cpu().numpy()))
        rows.append({"dt":dt,"rfft_correlation":v.obs.summary(products[0]),
                     "rpmdrate_correlation":v.obs.summary(products[1]),
                     "max_paired_q_error":float(np.max(abs(states[0][0]-states[1][0]))),
                     "max_paired_p_error":float(np.max(abs(states[0][1]-states[1][1]))),
                     "max_mean_correlation_difference_over_C0":float(np.max(abs((products[1]-products[0]).mean(0)))/C0)})
    return {"beads":P,"beta":beta,"trajectories":n,"seed":seed,"C0":C0,"thermostat":None,
            "duration":duration,"acceptance":acceptance,"rows":rows,
            "scope":"layout parity only; no new exact-quantum or digitized-paper reference"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--section",choices=("all","algebra","harmonic","quartic","correlation"),default="all")
    args=ap.parse_args();torch.set_num_threads(1)
    result={"upstream_repo":"GreenGroup/RPMDrate","upstream_commit":ref.REFERENCE_COMMIT,
            "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
                           "device":args.device,"cuda_available":torch.cuda.is_available(),"torch_threads":1},
            "source_sha256":{Path(inspect.getfile(obj)).name:hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
                             for obj in (TorchDynamics,RPMDrateNormalModes,ref.exact_free,main)}}
    start=time.perf_counter()
    for name,function in (("algebra",algebra),("harmonic",harmonic),("quartic",quartic),("correlation",correlation)):
        if args.section not in ("all",name):continue
        result[name]=function() if name=="algebra" else function(args.device)
        result["elapsed_seconds"]=time.perf_counter()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(name,"complete",flush=True)


if __name__=="__main__":main()
