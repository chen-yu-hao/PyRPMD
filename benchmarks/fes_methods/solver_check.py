"""Independent PyMBAR check on actual saved umbrella streams, without thinning."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from pymbar import MBAR
from pyrpmd.workflows.fes import write_json


def check(case):
    case=Path(case)
    beta=json.loads((case/"input-resolved.json").read_text())["config"]["beta"]
    with h5py.File(case/"run/fes.h5","r") as h:
        groups=[h["streams"][k] for k in sorted(h["streams"],key=int)]
        samples=[g["samples"][:] for g in groups]
        centers=np.asarray([g["center"][:] for g in groups])
        stiffness=np.asarray([g["stiffness"][:] for g in groups])
        weights=h["weights"][:];energies=h["window_free_energies"][:]
    joined=np.concatenate(samples);counts=np.asarray([len(s) for s in samples])
    # Independent expression: do not call PyRPMD's bias-matrix or MBAR solver.
    delta=joined[None,:,:]-centers[:,None,:]
    reduced=.5*beta*np.einsum("kni,kij,knj->kn",delta,stiffness,delta)
    reference=MBAR(np.vstack([reduced,np.zeros(len(joined))]),np.r_[counts,0],
                   relative_tolerance=1e-11,solver_protocol="robust")
    df=float(np.max(abs(reference.f_k[:-1]-energies)))
    dw=float(np.max(abs(reference.W_nk[:,-1]-weights)))
    assert df<1e-8 and dw<1e-10
    result={"case":str(case),"windows":len(counts),"samples":int(counts.sum()),
            "thinning":1,"max_window_free_energy_difference_dimensionless":df,
            "max_normalized_weight_difference":dw,"passed":True,
            "scope":"Same finite data, independent reduced bias expression and PyMBAR solver; does not establish sampling convergence."}
    write_json(case/"pymbar-check.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    print(json.dumps(check(p.parse_args().case),indent=2))
