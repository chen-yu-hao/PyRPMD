"""Fresh-process MBAR timing/RSS comparison against a pinned upstream solver.

Run each backend in a separate process with the same affinity and single-thread
BLAS environment. Includes a complete solve and optionally the full rate-PMF
bootstrap, rather than timing an isolated reduction kernel. Outputs carry the
actual native thread/call counters and numerical arrays for independent gates.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.analysis.mbar_problem import umbrella_problem
from pyrpmd.analysis.umbrella import umbrella_reduced_potentials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("baseline","numpy","native"),required=True)
    parser.add_argument("--baseline-ref",default="f410b97807d99d429ab1b63c6b1c7ff93b424b5c")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--states",type=int,default=48)
    parser.add_argument("--samples-per-state",type=int,default=2000)
    parser.add_argument("--bootstrap",type=int,default=0)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261110)
    centers=np.linspace(-.2,1.2,args.states)
    stiffness,beta=40.,1.3
    samples=[rng.normal((stiffness*c+.5)/(1+stiffness),
                        1/np.sqrt(beta*(1+stiffness)),args.samples_per_state) for c in centers]
    counts=np.full(args.states,args.samples_per_state)
    if args.backend=="baseline":
        source=subprocess.check_output(["git","show",args.baseline_ref+":src/pyrpmd/analysis/mbar.py"],cwd=ROOT,text=True)
        module=types.ModuleType("_mbar_baseline");sys.modules[module.__name__]=module
        exec(compile(source,"pinned_upstream_mbar.py","exec"),module.__dict__)
        solver=module.solve_mbar
    else:
        solver=solve_mbar
    wall=time.perf_counter();cpu=time.process_time()
    if args.bootstrap:
        import pyrpmd.analysis.rate_pmf as rate
        if args.backend=="baseline":
            rate.umbrella_problem=umbrella_reduced_potentials
            def selected(*a,**kw):
                kw.pop("threads",None)
                return solver(*a,**kw)
            options={}
        else:
            selected=solver
            options={"backend":args.backend,"threads":1}
        result=rate.mbar_rate_pmf(samples,centers,stiffness,beta,
                                 reference=0.,xi_star=1.,bins=80,ranges=[-.4,1.4],
                                 bootstrap=args.bootstrap,seed=4281,compare_ui=True,
                                 bootstrap_workers=args.threads,bootstrap_total_cores=args.threads,
                                 solver_options=options,_solve_mbar=selected)
        arrays={"free_energy":result["free_energy"],"weights":result["weights"],
                "barriers":result["pmf_bootstrap_barriers"],"standard_error":result["standard_error"]}
        diagnostic=result["solver"]["details"]
        gate={"ready":result["ready"],"status":result["status"],
              "bootstrap_valid":result["bootstrap"]["valid"],
              "bootstrap_workers":result["bootstrap"]["workers"],
              "bootstrap_backend":result["bootstrap"]["backend"]}
        diagnostic={**diagnostic,"bootstrap_draws":result["bootstrap"]["draws"]}
    else:
        x=np.concatenate(samples)[:,None]
        if args.backend=="baseline":
            u=umbrella_reduced_potentials(x,centers[:,None],[[stiffness]],beta)
            result=solver(u,counts,return_state_weights=False)
        else:
            u=umbrella_problem(x,centers[:,None],[[stiffness]],beta)
            result=solver(u,counts,return_state_weights=False,backend=args.backend,threads=args.threads)
        arrays={k:getattr(result,k) for k in ("free_energies","weights","overlap","log_weights")}
        diagnostic=result.diagnostics
        gate={"converged":result.converged,"residual":result.residual}
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    report={"backend":args.backend,"threads":args.threads,"states":args.states,
            "samples":args.states*args.samples_per_state,"bootstrap":args.bootstrap,
            "timed_scope":"potential construction + complete solve" if not args.bootstrap else "complete rate PMF with paired UI bootstrap",
            "wall_seconds":wall,"cpu_seconds":cpu,"cpu_to_wall":cpu/wall,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "baseline_commit":args.baseline_ref,"python":sys.version,"numpy":np.__version__,
            "affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            "diagnostics":diagnostic,"gates":gate}
    if args.backend=="native":
        from pyrpmd import _mbar
        report["loaded_extension"]=_mbar.__file__
    def encode(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=encode)+"\n")
    np.savez(args.output.with_suffix(".npz"),**arrays)
    print(json.dumps({k:report[k] for k in ("backend","threads","wall_seconds","cpu_to_wall","peak_rss_kib","gates")}))


if __name__=="__main__":
    main()
