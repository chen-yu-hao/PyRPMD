"""Opt-in native RPMD workflow with a user-specified physical active backend.

Run in a PyRPMD checkout containing this patch:
  python benchmarks/rpmdrate_shared/run_shared_rate.py --input rate.toml \
    --problem-factory my_active_solver:build_problem --output runs/shared \
    --pes-id '<versioned-PES-identity>' --normalization '<common-Qr-convention>' \
    --kinetic-mapping '<metric-and-bath-model-identity>' --plateau-start 100

Python provider loading is trusted-code execution, just like a Python PES.
The provider must implement FiniteBasisFluxBackend's matched-data contract.
It may build a 1--3D constrained active model with build_active_slice_hamiltonian,
while providing independently generated matched active-RPMD reference blocks.
"""
from pathlib import Path
import argparse
import importlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--problem-factory',required=True,help='Trusted Python module:function')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pes-id',required=True)
    p.add_argument('--normalization',required=True)
    p.add_argument('--kinetic-mapping',required=True)
    p.add_argument('--plateau-start',type=int,required=True)
    p.add_argument('--bootstrap-samples',type=int,default=400)
    a=p.parse_args()
    module,sep,attr=a.problem_factory.partition(':')
    if not sep or not module or not attr:raise ValueError('Use module:function for the active provider')
    from pyrpmd.input import JobSpec
    from pyrpmd.analysis.rpmdrate_quantum import FiniteBasisFluxBackend
    from pyrpmd.workflows.rate_quantum import run_shared_quantum_rate
    cfg,system=JobSpec.from_file(a.input).resolve()
    backend=FiniteBasisFluxBackend(getattr(importlib.import_module(module),attr))
    result=run_shared_quantum_rate(cfg,system,a.output,backend,pes_identity=a.pes_id,
        reactant_normalization=a.normalization,kinetic_mapping=a.kinetic_mapping,
        plateau_start=a.plateau_start,bootstrap_samples=a.bootstrap_samples)
    if backend.last_diagnostics is not None:
        (a.output/'finite_basis_diagnostics.json').write_text(json.dumps(backend.last_diagnostics,indent=2))
    print(json.dumps({'rate_estimate_cm3_s':result['rate_estimate_cm3_s'],
        'rate_cm3_s':result['rate_cm3_s'],'physical_converged':False,
        'withheld_reasons':result['withheld_reasons']},indent=2))


if __name__=='__main__':main()
