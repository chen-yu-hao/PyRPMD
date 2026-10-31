"""Measure Python/native boundary overhead, not an ab-initio PES speedup.

Run: python benchmarks/efficiency/native_ensembles.py --output results.json
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np
from pyrpmd.native_ensemble import NativeEnsemble


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('native-ensembles-benchmark.json'))
    args=parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    cases=[]
    for beads in (1, 8, 32):
        measurements={'one_native_block':[], 'python_call_every_step':[]}
        digest={}
        for name in measurements:
            for repeat in range(args.repeats):
                engine=NativeEnsemble(np.zeros((beads, 2, 3)), [1.,2.], 2., .025,
                                      potential={'kind':'harmonic','k':.7}, thermostat='pile-l', seed=912)
                start=time.perf_counter()
                if name=='one_native_block':
                    result=engine.run(args.steps, stride=args.steps)
                else:
                    for _ in range(args.steps):
                        result=engine.run(1, stride=args.steps)
                measurements[name].append(time.perf_counter()-start)
                digest[name]=engine.state()['digest']
                assert result['force_evaluations']==args.steps+1
        if digest['one_native_block']!=digest['python_call_every_step']:
            raise RuntimeError('boundary batching changed the numerical trajectory')
        medians={key:statistics.median(value) for key,value in measurements.items()}
        cases.append(dict(beads=beads, atoms=2, steps=args.steps, repeats=args.repeats,
                          seconds=measurements, medians=medians,
                          speedup=medians['python_call_every_step']/medians['one_native_block'],
                          final_state_identical=True, force_evaluations=args.steps+1))
    report=dict(purpose='Python/native call overhead on a cheap harmonic PES; not a general potential speedup',
                python=sys.version, platform=platform.platform(), processor=platform.processor(),
                numpy=np.__version__, cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
