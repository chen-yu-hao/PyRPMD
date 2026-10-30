"""Real BKMP2 validation of integrated production block loops (not rate convergence).

Measures the actual Dynamics and workflow entrypoints, not a stand-alone kernel.
Original stepwise paths remain the numerical controls. All gates precede timing.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.pes import BKMP2Potential
from benchmarks.h3.rpmd import configuration
from benchmarks.loop_native.benchmark import initial, MASS_H, BETA, DT_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.cv import Distance, CVStack
from pyrpmd.bias import UmbrellaBias
from pyrpmd.workflows.fes import _sample_window, resolve_umbrellas
from pyrpmd.workflows.rate import _prepare, _child_task, rate_coordinate, project_positions
from pyrpmd.workflows.drpmd import _trajectory, ConnectivityClassifier


def equal(a, b, *, atol=2e-9, rtol=2e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.


def window_case(stage, directory):
    cfg, system = configuration(stage, directory, workers=1)
    cfg['sampling'].update(equilibration_steps=43, production_steps=210,
                           sample_interval=5, andersen_interval=7, replicas=1)
    cfg['output']['checkpoint_steps'] = 13
    cfg['analysis']['bootstrap'] = 0
    if stage == 'rate':
        cfg = _prepare(cfg, system, directory / 'prepare')
    cv, specs, centers, stiffness = resolve_umbrellas(cfg, system)
    index = len(centers)//2
    outputs, timings = {}, {}
    for backend in ('python', 'native'):
        config = copy.deepcopy(cfg)
        config['dynamics'].update(loop_backend=backend, block_steps=17)
        task = dict(config=config, system=system, specs=specs, center=centers[index],
                    stiffness=stiffness, window=index, replica=0, output=str(directory/backend))
        started = time.perf_counter(); outputs[backend] = _sample_window(task)
        timings[backend] = time.perf_counter()-started
        again = _sample_window(task)
        np.testing.assert_array_equal(again['samples'], outputs[backend]['samples'])
        np.testing.assert_array_equal(again['energies'], outputs[backend]['energies'])
    # Interrupt after an actually persisted checkpoint, then resume with a
    # different block size. Tuning must not change RNG/event or physics identity.
    from unittest.mock import patch
    import pyrpmd.workflows.fes as fes_module
    task['output'] = str(directory/'interrupted')
    original_save = fes_module._atomic_npz
    def interrupt_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        raise InterruptedError('intentional persisted-checkpoint interruption')
    with patch.object(fes_module, '_atomic_npz', interrupt_after_save):
        try:
            _sample_window(task)
        except InterruptedError:
            pass
        else:
            raise AssertionError('checkpoint interruption did not run')
    task['config']['dynamics']['block_steps'] = 11
    resumed = _sample_window(task)
    np.testing.assert_array_equal(resumed['samples'], outputs['native']['samples'])
    np.testing.assert_array_equal(resumed['last_q'], outputs['native']['last_q'])
    np.testing.assert_array_equal(resumed['last_p'], outputs['native']['last_p'])
    a, b = outputs['native'], outputs['python']
    errors = {key: equal(a[key], b[key], atol=2e-8) for key in ('samples', 'energies', 'last_q', 'last_p')}
    assert a['propagation']['native_steps'] == 253
    return dict(stage=stage, samples=len(a['samples']), maximum_errors=errors,
                seconds=timings, propagation=a['propagation'], resume_exact=True, interrupted_resume_exact=True)


def child_case(directory):
    cfg, system = configuration('rate', directory, workers=1)
    cfg = _prepare(cfg, system, directory / 'prepare')
    xi = 1.00925
    surface = rate_coordinate(cfg, system).dividing_surface(xi)
    q = np.repeat(system.geometry[None], cfg['beads'], axis=0)
    q = project_positions(q, surface, system.masses)
    outputs, timings = {}, {}
    for backend in ('python', 'native'):
        config = copy.deepcopy(cfg); config['dynamics'].update(loop_backend=backend, block_steps=37)
        task = dict(config=config, system=system, chain=0, parent=0, pair=0,
                    child_steps=500, xi_star=xi, q=q, output=str(directory/backend))
        started = time.perf_counter(); outputs[backend] = _child_task(task)
        timings[backend] = time.perf_counter()-started
        again = _child_task(task)
        np.testing.assert_array_equal(again['numerator'], outputs[backend]['numerator'])
    a, b = outputs['native'], outputs['python']
    np.testing.assert_array_equal(a['numerator'], b['numerator'])
    equal(a['denominator'], b['denominator'])
    equal(a['endpoint_energy_error_hartree_per_bead'], b['endpoint_energy_error_hartree_per_bead'])
    assert a['propagation']['native_steps'] == 1000
    return dict(stage='recrossing_children', timepoints=500, paired_children=2,
                all_timepoint_numerators_exact=True, seconds=timings,
                propagation=a['propagation'], resume_exact=True)


def collision_case(directory):
    q, p = initial(16)
    # Separated outgoing H + H2 checks exact stopping/classification semantics.
    p[:, 0, 0] -= 20.
    classifier = ConnectivityClassifier(['H']*3, MASS_H, [[0], [1, 2]], {'H-H': 2.0}, {}, 6.)
    config = {'beads':16, 'dt_au':DT_AU, 'collision':{'max_steps':97},
              'outcomes':{'check_interval':7, 'persistence_steps':11},
              'output':{'trajectory_stride':9, 'checkpoint_steps':13}}
    outputs = {}
    for backend in ('python', 'native'):
        dyn = Dynamics(BKMP2Potential(), MASS_H, BETA, 16, DT_AU, loop_backend=backend, block_steps=19)
        outputs[backend] = _trajectory(dyn, State(q, p), classifier, config,
            directory/backend, 0, 0, {}, 'fused-regression', False)
    a, b = outputs['native'], outputs['python']
    assert (a['status'], a['channel'], a['steps']) == (b['status'], b['channel'], b['steps'])
    assert a['status'] == 'nonreactive' and a['steps'] < 97
    equal(a['max_energy_drift_per_bead'], b['max_energy_drift_per_bead'])
    for key in ('final_q','final_p','steps','centroid','energy_per_bead'):
        with np.load(directory/'native/trajectories/trajectory-00000000.npz') as x, np.load(directory/'python/trajectories/trajectory-00000000.npz') as y:
            equal(x[key], y[key])
    assert a['propagation']['native_steps'] == a['steps']
    return dict(stage='drpmd_production', status=a['status'], steps=a['steps'],
                propagation=a['propagation'], exact_stop_and_classification=True)


def run(output, steps=1500, repeats=5):
    checks, timings = [], []
    cv = CVStack([Distance([0,1]), Distance([1,2])])
    for beads in (1,2,8,16,32):
        q,p=initial(beads)
        dyn=Dynamics(BKMP2Potential(), MASS_H, BETA, beads, DT_AU)
        a,b=State(q,p),State(q,p)
        n=dyn.propagate(a,121,observe=cv,backend='native',sample_interval=7)
        r=dyn.propagate(b,121,observe=cv,backend='python',sample_interval=7)
        checks.append(dict(beads=beads,q_error=equal(a.q,b.q),p_error=equal(a.p,b.p),
                           energy_error=equal(n.energies,r.energies),cv_error=equal(n.values,r.values)))
        assert (a.step,a.time)==(b.step,b.time)
    rng=np.random.default_rng(8103)
    for beads in (16,32):
        q,p=initial(beads)
        for workload in ('nve_energy_cv','umbrella2d'):
            bias=None if workload=='nve_energy_cv' else UmbrellaBias(cv,cv.value(q),[[.0002,0.],[0.,.002]])
            data={k:[] for k in ('python','native')}
            for repeat in range(repeats+1):
                for backend in rng.permutation(['python','native']):
                    dyn=Dynamics(BKMP2Potential(),MASS_H,BETA,beads,DT_AU,loop_backend=str(backend))
                    state=State(q,p)
                    start=time.perf_counter()
                    result=dyn.propagate(state,steps,bias=bias,observe=cv,sample_interval=1)
                    seconds=time.perf_counter()-start
                    assert result.backend==backend
                    if repeat:data[backend].append(seconds)
            med={k:statistics.median(v) for k,v in data.items()}
            timings.append(dict(beads=beads,workload=workload,steps=steps,repeats=repeats,seconds=data,
                                median_seconds=med,steps_per_second={k:steps/v for k,v in med.items()},
                                speedup=med['python']/med['native']))
    with tempfile.TemporaryDirectory(prefix='pyrpmd-fused-') as tmp:
        base=Path(tmp)
        workflows=[window_case('fes2d',base/'fes2d'),window_case('rate',base/'pmf'),
                   child_case(base/'children'),collision_case(base/'collision')]
    report=dict(schema='pyrpmd.fused-propagation.v1',passed=True,
                source_commit=os.environ.get('PYRPMD_TESTED_SHA', os.environ.get('GITHUB_SHA')),
                environment=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),
                numerical_checks=checks,workflows=workflows,timings=timings,
                scope='Integrated stateless BCB loops; not physical rate convergence. Native RATTLE parents and valid-prefix blocks are covered by native_propagation.py. Arbitrary Python dynamics callbacks and MPI-bead/GPU providers retain reference execution.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/fused-native.json')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.steps<1 or args.repeats<2:parser.error('steps>=1 and repeats>=2 required')
    run(args.output,args.steps,args.repeats)
