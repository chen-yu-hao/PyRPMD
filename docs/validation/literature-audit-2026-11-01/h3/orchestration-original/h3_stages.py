"""Resource-bounded H3 audit orchestration; preserve every production CV.

Uses unchanged PyRPMD sampling/estimators, splitting sampling from analysis.
Only the redundant intermediate histogram MBAR bootstrap is disabled; the
final UI rate PMF receives 100 block draws, and recrossing retains its profile.
This script is archived with the raw dataset, not part of the public API.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/scratch/a5e4820daedd/PyRPMD')
sys.path.insert(0, str(ROOT))
from benchmarks.h3.sampling_control import resolved_control, source_snapshot
from benchmarks.published_rates.h3 import configuration
from pyrpmd.io import write_json
from pyrpmd.parallel import run_tasks
from pyrpmd.workflows import fes, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sample', 'analyze'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--beads', type=int, default=128)
    parser.add_argument('--workers', type=int, default=3)
    a = parser.parse_args()
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output/'run'
    out.mkdir(exist_ok=True)
    if a.beads == 128:
        config, system = configuration(a.output, a.library, profile='audit', seed=a.seed, workers=a.workers)
    else:
        config, system = resolved_control(a.output, a.library, beads=a.beads, seed=a.seed, workers=a.workers)
    snapshot = source_snapshot()
    write_json(a.output/('settings-'+a.stage+'.json'), {'config': config, 'system': system.to_dict(),
                                                       'source_at_start': snapshot})
    if not (a.output/'settings.json').exists():
        write_json(a.output/'settings.json', {'config': config, 'system': system.to_dict()})
    prepared = rate._prepare(config, system, out)
    print(json.dumps({'event':'start', 'stage':a.stage, 'seed':a.seed,'beads':a.beads}),flush=True)
    if a.stage == 'sample':
        cv, specs, centers, stiffness = fes.resolve_umbrellas(prepared, system)
        tasks = [dict(config=prepared, system=system, specs=specs,
                      window=i,replica=r,center=center,stiffness=stiffness,
                      output=str(out/'umbrella'))
                 for i,center in enumerate(centers)
                 for r in range(prepared['sampling']['replicas'])]
        streams = list(run_tasks(fes._sample_window, tasks, prepared['parallel']))
        detail = {'streams':len(streams), 'saved_samples':sum(len(s['samples']) for s in streams),
                  'propagation':[s['propagation'] for s in streams]}
    else:
        # reanalyze_fes validates every checkpoint's physics fingerprint.
        sampling = copy.deepcopy(prepared)
        sampling['analysis']['bootstrap'] = 0
        sampled = fes.reanalyze_fes(sampling, system, out/'umbrella', return_data=True)
        prepared['analysis']['bootstrap'] = 100
        options = rate._recrossing_options(prepared)
        pmf, parent_tasks = rate._pmf_and_parent_tasks(sampled, prepared, system, out, options)
        def dispatch(fun, tasks):
            return run_tasks(fun, tasks, prepared['parallel'])
        if parent_tasks:
            parents = dispatch(rate._parent_task, parent_tasks)
            children_tasks = rate._child_tasks(parents, prepared, system, out, options, pmf)
            children = rate._dispatch_children(children_tasks, prepared, dispatch)
        else:
            children = []
        detail = rate._aggregate(children, pmf, prepared, out, options)
        write_json(out/'summary.json', detail)
        (out/'input').mkdir(exist_ok=True)
        write_json(out/'input/resolved.json', prepared)
    final = source_snapshot()
    write_json(a.output/('stage-'+a.stage+'.json'), {
        'stage':a.stage, 'beads':a.beads, 'seed':a.seed,
        'elapsed_seconds':time.perf_counter()-started,
        'source_at_start':snapshot,'source_at_end':final,
        'sources_unchanged':snapshot==final,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'physics_converged':False,
        'analysis_changes':({'intermediate_mbar_bootstrap':0,'final_ui_bootstrap':100} if a.stage=='analyze' else {}),
        'detail':detail,
    })
    print(json.dumps({'event':'finished','stage':a.stage, 'elapsed_seconds':time.perf_counter()-started,
                      'rate':detail.get('rate_estimate_cm3_s'),'samples':detail.get('saved_samples')}),flush=True)

if __name__ == '__main__':
    main()
