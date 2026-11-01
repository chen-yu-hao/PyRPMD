"""Publish immutable adoption/start evidence while excluding active logs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'pyrpmd-convergence-integration'
ROOT=BASE/'literature-convergence-quantum/p256-fixed-eight-chain-v1'
PREFIX=Path('docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-child-launch')
OUT=REPO/PREFIX
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    files={}
    for name in ['authorization.json','resource-decision.json','preflight.json','launch.json','umbrella-control-at-adoption.json']:
        files[name]=ROOT/'child-execution'/name
    for name in ['protocol.json','coordinator-registration.json']:
        files[name]=ROOT/'recrossing'/name
    files.update({'parent-root-exec-exit.json':ROOT/'execution/root-exec-exit.json',
        'parents-complete.json':ROOT/'parents/parents-complete.json',
        'launch_p256_children.py':BASE/'launch_p256_children.py',
        'prepare_p256_child_adoption.py':BASE/'prepare_p256_child_adoption.py',
        'curate_launch.py':Path(__file__),
        'convergence-parent-complete-aggregate-update.json':BASE/'convergence-parent-complete-aggregate-update.json',
        'convergence-fourth-p256-coordinator-copy.json':BASE/'convergence-fourth-p256-coordinator-copy.json',
        'convergence-fourth-p256-parent-stage-copy.json':BASE/'convergence-fourth-p256-parent-stage-copy.json'})
    auth=json.loads(files['authorization.json'].read_text())
    if sha(files['authorization.json'].read_bytes())!='3cefa11143fbcd2868f84b7d4ebf034ca48c0412cd55a6c119e70fecbf9ff384':
        raise ValueError('Adopted authorization changed')
    for key in ['synthetic_gate','independent_coordinator_review','scientific_review','resource_decision','parent_manifest']:
        row=auth[key];path=Path(row['path'])
        if sha(path.read_bytes())!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Bound completed prerequisite changed')
    for name,source in files.items():
        with (OUT/name).open('xb') as stream:stream.write(source.read_bytes())
    attempt=ROOT/'recrossing/coordinator-attempts/abc9e44ad7c24313b23f64741046d2c1'
    journal=attempt/'events.jsonl';raw=journal.read_bytes()
    first_two=raw.splitlines(keepends=True)[:2];events=[json.loads(line) for line in first_two]
    if [(e['event'],e['seed'],e['cpu']) for e in events]!=[('worker_started',825601,6),('worker_started',825603,7)]:
        raise ValueError('Observed first admissions differ')
    observation=dict(schema='pyrpmd.p256-child-start-observation.v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        exec_session_id=30588,launch_before_subprocess_utc=json.loads((OUT/'launch.json').read_text())['before_subprocess_call_utc'],
        attempt=attempt.name,actual_first_worker_start_events=events,
        committed_event_prefix=dict(path=str(journal),bytes=len(b''.join(first_two)),sha256=sha(b''.join(first_two))),
        status='initial two workers running; no complete scientific result',
        cohort_complete_claimed=False,rate_or_kappa_inspected=False,active_full_log_copied=False,
        live_process_token_published=False)
    (OUT/'start-observation.json').write_text(json.dumps(observation,indent=2)+'\n')
    (OUT/'README.md').write_text('''# P256 complete-child campaign: actual adoption and start

The fixed eight-chain parent cohort completed normally and passed independent
original/extracted verification. The reviewed child tools passed their
three-version compatibility gate; the separately fixed actual P256 native
calibration passed all360 map steps, including exact native save/recreate/reload.

Root then adopted the fixed P256 child design. The read-only preflight verified
76 bound files. Actual launch occurred at **2026-11-01 20:25:26.775457 UTC**;
the first two seed workers started at20:25:27 on CPU IDs6 and7. The allocation
is two P256 workers, up to four umbrella workers and two remaining original
water workers. Every numerical worker has one thread. The separately chosen
36,000-second whole-campaign limit includes a120-second finalization reserve.

All eight seeds must execute the same ordered four branches: batch0 dt0.1,
batch0 dt0.05, batch1 dt0.1, batch1 dt0.05. Each branch uses all200 parents.
Global completion requires32 complete branches,800 raw chunks and76.8 million
native physical steps, after every owned worker has exited. No subset can
produce an eligible global result. The coordinator performs no rate analysis
and cannot automatically retry or resume a nonempty campaign output.

This is **start evidence, not a P256 rate or bead-convergence result**. The
future complete comparison uses matching P256 normalization, eight independent
chains per bead count, same-chain covariance and the registered whole-interval
equivalence criterion. The P128 supplementary parent-half failure remains
explicit. See the [design](../p256-fixed-eight-chain-design/README.md),
[parent archive](../p256-parent-stage/README.md),
[child contracts](../p256-child-tool/README.md), and
[coordinator review](../p256-child-coordinator/README.md).

Only immutable adoption, preflight, source and registration records are copied
here. The two-event journal prefix is explicitly a historical start snapshot.
Active logs, live process tokens and incomplete trajectory arrays are excluded.
Final process exit and complete-cohort numerical analysis require later records.
''')
    rows=[dict(path=str(p.relative_to(REPO)),sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
        for p in sorted(OUT.rglob('*')) if p.is_file()]
    inventory=dict(schema='pyrpmd.p256-child-launch.publication-inventory.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),files=rows,complete_scientific_result=False,
        raw_arrays_included=False,active_logs_included=False)
    path=OUT/'publication-inventory.json'
    with path.open('x') as stream:stream.write(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),files=len(rows),bytes=sum(x['bytes'] for x in rows))))

if __name__=='__main__':
    main()
