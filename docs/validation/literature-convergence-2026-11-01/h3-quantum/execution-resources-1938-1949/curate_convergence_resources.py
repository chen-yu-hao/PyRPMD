"""Freeze completed scheduling decisions; never copy mutable active logs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

BASE = Path('/workspace/scratch/a5e4820daedd')
SOURCE = BASE/'PyRPMD'
REPO = BASE/'pyrpmd-convergence-integration'
PREFIX = Path('docs/validation/literature-convergence-2026-11-01')
OUT = PREFIX/'h3-quantum/execution-resources-1938-1949'
sha = lambda raw: hashlib.sha256(raw).hexdigest()

def main():
    target = REPO/OUT
    target.mkdir(exist_ok=False)
    resource = BASE/'literature-convergence-quantum/p256-fixed-eight-chain-v1/execution/resource-decision.json'
    record = json.loads(resource.read_text())
    files = {}
    for row in record['evidence']:
        source = Path(row['path'])
        raw = source.read_bytes()
        if sha(raw) != row['sha256'] or len(raw) != row['bytes']:
            raise ValueError('Resource prerequisite changed')
        files[source.name] = (source,raw)
    for source in [resource,
        BASE/'increase_umbrella_worker_admission.py',
        BASE/'literature-convergence-quantum/umbrella-worker-control.json',
        BASE/'convergence-four-pair-aggregate-update.json',
        BASE/'convergence-fourth-p256-four-pair-copy.json',
        BASE/'convergence-fourth-hch4-water-plot-copy.json',
        Path(__file__)]:
        files[source.name] = (source,source.read_bytes())
    admission=json.loads(files['umbrella-worker-admission-2-to-4.json'][1])
    if sha(files['umbrella-worker-control.json'][1]) != admission['new_control_sha256']:
        raise ValueError('Current control no longer matches observed admission')
    if sha(files['increase_umbrella_worker_admission.py'][1]) != admission['driver_sha256']:
        raise ValueError('Admission driver changed')
    rows=[]
    for name,(source,raw) in sorted(files.items()):
        with (target/name).open('xb') as stream:
            stream.write(raw)
        rows.append(dict(path=str(OUT/name),sha256=sha(raw),bytes=len(raw)))
    readme='''# Completed resource decisions, 19:38–19:49 UTC

The complete P128 extra-child process pool exited normally. At 19:38:18 UTC
its two CPU slots were reassigned to the unchanged quantum umbrella scheduler,
raising its admission ceiling from two to four. The receipt binds the exact
control bytes, frozen scheduler, completed cohort and observed process exit.
The copied control file is a historical snapshot, not a claim about future
resource allocation.

Water queues a and b completed all their original work. The original bounded
coordinator subsequently exited normally at its 19:44 boundary and released
all eleven leases. Its unchanged continuation admits only the two remaining
water seeds. The per-queue and whole-coordinator receipts preserve that
distinction and bind the committed journal prefixes and complete manifests.

At 19:48:59 UTC root adopted the fixed P256 **parent** campaign with two
single-threaded workers pinned to CPU IDs 6 and 7 and a 7,200-second wall
budget. The resulting long-work allocation was umbrella four, water two,
parents two, totaling eight. The parent resource decision does not authorize
child dynamics or any water preparation extension. No sampler, RNG, physical
parameter or scientific threshold changed with these scheduling decisions.

This bundle copies only completed, checksummed decisions and source/copy
receipts. Mutable campaign logs, partial scientific estimates and active
checkpoint arrays are excluded. Parent completion and any later child
adoption are separate records. The aggregate-document update receipt makes
its newer literature inventory identity explicit; it does not retroactively
change the earlier 315-case compatibility scope.
'''
    with (target/'README.md').open('x') as stream:
        stream.write(readme)
    raw=(target/'README.md').read_bytes()
    rows.append(dict(path=str(OUT/'README.md'),sha256=sha(raw),bytes=len(raw)))
    inventory=dict(schema='pyrpmd.resource-decisions.publication-inventory.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),files=sorted(rows,key=lambda x:x['path']),
        raw_checkpoints_included=False,active_logs_included=False)
    output=target/'publication-inventory.json'
    with output.open('x') as stream:
        stream.write(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output.read_bytes()),files=len(rows),bytes=sum(r['bytes'] for r in rows))))

if __name__ == '__main__':
    main()
