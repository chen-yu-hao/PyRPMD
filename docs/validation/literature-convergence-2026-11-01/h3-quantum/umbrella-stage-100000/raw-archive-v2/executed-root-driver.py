"""Root adoption of the reviewed, read-only complete-cohort archive only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

WORK = Path('/workspace/scratch/a5e4820daedd')
ROOT = WORK / 'PyRPMD'
BASE = ROOT / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-100000'
OUT = BASE / 'archive-root-adoption-v2'


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify_inventory(relative, expected):
    path = BASE / relative / 'publication-inventory.json'
    assert identity(path)['sha256'] == expected
    inventory = json.loads(path.read_text())
    for row in inventory['files']:
        actual = identity(ROOT / row['path'])
        assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], row['path']
    return identity(path)


def main():
    inventories = [
        verify_inventory('archive-preparation', '6fff89937fa5e11ef58c4fb70ed3e676a03b6fffd9fce784a000970c20e0f79c'),
        verify_inventory('archive-preparation-v2', 'b69ac4725a8ada021d33b51f9e82e58b0fe59bdcd61d5cb5aee103d207b667a1'),
    ]
    review = BASE / 'archive-preparation-v2/independent-review.json'
    assert identity(review)['sha256'] == '4dbede9eccdcd532180baaee4463e1f722b4027c64624afd70ae3c3b3243c898'
    review_data = json.loads(review.read_text())
    assert review_data['approved_for_one_root_adopted_read_only_archive_and_exact_replay'] is True
    assert review_data['remaining_blockers_in_reviewed_scope'] == []
    template = BASE / 'archive-preparation-v2/root-adoption-template.json'
    assert identity(template)['sha256'] == 'c8f5630b6ab1ab9a1c181d2de971f0721f0f31f1cc95d7d96711c9e5c5deb4a7'
    adoption = json.loads(template.read_text())
    for kind, session in [('sampling', 57942), ('analysis', 1094)]:
        bound = adoption[kind + '_exit']['actual_tool_observation']
        assert identity(Path(bound['path'])) == bound
        observed = json.loads(Path(bound['path']).read_text())
        assert observed['exec_session_id'] == session
        assert type(observed['tool_result']['exit_code']) is int and observed['tool_result']['exit_code'] == 0
    control = WORK / 'literature-convergence-quantum/umbrella-worker-admission-2-to-4.json'
    control_data = json.loads(control.read_text())
    policy = json.loads((BASE / 'archive-preparation-v2/policy.json').read_text())
    adoption.update({
        'adopted': True,
        'adopted_utc': datetime.now(timezone.utc).isoformat(),
        'CPU_affinity': [5],
        'independent_review': identity(review),
        'worker_control_change': identity(control),
        'read_only_lease_anomaly_exception_adopted': True,
        'no_pending_or_running_300000': True,
        'root_no_writer_attestation': policy['required_root_no_writer_attestation'],
        'lease_absence_claimed': False,
        'source_inventory_verification': inventories,
        'root_driver': identity(Path(__file__)),
        'decision_scope': 'One read-only archive and one frozen-analyzer replay on fresh extraction. No sampler, source/data/lease mutation, automatic retry or 300000 continuation is authorized by this adoption.',
        'resource_decision': 'CPU5, one thread, nice10; unchanged observed aggregate300CPU-s,900wall-s,1GiB RSS envelope. P256 two workers remain on CPU6/7. No other physical process uses CPU5.',
        'no_writer_basis': 'Root observed actual normal exits for owned sessions57942 and1094. All eight owned sampling workers are recorded drained. Root has admitted no other legacy/direct/300000 writer; all agents are instructed not to write this original campaign. No writer will be admitted until this attempt ends. The two exact present lease records remain untouched; PID is not used as liveness proof.',
    })
    adoption.pop('template_status')
    OUT.mkdir(exist_ok=False)
    path = OUT / 'root-adoption.json'
    with path.open('x') as stream:
        json.dump(adoption, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'adoption': identity(path), 'verified_inventories': inventories, 'CPU_affinity': [5], 'physical_sampling_authorized': False}))


if __name__ == '__main__':
    main()
