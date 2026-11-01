"""Adopt the reviewed metadata waiter without asserting future completion."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/scratch/a5e4820daedd/PyRPMD')
BUNDLE = ROOT / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-completion-tool'


def identity(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    inventory_path = BUNDLE / 'publication-inventory.json'
    assert identity(inventory_path)['sha256'] == 'c16c2d38582639169cb707b85a956686e89cdf1c604a33ad2d3be32c50328cb5'
    inventory = json.loads(inventory_path.read_text())
    for row in inventory['files']:
        actual = identity(ROOT / row['path'])
        assert (actual['bytes'], actual['sha256']) == (row['bytes'], row['sha256']), row['path']
    assert len(inventory['files']) == 20
    policy_path = BUNDLE / 'policy.json'
    review_path = BUNDLE / 'independent-review.json'
    source_path = ROOT / 'benchmarks/h3/p256_complete_analysis.py'
    assert identity(policy_path)['sha256'] == 'ff5be96fe6412a4d98f03019d5c7107bf7bbd7a7dfcb2426455412562bb30858'
    assert identity(review_path)['sha256'] == 'd9e16de758b1c0fbfbcb78b11b08416a7923e365640ad52ec85f1130abfd6862'
    assert identity(source_path)['sha256'] == '67e4e3eb1b7a7cc875fd14c74829c71be265a8171c516b7b10af5dca461d7b7e'
    policy = json.loads(policy_path.read_text())
    review = json.loads(review_path.read_text())
    assert review['approved_for_complete_analysis_once'] is True
    assert review['remaining_blockers'] == []
    assert review['policy_sha256'] == identity(policy_path)['sha256']
    assert review['source_sha256'] == identity(source_path)['sha256']
    assert policy['analysis_resources']['CPU_affinity'] == [6]
    assert policy['analysis_resources']['complete_verification_analysis_and_finalization_wall_seconds'] == 600
    assert not Path(policy['paths']['execution']).exists()
    assert not Path(policy['paths']['analysis_output']).exists()
    assert not Path(policy['paths']['root_observed_exit']).exists()
    adoption = {
        'schema': 'pyrpmd.h3.p256-complete-analysis-adoption.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'adopted': True,
        'adopted_by': 'root after independent review and frozen inventory verification',
        'policy_sha256': identity(policy_path)['sha256'],
        'source_sha256': identity(source_path)['sha256'],
        'independent_review': identity(review_path),
        'preparation_inventory': identity(inventory_path),
        'analysis_resources': policy['analysis_resources'],
        'waiting': policy['waiting'],
        'resource_decision': 'Metadata polling only while the two-worker P256 campaign runs. The single CPU 6 analysis starts only after all P256 workers are drained and actual exec session 30588 has returned normally; it then replaces one released worker slot.',
        'no_claim_of_current_campaign_completion_or_convergence': True,
        'root_exit_attestation': 'The actual session 30588 terminal observation must be recorded later. No future or inferred success receipt is created by this adoption.',
        'scientific_failure_policy': 'Retain every complete-cohort gate result, including false; no automatic retry, extension, estimator change, subset or replacement.',
        'integration_tree_modified': False,
    }
    destination = BUNDLE / 'root-adoption.json'
    with destination.open('x') as stream:
        json.dump(adoption, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'adoption': identity(destination), 'verified_preparation_files': len(inventory['files']), 'campaign_completed': False}))


if __name__ == '__main__':
    main()
