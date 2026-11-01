"""Independent metadata review; no NumPy, NPZ decoding, PES or analysis call.

Reads the completed archive once as an opaque SHA256 stream. All other file
content reads are small source/JSON/log files. Writes only this review directory.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import stat
import time

WORKSPACE = Path('/workspace/scratch/a5e4820daedd')
ROOT = WORKSPACE / 'PyRPMD'
BASE = ROOT / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-100000'
RESTORED = WORKSPACE / 'h3-quantum-umbrella-stage-100000-restored-v2'
REPLAY = WORKSPACE / 'h3-quantum-umbrella-stage-100000-replay-v2'
OUT = Path(__file__).resolve().parent
SEEDS = [211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039]
observed = {}
checks = []


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def digest(path, *, opaque_archive=False):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'regular non-symlink file: ' + str(path))
    require(opaque_archive or (path.suffix != '.npz' and path.stat().st_size < 1_048_576),
            'small-file content read or explicitly opaque tar stream: ' + str(path))
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1_048_576), b''):
            sha.update(block)
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}
    observed[str(path)] = result
    return result


def bound(identity, *, opaque_archive=False):
    actual = digest(identity['path'], opaque_archive=opaque_archive)
    require(actual == identity, 'identity matches: ' + identity['path'])
    return actual


def load(path):
    digest(path)
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    cpu = time.process_time()
    inventory = load(BASE / 'raw-archive-v2/member-inventory.json')
    registered = load(BASE / 'raw-archive-v2/registered-members.json')
    execution = load(BASE / 'raw-archive-v2/execution.json')
    wrapper = load(BASE / 'raw-archive-v2/stage-snapshot.json')
    adoption = load(BASE / 'archive-root-adoption-v2/root-adoption.json')
    actual_exit = load(BASE / 'archive-root-adoption-v2/root-exec-observation.json')
    claim = load(BASE / 'raw-archive-v2/RUN_ONCE.json')
    replay_launch = load(BASE / 'raw-archive-v2/replay-launch.json')
    require(actual_exit['exec_session_id'] == 46558 and actual_exit['tool_result']['exit_code'] == 0,
            'actual root-observed archive session 46558 exit 0')
    require(execution['complete_archive'] and execution['exact_scientific_replay']
            and execution['owned_replay_drained'] and execution['replay_invocations'] == 1,
            'one complete, drained exact replay is reported')
    require(execution['all8_state_RNG_and_prefixes_equal']
            and execution['all_original_archived_extracted_bytes_verified'],
            'execution reports complete state/RNG/prefix and member-byte checks')
    for key in ('original_data_modified', 'original_lease_modified',
                'original_lease_absence_claimed', 'automatic300000_continuation',
                'physical_rate_or_dynamic_bead_validation_claimed', 'Git_mutated', 'upload_performed'):
        require(execution[key] is False, 'negative execution scope retained: ' + key)
    require(execution['new_PES_or_dynamics_calls'] == 0, 'replay performed no new PES/dynamics')
    require(inventory['members'] == registered['members'], 'exact registered/archive member list')
    require(inventory['archive'] == execution['archive'], 'archive identity agrees across receipts')
    bound(inventory['archive'], opaque_archive=True)
    bound(execution['member_inventory'])
    require(Path(execution['member_inventory']['path']).read_bytes()
            == (BASE / 'raw-archive-v2/member-inventory.json').read_bytes(), 'external/member copy byte identity')
    for identity in (execution['source'], execution['policy'], execution['frozen_predecessor'],
                     claim['adoption'], claim['policy'], claim['source'], adoption['independent_review']):
        bound(identity)
    require(claim['automatic_retry_or_continuation'] is False, 'permanent one-attempt claim retained')
    require(adoption['policy_sha256'] == execution['policy']['sha256']
            and adoption['source_sha256'] == execution['source']['sha256'], 'root adoption binds executed code/policy')
    members = {row['member']: row for row in inventory['members']}
    require(len(members) == len(inventory['members']) == inventory['member_count'] == 116,
            '116 unique archive members')
    require(sum(row['source']['bytes'] for row in members.values()) == inventory['member_bytes'],
            'member byte total')
    require(dict(Counter(row['category'] for row in members.values())) == inventory['category_counts'],
            'category count reconstruction')
    expected_npz = {'data/full-cohort-stage-100000/pooled-wham.npz'}
    for seed in SEEDS:
        expected_npz.update({f'data/campaign-run/seed-{seed}/checkpoint.npz',
                             f'data/campaign-run/seed-{seed}/stages/30000/state.npz',
                             f'data/campaign-run/seed-{seed}/stages/100000/state.npz',
                             f'data/full-cohort-stage-100000/seed-{seed}-wham.npz'})
    require({name for name in members if name.endswith('.npz')} == expected_npz
            and inventory['raw_NPZ_count'] == registered['raw_NPZ_count'] == 33,
            'exact 33-NPZ whitelist: 8 checkpoints + 16 states + 9 WHAM arrays')
    require({str(path.relative_to(RESTORED)) for path in RESTORED.rglob('*') if path.is_file()}
            == set(members), 'verified extraction contains exactly the registered files')
    small_members = 0
    for name, row in members.items():
        parts = PurePosixPath(name)
        require(not parts.is_absolute() and '..' not in parts.parts
                and '.quantum-scheduler-lock' not in parts.parts,
                'safe relative member without live lease: ' + name)
        extracted = RESTORED / name
        require(stat.S_ISREG(extracted.lstat().st_mode)
                and extracted.stat().st_size == row['source']['bytes'], 'extracted regular-file size: ' + name)
        if not name.endswith('.npz'):
            original = bound(row['source'])
            got = digest(extracted)
            require((got['bytes'], got['sha256']) == (original['bytes'], original['sha256']),
                    'original/extracted small-file equality: ' + name)
            small_members += 1
    small_sources = 0
    for identity in registered['sources']:
        if identity['bytes'] < 1_048_576 and not identity['path'].endswith('.npz'):
            bound(identity)
            small_sources += 1
    completion_path = RESTORED / 'data/campaign-run/stage-100000.json'
    completion = load(completion_path)
    require(completion['schema'] == 'pyrpmd.h3.quantum-umbrella-scheduler-completion.v1'
            and completion['production'] == 100000, 'valid 100k scheduler completion schema')
    require(completion['requested_seeds'] == completion['completed_seeds'] == SEEDS
            and completion['all_submitted_workers_drained']
            and completion['prior_prefix_and_state_files_unchanged'], 'complete fixed eight-seed/drained manifest')
    require(wrapper['seeds'] == completion['snapshot']['seeds'], 'archive snapshot rows equal global completion rows')
    require(set(wrapper) == {'global_completion', 'prior_snapshot', 'seeds'},
            'archive wrapper is explicitly not the scheduler input schema')
    bound(wrapper['global_completion'])
    bound(wrapper['prior_snapshot'])
    prior = load(wrapper['prior_snapshot']['path'])
    old = {row['seed']: row for row in prior['seeds']}
    protocol = load(RESTORED / 'reference/protocol.json')
    require(protocol['independent_ladder_seeds'] == SEEDS
            and protocol['fixed_refinement_prefixes'] == [30000, 100000, 300000], 'unchanged fixed protocol')
    for row in wrapper['seeds']:
        seed = row['seed']
        require(row['completed_sweeps'] == 105000
                and row['arrays']['samples']['shape'] == [100000, 48]
                and row['arrays']['diagnostics']['shape'] == [10000, 48, 7]
                and row['arrays']['R']['shape'] == row['arrays']['r']['shape'] == [48, 128, 3],
                f'seed {seed}: completed counts/shape metadata')
        require(set(row['arrays']) == {'R', 'r', 'completed', 'rng', 'accepted', 'proposed',
                                     'swaps', 'lineage', 'samples', 'diagnostics'},
                f'seed {seed}: full restart/RNG/observation inventory')
        require(row['stage_receipts'][0] == old[seed]['stage_receipts'][0],
                f'seed {seed}: identical registered 30k prefix state/receipt identities')
        require(row['settings'] == old[seed]['settings'] and row['source_sha256'] == old[seed]['source_sha256'],
                f'seed {seed}: unchanged settings/model/source identities')
        for field in ('checkpoint', 'settings', 'run_report'):
            metadata = row[field]
            listed = members['data/campaign-run/' + metadata['path']]['source']
            require((metadata['bytes'], metadata['sha256']) == (listed['bytes'], listed['sha256']),
                    f'seed {seed}: {field} member identity')
        for stage in row['stage_receipts']:
            receipt = load(RESTORED / 'data/campaign-run' / stage['receipt']['path'])
            for field in ('receipt', 'state'):
                meta = stage[field]
                listed = members['data/campaign-run/' + meta['path']]['source']
                require((meta['bytes'], meta['sha256']) == (listed['bytes'], listed['sha256']),
                        f'seed {seed} stage {stage["production_sweeps"]}: {field} member identity')
            require(receipt['seed'] == seed and receipt['production_sweeps'] == stage['production_sweeps']
                    and receipt['state_sha256'] == stage['state']['sha256']
                    and receipt['samples_prefix_sha256'] == stage['samples_prefix_sha256']
                    and receipt['diagnostics_prefix_sha256'] == stage['diagnostics_prefix_sha256'],
                    f'seed {seed} stage {stage["production_sweeps"]}: cross-receipt prefix identity')
            if stage['production_sweeps'] == 100000:
                require(receipt['checkpoint_sha256'] == row['checkpoint']['sha256']
                        and stage['samples_prefix_sha256'] == row['arrays']['samples']['sha256']
                        and stage['diagnostics_prefix_sha256'] == row['arrays']['diagnostics']['sha256'],
                        f'seed {seed}: 100k current checkpoint/prefix identity')
    json_names = ['campaign-comparison.json'] + [f'seed-{seed}.json' for seed in SEEDS]
    for name in json_names:
        original = load(RESTORED / 'data/full-cohort-stage-100000' / name)
        replayed = load(REPLAY / name)
        if name == 'campaign-comparison.json':
            for field in ('created_utc', 'elapsed_seconds'):
                original.pop(field)
                replayed.pop(field)
        require(original == replayed, 'independently compared saved scientific JSON: ' + name)
    report = load(RESTORED / 'data/full-cohort-stage-100000/campaign-comparison.json')
    stats = report['statistics']
    require(report['expected_independent_seeds'] == report['completed_independent_seeds'] == SEEDS,
            'analysis retains all eight seeds')
    require(stats['primary_log_standard_error']
            == max(stats['between_ladder_log_standard_error'], stats['jackknife_log_standard_error']),
            'reported conservative primary SE equals max of two registered SEs')
    require(stats['primary_log_standard_error'] > protocol['gates']['target_qtst_relative_standard_error']
            and report['criteria']['statistical_precision'] is False
            and report['registered_primary_qtst_gates_passed'] is False,
            'registered precision failure retained without threshold change')
    require(report['criteria'] == execution['original_scientific_criteria'], 'archive retains every original criterion')
    require(report['full_rpmd_rate_validated'] is False and report['transmission_coefficient'] is None,
            'equilibrium archive is not a full RPMD rate validation')
    result = {
        'schema': 'pyrpmd.h3.umbrella100k-completed-archive-small-file-review.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'all_checks_passed': True,
        'checks': checks, 'independently_observed_file_identities': list(observed.values()),
        'archive_identity': inventory['archive'], 'member_count': len(members),
        'small_members_hashed_in_original_and_extraction': small_members,
        'small_external_source_or_identity_files_checked': small_sources,
        'NPZ_files_decoded': 0, 'new_analyzer_invocations': 0, 'new_PES_or_dynamics_calls': 0,
        'RNG_calls': 0, 'independently_compared_saved_scientific_JSON_count': len(json_names),
        'raw_array_replay_scope': 'The previously reviewed executor reports nine NPZ array-byte comparisons and eight full checkpoint/prefix/RNG checks; this metadata review did not repeat them.',
        'trust_boundary': 'Root saved actual execution observations/no-writer attestations; this reviewer does not infer process liveness from a PID or resolve the unexplained lease reappearance.',
        'scientific_precision': {'observed_log_standard_error': stats['primary_log_standard_error'],
                                 'registered_limit': protocol['gates']['target_qtst_relative_standard_error'],
                                 'passed': False},
        'elapsed_wall_seconds': time.monotonic() - started,
        'review_CPU_seconds': time.process_time() - cpu,
    }
    (OUT / 'checks.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('all_checks_passed', 'member_count',
          'small_members_hashed_in_original_and_extraction', 'NPZ_files_decoded', 'review_CPU_seconds')}, sort_keys=True))


if __name__ == '__main__':
    main()
