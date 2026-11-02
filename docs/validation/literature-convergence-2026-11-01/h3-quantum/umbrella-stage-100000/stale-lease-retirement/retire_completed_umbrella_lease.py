"""Retire one independently reviewed, completed root-owned scheduler lease."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'PyRPMD'
STAGE = REPO / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-100000'
REVIEW = STAGE / 'stale-lease-review/independent-review.json'
OUT = STAGE / 'stale-lease-retirement'
SOURCE = ROOT / 'literature-convergence-quantum/campaign-run/.quantum-scheduler-lock'
DESTINATION = ROOT / 'literature-convergence-quantum/retired-leases/umbrella-stage100000-owner-6e0835f4f471'


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def metadata(path):
    s = path.lstat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'mode': s.st_mode,
            'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def main():
    assert identity(REVIEW)['sha256'] == '8147f3f39ddb126cf290a9c6c2699c03d7ba81486b2deaba1303b9c165c94672'
    review = json.loads(REVIEW.read_text())
    assert review['conditional_retirement_justified'] is True
    assert review['remaining_blockers_for_root_recorded_one_time_retirement'] == []
    assert review['event_audit']['final_active_jobs'] == 0
    assert review['actual_normal_exit']['exec_session_id'] == 57942
    assert review['actual_normal_exit']['actual_exit_code'] == 0
    for row in review['input_identities']:
        assert identity(Path(row['path'])) == row, row['path']
    assert SOURCE.is_dir() and not SOURCE.is_symlink()
    assert sorted(p.name for p in SOURCE.iterdir()) == ['owner.json']
    assert not (SOURCE / 'owner.json').is_symlink()
    assert not DESTINATION.exists()
    owner = identity(SOURCE / 'owner.json')
    assert owner['bytes'] == 68 and owner['sha256'] == '6e0835f4f471796640551c601603e7886a906fff20dbb18afd37f661b8893dff'
    assert not (SOURCE.parent / 'stage-300000.json').exists()
    assert not any((SOURCE.parent / 'scheduler-attempts').glob('stage-300000-*'))
    OUT.mkdir(exist_ok=False)
    before = {'directory': metadata(SOURCE), 'owner': metadata(SOURCE / 'owner.json')}
    adoption = {
        'schema': 'pyrpmd.h3.umbrella100000-completed-lease-retirement-adoption.v1',
        'created_before_action_utc': datetime.now(timezone.utc).isoformat(),
        'adopted_by': 'root, after actual session 57942 exit and independent complete-cohort lifecycle review',
        'independent_review': identity(REVIEW),
        'source_lease': str(SOURCE), 'exclusive_quarantine': str(DESTINATION),
        'exact_owner': owner, 'before_metadata': before,
        'root_no_new_writer_attestation': {
            'actual_57942_terminal_exit_observed': True,
            'all_submitted_workers_drained': True,
            'legacy_or_direct_sampler_writer_running_or_pending': False,
            'new100000_or300000_coordinator_admitted': False,
            'basis': 'Root owns the completed launch and scheduling decisions. Exact eight-worker drain and terminal tool observation are bound by the independent review. No subsequent sampler launch is authorized or issued by root or delegated agents.',
            'PID_alone_used_as_proof': False,
        },
        'action': 'One Linux renameat2 RENAME_NOREPLACE of this exact completed lease; preserve bytes and identity. Never delete a checkpoint, owner byte or failure record.',
        'expected_effect': 'The existing authorized metadata waiter 1094 may begin its one complete-cohort analysis. No second analyzer or new sampling is launched.',
        'cause_of_surviving_old_lease': 'unexplained; no general filesystem-failure claim',
        'automatic_retry': False,
        'scientific_inputs_changed': False,
    }
    write_new(OUT / 'root-adoption.json', adoption)
    with (OUT / 'owner-before.json').open('xb') as stream:
        stream.write((SOURCE / 'owner.json').read_bytes())
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    # Recheck all exact metadata immediately before the one mutation.
    assert metadata(SOURCE) == before['directory']
    assert metadata(SOURCE / 'owner.json') == before['owner']
    assert sorted(p.name for p in SOURCE.iterdir()) == ['owner.json']
    assert identity(SOURCE / 'owner.json') == owner
    for row in review['input_identities']:
        assert identity(Path(row['path'])) == row
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(SOURCE), -100, os.fsencode(DESTINATION), 1)
    if result != 0:
        code = ctypes.get_errno()
        write_new(OUT / 'failure.json', {'errno': code, 'error': os.strerror(code), 'no_retry': True})
        raise OSError(code, os.strerror(code))
    assert not SOURCE.exists()
    assert sorted(p.name for p in DESTINATION.iterdir()) == ['owner.json']
    assert identity(DESTINATION / 'owner.json')['sha256'] == owner['sha256']
    assert metadata(DESTINATION)['inode'] == before['directory']['inode']
    assert metadata(DESTINATION / 'owner.json')['inode'] == before['owner']['inode']
    for row in review['input_identities']:
        if row['path'] != owner['path']:
            assert identity(Path(row['path'])) == row
    receipt = {
        'schema': 'pyrpmd.h3.umbrella100000-completed-lease-retirement.v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'root_adoption': identity(OUT / 'root-adoption.json'),
        'actual_operation': 'renameat2 RENAME_NOREPLACE', 'actual_returncode': result,
        'old_lease_path_absent': not SOURCE.exists(),
        'quarantined_owner': identity(DESTINATION / 'owner.json'),
        'owner_original_bytes_preserved': (OUT / 'owner-before.json').read_bytes() == (DESTINATION / 'owner.json').read_bytes(),
        'directory_and_owner_inode_preserved': True,
        'after_metadata': {'directory': metadata(DESTINATION), 'owner': metadata(DESTINATION / 'owner.json')},
        'source_launch_global_and_journal_guards_unchanged': True,
        'new_sampling_or_analyzer_launched_by_this_action': False,
        'existing_authorized_waiter': 1094,
        'source_driver': identity(Path(__file__)),
    }
    write_new(OUT / 'retirement.json', receipt)
    print(json.dumps({'retirement': identity(OUT / 'retirement.json'), 'old_path_absent': True, 'quarantine': str(DESTINATION), 'new_analyzer_launches': 0}))


if __name__ == '__main__':
    main()
