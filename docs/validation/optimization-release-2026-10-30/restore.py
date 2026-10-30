#!/usr/bin/env python3
"""Restore the measured checkouts offline without modifying the original evidence.

Use the same Python environment intended to import the preserved native binaries.
--smoke invokes the unchanged original harness once with no timing measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

BASELINE = '6fde29d2039c01921f00393bf17830b939af131b'
BASELINE_TREE = '80e21b261a1b3b66084af982ffd696907843f4cc'
CANDIDATE_TREE = '899cc0618ba92db03e891c236845dac671704c00'
ARCHIVE_SHA256 = '00daad7cf6e48ecbc17625500dcbf0a2a18eb30d732f22f58d5be123a7493cae'
BUNDLE_SHA256 = 'ab064c3b1aadcaba5212a0ca6d4b8cb8450f40914616a1ce706ac4782a7867a6'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_members(archive: tarfile.TarFile):
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members')
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f'Unexpected archive member: {member.name}')
    return members


def verify_manifest(path: Path, root: Path):
    entries = json.loads(path.read_text())
    for name, item in entries.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unexpected manifest path: {name}')
        file = root / name
        if file.stat().st_size != item['bytes'] or digest(file) != item['sha256']:
            raise ValueError(f'Manifest mismatch: {file}')
    return {'files': len(entries), 'bytes': sum(item['bytes'] for item in entries.values()), 'all_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path,
                        help='Original perf-audit-optimization-evidence.tar.gz')
    parser.add_argument('--destination', required=True, type=Path,
                        help='New directory; must not exist')
    parser.add_argument('--smoke', action='store_true',
                        help='Run all eight reduced functional cases, with no timings')
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    destination = args.destination.resolve()
    git_bundle = Path(__file__).resolve().with_name('baseline.bundle')
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError('Original evidence archive SHA256 mismatch')
    if digest(git_bundle) != BUNDLE_SHA256:
        raise ValueError('Baseline Git bundle SHA256 mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        'schema': 'pyrpmd.audit-restore.v1', 'passed': False,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(archive_path), 'archive_sha256': ARCHIVE_SHA256,
        'bundle_sha256': BUNDLE_SHA256, 'destination': str(destination),
        'restore_script_sha256': digest(Path(__file__)),
        'formal_performance_measurement': False, 'commands': [],
    }
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith('GIT_'):
            environment.pop(name)
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_CEILING_DIRECTORIES=str(destination),
                       PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONHASHSEED='0')
    environment.update({name: '1' for name in THREAD_ENV})

    def run(command, cwd, *, env=None):
        command = [str(part) for part in command]
        completed = subprocess.run(command, cwd=cwd, env=env or environment,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result['commands'].append({'command': command, 'cwd': str(cwd),
                                   'returncode': completed.returncode,
                                   'stdout': completed.stdout, 'stderr': completed.stderr})
        if completed.returncode:
            raise RuntimeError(f'Command failed: {command}\n{completed.stderr}')
        return completed.stdout.strip()

    try:
        unpacked = destination / 'evidence'
        unpacked.mkdir()
        with tarfile.open(archive_path) as archive:
            members = safe_members(archive)
            archive.extractall(unpacked, members=members, filter='data')
            result['archive_members'] = len(members)
        evidence = unpacked / 'perf-audit-optimization-evidence'
        raw = unpacked / 'perf-audit-optimization'
        result['raw_manifest'] = verify_manifest(evidence / 'raw-files-sha256.json', raw)
        result['support_manifest'] = verify_manifest(evidence / 'bundle-files-sha256.json', evidence)
        original_report = json.loads((raw / 'report.json').read_text())
        run_context = json.loads((evidence / 'run-context.json').read_text())
        if digest(raw / 'report.json') != run_context['report_sha256']:
            raise ValueError('Original report SHA256 mismatch')
        if run_context['candidate_staged_tree'] != CANDIDATE_TREE:
            raise ValueError('Unexpected candidate identity')
        result['original_report_sha256'] = digest(raw / 'report.json')
        result['checkouts'] = {}
        for variant, expected_tree in [('baseline', BASELINE_TREE), ('candidate', CANDIDATE_TREE)]:
            checkout = destination / variant
            checkout.mkdir()
            run(['git', 'init', '--quiet'], checkout)
            run(['git', 'config', 'core.autocrlf', 'false'], checkout)
            run(['git', 'bundle', 'verify', git_bundle], checkout)
            heads = run(['git', 'bundle', 'list-heads', git_bundle], checkout)
            if heads != f'{BASELINE} refs/heads/measured-baseline':
                raise ValueError('Bundle contains unexpected refs')
            run(['git', 'fetch', '--quiet', '--no-tags', git_bundle,
                 'refs/heads/measured-baseline:refs/heads/measured-baseline'], checkout)
            run(['git', 'checkout', '--quiet', '--detach', BASELINE], checkout)
            if variant == 'candidate':
                patch = evidence / 'candidate.patch'
                run(['git', 'apply', '--check', '--index', patch], checkout)
                run(['git', 'apply', '--index', patch], checkout)
            tree = run(['git', 'write-tree'], checkout)
            if tree != expected_tree:
                raise ValueError(f'{variant} source tree mismatch: {tree}')
            if run(['git', 'rev-parse', 'HEAD'], checkout) != BASELINE:
                raise ValueError(f'{variant} baseline HEAD mismatch')
            file_count = 0
            with tarfile.open(evidence / (variant + '-source.tar')) as sources:
                for member in safe_members(sources):
                    if not member.isfile():
                        continue
                    file_count += 1
                    actual = checkout / member.name
                    if digest(actual) != hashlib.sha256(sources.extractfile(member).read()).hexdigest():
                        raise ValueError(f'Full source mismatch: {variant}/{member.name}')
                    if bool(actual.stat().st_mode & 0o111) != bool(member.mode & 0o111):
                        raise ValueError(f'Source executable mode mismatch: {variant}/{member.name}')
            manifest = original_report['manifests'][variant]
            for name, expected in manifest['sources_sha256'].items():
                if digest(checkout / name) != expected:
                    raise ValueError(f'Measured source mismatch: {variant}/{name}')
            native_files = list((evidence / variant / 'src/pyrpmd').glob('_native*.so'))
            if len(native_files) != 1 or digest(native_files[0]) != manifest['native_extension_sha256']:
                raise ValueError(f'{variant} native binary mismatch')
            native = checkout / 'src/pyrpmd' / native_files[0].name
            shutil.copy2(native_files[0], native)
            probe = (
                'import json, pathlib, sys; import pyrpmd; from pyrpmd import _native; '
                'root=pathlib.Path(sys.argv[1]).resolve(); '
                'assert pathlib.Path(pyrpmd.__file__).resolve().is_relative_to(root); '
                'assert pathlib.Path(_native.__file__).resolve().is_relative_to(root); '
                'print(json.dumps({"package":pyrpmd.__file__,"native":_native.__file__, '
                '"version":pyrpmd.__version__,"has_openmp":bool(_native.has_openmp()), '
                '"has_sample_free":hasattr(_native,"sample_free")}))'
            )
            import_environment = environment.copy()
            import_environment['PYTHONPATH'] = os.pathsep.join([str(checkout / 'src'), str(checkout)])
            imported = json.loads(run([sys.executable, '-c', probe, checkout], checkout, env=import_environment))
            result['checkouts'][variant] = {
                'path': str(checkout), 'head': BASELINE, 'index_tree': tree,
                'all_archived_source_files_match': file_count,
                'measured_source_hashes_match': len(manifest['sources_sha256']),
                'native_sha256': digest(native), 'import_probe': imported,
            }
        library = evidence / 'pes/libbkmp2.so'
        for variant in ('baseline', 'candidate'):
            manifest = original_report['manifests'][variant]
            if digest(library) != manifest['pes_library_sha256']:
                raise ValueError('PES binary mismatch')
            if digest(library.with_suffix('.build.json')) != manifest['pes_build_provenance_sha256']:
                raise ValueError('PES provenance mismatch')
        harness = destination / 'candidate/benchmarks/efficiency/audit_optimization.py'
        if digest(harness) != original_report['harness_sha256']:
            raise ValueError('Original harness mismatch')
        command = [sys.executable, str(harness), '--baseline', str(destination / 'baseline'),
                   '--candidate', str(destination / 'candidate'), '--library', str(library),
                   '--output', str(destination / 'smoke-output'), '--smoke']
        result['smoke_command'] = command
        if args.smoke:
            run(command, destination)
            smoke = json.loads((destination / 'smoke-output/report.json').read_text())
            if not smoke['passed'] or smoke['mode'] != 'smoke_functional_only':
                raise ValueError('Original harness smoke validation failed')
            for name, case in smoke['cases'].items():
                if any(case['seconds_raw'].values()) or 'speedup' in case:
                    raise ValueError(f'Unexpected timings in functional smoke: {name}')
            result['smoke'] = {'passed': True, 'case_count': len(smoke['cases']),
                               'cases': list(smoke['cases']),
                               'report_sha256': digest(destination / 'smoke-output/report.json'),
                               'timings_collected': False}
        result['passed'] = True
    except BaseException as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        (destination / 'restore-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'commands'}, indent=2))


if __name__ == '__main__':
    main()
