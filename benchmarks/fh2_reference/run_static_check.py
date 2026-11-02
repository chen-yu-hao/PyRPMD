"""Build unchanged author sources and check their static Fortran interfaces.

Each variant is evaluated in a separate single-threaded process whose working
directory contains its pinned parameter files. This is not a production
potential provider, a thermal-rate calculation or an HSW force implementation.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scalar_function(library, name, is_function=False):
    entry = getattr(library, name)
    pointer = ctypes.POINTER(ctypes.c_double)
    entry.argtypes = [pointer] * (3 if is_function else 4)
    entry.restype = ctypes.c_double if is_function else None

    def evaluate(r):
        values = [ctypes.c_double(float(x)) for x in r]
        if is_function:
            return entry(*(ctypes.byref(x) for x in values))
        result = ctypes.c_double()
        entry(*(ctypes.byref(x) for x in values), ctypes.byref(result))
        return result.value

    return evaluate


def worker(variant, library_path, output):
    import numpy as np
    protocol = json.loads((HERE / 'static_protocol.json').read_text())
    library = ctypes.CDLL(str(Path(library_path).resolve()), mode=ctypes.RTLD_LOCAL)
    rows = []
    if variant == 'sw1':
        for name in ['inifit_', 'prepot_', 'pote_']:
            entry = getattr(library, name)
            entry.argtypes, entry.restype = [], None
        library.inifit_()
        library.prepot_()
        common = (ctypes.c_double * 7).in_dll(library, 'potcm_')
        energy = scalar_function(library, 'sw_')

        def derivative(r):
            for i, value in enumerate(r):
                common[i] = float(value)
            library.pote_()
            return float(common[3]), np.array(common[4:7], dtype=float)
    else:
        energy = scalar_function(library, 'pot_', is_function=True)
        base_energy = scalar_function(library, 'sw1_')

    for case in protocol['geometries']:
        r = np.array(case['distances_bohr'], dtype=float)
        if np.any(r <= 0) or any(r[i] >= sum(r) - r[i] for i in range(3)):
            raise ValueError('Predeclared distances are not a nonsingular triangle')
        value = energy(r)
        swapped = energy(r[[2, 1, 0]])
        row = {'id': case['id'], 'distances_bohr': r.tolist(), 'energy_hartree': value,
               'H_exchange_energy_error_hartree': abs(value - swapped)}
        if variant == 'sw1':
            value_d, gradient = derivative(r)
            _, swapped_g = derivative(r[[2, 1, 0]])
            checks = []
            for step in protocol['finite_difference_steps_bohr']:
                finite = []
                for coordinate in range(3):
                    offset = np.eye(3)[coordinate] * step
                    finite.append((energy(r + offset) - energy(r - offset)) / (2 * step))
                finite = np.array(finite)
                checks.append({'step_bohr': step, 'gradient_hartree_per_bohr': finite.tolist(),
                               'max_absolute_error': float(np.max(np.abs(finite - gradient)))})
            row.update(analytic_gradient_hartree_per_bohr=gradient.tolist(),
                       independent_energy_code_difference_hartree=abs(value - value_d),
                       H_exchange_gradient_error=float(np.max(np.abs(gradient - swapped_g[[2, 1, 0]]))),
                       finite_differences=checks)
        else:
            row['unmodified_sw1_component_hartree'] = base_energy(r)
            row['HSW_minus_SW1_hartree'] = value - row['unmodified_sw1_component_hartree']
            if 'HSW_author_decimal_hartree' in case:
                reference = case['HSW_author_decimal_hartree']
                row['author_decimal_hartree'] = reference
                row['author_decimal_absolute_difference_hartree'] = abs(value - reference)
                row['author_verify_real32_literal_promoted_to_real64'] = float(np.float32(reference))
                row['author_verify_literal_rounding_hartree'] = float(np.float32(reference)) - reference
        if not np.isfinite(value):
            raise ValueError('Native potential returned a nonfinite energy')
        rows.append(row)
    write(output, {'schema': 'pyrpmd.fh2.author-static-worker.v1', 'variant': variant,
                   'protocol_sha256': sha(HERE / 'static_protocol.json'),
                   'runner_sha256': sha(__file__), 'library_sha256': sha(library_path),
                   'working_directory': str(Path.cwd()), 'python': sys.version,
                   'numpy': np.__version__, 'geometries': rows,
                   'HSW_analytic_gradient_implemented': False, 'dynamics_steps': 0})


def build_and_check(archives, compiler, output):
    archives, compiler, output = Path(archives).resolve(), Path(compiler).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol_path, source_path = HERE / 'static_protocol.json', HERE / 'sources.json'
    protocol, sources = json.loads(protocol_path.read_text()), json.loads(source_path.read_text())
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    identities = {str(p): sha(p) for p in [protocol_path, source_path, Path(__file__)]}
    commands = []
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    cpu = min(os.sched_getaffinity(0))

    def bounded_child():
        os.sched_setaffinity(0, {cpu})
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

    def execute(command, cwd, label):
        if time.monotonic() - start > protocol['whole_call_wall_admission_limit_seconds']:
            raise TimeoutError('Static audit wall budget exhausted before a new subprocess')
        tick = time.monotonic()
        with (output / (label + '.stdout.txt')).open('xb') as stdout, (output / (label + '.stderr.txt')).open('xb') as stderr:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                    timeout=30, preexec_fn=bounded_child)
        commands.append({'label': label, 'command': command, 'cwd': str(cwd),
                         'returncode': result.returncode, 'wall_seconds': time.monotonic()-tick})
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; original logs retained')

    report = {'schema': 'pyrpmd.fh2.author-static-check.v1',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': identities, 'CPU_affinity': [cpu], 'numeric_threads': 1,
              'PES_coefficients_or_source_modified': False, 'potential_rate_reference_match': False,
              'thermal_rate_computed': False, 'paper_reproduction': False, 'dynamics_steps': 0}
    try:
        assets = []
        for variant in ['sw1', 'hsw']:
            record = sources['variants'][variant]
            archive = archives / record['archive_name']
            if sha(archive) != record['archive_sha256']:
                raise ValueError('Archive identity differs')
            directory = output / variant
            directory.mkdir()
            expected = {item['name']: item for item in record['members']}
            with tarfile.open(archive, 'r:*') as source:
                members = list(source)
                if {m.name for m in members} != set(expected) or len(members) != len(expected):
                    raise ValueError('Archive membership differs')
                for member in members:
                    if not member.isfile() or Path(member.name).name != member.name:
                        raise ValueError('Only the pinned flat regular-file archive is accepted')
                    data = source.extractfile(member).read()
                    row = expected[member.name]
                    if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
                        raise ValueError('Member identity differs')
                    path = directory / member.name
                    with path.open('xb') as stream:
                        stream.write(data)
                    assets.append({'path': str(path), 'bytes': len(data), 'sha256': row['sha256']})
            flags = list(protocol['compiler_flags']) + ['-Wl,-rpath,' + str(compiler.parent.parent / 'lib')]
            library = directory / ('lib' + variant + '_static.so')
            command = [str(compiler), *flags, '-shared', '-fPIC', *record['library_sources'], '-o', str(library)]
            execute(command, directory, variant + '-build')
            execute([sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                     '--library', str(library), '--output', str(output / (variant + '-results.json'))], directory, variant + '-checks')
            if variant == 'hsw':
                verify = directory / 'verify-original'
                execute([str(compiler), *flags, *record['library_sources'], 'verify.f', '-o', str(verify)], directory, 'hsw-author-build')
                execute([str(verify)], directory, 'hsw-author-verify')
            assets.append({'path': str(library), 'bytes': library.stat().st_size, 'sha256': sha(library)})
        sw = json.loads((output / 'sw1-results.json').read_text())['geometries']
        hsw = json.loads((output / 'hsw-results.json').read_text())['geometries']
        limits = protocol['acceptance']
        differences = [abs(a['energy_hartree'] - b['unmodified_sw1_component_hartree']) for a,b in zip(sw,hsw,strict=True)]
        author_stdout = (output / 'hsw-author-verify.stdout.txt').read_text()
        actual = [float(x.replace('D', 'E')) for x in re.findall(r'your calculated value:\s*([+\-\d.EeDd]+)', author_stdout)]
        author = [r for r in hsw if 'author_decimal_hartree' in r]
        if len(actual) != 2 or len(author) != 2:
            raise ValueError('Original verification output does not contain its two expected values')
        gates = {
            'SW1_two_energy_implementations': max(r['independent_energy_code_difference_hartree'] for r in sw) <= limits['energy_absolute_hartree'],
            'SW1_analytic_distance_gradient_all_three_steps': max(c['max_absolute_error'] for r in sw for c in r['finite_differences']) <= limits['gradient_absolute_hartree_per_bohr'],
            'SW1_H_exchange_gradient': max(r['H_exchange_gradient_error'] for r in sw) <= limits['gradient_symmetry_absolute_hartree_per_bohr'],
            'both_H_exchange_energies': max(r['H_exchange_energy_error_hartree'] for r in sw+hsw) <= limits['energy_absolute_hartree'],
            'same_SW1_component_in_both_archives': max(differences) <= limits['energy_absolute_hartree'],
            'HSW_two_author_decimal_targets': max(r['author_decimal_absolute_difference_hartree'] for r in author) <= limits['author_reference_absolute_hartree'],
            'HSW_unchanged_author_program_vs_wrapper': max(abs(value-row['energy_hartree']) for value,row in zip(actual,author,strict=True)) <= limits['energy_absolute_hartree'],
            'sources_unchanged': identities == {p:sha(p) for p in identities} and all(sha(row['path']) == row['sha256'] for row in assets),
        }
        report.update(gates=gates, all_static_checks_passed=all(gates.values()),
                      SW1_cross_archive_absolute_differences_hartree=differences,
                      external_assets=assets, expected_geometry_count=len(protocol['geometries']),
                      HSW_author_reference_note='Decimal values are retained as written; unchanged verify.f uses default-real literals assigned to double precision. Both the original executable and the decimal targets are reported; no constants or energy shifts are changed.')
    except BaseException as error:
        report.update(all_static_checks_passed=False, execution_error=f'{type(error).__name__}: {error}')
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report.update(commands=commands, whole_call_wall_seconds=time.monotonic()-start,
                      child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(output / 'report.json', report)
    print(json.dumps({k:report[k] for k in ['all_static_checks_passed','gates','whole_call_wall_seconds','child_CPU_seconds','paper_reproduction']}))
    return 0 if report['all_static_checks_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-dir', type=Path)
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['sw1','hsw'])
    parser.add_argument('--library', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.library, args.output)
        return 0
    if args.archives_dir is None or args.compiler is None:
        parser.error('--archives-dir and --compiler are required for the static audit')
    return build_and_check(args.archives_dir, args.compiler, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
