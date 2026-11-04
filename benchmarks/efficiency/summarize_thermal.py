#!/usr/bin/env python3
"""Collect complete thermal benchmark JSONs and compute the T49 gates."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    names = ['baseline-dynamics', 'candidate-mt-dynamics', 'candidate-philox-dynamics',
             'baseline-native-matrix', 'candidate-mt-native-matrix',
             'candidate-philox-native-1', 'candidate-philox-native-2', 'candidate-philox-native-4']
    if (args.directory/'candidate-philox-dynamics-4.json').exists():
        names.append('candidate-philox-dynamics-4')
    reports = {name: json.loads((args.directory/(name+'.json')).read_text()) for name in names}
    first = reports[names[0]]['environment']
    for name, report in reports.items():
        if not report['complete']:
            raise ValueError(f'incomplete benchmark: {name}')
        for key in ('hostname', 'cpu', 'python', 'numpy', 'compiler', 'affinity', 'build_flags', 'script_sha256'):
            if report['environment'][key] != first[key]:
                raise ValueError(f'unmatched benchmark environment: {name}, {key}')
    def row(name, atoms, beads=32, thermostat='pile-l'):
        return next(r for r in reports[name]['measurements']
                    if (r['atoms'], r['beads'], r['thermostat']) == (atoms, beads, thermostat))
    gates = {}
    for atoms in (512, 4096):
        value = row('candidate-philox-dynamics', atoms)['steps_per_second']/row('candidate-philox-dynamics', atoms, thermostat='none')['steps_per_second']
        gates[f'philox_pile_l_over_nve_N{atoms}'] = {'value': value, 'minimum': .5, 'passed': value >= .5}
    value = row('candidate-philox-native-4', 4096)['steps_per_second']/row('candidate-philox-native-1', 4096)['steps_per_second']
    gates['native_philox_four_thread_speedup'] = {'value': value, 'minimum': 2.5, 'passed': value >= 2.5}
    value = row('candidate-mt-dynamics', 4096)['short_call_profile']['outside_over_native']
    gates['two_step_non_native_overhead'] = {'value': value, 'maximum': .1, 'passed': value <= .1}
    for beads in (1, 32):
        value = row('candidate-mt-dynamics', 3, beads)['steps_per_second']/row('baseline-dynamics', 3, beads)['steps_per_second']
        gates[f'small_legacy_P{beads}'] = {'value': value, 'minimum': .98, 'passed': value >= .98}
    output = {'performance_gates': gates, 'all_performance_gates_passed': all(g['passed'] for g in gates.values()),
              'measurements': sum(len(r['measurements']) for r in reports.values()), 'physical_converged': False}
    if 'candidate-philox-dynamics-4' in reports:
        output['public_philox_four_thread_speedup'] = row('candidate-philox-dynamics-4', 4096)['steps_per_second']/row('candidate-philox-dynamics', 4096)['steps_per_second']
    (args.directory/'performance-gates.json').write_text(json.dumps(output, indent=2)+'\n')
    with (args.directory/'throughput.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(['report', 'api', 'rng', 'threads', 'atoms', 'beads', 'thermostat', 'steps_per_second', 'seconds'])
        for name, report in reports.items():
            for r in report['measurements']:
                writer.writerow([name, r['api'], r['rng'], report['environment']['thread_environment']['OMP_NUM_THREADS'],
                                 r['atoms'], r['beads'], r['thermostat'], r.get('steps_per_second'), json.dumps(r.get('seconds'))])
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
