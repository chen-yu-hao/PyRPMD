#!/usr/bin/env python3
"""Fair CPU BCB loop-language benchmark on identical H3/BKMP2 and Cayley maps.

A measured row uses exactly the same initial positions/momenta and 2 physical
force evaluations per full NVE timestep for Python, C, C++, and Fortran.
Native loops reuse per-trajectory memory and never enter Python during steps.
No timing claim should be generalized beyond compiler, CPU, PES and workload.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
MASS_H = np.full(3, 1837.152645, dtype=np.float64)
K_B_AU = 3.166811563e-6
TEMPERATURE = 1500.0
BETA = 1.0 / (K_B_AU * TEMPERATURE)
DT_AU = 0.05 * 41.34137333518211


def run(command: list[str]) -> None:
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def compiler_version(name: str) -> str:
    result = subprocess.run([name, '--version'], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0]


def build(output: Path, pes_library: Path, *, omit_native: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    for compiler in ['gcc', 'g++', 'gfortran']:
        if shutil.which(compiler) is None:
            raise RuntimeError(f'{compiler} required for language comparison')
    lib = output / 'libbkmp2.so'
    if lib.resolve() != pes_library.resolve():
        shutil.copy2(pes_library, lib)
    common = ['-O3', '-fPIC', '-shared', '-fno-fast-math']
    # The explicit rpath is strictly local to the benchmark build directory.
    link = [f'-L{output}', '-lbkmp2', '-Wl,-rpath,$ORIGIN']
    commands = {
        'c': ['gcc', '-std=c11', *common, str(HERE/'loop_c.c'), '-o', str(output/'libloop_c.so'), *link],
        'cpp': ['g++', '-std=c++17', *common, str(HERE/'loop_cpp.cpp'), '-o', str(output/'libloop_cpp.so'), *link],
        'fortran': ['gfortran', '-ffree-line-length-none', *common, str(HERE/'loop_fortran.f90'), '-o', str(output/'libloop_fortran.so'), *link],
    }
    for cmd in commands.values():
        run(cmd)
    return {'compilers': {s: compiler_version(s) for s in ['gcc','g++','gfortran']}, 'commands': commands}


def initial(beads: int, seed: int = 284271) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = np.zeros((beads, 3, 3), dtype=np.float64)
    # Real H + H2 entrance asymptote, H2 equilibrium bond ~1.401472 bohr.
    q[:, 0, 0] = -8.0
    q[:, 1, 0] = 0.0
    q[:, 2, 0] = 1.401472
    if beads > 1:
        dq = rng.normal(0, 0.015, size=q.shape)
        dq -= dq.mean(axis=0, keepdims=True)
        q += dq
    p = rng.normal(size=q.shape) * np.sqrt(MASS_H[None, :, None] * beads / BETA)
    # Conserve COM momentum; removes a large trivial translation drift.
    p -= np.mean(p, axis=(0, 1), keepdims=True)
    return np.ascontiguousarray(q), np.ascontiguousarray(p)


class Harmonic:
    metadata = {'execution_device': 'cpu', 'thread_safety': 'context_serial'}
    def evaluate(self, q):
        return (.0025 * np.sum(q*q, axis=(1,2)), -.005 * q)


def make_dynamics(potential: str, lib: Path, beads: int, native: bool = True):
    from pyrpmd.dynamics import Dynamics
    if potential == 'real':
        from benchmarks.h3.pes import BKMP2Potential
        provider = BKMP2Potential(lib)
    else:
        provider = Harmonic()
    return Dynamics(provider, MASS_H, BETA, beads, DT_AU, native=native, threads=1)


def python_loop(potential: str, lib: Path, q: np.ndarray, p: np.ndarray, steps: int, native: bool=True):
    from pyrpmd.dynamics import State
    dyn = make_dynamics(potential, lib, len(q), native=native)
    state = State(q, p)
    try:
        for _ in range(steps):
            dyn.step(state)
        return state.q.copy(), state.p.copy()
    finally:
        if hasattr(dyn.potential, 'close'):
            dyn.potential.close()


def external_loop(kind: str, libs: dict, mode: int, q: np.ndarray, p: np.ndarray, steps: int):
    call = libs[kind]
    xx = q.copy(order='C'); yy = p.copy(order='C')
    rc = call(len(q), MASS_H.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
              BETA, DT_AU, steps,
              xx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
              yy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mode)
    if rc != 0:
        raise RuntimeError(f'{kind} native loop failed rc={rc} at step={rc} mode={mode}')
    return xx, yy


def make_libraries(output: Path) -> dict:
    # Explicitly load the shared Fortran PES first; all 3 loops call the same .so.
    ctypes.CDLL(str(output/'libbkmp2.so'), mode=ctypes.RTLD_GLOBAL)
    libs = {}
    for kind in ('c', 'cpp', 'fortran'):
        lib = ctypes.CDLL(str(output/f'libloop_{kind}.so'))
        symbol = getattr(lib, f'pyrpmd_loop_{kind}')
        symbol.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double),
                           ctypes.c_double,ctypes.c_double,ctypes.c_int64,
                           ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int32]
        symbol.restype = ctypes.c_int32
        libs[kind] = symbol
    return libs


def check_correctness(kinds, libs, potential, pes_library):
    mode = 0 if potential == 'real' else 1
    checks=[]
    # P=1 and 2 exercise special cases. 16/32 are scientific sizes.
    for beads in (1,2,8,16,32):
        q,p=initial(beads)
        short_steps=9
        ref_q, ref_p=python_loop(potential, pes_library, q, p, short_steps, native=False)
        for kind in kinds:
            if kind in ('python','numpy'):
                got_q,got_p=python_loop(potential, pes_library, q, p, short_steps, native=kind=='python')
            else:
                got_q,got_p=external_loop(kind,libs,mode,q,p,short_steps)
            qerr=float(np.max(np.abs(ref_q-got_q)))
            perr=float(np.max(np.abs(ref_p-got_p)))
            # The FFT and real-space midpoint Cayley maps use equivalent algebra,
            # so tolerance is tight but not exact bit identity.
            if not np.allclose(ref_q, got_q, rtol=5e-10, atol=5e-10) or \
               not np.allclose(ref_p, got_p, rtol=5e-10, atol=5e-10):
                raise AssertionError((beads, kind, qerr, perr))
            checks.append({'beads':beads,'implementation':kind,'steps':short_steps,
                           'max_q_abs_error_bohr':qerr,'max_p_abs_error_au':perr,
                           'passed':True})
    return checks


def measure(fn, repeats: int):
    durations=[]
    result=None
    for _ in range(repeats):
        t=time.perf_counter()
        result=fn()
        duration=time.perf_counter()-t
        durations.append(duration)
    return durations, result


def benchmark(kinds, libs, potential, pes_library, sizes, steps, repeats):
    mode=0 if potential=='real' else 1
    records=[]
    for beads in sizes:
        q,p=initial(beads)
        for kind in kinds:
            if kind in ('python','numpy'):
                f=lambda kind=kind: python_loop(potential,pes_library,q,p,steps,native=kind=='python')
            else:
                f=lambda kind=kind: external_loop(kind,libs,mode,q,p,steps)
            f() if steps <= 80 else None
            # Always run a short warm-up, not a full measurement-length trajectory.
            if kind in ('python','numpy'):
                python_loop(potential,pes_library,q,p,5,native=kind=='python')
            else:
                external_loop(kind,libs,mode,q,p,5)
            durations,(qo,po)=measure(f,repeats)
            record={
                'potential':potential,'beads':beads,'implementation':kind,
                'steps':steps,'repeats':repeats,'seconds':durations,
                'median_seconds':statistics.median(durations),
                'steps_per_second':steps/statistics.median(durations),
                'frame_force_calls':2*steps*beads,
                'final_q_sum':float(np.sum(qo)),
                'final_p_sum':float(np.sum(po)),
                'last_state_finite':bool(np.isfinite(qo).all() and np.isfinite(po).all()),
            }
            if not record['last_state_finite']:
                raise FloatingPointError(f'{kind} produced NaNs')
            records.append(record)
            print(json.dumps({'result': record}, sort_keys=True),flush=True)
    # Three native backend states should agree numerically; no overclaim for
    # chaotic long trajectories, but a divergence would be diagnostic.
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--potential',choices=['real','toy'],default='real')
    parser.add_argument('--pes-library',type=Path,help='Built real BKMP2 .so (real mode)')
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build'/'loop_language_bench')
    parser.add_argument('--steps',type=int,default=2500)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--sizes',type=int,nargs='+',default=[16,32])
    parser.add_argument('--output',type=Path,default=ROOT/'runs'/'loop-language'/'report.json')
    parser.add_argument('--include-numpy',action='store_true')
    parser.add_argument('--pin-cpu',action='store_true',help='Linux: pin all measurements to one CPU')
    args=parser.parse_args()
    if args.steps<1 or args.repeats<1 or min(args.sizes)<1:
        parser.error('steps, repeats, sizes must be positive')
    args.build_dir=args.build_dir.resolve()
    if args.pin_cpu and hasattr(os,'sched_setaffinity'):
        affinity=os.sched_getaffinity(0)
        os.sched_setaffinity(0,{min(affinity)})
    pes_library=args.pes_library
    if args.potential=='real' and pes_library is None:
        from benchmarks.h3.pes import DEFAULT_LIBRARY
        pes_library=DEFAULT_LIBRARY
    if args.potential=='toy' and pes_library is None:
        # A test-only shared symbol for linkage; never used in toy mode.
        pes_library=args.build_dir/'libtoy_pes.so'
        args.build_dir.mkdir(parents=True,exist_ok=True)
        run(['gcc','-O3','-shared','-fPIC',str(HERE/'toy_pes.c'),'-o',str(pes_library)])
    pes_library=Path(pes_library).resolve()
    if not pes_library.is_file():
        raise FileNotFoundError(f'PES library missing: {pes_library}')
    build_info=build(args.build_dir,pes_library)
    libs=make_libraries(args.build_dir)
    kinds=['python','c','cpp','fortran']
    if args.include_numpy:
        kinds.append('numpy')
    checks=check_correctness(kinds,libs,args.potential,pes_library)
    print(f'Correctness passed: {len(checks)} comparisons with independent FFT/NumPy',flush=True)
    rows=benchmark(kinds,libs,args.potential,pes_library,args.sizes,args.steps,args.repeats)
    for beads in args.sizes:
        group={r['implementation']:r for r in rows if r['beads']==beads}
        control=group['python']['median_seconds']
        for r in rows:
            if r['beads']==beads:
                r['speedup_vs_current_python']=control/r['median_seconds']
    report={
        'schema':'pyrpmd.loop_languages.v1',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'physical_scope':'BCB NVE; two identical Fortran BKMP2 force evaluations each step in real mode. No thermostat, umbrella, constraints, or classifier.',
        'potential':args.potential,
        'input':{'temperature_K':TEMPERATURE,'beta_hartree_inverse':BETA,'timestep_fs':0.05,
                 'timestep_au':DT_AU,'masses_au':MASS_H.tolist(), 'sizes':args.sizes,
                 'steps_per_repeat':args.steps,'repeats':args.repeats,'seed':284271},
        'environment':{'platform':platform.platform(),'python':sys.version.split()[0],
                       'cpu_count':os.cpu_count(), 'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                       'numpy':np.__version__, **build_info},
        'checks':checks,'records':rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('\n| PES | Beads | Python steps/s | C steps/s | C++ steps/s | Fortran steps/s | Winner |',flush=True)
    print('|---|---:|---:|---:|---:|---:|---|',flush=True)
    for beads in args.sizes:
        group={r['implementation']:r for r in rows if r['beads']==beads}
        winner=max((k for k in group if k!='numpy'), key=lambda k:group[k]['steps_per_second'])
        print(f'| {args.potential} | {beads} | '+ ' | '.join(f'{group[k]["steps_per_second"]:.1f}' for k in ('python','c','cpp','fortran'))+ f' | {winner} |',flush=True)
    print('BENCHMARK_JSON '+str(args.output),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
