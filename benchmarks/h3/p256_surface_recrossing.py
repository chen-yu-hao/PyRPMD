"""Strict eight-chain P256 binding around the unchanged native child kernel.

Only the separately completed, hash-bound P256 parent campaign is eligible.
All four registered momentum pairs and both timesteps are primary controls.
There is no calibration bypass or partial-parent route in this adapter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3 import quantum_surface_recrossing as original
from benchmarks.h3.quantum_surface_recrossing import (
    BETA, MASS, MASS_AU, PAIRS, CHUNK_PARENTS, BKMP2Potential, Dynamics, State,
    FS_TO_AU, atomic_savez, load_checkpoint, write_json, fingerprint, sha,
    surface_coordinate, minimum_pair, assign_channel, maxwell_momenta,
    flux_velocity, paired_flux, read_chunk, run_directory,
)

EVIDENCE = ROOT / 'docs/validation/literature-convergence-2026-11-01'
DESIGN = EVIDENCE / 'h3-quantum/p256-fixed-eight-chain-design/proposal.json'
DESIGN_SHA = 'c668c15fd89ca0a535afc4082521e3f42ba9e84f5736b5b481bc1d7c376d791d'
PARENTS = ROOT.parent / 'literature-convergence-quantum/p256-fixed-eight-chain-v1/parents'
BASELINE = ROOT.parent / 'literature-convergence-quantum/recrossing-analysis-200-four-pairs/quantum-rate-comparison.json'
BASELINE_SHA = '62ce4fea23a55955cf404ae8aa00a2c70204bf20dd48e74de520169a83724720'
SEEDS = (825601, 825603, 825607, 825609, 825613, 825617, 825621, 825623)
P128_SEEDS = (721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323)
FILES = ('settings.json', 'report.json', 'reference.json', 'reference-domain.json', 'checkpoint.npz')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(path, expected):
    path = Path(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f'Input hash differs: {path}')
    return json.loads(raw)


def source_identity():
    values = original.source_identity()
    for name in ('benchmarks/h3/p256_surface_recrossing.py', 'benchmarks/h3/p256_parent_campaign.py',
                 'benchmarks/h3/p256_recrossing_analyze.py', 'benchmarks/h3/quantum_recrossing_analyze.py'):
        values[name] = sha(ROOT / name)
    return values


def verify_manifest(parent_root, design):
    """Read a complete eight-chain manifest, with no PES or sampling calls."""
    parent_root = Path(parent_root).resolve()
    path = parent_root / 'parents-complete.json'
    manifest = json.loads(path.read_text())
    require(manifest['schema'] == 'pyrpmd.h3.p256-eight-parent-completion.v1'
            and manifest['eligible_for_separately_authorized_children'] is True
            and manifest['all_eight_counts_complete_and_verified'] is True
            and manifest['all_owned_workers_drained'] is True
            and manifest['four_chain_interim_selection'] is False
            and manifest['independent_parent_chains'] == 8
            and manifest['errors'] == [] and manifest['unstarted_seeds'] == [],
            'Complete, drained, eligible eight-chain parents are required')
    require(tuple(manifest['seed_order']) == SEEDS
            and tuple(row['seed'] for row in manifest['parents']) == SEEDS,
            'Parent cohort order or membership differs')
    require(manifest['design_sha256'] == DESIGN_SHA
            and manifest['sampling_flags'] == design['parent_generation']['sampling_flags'],
            'Parent design or sampling flags differ')
    registration = bound_json(parent_root / 'registration.json', manifest['registration_sha256'])
    require(registration['schema'] == 'pyrpmd.h3.p256-eight-parent-registration.v1'
            and registration['design_sha256'] == DESIGN_SHA
            and tuple(registration['fixed_seeds']) == SEEDS
            and Path(registration['output']).resolve() == parent_root,
            'Parent registration differs')
    require(registration['driver_sha256'] == sha(ROOT / 'benchmarks/h3/p256_parent_campaign.py'),
            'The executed parent producer changed')
    expected = {row['seed']: row for row in design['parent_generation']['cohorts']}
    flags = design['parent_generation']['sampling_flags']
    for row in manifest['parents']:
        require(row['complete'] is True and row['identity_and_geometry_verified'] is True
                and row['sampling_flags_passed'] is True
                and row['parents'] == 200 and row['beads'] == 256
                and row['completed_sweeps'] == 22000,
                'Incomplete or ineligible parent row')
        require(row['cohort_role'] == expected[row['seed']]['cohort_role'], 'Parent cohort role differs')
        require(row['settings_fingerprint'] == expected[row['seed']]['expected_settings_fingerprint'],
                'Parent expected-settings identity differs')
        require(set(row['files']) == set(FILES), 'Parent file inventory differs')
        require(row['checkpoint_sha256'] == row['files']['checkpoint.npz']['sha256'],
                'Parent checkpoint summary and file inventory differ')
        numbers = [row['minimum_BAR_overlap'], row['minimum_energy_ESS'],
                   row['half_log_partition_difference'], row['BAR_log_TS_quantum_classical_ratio']]
        require(np.isfinite(numbers).all()
                and numbers[0] >= flags['minimum_BAR_overlap']
                and numbers[1] >= flags['minimum_energy_ESS']
                and abs(numbers[2]) <= flags['maximum_absolute_half_log_difference'],
                'Parent sampling numbers fail the registered flags')
        if row['seed'] in SEEDS[:4]:
            replay = row['historical_replay']
            require(replay is not None and replay['bitwise_verified_array_count'] == 16
                    and sorted(replay['bitwise_verified_arrays']) == sorted(expected[row['seed']]['exact_original_payloads'])
                    and replay['reference_bytes_equal'] is True
                    and replay['scientific_report_equal'] is True,
                    'Historical parent replay is not verified')
        else:
            require(row['historical_replay'] is None, 'New chains must not impersonate historical replay')
        directory = parent_root / expected[row['seed']]['relative_directory']
        require(directory.resolve().is_relative_to(parent_root), 'Parent directory escapes the campaign')
        for name, item in row['files'].items():
            target = directory / name
            require(Path(item['path']).resolve() == target.resolve() and not target.is_symlink(),
                    'Parent file route differs')
            require(target.stat().st_size == item['bytes'] and sha(target) == item['sha256'],
                    'Bound parent file changed')
    return manifest


def design(library):
    specification = bound_json(DESIGN, DESIGN_SHA)
    require(tuple(specification['independent_P256_seeds']) == SEEDS, 'P256 seed design differs')
    child = specification['child_experiment_definition']
    require(child['beads'] == 256 and child['horizon_fs'] == 200
            and child['timestep_fs'] == [.1, .05] and child['pair_indices'] == [0, 1, 2, 3]
            and child['parents_per_chain'] == 200 and child['mass_au'] == MASS_AU
            and child['beta'] == BETA, 'Child scientific design differs')
    identity = source_identity()
    require(all(identity.get(name) == expected
                for name, expected in child['PES_and_all_scientific_dependencies'].items()),
            'Frozen physical source or native library differs')
    parents = verify_manifest(PARENTS, specification)
    baseline = bound_json(BASELINE, BASELINE_SHA)
    require(tuple(baseline['completed_seeds']) == tuple(baseline['expected_seeds']) == P128_SEEDS
            and baseline['criteria']['complete_registered_cohort'] is True
            and baseline['pair_batches'] == [0, 1] and baseline['horizon_fs'] == 200,
            'Complete P128 four-pair baseline required')
    with BKMP2Potential(library) as potential:
        model = potential.metadata
    expected_model = specification['parent_generation']['cohorts'][0]['expected_settings']['model']
    require(model == expected_model, 'The child PES differs from the fixed parent PES')
    return {
        'schema': 'pyrpmd.h3.p256-surface-recrossing.v1',
        'design_path': str(DESIGN), 'design_sha256': DESIGN_SHA,
        'parent_root': str(PARENTS), 'parent_manifest_sha256': sha(PARENTS / 'parents-complete.json'),
        'parent_registration_sha256': parents['registration_sha256'],
        'P128_complete_four_pair_report': {'path': str(BASELINE), 'sha256': BASELINE_SHA},
        'seeds': list(SEEDS), 'beads': 256, 'temperature_K': 300., 'beta': BETA, 'mass_au': MASS_AU,
        'parents_per_seed': 200, 'parent_completed_sweeps': list(range(2100, 22001, 100)),
        'momentum_pairs_per_batch': PAIRS, 'registered_primary_pair_batches': [0, 1],
        'momentum_pair_indices': [0, 1, 2, 3], 'children_per_pair': 2,
        'timestep_fs': [.1, .05], 'primary_timestep_fs': .05, 'horizon_fs': 200,
        'surface_observation_interval_fs': 1, 'minimum_pair_observation_interval_fs': 1,
        'static_domain_minimum_pair_bohr': .5, 'maximum_centroid_bond_bohr': 8.,
        'integrator': 'unchanged native PyRPMD Cayley-BCB, no thermostat',
        'momentum_source': 'ordinary Maxwell variance=m*P/beta, keyed by seed/parent/pair; both signs',
        'flux_estimator': 'sum[(v/fs)*(H(s_plus)-H(s_minus))]/sum[abs(v)/fs]',
        'expected_denominator': 2., 'uncertainty_unit': 'eight independent equilibrium parent chains',
        'plateau_windows_fs': [[80, 100], [180, 200]],
        'plateau_equivalence_absolute_margin': .01, 'timestep_equivalence_absolute_margin': .01,
        'precision_gate_relative_kappa_95_halfwidth': .03,
        'parent_time_half_equivalence': 'supplementary; unchanged absolute margin0.03, failures retained',
        'full_rate_bead_comparison': 'independent8-versus8Welch95interval of chain log(QTST*kappa) wholly inside +/-log(1.05)',
        'extensions': 'none automatic; all registered children are retained regardless of agreement',
        'source_sha256': identity, 'model': model,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
    }


def register(output, library):
    expected = design(library)
    output = Path(output)
    path = output / 'protocol.json'
    if path.exists():
        actual = json.loads(path.read_text())
        actual.pop('created_before_primary_children_utc')
        require(actual == expected, 'Registered P256 children or sources changed')
    else:
        output.mkdir(parents=True, exist_ok=True)
        require(not any(output.iterdir()), 'A new P256 child campaign must use an empty directory')
        # This registration is made by the single coordinator before workers.
        with path.open('x') as stream:
            json.dump(dict(expected, created_before_primary_children_utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
            stream.write('\n')
    return expected


def load_parents(directory, protocol, *, calibration=False, count=None):
    require(not calibration and count is None, 'P256 adapter has no calibration/subset bypass')
    specification = bound_json(DESIGN, protocol['design_sha256'])
    parent_root = Path(protocol['parent_root'])
    manifest = bound_json(parent_root / 'parents-complete.json', protocol['parent_manifest_sha256'])
    require(tuple(manifest['seed_order']) == SEEDS
            and manifest['eligible_for_separately_authorized_children'] is True,
            'All eight eligible parent chains are required')
    directory = Path(directory)
    rows = {row['seed']: row for row in manifest['parents']}
    settings = json.loads((directory / 'settings.json').read_text())
    seed = settings['seed']
    require(seed in SEEDS, 'Unregistered parent seed')
    row = rows[seed]
    expected = next(item for item in specification['parent_generation']['cohorts'] if item['seed'] == seed)
    require(directory.resolve() == (parent_root / expected['relative_directory']).resolve(), 'Parent route differs')
    require(settings == expected['expected_settings'], 'Parent physical or sampling settings differ')
    for name in FILES:
        require(sha(directory / name) == row['files'][name]['sha256'], 'Parent payload changed')
    require(row['checkpoint_sha256'] == row['files']['checkpoint.npz']['sha256'],
            'Parent checkpoint summary and file inventory differ')
    report = json.loads((directory / 'report.json').read_text())
    require(report['complete'] and report['source_unchanged'] and report['phase'] == 'production'
            and report['surface'] == 'ts' and report['seed'] == seed and report['beads'] == 256,
            'Complete unchanged P256 TS report required')
    require(report['checkpoint_sha256'] == row['checkpoint_sha256']
            and report['analysis']['log_quantum_classical_ratio'] == row['BAR_log_TS_quantum_classical_ratio'],
            'Parent report and manifest differ')
    with load_checkpoint(directory / 'checkpoint.npz', expected_kind='h3.surface-partition', require_integrity=True) as saved:
        require(str(saved['fingerprint'].item()) == row['settings_fingerprint'], 'Parent checkpoint settings differ')
        require(str(saved['domain_sha256'].item()) == row['files']['reference-domain.json']['sha256'], 'Parent normalization differs')
        require(int(saved['completed']) == 22000, 'Parent chain is incomplete')
        q, sweeps = saved['endpoint_q'], saved['endpoint_completed_sweeps']
    require(q.shape == (200, 256, 3, 3) and np.isfinite(q).all()
            and np.array_equal(sweeps, np.arange(2100, 22001, 100)), 'Parent sequence/count differs')
    require(settings['beta'] == protocol['beta'] and settings['mass_au'] == protocol['mass_au']
            and settings['model'] == protocol['model'], 'Parent and child Hamiltonians differ')
    for parent in q:
        flux_velocity(parent, np.zeros_like(parent))
    centers = q.mean(axis=1)
    require(minimum_pair(q) >= .5 - 1e-12 and np.max(abs(q.mean(axis=2))) <= 1e-10,
            'Parent bead domain or COM convention differs')
    require(np.linalg.norm(centers[:, 1] - centers[:, 2], axis=-1).max() <= 8. + 1e-12,
            'Parent centroid domain differs')
    identity = {'checkpoint_sha256': row['checkpoint_sha256'],
                'settings_sha256': row['files']['settings.json']['sha256'],
                'report_sha256': row['files']['report.json']['sha256'], 'seed': seed,
                'BAR_log_TS_quantum_classical_ratio': row['BAR_log_TS_quantum_classical_ratio'],
                'parent_manifest_sha256': protocol['parent_manifest_sha256'],
                'parent_registration_sha256': protocol['parent_registration_sha256'],
                'P256_design_sha256': protocol['design_sha256']}
    return q, sweeps, identity


def run_seed(output, library, seed, timestep, *, horizon=200, pair_batch=0, stop_after_chunks=None):
    """Run every parent in one of the two fixed primary momentum batches."""
    output = Path(output)
    require(type(horizon) is int and horizon == 200, 'Only the fixed P256 200-fs control is registered')
    require(type(seed) is int and seed in SEEDS, 'Unregistered P256 parent seed')
    require(type(pair_batch) is int and pair_batch in (0, 1), 'Both registered primary pair batches are required')
    require(timestep in (.1, .05), 'Unregistered child timestep')
    require(stop_after_chunks is None or (type(stop_after_chunks) is int and stop_after_chunks > 0),
            'A bounded incomplete-run chunk count must be positive')
    calibration = False
    protocol = register(output, library)
    specification = bound_json(DESIGN, protocol['design_sha256'])
    parent_row = next(row for row in specification['parent_generation']['cohorts'] if row['seed'] == seed)
    parent_directory = Path(protocol['parent_root']) / parent_row['relative_directory']
    parents, sweeps, parent_identity = load_parents(parent_directory, protocol)
    require(seed == parent_identity['seed'], 'Child seed must identify its original parent chain')
    before = source_identity()
    run = run_directory(output, seed, timestep, horizon, pair_batch)
    run.mkdir(parents=True, exist_ok=True)
    start_horizon = 200 if not calibration and horizon == 400 else 0
    previous = run.parent / "horizon-200"
    settings = {"schema": protocol["schema"], "phase": "excluded calibration" if calibration else "primary",
                "seed": seed, "beads": protocol["beads"], "parents": len(parents),
                "timestep_fs": timestep, "horizon_fs": horizon, "start_horizon_fs": start_horizon,
                "pair_batch": pair_batch, "momentum_pair_indices": list(range(PAIRS*pair_batch, PAIRS*(pair_batch+1))),
                "protocol_sha256": fingerprint(protocol), "parent_identity": parent_identity,
                "source_sha256": before, "chunk_parents": CHUNK_PARENTS}
    identity = fingerprint(settings)
    settings_path = run / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("child run settings changed")
    else:
        write_json(settings_path, settings)
    prior_identity = None
    prior_hashes = {}
    if start_horizon:
        prior_settings = dict(settings, horizon_fs=200, start_horizon_fs=0)
        if json.loads((previous / "settings.json").read_text()) != prior_settings:
            raise ValueError("extension is not the same registered child ensemble")
        prior_identity = fingerprint(prior_settings)
        prior_report = json.loads((previous / "report.json").read_text())
        if not prior_report["complete"]:
            raise ValueError("extend only a complete 200-fs child cohort")
        if prior_report["settings_sha256"] != sha(previous / "settings.json"):
            raise ValueError("predecessor report/settings checksum mismatch")
        prior_hashes = {item["file"]: item["sha256"] for item in prior_report["chunks"]}
        expected_names = {f"chunk-{left:04d}.npz" for left in range(0, len(parents), CHUNK_PARENTS)}
        if set(prior_hashes) != expected_names:
            raise ValueError("predecessor report has an incomplete child inventory")
        for name, expected_hash in prior_hashes.items():
            if sha(previous / name) != expected_hash:
                raise ValueError("predecessor chunk differs from its completed report")
    times = np.arange(1, horizon + 1, dtype=float)
    block_steps = round(1. / timestep)
    if abs(block_steps * timestep - 1.) > 1e-12:
        raise ValueError("timestep must exactly divide the 1-fs observation interval")
    started = time.perf_counter()
    surface = surface_coordinate()
    new_chunks = 0
    with BKMP2Potential(library) as potential:
        dynamics = Dynamics(potential, MASS, BETA, protocol["beads"], timestep * FS_TO_AU,
                            native="required", loop_backend="native")
        if not dynamics.can_fuse(observe=surface):
            raise ValueError("native RPMD child path unavailable")
        for left in range(0, len(parents), CHUNK_PARENTS):
            indices = np.arange(left, min(left + CHUNK_PARENTS, len(parents)))
            path = run / f"chunk-{left:04d}.npz"
            if path.exists():
                saved = read_chunk(path, identity, indices)
                if start_horizon and str(saved["previous_chunk_sha256"].item()) != prior_hashes[path.name]:
                    raise ValueError("existing extension used a different predecessor chunk")
                continue
            if stop_after_chunks is not None and new_chunks >= stop_after_chunks:
                break
            old = read_chunk(previous / path.name, prior_identity, indices) if start_horizon else None
            shape = (len(indices), PAIRS, 2)
            surfaces = np.empty(shape + (horizon,))
            minima = np.empty(shape + (horizon + 1,))
            extrema = np.empty(shape + (3,))
            initial_p = np.empty((len(indices), PAIRS, protocol["beads"], 3, 3))
            final_q = np.empty(shape + (protocol["beads"], 3, 3))
            final_p = np.empty_like(final_q)
            final_step = np.empty(shape, dtype=np.int64)
            final_time = np.empty(shape)
            velocities = np.empty((len(indices), PAIRS))
            fluxes = np.empty(len(indices))
            channels = np.empty(len(indices), dtype=np.int64)
            numerical_before = dict(dynamics.propagation_stats)
            for local, index in enumerate(indices):
                q, channels[local] = assign_channel(parents[index], seed, int(index))
                for pair in range(PAIRS):
                    p = maxwell_momenta(seed, int(index), PAIRS * pair_batch + pair, protocol["beads"])
                    initial_p[local, pair] = p
                    fluxes[local], velocities[local, pair] = flux_velocity(q, p, surface=surface)
                    if old is not None and not np.array_equal(p, old["initial_p"][local, pair]):
                        raise ValueError("extension momentum replay differs")
                    for side, sign in enumerate((1., -1.)):
                        if old is None:
                            state = State(q, sign * p)
                            initial_energy = dynamics.energy(state)
                            extrema[local, pair, side] = [initial_energy, initial_energy, initial_energy]
                            minima[local, pair, side, 0] = minimum_pair(q)
                        else:
                            state = State(old["final_q"][local, pair, side], old["final_p"][local, pair, side],
                                          step=int(old["final_step"][local, pair, side]),
                                          time=float(old["final_time_au"][local, pair, side]))
                            extrema[local, pair, side] = old["energy_extrema"][local, pair, side]
                            minima[local, pair, side, :start_horizon + 1] = old["minimum_pair_at_1fs"][local, pair, side]
                            surfaces[local, pair, side, :start_horizon] = old["surface_values"][local, pair, side]
                        for frame in range(start_horizon, horizon):
                            try:
                                result = dynamics.propagate(state, block_steps, observe=surface,
                                                            sample_interval=block_steps, record_centroids=False)
                            except BaseException as error:
                                atomic_savez(run / "failed-child-state.npz", kind="h3.failed-quantum-child",
                                             q=state.q, p=state.p, parent_index=index, pair=pair, side=side,
                                             last_observed_boundary_fs=frame, state_step=state.step,
                                             state_time_au=state.time, fingerprint=identity)
                                write_json(run / "failure.json", {"parent_index": int(index), "pair": pair,
                                           "side": side, "last_observed_boundary_fs": frame,
                                           "saved_state_step": state.step, "saved_state_time_au": state.time,
                                           "error": repr(error),
                                           "policy": "No child removed; the registered job is incomplete."})
                                raise
                            if result.backend != "native":
                                raise ValueError("unexpected Python child propagation")
                            surfaces[local, pair, side, frame] = result.values[-1, 0]
                            minima[local, pair, side, frame + 1] = minimum_pair(state.q)
                            extrema[local, pair, side, 1] = min(extrema[local, pair, side, 1], result.minimum_energy)
                            extrema[local, pair, side, 2] = max(extrema[local, pair, side, 2], result.maximum_energy)
                        final_q[local, pair, side], final_p[local, pair, side] = state.q, state.p
                        final_step[local, pair, side], final_time[local, pair, side] = state.step, state.time
            numerator = velocities[..., None] / fluxes[:, None, None] * (
                (surfaces[:, :, 0] > 0).astype(float) - (surfaces[:, :, 1] > 0))
            denominator = abs(velocities) / fluxes[:, None]
            native_steps = dynamics.propagation_stats["native_steps"] - numerical_before["native_steps"]
            expected_steps = len(indices) * PAIRS * 2 * (horizon - start_horizon) * block_steps
            if native_steps != expected_steps or dynamics.propagation_stats["python_steps"]:
                raise ValueError("child native step accounting differs")
            if source_identity() != before:
                raise ValueError("child source changed before checkpoint commit")
            atomic_savez(path, kind="h3.quantum-surface-children", fingerprint=identity,
                         parent_index=indices, parent_completed_sweeps=sweeps[indices], channel=channels,
                         momentum_pair_index=np.arange(PAIRS*pair_batch, PAIRS*(pair_batch+1)),
                         time_fs=times, surface_values=surfaces, minimum_pair_at_1fs=minima,
                         energy_extrema=extrema, initial_p=initial_p, final_q=final_q, final_p=final_p,
                         final_step=final_step, final_time_au=final_time,
                         flux_metric=fluxes, velocity=velocities, numerator=numerator, denominator=denominator,
                         native_steps_this_stage=native_steps, previous_chunk_sha256=sha(previous / path.name) if old is not None else "")
            new_chunks += 1
            print(json.dumps({"seed": seed, "dt_fs": timestep, "horizon_fs": horizon,
                              "pair_batch": pair_batch,
                              "parents_completed": int(indices[-1]) + 1,
                              "elapsed_this_call_seconds": time.perf_counter() - started}), flush=True)
    return summarize(run, settings, identity, parents, elapsed=time.perf_counter() - started)


def summarize(run, settings, identity, parents, *, elapsed=0.):
    run = Path(run)
    specification = bound_json(DESIGN, DESIGN_SHA)
    require(sha(PARENTS / 'parents-complete.json') == settings['parent_identity']['parent_manifest_sha256'],
            'Parent manifest changed during child propagation')
    verify_manifest(PARENTS, specification)
    numerator = np.zeros(settings["horizon_fs"])
    denominator = 0.
    errors, minimum, crossed = [], np.inf, 0
    count, native_steps, chunks = 0, 0, []
    for left in range(0, len(parents), CHUNK_PARENTS):
        indices = np.arange(left, min(left + CHUNK_PARENTS, len(parents)))
        path = run / f"chunk-{left:04d}.npz"
        if not path.exists():
            break
        saved = read_chunk(path, identity, indices)
        numerator += saved["numerator"].sum(axis=(0, 1))
        denominator += float(saved["denominator"].sum())
        extrema = saved["energy_extrema"]
        error = np.maximum(abs(extrema[..., 1] - extrema[..., 0]), abs(extrema[..., 2] - extrema[..., 0]))
        errors.extend((error * BETA / settings["beads"]).ravel().tolist())
        minimum = min(minimum, float(saved["minimum_pair_at_1fs"].min()))
        crossed += int(np.sum(saved["minimum_pair_at_1fs"].min(axis=-1) < .5))
        native_steps += int(saved["native_steps_this_stage"])
        count += len(indices)
        chunks.append({"file": path.name, "sha256": sha(path)})
    times = np.arange(1, settings["horizon_fs"] + 1, dtype=float)
    kappa = numerator / denominator if denominator else None
    plateaus = {f"{lo}-{hi}": float(kappa[(times >= lo) & (times <= hi)].mean())
                for lo, hi in ((80, 100), (180, 200), (380, 400))
                if kappa is not None and hi <= settings["horizon_fs"]}
    report = {"schema": settings["schema"], "phase": settings["phase"], "seed": settings["seed"],
              "timestep_fs": settings["timestep_fs"], "horizon_fs": settings["horizon_fs"],
              "pair_batch": settings["pair_batch"], "momentum_pair_indices": settings["momentum_pair_indices"],
              "complete": count == len(parents), "parents_completed": count, "pairs_per_parent": PAIRS,
              "plateaus": plateaus, "mean_denominator": denominator / (count * PAIRS) if count else None,
              "expected_mean_denominator": 2., "native_steps_this_stage": native_steps,
              "unique_native_steps_all_stages": count * PAIRS * 2 * round(settings["horizon_fs"] / settings["timestep_fs"]),
              "python_steps": 0, "maximum_energy_error_kbt_per_bead": max(errors) if errors else None,
              "energy_error_quantiles_kbt_per_bead": np.quantile(errors, [0., .5, .95, .99, 1.]) if errors else None,
              "minimum_pair_at_1fs_bohr": minimum if count else None,
              "children_with_sampled_pair_below_static_domain": crossed,
              "minimum_pair_scope": "Only initial and 1-fs boundaries; every-step PES guard and full-H extrema are separate checks.",
              "source_unchanged": source_identity() == settings["source_sha256"],
              "settings_sha256": sha(run / "settings.json"), "chunks": chunks,
              "parent_identity": settings["parent_identity"], "elapsed_this_call_seconds": elapsed,
              "created_utc": datetime.now(timezone.utc).isoformat(),
               "literature_reproduced": False}
    write_json(run / "report.json", report)
    if kappa is not None:
        np.savetxt(run / "kappa.csv", np.column_stack((times, kappa)), delimiter=",",
                   header="time_fs,kappa", comments="")
    return report



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--timestep', type=float, choices=(.1, .05), default=.05)
    parser.add_argument('--pair-batch', type=int, choices=(0, 1), default=0)
    parser.add_argument('--stop-after-chunks', type=int)
    args = parser.parse_args()
    if args.register_only:
        protocol = register(args.output, args.library)
        print(json.dumps({'registered':True, 'protocol_sha256':sha(args.output/'protocol.json'),
                          'seeds':protocol['seeds'], 'PES_dynamics_steps':0}))
        return
    if args.seed is None:
        parser.error('--seed is required for child propagation')
    result = run_seed(args.output, args.library, args.seed, args.timestep,
                      pair_batch=args.pair_batch, stop_after_chunks=args.stop_after_chunks)
    print(json.dumps({key:result[key] for key in ('seed','pair_batch','timestep_fs','complete','parents_completed',
                                               'native_steps_this_stage','source_unchanged')}))


if __name__ == '__main__':
    main()
