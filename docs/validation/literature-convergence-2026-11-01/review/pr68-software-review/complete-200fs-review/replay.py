"""Independent read-only replay of the complete, original 200-fs child cohort.

No producer/analyzer entry point is called. The checkpoint reader verifies its
existing integrity envelopes; flux, geometry, RNG and statistics are rebuilt
here from the raw arrays. No potential or dynamics is evaluated.
"""
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from scipy.stats import t
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import FS_TO_AU

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
DATA = Path("/workspace/scratch/a5e4820daedd/literature-convergence-quantum/recrossing-primary")
ANALYSIS = DATA.parent / "recrossing-analysis-200"
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path)
    digest = sha(path)
    INPUTS[str(path.resolve())] = digest
    return digest


def read(path):
    bind(path)
    return json.loads(Path(path).read_text())


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def stats(values):
    values = [float(x) for x in values]
    n = len(values)
    mean = math.fsum(values) / n
    se = math.sqrt(math.fsum((x-mean)**2 for x in values) / (n*(n-1)))
    half = float(t.ppf(.975, n-1)) * se
    return dict(mean=mean, standard_error=se, halfwidth95=half,
                CI95=[mean-half, mean+half], degrees_of_freedom=n-1)


def gradient(centroid):
    """Explicit s1=r12-min(r01,r02) derivative, independent of CV helpers."""
    lengths = {pair: np.linalg.norm(centroid[pair[0]]-centroid[pair[1]])
               for pair in [(0, 1), (0, 2), (1, 2)]}
    forming = (0, 1) if lengths[(0, 1)] <= lengths[(0, 2)] else (0, 2)
    g = np.zeros((3, 3))
    for sign, (i, j) in [(1., (1, 2)), (-1., forming)]:
        vector = (centroid[i]-centroid[j]) / lengths[(i, j)]
        g[i] += sign*vector
        g[j] -= sign*vector
    return lengths[(1, 2)]-lengths[forming], g


def endpoint_diagnostics(q, p, mass):
    centers = q.mean(axis=-3).reshape(-1, 3, 3)
    velocities = p.mean(axis=-3).reshape(-1, 3, 3) / mass
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    lengths = np.stack([np.linalg.norm(centers[:, i]-centers[:, j], axis=1)
                        for i, j, _ in pairs], axis=1)
    selected = lengths.argmin(axis=1)
    bond, distance, radial = [], [], []
    for row, which in enumerate(selected):
        i, j, k = pairs[which]
        displacement = centers[row, k] - (centers[row, i]+centers[row, j])/2
        relative_velocity = velocities[row, k] - (velocities[row, i]+velocities[row, j])/2
        radius = float(np.linalg.norm(displacement))
        bond.append(float(lengths[row, which]))
        distance.append(radius)
        radial.append(float(np.dot(displacement, relative_velocity)/radius*FS_TO_AU))
    return dict(nearest_pair_bohr=bond, spectator_separation_bohr=distance,
                radial_velocity_bohr_per_fs=radial)


def main():
    started = time.perf_counter()
    assert len(os.sched_getaffinity(0)) == 1
    report = read(ANALYSIS / "quantum-rate-comparison.json")
    decision = read(ANALYSIS / "precision-extension-decision.json")
    execution = read(ANALYSIS / "execution.json")
    protocol_full = read(DATA / "protocol.json")
    protocol = dict(protocol_full)
    protocol.pop("created_before_primary_children_utc")
    parent_protocol_path = EVIDENCE / "surface-partition/primary-protocol.json"
    parent_protocol = read(parent_protocol_path)
    classical_path = EVIDENCE / "h3-classical/transmission/transmission-comparison.json"
    classical = read(classical_path)
    transfer_path = EVIDENCE / "h3-quantum/diatom-transfer.json"
    transfer = read(transfer_path)
    prespec_path = EVIDENCE / "h3-quantum/recrossing-analysis-prespecification.json"
    prespec = read(prespec_path)
    assert report["completed_seeds"] == report["expected_seeds"] == protocol["seeds"]
    assert len(protocol["seeds"]) == 8 and report["pair_batches"] == [0]
    assert report["horizon_fs"] == 200 and protocol["parents_per_seed"] == 200
    assert bind(DATA / "protocol.json") == report["protocol_sha256"] == decision["protocol_sha256"]
    assert bind(parent_protocol_path) == report["parent_protocol_sha256"]
    assert bind(classical_path) == report["classical_reference_sha256"]
    assert bind(transfer_path) == report["reactant_reference_sha256"]
    analyzer = ROOT / "benchmarks/h3/quantum_recrossing_analyze.py"
    assert bind(analyzer) == report["analysis_source_sha256"] == prespec["source_sha256"]
    for name, digest in protocol["source_sha256"].items():
        path = (ROOT / "src/pyrpmd" / name.split(":", 1)[1]
                if name.startswith("loaded_native:") else ROOT / name)
        assert bind(path) == digest, name
    beta, mass, beads = protocol["beta"], protocol["mass_au"], protocol["beads"]
    assert transfer["temperature_K"] == protocol["temperature_K"] == 300.
    assert transfer["reduced_mass_au"] == mass/2
    assert transfer["primary_beads"] == beads == 128
    assert transfer["provenance"]["pes"] == protocol["model"]
    rows = [row for row in transfer["bead_refinement"] if row["beads"] == beads]
    assert len(rows) == 1 and rows[0]["domain_bohr"] == [.5, 8.]
    assert np.isclose(rows[0]["free_kernel_width_bohr"]**2*mass/2*beads, beta, rtol=1e-14)
    logs, per_dt = [], {"0.1": [], "0.05": []}
    endpoints = {dt: {key: [] for key in ["nearest_pair_bohr", "spectator_separation_bohr",
                                         "radial_velocity_bohr_per_fs"]} for dt in per_dt}
    maximum_metric_relative_error = maximum_velocity_absolute_error = 0.
    maximum_final_surface_error = maximum_final_minimum_pair_error = 0.
    maximum_final_clock_error_au = 0.
    chunks_checked = total_rng_replays = channel_replays = 0
    late_changes = {dt: 0 for dt in per_dt}
    signed_nonpositive = {dt: 0 for dt in per_dt}
    parent_flags = []
    for seed in protocol["seeds"]:
        parent_dir = parent_protocol_path.parent / "primary/ts" / f"seed-{seed}"
        parent_settings = read(parent_dir / "settings.json")
        parent_report = read(parent_dir / "report.json")
        parent_checkpoint = parent_dir / "checkpoint.npz"
        parent_hash = bind(parent_checkpoint)
        domain_hash = bind(parent_dir / "reference-domain.json")
        assert parent_hash == parent_report["checkpoint_sha256"]
        assert parent_report["complete"] and parent_report["source_unchanged"]
        assert parent_settings["beta"] == beta and parent_settings["mass_au"] == mass
        assert parent_settings["model"] == protocol["model"]
        with load_checkpoint(parent_checkpoint, expected_kind="h3.surface-partition", require_integrity=True) as saved:
            parents = saved["endpoint_q"].copy()
            sweeps = saved["endpoint_completed_sweeps"].copy()
            assert str(saved["domain_sha256"]) == domain_hash
            assert str(saved["fingerprint"]) == hashlib.sha256(json.dumps(parent_settings, sort_keys=True).encode()).hexdigest()
            assert int(saved["completed"]) == parent_settings["warmup"]+parent_settings["production"]
        assert parents.shape == (200, 128, 3, 3)
        assert np.array_equal(sweeps, protocol["parent_completed_sweeps"])
        identity = dict(seed=seed, checkpoint_sha256=parent_hash,
                        settings_sha256=bind(parent_dir / "settings.json"),
                        report_sha256=bind(parent_dir / "report.json"),
                        BAR_log_TS_quantum_classical_ratio=parent_report["analysis"]["log_quantum_classical_ratio"])
        logs.append(identity["BAR_log_TS_quantum_classical_ratio"])
        flags = parent_protocol["seed_sampling_flags"]
        pa = parent_report["analysis"]
        parent_flags.append(dict(seed=seed,
            BAR_overlap=pa["minimum_BAR_overlap"] >= flags["minimum_BAR_overlap"],
            energy_ESS=min(x for x in pa["diagnostic_energy_effective_samples"] if x is not None) >= flags["minimum_energy_ESS"],
            half_log_difference=abs(pa["second_half_log_ratio"]-pa["first_half_log_ratio"]) <= flags["maximum_absolute_half_log_difference"]))
        paired_hashes = {}
        for dt in per_dt:
            run = DATA / f"seed-{seed}" / f"dt-{dt}" / "horizon-200"
            settings, child_report = read(run / "settings.json"), read(run / "report.json")
            assert settings["parent_identity"] == child_report["parent_identity"] == identity
            assert settings["source_sha256"] == protocol["source_sha256"]
            assert settings["protocol_sha256"] == fp(protocol)
            assert settings["phase"] == "primary" and settings["seed"] == seed
            assert settings["timestep_fs"] == float(dt) and settings["horizon_fs"] == 200
            assert settings["parents"] == 200 and settings["beads"] == 128
            assert settings["pair_batch"] == 0 and settings["momentum_pair_indices"] == [0, 1]
            assert child_report["complete"] and child_report["source_unchanged"]
            assert child_report["parents_completed"] == 200 and child_report["python_steps"] == 0
            assert bind(run / "settings.json") == child_report["settings_sha256"]
            inventory = {entry["file"]: entry["sha256"] for entry in child_report["chunks"]}
            expected_names = {f"chunk-{i:04d}.npz" for i in range(0, 200, 8)}
            assert len(child_report["chunks"]) == 25 and set(inventory) == expected_names
            assert {p.name for p in run.glob("chunk-*.npz")} == expected_names
            num, den = np.zeros((200, 200)), np.zeros(200)
            min_pair, max_energy, crossings, native_steps = np.inf, 0., 0, 0
            pairing = []
            for left in range(0, 200, 8):
                path = run / f"chunk-{left:04d}.npz"
                assert bind(path) == inventory[path.name]
                with load_checkpoint(path, expected_kind="h3.quantum-surface-children", require_integrity=True) as saved:
                    z = {key: saved[key] for key in saved.files}
                indices = np.arange(left, left+8)
                assert str(z["fingerprint"]) == fp(settings)
                assert np.array_equal(z["parent_index"], indices)
                assert np.array_equal(z["parent_completed_sweeps"], sweeps[indices])
                assert np.array_equal(z["momentum_pair_index"], [0, 1])
                assert np.array_equal(z["time_fs"], np.arange(1, 201))
                assert z["surface_values"].shape == (8, 2, 2, 200)
                for local, parent in enumerate(indices):
                    channel = int(np.random.default_rng(np.random.SeedSequence([seed, 95321, int(parent)])).integers(2))
                    assert channel == int(z["channel"][local])
                    q = parents[parent][:, [0, 2, 1] if channel else [0, 1, 2]]
                    value, grad = gradient(q.mean(axis=0))
                    assert abs(value) < 1e-10
                    fs = math.sqrt(float(np.sum(grad*grad/mass))/(2*math.pi*beta))
                    maximum_metric_relative_error = max(maximum_metric_relative_error, abs(fs/z["flux_metric"][local]-1))
                    assert np.isclose(fs, z["flux_metric"][local], rtol=1e-12, atol=0)
                    for pair in range(2):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, 173017, int(parent), pair]))
                        momentum = rng.normal(size=(beads, 3, 3))*np.sqrt(np.full(3, mass)[None, :, None]*beads/beta)
                        assert np.array_equal(momentum, z["initial_p"][local, pair])
                        velocity = float(np.sum(grad*momentum.mean(axis=0)/mass))
                        maximum_velocity_absolute_error = max(maximum_velocity_absolute_error, abs(velocity-z["velocity"][local, pair]))
                        assert np.isclose(velocity, z["velocity"][local, pair], rtol=1e-11, atol=1e-14)
                        total_rng_replays += 1
                    channel_replays += 1
                sign = ((z["surface_values"][:, :, 0] > 0).astype(np.int8)
                        - (z["surface_values"][:, :, 1] > 0).astype(np.int8))
                replay_num = (z["velocity"] / z["flux_metric"][:, None])[..., None]*sign
                replay_den = np.abs(z["velocity"])/z["flux_metric"][:, None]
                assert np.array_equal(replay_num, z["numerator"])
                assert np.array_equal(replay_den, z["denominator"])
                num[indices], den[indices] = replay_num.sum(axis=1), replay_den.sum(axis=1)
                signed_nonpositive[dt] += int((replay_num[..., 179:200].mean(axis=-1) <= 0).sum())
                signs_late = z["surface_values"][..., 79:200] > 0
                late_changes[dt] += int(np.any(signs_late[..., 1:] != signs_late[..., :-1], axis=-1).sum())
                min_pair = min(min_pair, float(z["minimum_pair_at_1fs"].min()))
                crossings += int((z["minimum_pair_at_1fs"].min(axis=-1) < .5).sum())
                extrema = z["energy_extrema"]
                max_energy = max(max_energy, float(np.max(abs(extrema[..., 1:]-extrema[..., :1]))*beta/beads))
                native_steps += int(z["native_steps_this_stage"])
                assert np.all(z["final_step"] == round(200/float(dt)))
                maximum_final_clock_error_au = max(maximum_final_clock_error_au, float(np.max(abs(z["final_time_au"]-200*FS_TO_AU))))
                centers = z["final_q"].mean(axis=-3)
                r12 = np.linalg.norm(centers[..., 1, :]-centers[..., 2, :], axis=-1)
                r01 = np.linalg.norm(centers[..., 0, :]-centers[..., 1, :], axis=-1)
                r02 = np.linalg.norm(centers[..., 0, :]-centers[..., 2, :], axis=-1)
                maximum_final_surface_error = max(maximum_final_surface_error, float(np.max(abs(r12-np.minimum(r01, r02)-z["surface_values"][..., -1]))))
                final_minimum = np.minimum.reduce([np.linalg.norm(z["final_q"][..., i, :]-z["final_q"][..., j, :], axis=-1).min(axis=-1) for i, j in [(0, 1), (0, 2), (1, 2)]])
                maximum_final_minimum_pair_error = max(maximum_final_minimum_pair_error, float(np.max(abs(final_minimum-z["minimum_pair_at_1fs"][..., -1]))))
                for key, values in endpoint_diagnostics(z["final_q"], z["final_p"], mass).items():
                    endpoints[dt][key].extend(values)
                pairing.append((sha_array(z["initial_p"]), sha_array(z["channel"])))
                chunks_checked += 1
            assert native_steps == child_report["native_steps_this_stage"] == 800*round(200/float(dt))
            curve = np.sum(num, axis=0)/math.fsum(den)
            halves = [np.sum(num[s], axis=0)/math.fsum(den[s]) for s in [slice(0, 100), slice(100, 200)]]
            plateaus = {"80-100": float(curve[79:100].mean()), "180-200": float(curve[179:200].mean())}
            published = next(row for row in report["per_timestep_per_seed"][dt] if row["seed"] == seed)
            assert all(abs(plateaus[k]-published["plateaus"][k]) < 2e-14 for k in plateaus)
            assert abs(math.fsum(den)/400-published["mean_denominator"]) < 2e-14
            assert min_pair == published["minimum_pair_at_1fs_bohr"] and crossings == published["children_with_sampled_domain_crossing"]
            assert abs(max_energy-published["maximum_energy_error_kbt_per_bead"]) < 1e-14
            per_dt[dt].append(dict(seed=seed, plateaus=plateaus, mean_denominator=math.fsum(den)/400,
                                  first_parent_half=float(halves[0][179:200].mean()),
                                  second_parent_half=float(halves[1][179:200].mean()),
                                  minimum_pair_bohr=min_pair, sampled_crossings=crossings,
                                  maximum_energy_error_kbt_per_bead=max_energy, native_steps=native_steps))
            paired_hashes[dt] = pairing
        assert paired_hashes["0.1"] == paired_hashes["0.05"]
    assert chunks_checked == 400 and parent_flags == report["parent_sampling_flags"]
    fine = per_dt["0.05"]
    kappa = np.asarray([row["plateaus"]["180-200"] for row in fine])
    log_ts, log_kappa = np.asarray(logs), np.log(kappa)
    log_qtst = np.log(classical["independent_classical_tst_cm3_molecule_s"])+log_ts-transfer["primary_log_quantum_classical_ratio"]
    log_rate = log_qtst+log_kappa
    qstat, kstat, rstat = stats(log_qtst), stats(kappa), stats(log_rate)
    covariance = [[math.fsum((float(x)-float(a.mean()))*(float(y)-float(c.mean())) for x, y in zip(a, c))/56
                   for c in [log_ts, log_kappa]] for a in [log_ts, log_kappa]]
    timestep = stats([coarse["plateaus"]["180-200"]-primary["plateaus"]["180-200"] for coarse, primary in zip(per_dt["0.1"], fine)])
    plateau = stats([row["plateaus"]["180-200"]-row["plateaus"]["80-100"] for row in fine])
    parent_half = stats([row["second_parent_half"]-row["first_parent_half"] for row in fine])
    normalization = stats([row["mean_denominator"] for row in fine])
    for ours, reference in [(qstat, report["statistics"]["log_qtst_statistics"]),
                            (kstat, report["statistics"]["kappa_statistics"]),
                            (rstat, report["statistics"]["log_rate_statistics"]),
                            (timestep, report["paired_timestep_comparison"]),
                            (plateau, report["time_plateau_comparison"]),
                            (parent_half, report["parent_sampling_time_halves"]),
                            (normalization, report["mean_denominator_diagnostic"])]:
        assert all(np.allclose(ours[key], reference[key], rtol=1e-11, atol=2e-14) for key in ours)
    assert np.allclose(covariance, report["statistics"]["same_chain_log_partition_log_kappa_covariance_of_mean"], rtol=1e-11, atol=1e-17)
    assert np.isclose(sum(map(sum, covariance)), rstat["standard_error"]**2, rtol=1e-11)
    relative_precision = kstat["halfwidth95"]/kstat["mean"]
    assert relative_precision > protocol["precision_gate_relative_kappa_95_halfwidth"]
    assert abs(relative_precision-decision["trigger"]["observed_relative_95_halfwidth"]) < 1e-14
    assert decision["complete_report_sha256"] == bind(ANALYSIS / "quantum-rate-comparison.json")
    assert decision["complete_execution_sha256"] == bind(ANALYSIS / "execution.json")
    assert decision["all_original_gate_results_retained"] == report["criteria"]
    assert [k for k, v in report["criteria"].items() if not v] == ["kappa_statistical_precision"]
    assert decision["rule"] == protocol["precision_extension_rule"]
    assert decision["extension"]["seeds"] == protocol["seeds"]
    assert decision["extension"]["momentum_pair_indices"] == protocol["precision_extension_pair_indices"] == [2, 3]
    assert decision["extension"]["timestep_fs"] == protocol["timestep_fs"]
    assert decision["extension"]["parents_per_seed"] == 200 and decision["extension"]["horizon_fs"] == 200
    assert not execution["available_only_used"] and execution["complete_cohort_required"]
    assert execution["returncode"] == 0 and execution["no_continuation_launched"]
    assert prespec["registered_utc"] < report["created_utc"] < decision["decided_utc"]
    endpoint_summary = {}
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    for dt, arrays in endpoints.items():
        endpoint_summary[dt] = dict(children=6400, quantile_probabilities=quantiles,
            quantiles={key: np.quantile(values, quantiles).tolist() for key, values in arrays.items()},
            nearest_pair_not_energy_based_binding=True,
            separation_at_least_counts={str(threshold): int((np.asarray(arrays["spectator_separation_bohr"]) >= threshold).sum()) for threshold in [5, 10, 20, 30]},
            radial_velocity_nonpositive_count=int((np.asarray(arrays["radial_velocity_bohr_per_fs"]) <= 0).sum()),
            children_with_observed_sign_change_from_80_to_200fs=late_changes[dt],
            nonpositive_antithetic_pair_mean_contribution_count=signed_nonpositive[dt])
    assert all(sha(path) == digest for path, digest in INPUTS.items())
    result = dict(schema="pyrpmd.complete-200fs-independent-review.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        reviewed_report_sha256=bind(ANALYSIS/"quantum-rate-comparison.json"),
        reviewed_decision_sha256=bind(ANALYSIS/"precision-extension-decision.json"),
        reviewer_source_sha256=sha(__file__), input_file_count=len(INPUTS),
        complete_registered_chains=8, parent_checkpoints=8, raw_chunks_checked=chunks_checked,
        children_per_timestep=6400, timesteps_fs=[.1, .05], independent_uncertainty_units=8,
        keyed_Maxwell_arrays_replayed=total_rng_replays, keyed_channels_replayed=channel_replays,
        all_chunk_envelopes_and_file_hashes_valid=True, all_signed_arrays_replay_exactly=True,
        same_momenta_and_channels_at_both_timesteps=True, inputs_unchanged_during_review=True,
        maximum_flux_metric_relative_difference=maximum_metric_relative_error,
        maximum_surface_velocity_absolute_difference_au=maximum_velocity_absolute_error,
        maximum_final_surface_difference_bohr=maximum_final_surface_error,
        maximum_final_minimum_pair_difference_bohr=maximum_final_minimum_pair_error,
        maximum_final_clock_error_au=maximum_final_clock_error_au,
        independently_recomputed=dict(log_qtst=qstat, kappa=kstat, log_rate=rstat,
            rate_geometric_mean=math.exp(rstat["mean"]), rate_CI95=np.exp(rstat["CI95"]).tolist(),
            covariance_of_mean=covariance, paired_time=plateau, paired_timestep=timestep,
            parent_time_halves=parent_half, denominator=normalization,
            kappa_relative_95_halfwidth=relative_precision),
        per_seed=per_dt, supplementary_endpoints=endpoint_summary,
        precision_extension_trigger_matches_registered_rule=True,
        mandatory_gate_status_remains_false=True,
        no_new_dynamics_or_PES_calls=True, partial_extra_batch_results_read=False,
        elapsed_seconds=time.perf_counter()-started,
        limitations=[
            "Parent BAR estimates are bound to their immutable raw checkpoints/reports; this review does not rerun BAR or a sampler.",
            "Eight-chain t intervals are nominal approximations; finite-chain ratio/log and geometric-mean bias is not corrected, and conditional extension is not sequential coverage adjustment.",
            "Exactly zero observed late-window differences are not a population bound on rare later recrossings.",
            "Endpoint nearest-pair/outgoing diagnostics are supplementary, not new gates or energy-based binding determinations.",
            "Minimum pair observations are at1-fs boundaries, not every physical integration step.",
            "The failed supplementary parent-half equivalence check remains visible; further momenta do not add parent equilibration.",
        ])
    (OUT/"raw-replay.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT/"input-file-hashes.json").write_text(json.dumps(INPUTS, indent=2)+"\n")
    print(json.dumps({"chunks": chunks_checked, "rate": result["independently_recomputed"]["rate_geometric_mean"],
                      "relative_precision": relative_precision, "endpoints": endpoint_summary}, indent=2))


def sha_array(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
