"""Independent complete four-pair raw replay; no producer, PES or propagation.

The same function can inspect the original tree and a freshly extracted union
of the immutable base and incremental archives. The existing integrity reader
is shared; explicit geometry, RNG, signed flux, aggregation and statistics are
reconstructed here without calling the production analyzer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import t as student_t
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import FS_TO_AU

EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
SCRATCH = ROOT.parent
SEEDS = (721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323)
REPORT_SHA = "62ce4fea23a55955cf404ae8aa00a2c70204bf20dd48e74de520169a83724720"
PROTOCOL_SHA = "53203aeca4fdf66d9fbf1a23fe7a430ab99e24d18c351e85bd2eb383a87d3589"
PUBLICATION_SHA = "40dc3e500115a0bbff95be18ab25d82cfb1b6459da9e692b184606d65804ab46"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def interval(values):
    values = [float(x) for x in values]
    assert len(values) == 8 and all(math.isfinite(x) for x in values)
    mean = math.fsum(values) / 8
    se = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / 56)
    half = float(student_t.ppf(.975, 7)) * se
    return {"mean": mean, "standard_error": se, "halfwidth95": half,
            "CI95": [mean - half, mean + half], "degrees_of_freedom": 7}


def s1_gradient(c):
    grad = np.zeros((3, 3))
    d12 = c[1] - c[2]
    r12 = float(np.linalg.norm(d12))
    grad[1] += d12 / r12
    grad[2] -= d12 / r12
    nearest = 1 if np.linalg.norm(c[0] - c[1]) <= np.linalg.norm(c[0] - c[2]) else 2
    delta = c[0] - c[nearest]
    length = float(np.linalg.norm(delta))
    grad[0] -= delta / length
    grad[nearest] += delta / length
    return r12 - length, grad


def equal_numbers(actual, expected, *, atol=2e-14, rtol=1e-11):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replay(data_root, parent_evidence, output, *, reference_root=None):
    started = time.monotonic()
    data_root, parent_evidence, output = map(Path, (data_root, parent_evidence, output))
    output.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    inputs, raw_rows, parent_rows = {}, [], []
    data = data_root / "recrossing-primary"
    analyses = {1: data_root / "recrossing-analysis-200", 2: data_root / "recrossing-analysis-200-four-pairs"}

    def bind(path, *, raw=None):
        path = Path(path).resolve()
        raw = path.read_bytes() if raw is None else raw
        row = {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}
        assert str(path) not in inputs or inputs[str(path)] == row
        inputs[str(path)] = row
        return row

    def read(path):
        raw = Path(path).read_bytes()
        bind(path, raw=raw)
        return json.loads(raw)

    def checkpoint(path, kind, expected=None):
        raw = Path(path).read_bytes()
        row = bind(path, raw=raw)
        assert expected is None or row["sha256"] == expected
        with load_checkpoint(io.BytesIO(raw), expected_kind=kind, require_integrity=True) as saved:
            arrays = {name: saved[name] for name in saved.files}
            envelope = {"kind": saved.envelope["kind"], "schema_version": saved.envelope["schema_version"]}
        for name, value in arrays.items():
            if value.dtype.kind in "fc":
                assert np.isfinite(value).all(), name
        return arrays, row, envelope

    protocol_full = read(data / "protocol.json")
    assert bind(data / "protocol.json")["sha256"] == PROTOCOL_SHA
    protocol = dict(protocol_full)
    protocol.pop("created_before_primary_children_utc")
    assert tuple(protocol["seeds"]) == SEEDS and protocol["beads"] == 128
    assert protocol["parents_per_seed"] == 200
    reports = {n: read(directory / "quantum-rate-comparison.json") for n, directory in analyses.items()}
    assert bind(analyses[2] / "quantum-rate-comparison.json")["sha256"] == REPORT_SHA
    for batches, report in reports.items():
        assert tuple(report["completed_seeds"]) == tuple(report["expected_seeds"]) == SEEDS
        assert report["pair_batches"] == list(range(batches)) and report["horizon_fs"] == 200
        assert report["protocol_sha256"] == PROTOCOL_SHA
        execution = read(analyses[batches] / "execution.json")
        assert execution["returncode"] == 0 and execution["complete_cohort_required"]
        assert not execution["available_only_used"]
    for batch in (0, 1):
        schedule = read(data / f"schedule-200-pairs-{batch}.json")
        complete = read(data / f"schedule-200-pairs-{batch}-completed.json")
        assert complete["inventory"] == schedule and schedule["pair_batch"] == batch
        assert tuple(row["seed"] for row in complete["completed"]) == SEEDS
        assert all(row["complete"] for row in complete["completed"])
    if reference_root is None:
        references = {
            "parent": parent_evidence / "primary-protocol.json",
            "classical": EVIDENCE / "h3-classical/transmission/transmission-comparison.json",
            "reactant": EVIDENCE / "h3-quantum/diatom-transfer.json",
            "analysis": EVIDENCE / "h3-quantum/recrossing-analysis-prespecification.json",
        }
    else:
        reference_root = Path(reference_root)
        references = {key: reference_root / name for key, name in {
            "parent": "parent-protocol.json", "classical": "classical-transmission-reference.json",
            "reactant": "finite-P-reactant-reference.json", "analysis": "analysis-prespecification.json"}.items()}
    parent_protocol, classical, transfer, prespec = (read(references[key]) for key in ("parent", "classical", "reactant", "analysis"))
    for key, field in (("parent", "parent_protocol_sha256"), ("classical", "classical_reference_sha256"), ("reactant", "reactant_reference_sha256")):
        assert bind(references[key])["sha256"] == reports[1][field] == reports[2][field]
    analyzer = ROOT / "benchmarks/h3/quantum_recrossing_analyze.py"
    assert bind(analyzer)["sha256"] == reports[2]["analysis_source_sha256"] == prespec["source_sha256"]
    for name, expected in protocol["source_sha256"].items():
        path = ROOT / ("src/pyrpmd/" + name.split(":", 1)[1] if name.startswith("loaded_native:") else name)
        assert bind(path)["sha256"] == expected
    beta, mass, beads = protocol["beta"], protocol["mass_au"], protocol["beads"]
    assert transfer["numerical_converged"] and transfer["temperature_K"] == protocol["temperature_K"] == 300.
    assert transfer["reduced_mass_au"] == mass / 2 and transfer["primary_beads"] == beads
    assert transfer["provenance"]["pes"] == protocol["model"]
    transfer_rows = [row for row in transfer["bead_refinement"] if row["beads"] == beads]
    assert len(transfer_rows) == 1 and transfer_rows[0]["domain_bohr"] == [.5, 8.]
    equal_numbers(transfer_rows[0]["free_kernel_width_bohr"] ** 2 * mass / 2 * beads, beta, atol=0, rtol=1e-14)

    results = {n: {dt: [] for dt in (.1, .05)} for n in (1, 2)}
    curves = {n: [] for n in (1, 2)}
    log_ts, parent_flags = [], []
    errors = {"flux_metric_relative": 0., "velocity_au": 0., "final_surface_bohr": 0.,
              "final_minimum_pair_bohr": 0., "final_clock_au": 0.}
    momenta_replayed = channels_replayed = native_steps_total = 0
    for seed in SEEDS:
        parent = parent_evidence / "primary/ts" / f"seed-{seed}"
        ps, pr = read(parent / "settings.json"), read(parent / "report.json")
        pz, parent_id, _ = checkpoint(parent / "checkpoint.npz", "h3.surface-partition", pr["checkpoint_sha256"])
        assert pr["complete"] and pr["source_unchanged"] and ps["beta"] == beta and ps["mass_au"] == mass
        assert ps["model"] == protocol["model"] and ps["beads"] == beads
        parent_files = {name: bind(parent / name) for name in ("checkpoint.npz", "settings.json", "report.json", "reference.json", "reference-domain.json")}
        assert str(pz["domain_sha256"].item()) == parent_files["reference-domain.json"]["sha256"]
        assert str(pz["fingerprint"].item()) == digest(json.dumps(ps, sort_keys=True).encode())
        assert int(pz["completed"]) == ps["warmup"] + ps["production"] == 22000
        parents, sweeps = pz["endpoint_q"], pz["endpoint_completed_sweeps"]
        assert parents.shape == (200, beads, 3, 3)
        np.testing.assert_array_equal(sweeps, protocol["parent_completed_sweeps"])
        identity = {"seed": seed, "checkpoint_sha256": parent_id["sha256"],
                    "settings_sha256": parent_files["settings.json"]["sha256"],
                    "report_sha256": parent_files["report.json"]["sha256"],
                    "BAR_log_TS_quantum_classical_ratio": pr["analysis"]["log_quantum_classical_ratio"]}
        log_ts.append(identity["BAR_log_TS_quantum_classical_ratio"])
        parent_rows.append({"seed": seed, "files": parent_files, "endpoint_q_shape": list(parents.shape)})
        flags, pa = parent_protocol["seed_sampling_flags"], pr["analysis"]
        parent_flags.append({"seed": seed,
            "BAR_overlap": pa["minimum_BAR_overlap"] >= flags["minimum_BAR_overlap"],
            "energy_ESS": min(x for x in pa["diagnostic_energy_effective_samples"] if x is not None) >= flags["minimum_energy_ESS"],
            "half_log_difference": abs(pa["second_half_log_ratio"] - pa["first_half_log_ratio"]) <= flags["maximum_absolute_half_log_difference"]})
        paired = {}
        for dt in (.1, .05):
            numerator = np.zeros((200, 4, 200))
            denominator = np.zeros((200, 4))
            minimum, maximum_energy, crossings, native_steps = math.inf, 0., 0, 0
            for batch in (0, 1):
                prefix = data if batch == 0 else data / "extra-pair-batch-1"
                run = prefix / f"seed-{seed}" / f"dt-{dt:g}" / "horizon-200"
                settings, child_report = read(run / "settings.json"), read(run / "report.json")
                pairs = list(range(2 * batch, 2 * batch + 2))
                assert settings["parent_identity"] == child_report["parent_identity"] == identity
                assert settings["source_sha256"] == protocol["source_sha256"]
                assert settings["protocol_sha256"] == compact_fingerprint(protocol)
                expected = {"phase": "primary", "seed": seed, "beads": beads, "parents": 200,
                            "timestep_fs": dt, "horizon_fs": 200, "start_horizon_fs": 0,
                            "pair_batch": batch, "momentum_pair_indices": pairs, "chunk_parents": 8}
                assert all(settings[key] == value for key, value in expected.items())
                assert child_report["complete"] and child_report["source_unchanged"] and child_report["python_steps"] == 0
                assert child_report["parents_completed"] == 200
                assert bind(run / "settings.json")["sha256"] == child_report["settings_sha256"]
                names = {f"chunk-{i:04d}.npz" for i in range(0, 200, 8)}
                inventory = {row["file"]: row["sha256"] for row in child_report["chunks"]}
                assert len(child_report["chunks"]) == 25 and set(inventory) == names
                assert {path.name for path in run.iterdir()} == names | {"settings.json", "report.json", "kappa.csv"}
                bind(run / "kappa.csv")
                for n in (batch + 1, 2):
                    published = next(row for row in reports[n]["per_timestep_per_seed"][str(dt)] if row["seed"] == seed)
                    receipt = next(row for row in published["receipts"] if row["batch"] == batch)
                    assert receipt["report_sha256"] == bind(run / "report.json")["sha256"]
                    assert receipt["settings_sha256"] == bind(run / "settings.json")["sha256"]
                    assert receipt["chunks"] == child_report["chunks"]
                batch_steps = 0
                for left in range(0, 200, 8):
                    path = run / f"chunk-{left:04d}.npz"
                    z, file_id, envelope = checkpoint(path, "h3.quantum-surface-children", inventory[path.name])
                    indices = np.arange(left, left + 8)
                    assert str(z["fingerprint"].item()) == compact_fingerprint(settings)
                    for key, expected_value in (("parent_index", indices), ("parent_completed_sweeps", sweeps[indices]),
                                                ("momentum_pair_index", pairs), ("time_fs", np.arange(1, 201))):
                        np.testing.assert_array_equal(z[key], expected_value)
                    assert str(z["previous_chunk_sha256"].item()) == ""
                    shapes = {"surface_values": (8, 2, 2, 200), "initial_p": (8, 2, 128, 3, 3),
                              "final_q": (8, 2, 2, 128, 3, 3), "final_p": (8, 2, 2, 128, 3, 3),
                              "energy_extrema": (8, 2, 2, 3), "minimum_pair_at_1fs": (8, 2, 2, 201)}
                    assert all(z[key].shape == shape for key, shape in shapes.items())
                    for local, index in enumerate(indices):
                        channel = int(np.random.default_rng(np.random.SeedSequence([seed, 95321, int(index)])).integers(2))
                        assert int(z["channel"][local]) == channel
                        q = parents[index][:, [0, 2, 1] if channel else [0, 1, 2]]
                        s, grad = s1_gradient(q.mean(axis=0))
                        assert abs(s) < 1e-10
                        metric = math.sqrt(float(np.sum(grad * grad / mass)) / (2 * math.pi * beta))
                        errors["flux_metric_relative"] = max(errors["flux_metric_relative"], abs(metric / z["flux_metric"][local] - 1))
                        equal_numbers(metric, z["flux_metric"][local], atol=0, rtol=1e-12)
                        for local_pair, pair in enumerate(pairs):
                            rng = np.random.default_rng(np.random.SeedSequence([seed, 173017, int(index), pair]))
                            momentum = rng.normal(size=(beads, 3, 3)) * np.sqrt(np.full(3, mass)[None, :, None] * beads / beta)
                            np.testing.assert_array_equal(momentum, z["initial_p"][local, local_pair])
                            velocity = float(np.sum(grad * momentum.mean(axis=0) / mass))
                            errors["velocity_au"] = max(errors["velocity_au"], abs(velocity - z["velocity"][local, local_pair]))
                            equal_numbers(velocity, z["velocity"][local, local_pair], atol=1e-14)
                            momenta_replayed += 1
                        channels_replayed += 1
                    sign = (z["surface_values"][:, :, 0] > 0).astype(int) - (z["surface_values"][:, :, 1] > 0).astype(int)
                    n = (z["velocity"] / z["flux_metric"][:, None])[..., None] * sign
                    d = np.abs(z["velocity"]) / z["flux_metric"][:, None]
                    np.testing.assert_array_equal(n, z["numerator"])
                    np.testing.assert_array_equal(d, z["denominator"])
                    numerator[left:left + 8, pairs] = n
                    denominator[left:left + 8, pairs] = d
                    minimum = min(minimum, float(z["minimum_pair_at_1fs"].min()))
                    crossings += int(np.sum(z["minimum_pair_at_1fs"].min(axis=-1) < .5))
                    e = z["energy_extrema"]
                    maximum_energy = max(maximum_energy, float(np.max(abs(e[..., 1:] - e[..., :1])) * beta / beads))
                    np.testing.assert_array_equal(z["final_step"], np.full((8, 2, 2), round(200 / dt)))
                    assert int(z["native_steps_this_stage"]) == 32 * round(200 / dt)
                    batch_steps += int(z["native_steps_this_stage"])
                    errors["final_clock_au"] = max(errors["final_clock_au"], float(np.max(abs(z["final_time_au"] - 200 * FS_TO_AU))))
                    c = z["final_q"].mean(axis=-3)
                    r01 = np.linalg.norm(c[..., 0, :] - c[..., 1, :], axis=-1)
                    r02 = np.linalg.norm(c[..., 0, :] - c[..., 2, :], axis=-1)
                    r12 = np.linalg.norm(c[..., 1, :] - c[..., 2, :], axis=-1)
                    reconstructed_s = r12 - np.minimum(r01, r02)
                    errors["final_surface_bohr"] = max(errors["final_surface_bohr"], float(np.max(abs(reconstructed_s - z["surface_values"][..., -1]))))
                    equal_numbers(reconstructed_s, z["surface_values"][..., -1], atol=2e-12)
                    reconstructed_min = np.minimum.reduce([np.linalg.norm(z["final_q"][..., a, :] - z["final_q"][..., b, :], axis=-1).min(axis=-1) for a, b in ((0, 1), (0, 2), (1, 2))])
                    errors["final_minimum_pair_bohr"] = max(errors["final_minimum_pair_bohr"], float(np.max(abs(reconstructed_min - z["minimum_pair_at_1fs"][..., -1]))))
                    equal_numbers(reconstructed_min, z["minimum_pair_at_1fs"][..., -1], atol=2e-12)
                    pair_key = (batch, left)
                    pair_value = (digest(z["initial_p"].tobytes()), digest(z["channel"].tobytes()))
                    if dt == .1:
                        paired[pair_key] = pair_value
                    else:
                        assert paired[pair_key] == pair_value
                    raw_rows.append({**file_id, "member": str(path.relative_to(data_root)), "seed": seed,
                        "timestep_fs": dt, "pair_batch": batch, "pair_indices": pairs, "first_parent": left,
                        "settings_sha256": bind(run / "settings.json")["sha256"],
                        "parent_checkpoint_sha256": parent_id["sha256"], "envelope": envelope,
                        "initial_p_sha256": pair_value[0], "channel_sha256": pair_value[1],
                        "native_steps": int(z["native_steps_this_stage"]), "signed_flux_exact": True})
                assert batch_steps == child_report["native_steps_this_stage"] == 800 * round(200 / dt)
                assert child_report["unique_native_steps_all_stages"] == batch_steps
                native_steps += batch_steps
                native_steps_total += batch_steps
                pairs_used = 2 * (batch + 1)
                n_by_parent = numerator[:, :pairs_used].sum(axis=1)
                d_by_parent = denominator[:, :pairs_used].sum(axis=1)
                total_denominator = math.fsum(d_by_parent)
                curve = n_by_parent.sum(axis=0) / total_denominator
                half_curves = [n_by_parent[h].sum(axis=0) / math.fsum(d_by_parent[h]) for h in (slice(0, 100), slice(100, 200))]
                row = {"seed": seed, "plateaus": {"80-100": float(curve[79:100].mean()), "180-200": float(curve[179:200].mean())},
                       "first_parent_half": float(half_curves[0][179:200].mean()), "second_parent_half": float(half_curves[1][179:200].mean()),
                       "mean_denominator": total_denominator / (200 * pairs_used), "minimum_pair_at_1fs_bohr": minimum,
                       "children_with_sampled_domain_crossing": crossings, "maximum_energy_error_kbt_per_bead": maximum_energy,
                       "unique_native_steps": native_steps}
                published = next(value for value in reports[batch + 1]["per_timestep_per_seed"][str(dt)] if value["seed"] == seed)
                for key in row:
                    if key == "plateaus":
                        for window, value in row[key].items(): equal_numbers(value, published[key][window])
                    elif key not in ("first_parent_half", "second_parent_half"):
                        equal_numbers(row[key], published[key])
                equal_numbers([row["first_parent_half"], row["second_parent_half"]], published["parent_time_half_plateaus"]["180-200"])
                results[batch + 1][dt].append(row)
                if dt == .05: curves[batch + 1].append(curve)
            print(f"Verified seed {seed}, dt={dt:g}: both batches, 50 complete chunks", flush=True)

    assert len(raw_rows) == 800 and native_steps_total == 76800000
    summaries = {}
    classical_tst = classical["independent_classical_tst_cm3_molecule_s"]
    log_r = transfer["primary_log_quantum_classical_ratio"]
    for batches in (1, 2):
        report, fine, coarse = reports[batches], results[batches][.05], results[batches][.1]
        kappas = [row["plateaus"]["180-200"] for row in fine]
        log_k = [math.log(k) for k in kappas]
        log_q = [math.log(classical_tst) + z - log_r for z in log_ts]
        log_rate = [q + k for q, k in zip(log_q, log_k)]
        qstat, kstat, rstat = map(interval, (log_q, kappas, log_rate))
        covariance = [[math.fsum((x - math.fsum(a) / 8) * (y - math.fsum(b) / 8) for x, y in zip(a, b)) / 56 for b in (log_ts, log_k)] for a in (log_ts, log_k)]
        time_check = interval([row["plateaus"]["180-200"] - row["plateaus"]["80-100"] for row in fine])
        dt_check = interval([a["plateaus"]["180-200"] - b["plateaus"]["180-200"] for a, b in zip(coarse, fine)])
        parent_check = interval([row["second_parent_half"] - row["first_parent_half"] for row in fine])
        normalization = interval([row["mean_denominator"] for row in fine])
        comparisons = ((qstat, report["statistics"]["log_qtst_statistics"]), (kstat, report["statistics"]["kappa_statistics"]),
                       (rstat, report["statistics"]["log_rate_statistics"]), (time_check, report["time_plateau_comparison"]),
                       (dt_check, report["paired_timestep_comparison"]), (parent_check, report["parent_sampling_time_halves"]),
                       (normalization, report["mean_denominator_diagnostic"]))
        for computed, published in comparisons:
            for key, value in computed.items(): equal_numbers(value, published[key])
        equal_numbers(covariance, report["statistics"]["same_chain_log_partition_log_kappa_covariance_of_mean"], atol=1e-17)
        equal_numbers(sum(map(sum, covariance)), rstat["standard_error"] ** 2, atol=1e-18)
        rate, qtst = math.exp(rstat["mean"]), math.exp(qstat["mean"])
        equal_numbers(rate, report["statistics"]["rate_geometric_mean_cm3_molecule_s"], atol=0)
        equal_numbers(np.exp(rstat["CI95"]), report["statistics"]["rate_CI95_cm3_molecule_s"], atol=0)
        def within(stat, margin): return stat["CI95"][0] >= -margin and stat["CI95"][1] <= margin
        paper = report["paper"]
        assert paper["doi"] == "10.1016/j.cpc.2012.10.017" and paper["table"] == 2
        assert paper["rate_cm3_molecule_s"] == 1.95e-16 and paper["qtst_cm3_molecule_s"] == 2.51e-16 and paper["kappa"] == .776
        criteria = {"complete_registered_cohort": True,
            "parent_sampling_flags": all(all(v for k, v in row.items() if k != "seed") for row in parent_flags),
            "time_plateau_equivalence": within(time_check, protocol["plateau_equivalence_absolute_margin"]),
            "timestep_equivalence": within(dt_check, protocol["timestep_equivalence_absolute_margin"]),
            "kappa_statistical_precision": kstat["halfwidth95"] / kstat["mean"] <= protocol["precision_gate_relative_kappa_95_halfwidth"],
            "no_sampled_static_domain_crossings": all(row["children_with_sampled_domain_crossing"] == 0 for dt in (.1, .05) for row in results[batches][dt]),
            "primary_maximum_energy_error_at_most_01_kbt_per_bead": max(row["maximum_energy_error_kbt_per_bead"] for row in fine) <= .1,
            "full_rate_95_log_halfwidth_at_most_5_percent": rstat["halfwidth95"] <= math.log(1.05),
            "paper_full_rate_point_within_reported_5_percent": abs(rate / paper["rate_cm3_molecule_s"] - 1) <= .05,
            "paper_qtst_point_within_reported_5_percent": abs(qtst / paper["qtst_cm3_molecule_s"] - 1) <= .05,
            "paper_kappa_point_within_reported_5_percent": abs(kstat["mean"] / paper["kappa"] - 1) <= .05}
        assert criteria == report["criteria"] and parent_flags == report["parent_sampling_flags"]
        assert all(criteria.values()) == report["fixed_P_independent_rate_gates_passed"]
        assert within(parent_check, .03) == report["parent_sampling_time_halves"]["CI95_wholly_within_margin"]
        raw_curve = Path(analyses[batches] / "kappa-curve.csv").read_bytes()
        bind(analyses[batches] / "kappa-curve.csv", raw=raw_curve)
        published_curve = np.loadtxt(io.BytesIO(raw_curve), delimiter=",", skiprows=1)
        ours_curve = np.asarray(curves[batches])
        half = student_t.ppf(.975, 7) * ours_curve.std(axis=0, ddof=1) / math.sqrt(8)
        reconstructed_curve = np.column_stack((np.arange(1, 201), ours_curve.mean(axis=0), ours_curve.mean(axis=0) - half, ours_curve.mean(axis=0) + half))
        equal_numbers(reconstructed_curve, published_curve)
        summaries[str(batches * 2)] = {"pair_indices": list(range(batches * 2)), "log_qtst": qstat, "kappa": kstat,
            "log_rate": rstat, "rate_cm3_molecule_s": rate, "rate_CI95_cm3_molecule_s": np.exp(rstat["CI95"]).tolist(),
            "same_chain_covariance_of_mean": covariance, "paired_time": time_check, "paired_timestep": dt_check,
            "parent_time_halves": parent_check, "supplementary_parent_half_equivalence_passed": within(parent_check, .03),
            "denominator_diagnostic": normalization, "relative_kappa_95_halfwidth": kstat["halfwidth95"] / kstat["mean"],
            "criteria": criteria, "fixed_P_primary_gates_passed": all(criteria.values()),
            "per_timestep_per_seed": {str(dt): results[batches][dt] for dt in (.1, .05)}}
    decision = read(analyses[1] / "precision-extension-decision.json")
    assert decision["complete_report_sha256"] == bind(analyses[1] / "quantum-rate-comparison.json")["sha256"]
    equal_numbers(decision["trigger"]["observed_relative_95_halfwidth"], summaries["2"]["relative_kappa_95_halfwidth"])
    assert decision["trigger"]["criterion"] == "kappa_statistical_precision"
    assert decision["extension"]["momentum_pair_indices"] == [2, 3] and tuple(decision["extension"]["seeds"]) == SEEDS
    assert not summaries["2"]["fixed_P_primary_gates_passed"] and summaries["4"]["fixed_P_primary_gates_passed"]
    assert not summaries["4"]["supplementary_parent_half_equivalence_passed"]
    assert all(sha(path) == row["sha256"] for path, row in inputs.items())
    assert sha(__file__) == source_sha
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"schema": "pyrpmd.h3.quantum-four-pair-independent-replay.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_source_sha256": source_sha, "reviewed_report_sha256": REPORT_SHA, "original_protocol_sha256": PROTOCOL_SHA,
        "complete_independent_chains": 8, "parent_checkpoints": 8, "parent_configurations": 1600,
        "raw_chunks_verified": len(raw_rows), "raw_bytes": sum(row["bytes"] for row in raw_rows),
        "children_per_timestep": 12800, "timesteps_fs": [.1, .05], "native_steps_preserved": native_steps_total,
        "keyed_Maxwell_array_checks_including_timestep_repetition": momenta_replayed,
        "unique_parent_momentum_pairs": 6400, "keyed_channel_checks_including_batch_and_timestep_repetition": channels_replayed,
        "all_raw_envelopes_and_report_hashes_verified": True, "same_momenta_and_channels_at_both_timesteps": True,
        "all_signed_flux_numerators_and_denominators_exact": True, "input_bytes_unchanged": True,
        "independent_uncertainty_units": 8, "summaries": summaries, "numerical_replay_maximum_errors": errors,
        "external_PES_calls": 0, "dynamics_steps": 0, "producer_or_analysis_functions_called": False,
        "elapsed_seconds": time.monotonic() - started, "CPU_seconds": usage.ru_utime + usage.ru_stime,
        "peak_RSS_KiB": usage.ru_maxrss, "CPU_affinity": sorted(os.sched_getaffinity(0)),
        "limitations": ["Raw replay verifies the recorded estimator and identities; it does not re-integrate trajectories or rerun BAR.",
            "Eight-chain t intervals are nominal approximations; ratio/geometric-mean bias and conditional-extension coverage are not separately corrected.",
            "The failed supplementary parent-half diagnostic remains visible; extra momenta did not add parent equilibration.",
            "The minimum-distance record samples 1-fs boundaries, not every integration step.",
            "A fixed-P128 reference does not establish dynamic bead convergence, validate production umbrella sampling or validate another PES/reaction."]}
    write_new(output / "raw-replay.json", result)
    write_new(output / "input-file-identities.json", list(inputs.values()))
    write_new(output / "raw-chunk-identities.json", raw_rows)
    write_new(output / "parent-identities.json", parent_rows)
    print(json.dumps({"chunks": len(raw_rows), "rate": summaries["4"]["rate_cm3_molecule_s"],
        "relative_kappa_95_halfwidth": summaries["4"]["relative_kappa_95_halfwidth"],
        "primary_gates": summaries["4"]["criteria"], "supplementary_parent_half_pass": False}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCRATCH / "literature-convergence-quantum")
    parser.add_argument("--parent-evidence", type=Path, default=EVIDENCE / "surface-partition")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    replay(args.data_root, args.parent_evidence, args.output, reference_root=args.reference_root)
