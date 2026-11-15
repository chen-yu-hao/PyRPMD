"""Parallel multi-session shard runner and merger for ``benchmarks/subspace_instanton``.

Run one shard on an N-core worker (e.g. a Google Colab High-RAM VM):

    python -m benchmarks.subspace_instanton.run_shard \\
        --qm benchmarks/subspace_instanton/results/qm.json \\
        --shard-index 0 --num-shards 4 --workers 8 --out-dir /content/shard_0

Merge completed task files into a single committed result file:

    python -m benchmarks.subspace_instanton.run_shard --merge /tmp/shards \\
        --qm benchmarks/subspace_instanton/results/qm.json \\
        --skeleton benchmarks/subspace_instanton/results/skeleton.json \\
        --out benchmarks/subspace_instanton/results/rpmd.json
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.subspace_instanton.greens import barrier_rates
from benchmarks.subspace_instanton.instanton import find_instanton, steepest_descent_factor
from benchmarks.subspace_instanton.models import eckart, family
from benchmarks.subspace_instanton.reductions import minimum_energy_path, reductions
from benchmarks.subspace_instanton.run_rpmd import (
    BEADS_FULL,
    BEADS_REDUCED,
    REDUCTIONS,
    full_model,
    solve,
    tabulated_model,
)
from benchmarks.subspace_instanton.run_skeleton import SKELETON
from pyrpmd.analysis.subspace_correction import harmonic_bead_factor

RATIOS = (1.25, 1.0, 0.8, 0.65, 0.55)


def adiabatic_v(model_name: str, qm_model: dict, valley) -> float:
    """Dimensionless adiabatic barrier height ``V_ad^ddagger / (hbar * lambda^ddagger)``."""
    v_ad = valley.V0 + 0.5 * (float(qm_model["stable_frequency"]) - valley.omega_inf)
    return float(v_ad / float(qm_model["unstable_frequency"]))


def build_tasks(qm: dict, ratios=RATIOS) -> list[dict]:
    """Deterministic LPT-ordered list of all full, reduced and Eckart RPMD tasks."""
    models = family()
    tasks = []
    seen_eckart: dict[tuple[float, str], dict] = {}
    for name in ("M1", "M2", "M3", "M4"):
        valley = models[name]
        qm_model = qm["models"][name]
        v_ad = adiabatic_v(name, qm_model, valley)
        v_key = f"{v_ad:.6f}"
        for ratio in ratios:
            entry = qm_model["temperatures"][str(ratio)]
            beta = float(entry["beta"])
            nbeads_full = BEADS_FULL[name][ratio]
            cost_full = nbeads_full * (2.2 if name == "M4" else 1.5)
            tasks.append({
                "id": f"full_{name}_{ratio}",
                "kind": "full",
                "model": name,
                "ratio": ratio,
                "beta": beta,
                "nbeads": nbeads_full,
                "k_qm": float(entry["k_qm"]),
                "stable_frequency": float(qm_model["stable_frequency"]),
                "omega_inf": float(valley.omega_inf),
                "cost": cost_full,
            })
            for key in REDUCTIONS[name]:
                nbeads_red = BEADS_REDUCED[ratio]
                tasks.append({
                    "id": f"red_{name}_{ratio}_{key}",
                    "kind": "reduced",
                    "model": name,
                    "ratio": ratio,
                    "reduction": key,
                    "beta": beta,
                    "nbeads": nbeads_red,
                    "k_qm": float(entry["reductions"][key]["k_qm"]),
                    "cost": 0.25 * nbeads_red,
                })
            u = 2.0 * np.pi / ratio
            if (ratio, v_key) not in seen_eckart:
                nbeads_eck = 32 if u <= 6.0 else (48 if u <= 9.0 else 64)
                task = {
                    "id": f"eckart_{ratio}_{v_key}",
                    "kind": "eckart",
                    "ratio": ratio,
                    "u": float(u),
                    "v": v_ad,
                    "v_key": v_key,
                    "nbeads": nbeads_eck,
                    "models": [name],
                    "cost": 0.25 * nbeads_eck,
                }
                seen_eckart[(ratio, v_key)] = task
                tasks.append(task)
            else:
                seen_eckart[(ratio, v_key)]["models"].append(name)
    tasks.sort(key=lambda item: (-item["cost"], item["id"]))
    return tasks


def execute_task(task: dict) -> dict:
    """Execute a single RPMD task with the exact seeds and settings of run_rpmd/run_eckart."""
    kind = task["kind"]
    if kind == "full":
        name = task["model"]
        valley = family()[name]
        beta = task["beta"]
        nbeads = task["nbeads"]
        rec = solve(
            full_model(valley),
            beta,
            nbeads,
            seed=11,
            md_time=50.0 if name == "M4" else None,
        )
        bead = harmonic_bead_factor([task["stable_frequency"]], [task["omega_inf"]], beta, nbeads)
        rec["bead_factor"] = float(bead["factor"] if isinstance(bead, dict) else bead)
        rec["rate_bead_corrected"] = rec["rate"] / rec["bead_factor"]
        rec["k_qm"] = task["k_qm"]
        rec["eps"] = task["k_qm"] / rec["rate_bead_corrected"]
        return {"task": task, "result": rec}

    if kind == "reduced":
        name = task["model"]
        key = task["reduction"]
        beta = task["beta"]
        valley = family()[name]
        reduced, _ = reductions(valley, beta, minimum_energy_path(valley))
        x, potential = reduced[key]
        rec = solve(tabulated_model(x, potential, key), beta, task["nbeads"], seed=5, walkers=16, sweeps=800)
        rec["k_qm"] = task["k_qm"]
        rec["eps"] = task["k_qm"] / rec["rate"]
        return {"task": task, "result": rec}

    if kind == "eckart":
        u, v = task["u"], task["v"]
        potential, slope, curvature, _ = eckart(v)
        x = np.arange(-18.0, 18.005, 0.01)
        values = potential(x)
        rec = solve(tabulated_model(x, values, "eckart"), u, task["nbeads"], seed=7, walkers=16, sweeps=800)
        rec["k_qm"] = barrier_rates(x, values, [u])[u]
        rec["eps"] = rec["k_qm"] / rec["rate"]
        inst = find_instanton(
            lambda b: slope(b[:, 0])[:, None],
            lambda b: curvature(b[:, 0])[:, None, None],
            [0.0],
            [1.0],
            1.0,
            u,
            nbeads=256,
        )
        rec.update(u=u, v=v, ln_eps_sd=steepest_descent_factor(u, 1.0, inst))
        rec["remainder"] = float(np.log(rec["eps"]) - rec["ln_eps_sd"])
        return {"task": task, "result": rec}

    raise ValueError(f"unknown task kind: {kind}")


def run_shard(args) -> int:
    qm = json.loads(Path(args.qm).read_text())
    all_tasks = build_tasks(qm)
    shard_tasks = all_tasks[args.shard_index :: args.num_shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for task in shard_tasks:
        target = out_dir / f"{task['id']}.json"
        if target.is_file():
            continue
        pending.append(task)

    print(
        f"[shard {args.shard_index}/{args.num_shards}] total={len(shard_tasks)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    t0 = time.time()
    done = len(shard_tasks) - len(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_task, task): task for task in pending}
        for future in as_completed(futures):
            payload = future.result()
            task = payload["task"]
            res = payload["result"]
            target = out_dir / f"{task['id']}.json"
            tmp = out_dir / f".{task['id']}.json.tmp"
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(target)
            done += 1
            print(
                f"[{done}/{len(shard_tasks)} +{time.time() - t0:5.1f}s] {task['id']:28s} "
                f"eps={res['eps']:.4f}+-{res['eps'] * res['relative_error']:.4f} "
                f"kappa={res['kappa']:.3f} ({res['seconds']:.1f}s)",
                flush=True,
            )
    print(f"[shard {args.shard_index}/{args.num_shards}] ALL {len(shard_tasks)} TASKS COMPLETE in {time.time() - t0:.1f}s", flush=True)
    return 0


def merge_shards(args) -> int:
    qm = json.loads(Path(args.qm).read_text())
    skeleton = json.loads(Path(args.skeleton).read_text())
    expected = build_tasks(qm)
    records = {}
    for directory in args.merge:
        for path in sorted(Path(directory).rglob("*.json")):
            if path.name.startswith("."):
                continue
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "task" in data and "result" in data:
                records[data["task"]["id"]] = data
    missing = [t["id"] for t in expected if t["id"] not in records]
    if missing:
        raise RuntimeError(f"missing {len(missing)} task results: {missing[:10]}")

    models = family()
    eckart_table: dict[str, dict] = {}
    for item in expected:
        if item["kind"] == "eckart":
            eckart_table[f"{item['ratio']}:{item['v_key']}"] = {
                **records[item["id"]]["result"],
                "temperature_ratio": item["ratio"],
                "models": item["models"],
            }

    out = {
        "temperature_ratios": list(RATIOS),
        "skeleton_of": SKELETON,
        "eckart": eckart_table,
        "models": {},
    }
    for name in ("M1", "M2", "M3", "M4"):
        valley = models[name]
        qm_model = qm["models"][name]
        v_ad = adiabatic_v(name, qm_model, valley)
        v_key = f"{v_ad:.6f}"
        model_entry = {
            "unstable_frequency": float(qm_model["unstable_frequency"]),
            "stable_frequency": float(qm_model["stable_frequency"]),
            "omega_inf": float(valley.omega_inf),
            "v_adiabatic": v_ad,
            "temperatures": {},
        }
        for ratio in RATIOS:
            r_str = str(ratio)
            qm_temp = qm_model["temperatures"][r_str]
            skel_temp = skeleton["models"][name][r_str]
            full_res = dict(records[f"full_{name}_{ratio}"]["result"])
            ln_eps_h = float(np.log(full_res["eps"]))
            a_h = float(qm_temp["ln_eps_sd"])
            sigma_h = float(full_res["relative_error"])
            full_res.update(
                ln_eps=ln_eps_h,
                ln_eps_sd=a_h,
                remainder=ln_eps_h - a_h,
            )

            eck = eckart_table[f"{ratio}:{v_key}"]
            ln_c_eck = a_h + float(eck["remainder"])
            sigma_eck = float(np.hypot(sigma_h, eck["relative_error"]))

            red_entries = {}
            for key in REDUCTIONS[name]:
                red_res = dict(records[f"red_{name}_{ratio}_{key}"]["result"])
                ln_c_s = float(np.log(red_res["eps"]))
                sigma_s = float(red_res["relative_error"])
                sigma_comb = float(np.hypot(sigma_h, sigma_s))
                a_s = float(qm_temp["reductions"][key]["ln_eps_sd"])
                skel_key = SKELETON[key]
                a_s0 = float(skel_temp[skel_key])
                delta = ln_eps_h - ln_c_s
                ln_c_skel = ln_c_s + (a_h - a_s0)
                ln_c_naive = ln_c_s + (a_h - a_s)
                red_res.update(
                    ln_eps=ln_c_s,
                    ln_eps_sd=a_s,
                    skeleton=skel_key,
                    ln_eps_sd_skeleton=a_s0,
                    delta_sd_skeleton=a_h - a_s0,
                    delta_sd_naive=a_h - a_s,
                    transfer_error=delta,
                    transfer_standard_error=sigma_comb,
                    ln_c_anchored_skeleton=ln_c_skel,
                    anchored_skeleton_error=ln_eps_h - ln_c_skel,
                    ln_c_anchored_naive=ln_c_naive,
                    anchored_naive_error=ln_eps_h - ln_c_naive,
                )
                red_entries[key] = red_res

            model_entry["temperatures"][r_str] = {
                "beta": float(qm_temp["beta"]),
                "full": full_res,
                "eckart_estimate": {
                    "u": float(eck["u"]),
                    "v": float(eck["v"]),
                    "remainder": float(eck["remainder"]),
                    "ln_c_eckart": ln_c_eck,
                    "c_eckart": float(np.exp(ln_c_eck)),
                    "error": ln_eps_h - ln_c_eck,
                    "standard_error": sigma_eck,
                },
                "reductions": red_entries,
            }
        out["models"][name] = model_entry

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(f"Wrote {target} ({len(expected)} tasks merged across {len(RATIOS)} temperature ratios)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm", default="benchmarks/subspace_instanton/results/qm.json")
    parser.add_argument("--skeleton", default="benchmarks/subspace_instanton/results/skeleton.json")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", default="benchmarks/subspace_instanton/results/shards/0")
    parser.add_argument("--merge", nargs="+", help="Shard directories to merge")
    parser.add_argument("--out", default="benchmarks/subspace_instanton/results/rpmd.json")
    args = parser.parse_args()
    if args.merge:
        return merge_shards(args)
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
