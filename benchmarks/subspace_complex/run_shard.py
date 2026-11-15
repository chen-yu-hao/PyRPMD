"""Sharded high-dimensional complex PES validation for subspace & instanton anchoring.

Evaluates four distinct chemical PES topologies across 6D--10D coordinates:

* Shard 0: ``twisted_10d_220K`` -- 10D ``SO(10)``-rotated non-planar curved +
  slow-torsion valley against exact 3D ``flux_eigenstates`` QM + 3D RPMD.
* Shard 1: ``prereaction_well_8d_200K`` and ``prereaction_well_8d_175K`` -- 8D
  ``SO(8)``-rotated barrier + pre-reaction van der Waals well against exact 2D
  ``sylvester`` QM + 2D RPMD across two deep-tunneling temperatures.
* Shard 2: ``prereaction_complex_8d_200K`` and ``submerged_complex_6d_200K`` --
  8D positive-barrier and 6D submerged-barrier (``V_TS < V_asymptote``) deep
  pre-/post-reaction ion--dipole complexes against exact 2D QM + 2D RPMD.
* Shard 3: ``leps_ion_complex_9d_220K`` and ``leps_ion_complex_9d_180K`` -- full
  9D 3-atom Cartesian LEPS + ion--dipole molecular PES (``rigid_body=True``,
  ``RPMDrateCoordinate``) with 1D/SCT/2D-RPH subspace models and 9D
  Cartesian instanton skeleton anchoring cross-checked against
  :func:`pyrpmd.instanton.optimize_instanton`.

Usage:

    # Run shard k of 4 on an 8-vCPU VM:
    python -m benchmarks.subspace_complex.run_shard \\
        --shard-index 0 --num-shards 4 --out-dir /tmp/complex_shard_0

    # Merge all 4 shards into the canonical results JSON:
    python -m benchmarks.subspace_complex.run_shard \\
        --merge /tmp/complex_shards \\
        --out benchmarks/subspace_complex/results/complex_pes.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.subspace_complex.models import (  # noqa: E402
    LepsIonComplex9D,
    PreReactionComplex,
    PreReactionWell,
    RotatedEmbedding,
    TwistedCurved,
    abstract_rate_run,
    complex_suite,
    molecular_leps_rate_run,
)
from benchmarks.subspace_rph.reference import QUANTUM, build_path, subspace_models  # noqa: E402
from pyrpmd.analysis.subspace_correction import (  # noqa: E402
    SubspaceCorrection,
    full_instanton_defect,
    instanton_anchor,
    nested_subspace_sensitivity,
    subspace_instanton_defect,
)
from pyrpmd.analysis.subspace_model import SubspaceModel  # noqa: E402
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate  # noqa: E402
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate  # noqa: E402
from pyrpmd.instanton import instanton_seed, optimize_instanton, steepest_descent_defect  # noqa: E402

HARTREE_TO_CM = 219474.6313632


CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "twisted_10d_220K",
        "shard": 0,
        "suite_key": "twisted_10d",
        "topology": "nonplanar_curved_torsion",
        "temperature_K": 220.0,
        "nbeads": 48,
        "instanton_beads": 96,
        "spacing_2d": [2.8, 2.5],
        "path_step": 3.5,
        "has_exact_full_reference": True,
        "full_spacing": [2.6, 2.6, 3.2],
        "full_bounds": [(-32.0, 48.0), (-40.0, 40.0)],
        "full_reference": 92.0,
        "full_cap": 36.0,
        "truncation_kt": 28.0,
        "truncation_min": 0.050,
        "sweeps": 500,
        "trajectories": 768,
        "seed": 101,
    },
    {
        "key": "prereaction_well_8d_200K",
        "shard": 1,
        "suite_key": "prereaction_well_8d",
        "topology": "barrier_with_prereaction_well",
        "temperature_K": 200.0,
        "nbeads": 48,
        "instanton_beads": 96,
        "spacing_2d": [2.8, 2.5],
        "path_step": 3.5,
        "has_exact_full_reference": True,
        "full_spacing": [2.5, 2.4],
        "full_bounds": [(-32.0, 48.0)],
        "full_reference": 112.0,
        "full_cap": 36.0,
        "truncation_kt": 28.0,
        "truncation_min": 0.050,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 201,
    },
    {
        "key": "prereaction_well_8d_175K",
        "shard": 1,
        "suite_key": "prereaction_well_8d",
        "topology": "barrier_with_prereaction_well",
        "temperature_K": 175.0,
        "nbeads": 64,
        "instanton_beads": 96,
        "spacing_2d": [2.8, 2.5],
        "path_step": 3.5,
        "has_exact_full_reference": True,
        "full_spacing": [2.5, 2.4],
        "full_bounds": [(-32.0, 48.0)],
        "full_reference": 112.0,
        "full_cap": 36.0,
        "truncation_kt": 28.0,
        "truncation_min": 0.050,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 202,
    },
    {
        "key": "prereaction_complex_8d_200K",
        "shard": 2,
        "suite_key": "prereaction_complex_8d",
        "topology": "deep_pre_and_post_complex",
        "temperature_K": 200.0,
        "nbeads": 48,
        "instanton_beads": 96,
        "spacing_2d": [3.2, 2.8],
        "path_step": 3.5,
        "has_exact_full_reference": True,
        "full_spacing": [2.6, 2.5],
        "full_bounds": [(-32.0, 50.0)],
        "full_reference": 132.0,
        "full_cap": 36.0,
        "truncation_kt": 28.0,
        "truncation_min": 0.050,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 301,
    },
    {
        "key": "submerged_complex_6d_200K",
        "shard": 2,
        "suite_key": "submerged_complex_6d",
        "topology": "submerged_barrier_complex",
        "temperature_K": 200.0,
        "nbeads": 48,
        "instanton_beads": 96,
        "spacing_2d": [3.2, 2.8],
        "path_step": 3.5,
        "has_exact_full_reference": True,
        "full_spacing": [2.6, 2.5],
        "full_bounds": [(-32.0, 50.0)],
        "full_reference": 132.0,
        "full_cap": 36.0,
        "truncation_kt": 28.0,
        "truncation_min": 0.050,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 302,
    },
    {
        "key": "leps_ion_complex_9d_220K",
        "shard": 3,
        "suite_key": "leps_ion_complex_9d",
        "topology": "molecular_cartesian_3atom_complex",
        "temperature_K": 220.0,
        "nbeads": 48,
        "instanton_beads": 64,
        "spacing_2d": [4.5, 3.5],
        "path_step": 3.0,
        "has_exact_full_reference": False,
        "truncation_kt": 28.0,
        "truncation_min": 0.048,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 401,
    },
    {
        "key": "leps_ion_complex_9d_180K",
        "shard": 3,
        "suite_key": "leps_ion_complex_9d",
        "topology": "molecular_cartesian_3atom_complex",
        "temperature_K": 180.0,
        "nbeads": 64,
        "instanton_beads": 64,
        "spacing_2d": [4.5, 3.5],
        "path_step": 3.0,
        "has_exact_full_reference": False,
        "truncation_kt": 28.0,
        "truncation_min": 0.048,
        "sweeps": 600,
        "trajectories": 768,
        "seed": 402,
    },
)


def _run_rpmd(model: SubspaceModel, beta: float, nbeads: int, options: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    res = ring_polymer_rate(model, beta, nbeads, **options)
    return {
        "rate": float(res.rate),
        "rate_standard_error": float(res.rate_standard_error),
        "log_standard_error": float(res.rate_standard_error / res.rate),
        "transmission": float(res.transmission),
        "seconds": float(time.time() - t0),
    }


def _run_quantum(model: SubspaceModel, beta: float, options: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    res = exact_quantum_rate(model, beta, **options)
    return {
        "rate": float(res.rate),
        "method": str(res.method),
        "retained_points": res.diagnostics.get("retained_points"),
        "seconds": float(time.time() - t0),
    }


def _build_core_reference_model(base_pot, beta: float, spec: dict[str, Any]) -> tuple[SubspaceModel, dict[str, Any]]:
    """Build the exact core-dimensional tensor-grid reference model (2D or 3D)."""
    dim = int(base_pot.dimension)
    reference = float(spec["full_reference"])
    cap = float(spec["full_cap"])
    spacing = [float(s) for s in spec["full_spacing"]]
    length = reference + cap + 2.0 * spacing[0]
    x = np.arange(-length, length + 1e-9, spacing[0])
    x = x - x[int(np.argmin(np.abs(x + reference)))] - reference
    axes = [x]
    for i in range(1, dim):
        lo, hi = spec["full_bounds"][i - 1]
        axes.append(np.arange(float(lo), float(hi) + 1e-9, spacing[i]))
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = np.asarray(base_pot.energy(points), dtype=float)
    # Place dividing surface at the saddle's x coordinate on the core potential.
    ts_guess = np.zeros(dim, dtype=float)
    if hasattr(base_pot, "b_y"):
        y_off = float(getattr(base_pot, "y_offset", 0.0))
        ts_guess[1] = float(base_pot.b_y) * np.exp(-((0.0 - y_off) / float(base_pot.c_y)) ** 2)
    if dim >= 3 and hasattr(base_pot, "b_z"):
        z_off = float(getattr(base_pot, "z_offset", 0.0))
        ts_guess[2] = float(base_pot.b_z) * np.exp(-((0.0 - z_off) / float(base_pot.c_z)) ** 2)
    from pyrpmd.analysis.subspace_paths import refine_saddle

    saddle_core, _ = refine_saddle(
        lambda u: base_pot.gradient(u), ts_guess, np.ones(dim), rigid_body=False, gradient_tolerance=1e-9
    )
    div_x = float(saddle_core[0])
    model = SubspaceModel(
        tuple(axes),
        values,
        np.ones(dim, dtype=float),
        div_x,
        "asymptotic",
        -reference,
        {"model": "full_core"},
        function=base_pot.energy,
        gradient=base_pot.gradient,
    )
    options: dict[str, Any] = {"cap_width": cap}
    if dim == 2:
        floor = 0.5 * float(base_pot.omega_y) - 1.0 / beta
        offset = max(float(spec["truncation_kt"]) / beta, float(spec["truncation_min"]))
        options.update(energy_floor=floor, truncation_energy=floor + offset,
                       dense_limit=16000, method="sylvester")
    else:
        options.update(method="flux_eigenstates")
    return model, options


def run_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one high-dimensional complex PES benchmark case."""
    t_case = time.time()
    suite = complex_suite()
    pot = suite[spec["suite_key"]]
    temp_k = float(spec["temperature_K"])
    nbeads = int(spec["nbeads"])
    inst_beads = int(spec["instanton_beads"])

    is_molecular = isinstance(pot, LepsIonComplex9D)
    if is_molecular:
        run = molecular_leps_rate_run(pot, temp_k, nbeads)
    else:
        ref_dist = max(175.0, float(spec.get("full_reference", 165.0)) + 20.0)
        run = abstract_rate_run(pot, temp_k, nbeads, reference_distance=ref_dist)

    t_path = time.time()
    path, path_info = build_path(run, step=float(spec["path_step"]))
    mep_rec = path_info["mep"]
    v_ts = float(run.energy(run.saddle))
    v_react_end = float(run.energy(path[0]))
    v_prod_end = float(run.energy(path[-1]))

    # 1. Full high-dimensional instanton steepest-descent defect A_H.
    t_inst = time.time()
    if is_molecular:
        full_def = full_instanton_defect(
            run.energy,
            run.gradient,
            pot.masses_au,
            run.beta,
            run.system.geometry.ravel(),
            rigid_body=True,
            refine=True,
            nbeads=inst_beads,
        )
        # Cross-check against standalone pyrpmd.instanton.optimize_instanton.
        class _PotWrap:
            def evaluate(self, q):
                arr = np.asarray(q, dtype=float)
                e, g = pot._parts(arr)
                return SimpleNamespace(
                    energy=np.asarray(e, float), forces=-np.asarray(g, float).reshape(arr.shape)
                )

        seed_path = instanton_seed(
            run.saddle.reshape(3, 3),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [0.0, 0.0, 1.0]]),
            inst_beads,
            0.35,
        )
        opt_res = optimize_instanton(_PotWrap(), pot.masses_au, temp_k, seed_path, max_iterations=600)
        opt_def = steepest_descent_defect(opt_res, temperature_kelvin=temp_k)
        instanton_crosscheck = {
            "optimize_instanton_factor": float(opt_def["factor"]),
            "optimize_instanton_log_factor": float(opt_def["log_factor"]),
            "log_difference": float(full_def["log_factor"] - opt_def["log_factor"]),
        }
    else:
        full_def = full_instanton_defect(
            run.energy,
            run.gradient,
            run.cartesian_masses,
            run.beta,
            run.system.geometry,
            rigid_body=False,
            refine=True,
            nbeads=inst_beads,
            gradient_batch=pot.gradient,
        )
        instanton_crosscheck = None
    full_def_seconds = float(time.time() - t_inst)

    # 2. Build and solve subspace models (1d, 1d_sct, rph) on the high-dimensional PES.
    extra_q = {
        "truncation_kt": float(spec["truncation_kt"]),
        "truncation_min": float(spec["truncation_min"]),
        "spacing_2d": list(spec["spacing_2d"]),
        "dense_limit": 16000,
    }
    saved_q = dict(QUANTUM)
    QUANTUM.update(extra_q)
    try:
        built = subspace_models(run, path, path_info, which=("1d", "1d_sct", "rph"))
    finally:
        QUANTUM.clear()
        QUANTUM.update(saved_q)

    rpmd_opts = {
        "windows": int(spec.get("windows", 48)),
        "walkers": int(spec.get("walkers", 32)),
        "burn_in": int(spec.get("burn_in", 200)),
        "sweeps": int(spec["sweeps"]),
        "trajectories": int(spec["trajectories"]),
        "seed": int(spec["seed"]),
        "threads": int(spec.get("threads", os.cpu_count() or 1)),
    }

    def _finite(val: Any) -> float | None:
        if val is None:
            return None
        f = float(val)
        return f if math.isfinite(f) else None

    subspace_out: dict[str, Any] = {}
    corr_unanchored: dict[str, SubspaceCorrection] = {}
    corr_anchored: dict[str, SubspaceCorrection] = {}

    for idx_m, (m_name, (model, q_opts, m_info, rpmd_model)) in enumerate(built.items()):
        m_rpmd_opts = {**rpmd_opts, "seed": int(spec["seed"]) + 10 * (idx_m + 1)}
        qm_res = _run_quantum(model, run.beta, q_opts)
        rp_res = _run_rpmd(rpmd_model, run.beta, nbeads, m_rpmd_opts)
        c_s = float(qm_res["rate"] / rp_res["rate"])
        ln_c_s = float(math.log(c_s))
        corr_obj = SubspaceCorrection(
            beta=run.beta,
            dimension=model.dimension,
            factor=c_s,
            log_standard_error=rp_res["log_standard_error"],
            quantum_rate=qm_res["rate"],
            rpmd_rate=rp_res["rate"],
            nbeads=nbeads,
            model=rpmd_model.describe(),
        )
        corr_unanchored[m_name] = corr_obj

        skel_beads = max(inst_beads, 384)
        anch = instanton_anchor(corr_obj, full_def, rpmd_model, nbeads=skel_beads)
        corr_anchored[m_name] = anch["anchored_correction"]

        # Also evaluate naive (non-skeleton) subspace instanton defect when supported (metric is None).
        if rpmd_model.metric is None:
            naive_def = subspace_instanton_defect(rpmd_model, run.beta, use_skeleton=False, nbeads=skel_beads)
            a_s_naive = _finite(naive_def["log_factor"]) if naive_def["valid"] else None
            naive_shift = _finite(full_def["log_factor"] - a_s_naive) if a_s_naive is not None else None
        else:
            a_s_naive = None
            naive_shift = None

        subspace_out[m_name] = {
            "dimension": int(model.dimension),
            "shape": list(model.shape),
            "captured_curvature": _finite(m_info.get("captured_curvature")),
            "bath_stable": model.provenance.get("bath_stable"),
            "quantum": qm_res,
            "rpmd": rp_res,
            "C_S": c_s,
            "ln_C_S": ln_c_s,
            "log_standard_error": rp_res["log_standard_error"],
            "anchor_valid": bool(anch["valid"]),
            "skeleton_defect_log": _finite(anch["skeleton_defect"].get("log_factor")),
            "skeleton_defect_valid": bool(anch["skeleton_defect"].get("valid", False)),
            "anchor_log_shift": _finite(anch["log_shift"]),
            "C_S_anchored": _finite(anch["anchored_correction"].factor),
            "ln_C_S_anchored": _finite(anch["anchored_correction"].log_factor),
            "naive_defect_log": a_s_naive,
            "naive_log_shift": naive_shift,
            "ln_C_S_naive": _finite(ln_c_s + naive_shift) if naive_shift is not None else None,
        }
        shift_disp = anch["log_shift"] if anch["log_shift"] is not None else float("nan")
        print(
            f"  [{spec['key']}] {m_name}: ln_C_S={ln_c_s:+.3f}+-{rp_res['log_standard_error']:.3f} "
            f"shift={shift_disp:+.3f} (qm {qm_res['seconds']:.1f}s, rpmd {rp_res['seconds']:.1f}s)",
            flush=True,
        )

    # Nested 1D -> 2D RPH sensitivity before and after skeleton anchoring.
    sens_unanchored = nested_subspace_sensitivity([corr_unanchored["1d"], corr_unanchored["rph"]])
    sens_anchored = (
        nested_subspace_sensitivity([corr_anchored["1d"], corr_anchored["rph"]])
        if corr_anchored["1d"].valid and corr_anchored["rph"].valid
        else None
    )

    # 3. Exact full-core quantum/RPMD reference when available (2D/3D core + orthogonal bath).
    full_ref_out = None
    if spec.get("has_exact_full_reference"):
        core_pot = pot.base if isinstance(pot, RotatedEmbedding) else pot
        core_model, core_q_opts = _build_core_reference_model(core_pot, run.beta, spec)
        core_qm = _run_quantum(core_model, run.beta, core_q_opts)
        core_rp = _run_rpmd(
            core_model,
            run.beta,
            nbeads,
            {**rpmd_opts, "seed": int(spec["seed"]) + 99},
        )
        c_exact = float(core_qm["rate"] / core_rp["rate"])
        ln_c_exact = float(math.log(c_exact))
        full_ref_out = {
            "core_dimension": int(core_pot.dimension),
            "embedded_dimension": int(pot.dimension),
            "shape": list(core_model.shape),
            "quantum": core_qm,
            "rpmd": core_rp,
            "C_exact": c_exact,
            "ln_C_exact": ln_c_exact,
            "log_standard_error": core_rp["log_standard_error"],
        }
        for entry in subspace_out.values():
            combined_se = float(math.hypot(core_rp["log_standard_error"], entry["log_standard_error"]))
            delta_unanch = float(ln_c_exact - entry["ln_C_S"])
            delta_skel = _finite(ln_c_exact - entry["ln_C_S_anchored"]) if entry["ln_C_S_anchored"] is not None else None
            entry["transfer_error_log"] = delta_unanch
            entry["anchored_skeleton_error_log"] = delta_skel
            entry["combined_log_standard_error"] = combined_se
            entry["relative_error_unanchored"] = float(entry["C_S"] / c_exact - 1.0)
            entry["relative_error_skeleton"] = (
                _finite(entry["C_S_anchored"] / c_exact - 1.0) if entry["C_S_anchored"] is not None else None
            )
            if entry["ln_C_S_naive"] is not None:
                entry["anchored_naive_error_log"] = float(ln_c_exact - entry["ln_C_S_naive"])
            else:
                entry["anchored_naive_error_log"] = None
    else:
        # Molecular 9D Cartesian case: compare 1D / 1D+SCT / 2D-plane against 2D RPH (captured_curvature = 1.0).
        ln_rph = float(subspace_out["rph"]["ln_C_S"])
        rph_se = float(subspace_out["rph"]["log_standard_error"])
        ln_rph_anch = subspace_out["rph"]["ln_C_S_anchored"]
        for m_name, entry in subspace_out.items():
            combined_se = 0.0 if m_name == "rph" else float(math.hypot(rph_se, entry["log_standard_error"]))
            entry["vs_rph_unanchored_log"] = float(ln_rph - entry["ln_C_S"])
            entry["vs_rph_skeleton_log"] = (
                _finite(ln_rph_anch - entry["ln_C_S_anchored"])
                if (ln_rph_anch is not None and entry["ln_C_S_anchored"] is not None)
                else None
            )
            entry["vs_rph_log_standard_error"] = combined_se

    return {
        "key": spec["key"],
        "suite_key": spec["suite_key"],
        "topology": spec["topology"],
        "dimension": int(pot.dimension),
        "rigid_body": bool(is_molecular),
        "temperature_K": temp_k,
        "beta": float(run.beta),
        "nbeads": nbeads,
        "crossover_temperature_K": float(full_def.get("crossover_temperature_kelvin")
                                         or (1.0 / (float(full_def["crossover_beta"]) * (1.0 / (run.beta * temp_k))))),
        "t_over_tc": float(full_def["crossover_beta"] / run.beta),
        "barrier_energy_eh": v_ts,
        "barrier_submerged": bool(v_ts < v_react_end - 5e-4),
        "reactant_end_energy_eh": v_react_end,
        "product_end_energy_eh": v_prod_end,
        "path": {
            "points": int(len(path)),
            "seconds": float(t_inst - t_path),
            "imaginary_frequency_cm": float(mep_rec["imaginary_frequency"] * HARTREE_TO_CM),
            "reactant_status": path_info["reactant_termination"]["status"],
            "reactant_accepted": path_info["reactant_termination"]["accepted"],
            "reactant_minima_climbed": int(len(path_info["reactant_termination"].get("minima", []))),
            "product_status": path_info["product_termination"]["status"],
            "product_accepted": path_info["product_termination"]["accepted"],
        },
        "full_instanton": {
            "nbeads": inst_beads,
            "valid": bool(full_def["valid"]),
            "below_crossover": bool(full_def["below_crossover"]),
            "factor": float(full_def["factor"]),
            "log_factor": float(full_def["log_factor"]),
            "spread": float(full_def["spread"]),
            "phase_mode_relative_residual": float(full_def["phase_mode_relative_residual"]),
            "seconds": full_def_seconds,
            "crosscheck": instanton_crosscheck,
        },
        "full_reference": full_ref_out,
        "subspace": subspace_out,
        "nested_sensitivity": {
            "unanchored_1d_to_rph": sens_unanchored,
            "skeleton_anchored_1d_to_rph": sens_anchored,
        },
        "exact_general_rate_claim": False,
        "wall_seconds": float(time.time() - t_case),
    }


def merge_shards(shards_dir: Path, out_path: Path) -> dict[str, Any]:
    """Merge per-case JSON outputs from shard directories into ``out_path``."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(shards_dir.rglob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "key" in data and "subspace" in data:
            found[str(data["key"])] = data
    expected = [spec["key"] for spec in CASE_SPECS]
    missing = [k for k in expected if k not in found]
    if missing:
        raise RuntimeError(f"missing complex PES benchmark cases: {missing}")

    cases = {k: found[k] for k in expected}
    payload = {
        "schema": "pyrpmd.benchmark.subspace_complex",
        "schema_version": 1,
        "case_count": len(cases),
        "cases": cases,
        "exact_general_rate_claim": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--case", type=str, default=None, help="run a single named case")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--merge", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.merge is not None:
        if args.out is None:
            raise SystemExit("--out is required with --merge")
        merged = merge_shards(args.merge, args.out)
        print(f"Wrote {args.out} ({merged['case_count']} high-dimensional complex PES cases merged)")
        return 0

    if args.out_dir is None:
        raise SystemExit("--out-dir is required when running benchmark shards")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.case is not None:
        selected = [s for s in CASE_SPECS if s["key"] == args.case]
        if not selected:
            raise SystemExit(f"unknown --case {args.case!r}")
    elif args.shard_index is not None:
        selected = [s for s in CASE_SPECS if int(s["shard"]) % int(args.num_shards) == int(args.shard_index)]
    else:
        selected = list(CASE_SPECS)

    t0 = time.time()
    print(f"[complex shard] running {len(selected)} cases -> {args.out_dir}", flush=True)
    for idx, spec in enumerate(selected, start=1):
        out_file = args.out_dir / f"{spec['key']}.json"
        if out_file.is_file():
            print(f"[{idx}/{len(selected)}] reuse {spec['key']}", flush=True)
            continue
        print(f"[{idx}/{len(selected)}] start {spec['key']} (D={complex_suite()[spec['suite_key']].dimension}, T={spec['temperature_K']}K)...", flush=True)
        res = run_case(spec)
        out_file.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
        a_h = res["full_instanton"]["log_factor"]
        s1_shift = res["subspace"]["1d"]["anchor_log_shift"] or 0.0
        rph_shift = res["subspace"]["rph"]["anchor_log_shift"] or 0.0
        ref_str = ""
        if res["full_reference"] is not None:
            e1 = res["subspace"]["1d"]["transfer_error_log"]
            e1_s = res["subspace"]["1d"]["anchored_skeleton_error_log"]
            er = res["subspace"]["rph"]["transfer_error_log"]
            er_s = res["subspace"]["rph"]["anchored_skeleton_error_log"]
            e1_s_str = f"{e1_s:+.3f}" if e1_s is not None else "invalid"
            er_s_str = f"{er_s:+.3f}" if er_s is not None else "invalid"
            ref_str = f" | 1D err {e1:+.3f}->{e1_s_str}, RPH err {er:+.3f}->{er_s_str}"
        print(
            f"[{idx}/{len(selected)} +{time.time() - t0:.1f}s] {spec['key']}: "
            f"A_H={a_h:+.3f} shift(1D)={s1_shift:+.3f} shift(RPH)={rph_shift:+.3f}{ref_str} "
            f"({res['wall_seconds']:.1f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
