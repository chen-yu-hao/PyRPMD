"""High-dimensional and complex-PES validation tests for subspace & instanton anchoring.

Covers curved planar & non-planar reaction valleys, barrier + pre-reaction van der
Waals wells, deep pre-/post-reaction complexes (including submerged central
barriers ``V_TS < V_asymptote``), L-shaped 2D valleys, dense ``SO(D)`` rotated
bath embeddings up to 36D (exercising the sparse ``LinearOperator`` + ``eigsh``
ring-Hessian solver), and full 9D 3-atom Cartesian LEPS + ion--dipole complexes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.subspace_complex.models import (  # noqa: E402
    CM,
    RotatedEmbedding,
    TwistedCurved,
    abstract_rate_run,
    complex_suite,
    molecular_leps_rate_run,
)
from benchmarks.subspace_rph.models import Curved, model_run  # noqa: E402
from benchmarks.subspace_rph.reference import QUANTUM, build_path  # noqa: E402

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_correction import (
    SubspaceCorrection,
    _initial_subspace_saddle_guess,
    full_instanton_defect,
    instanton_anchor,
    subspace_instanton_defect,
)
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.constants import KB_AU
from pyrpmd.instanton import instanton_seed, optimize_instanton, steepest_descent_defect


def _fd_gradient(energy_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    grad = np.empty_like(x)
    for k in range(x.size):
        step = np.zeros_like(x)
        step[k] = eps
        grad[k] = (float(energy_fn(x + step)) - float(energy_fn(x - step))) / (2.0 * eps)
    return grad


def test_complex_pes_analytic_gradients_and_topologies():
    suite = complex_suite()
    rng = np.random.default_rng(2026)

    for name, pot in suite.items():
        if name == "leps_ion_complex_9d":
            pts = [
                np.array([0.0, 0.1, -1.9, 0.05, -0.04, 0.02, -0.03, 0.08, 1.85]),
                np.array([0.2, -0.3, -3.4, 0.0, 0.1, 0.1, -0.1, 0.05, 1.52]),
            ]
        else:
            pts = [rng.normal(scale=1.8, size=pot.dimension) for _ in range(2)]
        for pt in pts:
            analytic = np.asarray(pot.gradient(pt), dtype=float).ravel()
            numeric = _fd_gradient(pot.energy, pt)
            assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7), name

    # 9D 3-atom LEPS + ion complex must be invariant to rigid translation and 3D rotation.
    leps = suite["leps_ion_complex_9d"]
    geom = np.array([[0.1, -0.2, -2.1], [0.0, 0.05, 0.02], [-0.08, 0.12, 1.72]])
    q_rot, r_rot = np.linalg.qr(rng.normal(size=(3, 3)))
    q_rot *= np.sign(np.diag(r_rot))
    if np.linalg.det(q_rot) < 0:
        q_rot[:, 0] = -q_rot[:, 0]
    transformed = geom @ q_rot.T + np.array([3.5, -2.0, 1.25])
    assert leps.energy(transformed.ravel()) == pytest.approx(leps.energy(geom.ravel()), abs=1e-12)
    g_orig = leps.gradient(geom.ravel()).reshape(3, 3) @ q_rot.T
    g_trans = leps.gradient(transformed.ravel()).reshape(3, 3)
    assert g_trans == pytest.approx(g_orig, abs=1e-11)

    # Submerged complex 6D must have V(TS) < 0 = V(asymptote) while sitting above pre/post complexes.
    sub6 = suite["submerged_complex_6d"]
    run_sub = abstract_rate_run(sub6, 200.0, 32)
    v_ts = run_sub.energy(run_sub.saddle)
    v_asym = sub6.base.energy(np.array([-220.0, 0.0]))
    x_pre, x_post = sub6.base.pre_center, sub6.base.post_center
    v_pre = sub6.base.energy(np.array([x_pre, sub6.base.b_y * np.exp(-((x_pre + 3.0) / sub6.base.c_y) ** 2)]))
    v_post = sub6.base.energy(np.array([x_post, sub6.base.b_y * np.exp(-((x_post + 3.0) / sub6.base.c_y) ** 2)]))
    assert v_ts < v_asym - 1.5e-3
    assert v_pre < v_ts - 6.0e-3
    assert v_post < v_pre - 8.0e-3


def test_full_instanton_defect_molecular_9d_rotation_invariance_and_sparse_branch():
    suite = complex_suite()
    leps = suite["leps_ion_complex_9d"]
    run_leps = molecular_leps_rate_run(leps, 220.0, 32)

    # Cross-check full_instanton_defect (from bent off-saddle guess) vs optimize_instanton.
    defect_9d = full_instanton_defect(
        run_leps.energy,
        run_leps.gradient,
        leps.masses_au,
        run_leps.beta,
        run_leps.system.geometry.ravel(),
        rigid_body=True,
        refine=True,
        nbeads=32,
    )
    assert defect_9d["valid"] and defect_9d["below_crossover"]

    class _LepsPot:
        def evaluate(self, q):
            from types import SimpleNamespace

            arr = np.asarray(q, dtype=float)
            e, g = leps._parts(arr)
            return SimpleNamespace(energy=np.asarray(e, float), forces=-np.asarray(g, float).reshape(arr.shape))

    seed_9d = instanton_seed(
        run_leps.saddle.reshape(3, 3),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [0.0, 0.0, 1.0]]),
        32,
        0.6,
    )
    opt = optimize_instanton(_LepsPot(), leps.masses_au, 220.0, seed_9d)
    ref_9d = steepest_descent_defect(opt, temperature_kelvin=220.0)
    assert defect_9d["factor"] == pytest.approx(ref_9d["factor"], rel=2e-3)

    # Orthogonal SO(D) rotation + harmonic bath invariance (3D vs 10D) and sparse eigsh branch (36D * 64 = 2304 > 2048).
    beta_220 = 1.0 / (KB_AU * 220.0)
    t3 = suite["twisted_3d"]
    t10 = suite["twisted_10d"]
    bath_33 = tuple(np.linspace(600.0, 3200.0, 33) * CM)
    t36 = RotatedEmbedding(TwistedCurved(), bath_33, seed=41)

    run_3 = abstract_rate_run(t3, 220.0, 64)
    run_10 = abstract_rate_run(t10, 220.0, 64)
    run_36 = abstract_rate_run(t36, 220.0, 64)

    d3 = full_instanton_defect(
        run_3.energy, run_3.gradient, run_3.cartesian_masses, beta_220, run_3.system.geometry,
        rigid_body=False, nbeads=64, gradient_batch=t3.gradient,
    )
    d10 = full_instanton_defect(
        run_10.energy, run_10.gradient, run_10.cartesian_masses, beta_220, run_10.system.geometry,
        rigid_body=False, nbeads=64, gradient_batch=t10.gradient,
    )
    d36 = full_instanton_defect(
        run_36.energy, run_36.gradient, run_36.cartesian_masses, beta_220, run_36.system.geometry,
        rigid_body=False, nbeads=64, gradient_batch=t36.gradient,
    )
    assert d3["valid"] and d10["valid"] and d36["valid"]
    assert d10["factor"] == pytest.approx(d3["factor"], rel=2e-4)
    assert d36["factor"] == pytest.approx(d3["factor"], rel=5e-4)


def test_curved_2d_saved_benchmark_and_nonplanar_twisted_10d_hierarchy():
    saved_path = Path(__file__).resolve().parents[1] / "benchmarks" / "subspace_rph" / "results" / "curved_2d_200K.json"
    saved = json.loads(saved_path.read_text())
    beta = float(saved["beta"])
    c_exact = float(saved["full"]["C"])

    pot = Curved(2)
    run = model_run(pot, 200.0, 48)
    path, info = build_path(run)
    gating = {"path_info": info, "validity": {}, "hessian_displacement": 5e-3}
    opts = {**QUANTUM, "environment": "adiabatic", **gating}
    model_1d, _, _ = pipeline.path_model(run, path, opts)
    _, _, info_rph, rpmd_rph = pipeline.rph_model(run, path, opts)
    assert info_rph["captured_curvature"] == pytest.approx(1.0, abs=1e-5)

    full_def = full_instanton_defect(
        run.energy, run.gradient, run.cartesian_masses, beta, run.system.geometry,
        rigid_body=False, refine=True, nbeads=96, gradient_batch=pot.gradient,
    )
    assert full_def["valid"]

    corr_1d = SubspaceCorrection(
        beta=beta, dimension=1,
        factor=float(saved["subspace"]["1d"]["C_S"]),
        log_standard_error=float(saved["subspace"]["1d"]["log_standard_error"]),
        quantum_rate=float(saved["subspace"]["1d"]["quantum"]["rate"]),
        rpmd_rate=float(saved["subspace"]["1d"]["rpmd"]["rate"]),
        nbeads=48,
        model=model_1d.describe(),
    )
    anch_1d = instanton_anchor(corr_1d, full_def, model_1d, nbeads=96)
    assert anch_1d["valid"]
    unanchored_err = abs(corr_1d.factor / c_exact - 1.0)
    anchored_err = abs(anch_1d["anchored_correction"].factor / c_exact - 1.0)
    assert unanchored_err > 0.13
    assert anchored_err < unanchored_err - 0.025

    corr_rph = SubspaceCorrection(
        beta=beta, dimension=2,
        factor=float(saved["subspace"]["rph"]["C_S"]),
        log_standard_error=float(saved["subspace"]["rph"]["log_standard_error"]),
        quantum_rate=float(saved["subspace"]["rph"]["quantum"]["rate"]),
        rpmd_rate=float(saved["subspace"]["rph"]["rpmd"]["rate"]),
        nbeads=48,
        model=rpmd_rph.describe(),
    )
    anch_rph = instanton_anchor(corr_rph, full_def, rpmd_rph, nbeads=96)
    assert anch_rph["valid"]
    assert abs(anch_rph["log_shift"]) < 1.5e-2

    # Non-planar 10D twisted valley: 2D RPH captures planar bend but leaves slow z torsion in the bath,
    # so A_H(10D) > A_{RPH^0} > A_{1D^0} > 0 strictly.
    t10 = complex_suite()["twisted_10d"]
    run_10 = abstract_rate_run(t10, 220.0, 48)
    path_10, info_10 = build_path(run_10)
    opts_10 = {**QUANTUM, "environment": "adiabatic", "path_info": info_10, "validity": {}, "hessian_displacement": 5e-3}
    m1_10, _, _ = pipeline.path_model(run_10, path_10, opts_10)
    _, _, rph_info_10, rph_rpmd_10 = pipeline.rph_model(run_10, path_10, {**opts_10, "spacing_2d": [3.5, 3.0]})
    assert 0.50 < rph_info_10["captured_curvature"] < 0.995

    def_full_10 = full_instanton_defect(
        run_10.energy, run_10.gradient, run_10.cartesian_masses, run_10.beta, run_10.system.geometry,
        rigid_body=False, nbeads=96, gradient_batch=t10.gradient,
    )
    def_rph_skel = subspace_instanton_defect(rph_rpmd_10, run_10.beta, use_skeleton=True, nbeads=128)
    def_1d_skel = subspace_instanton_defect(m1_10, run_10.beta, use_skeleton=True, nbeads=128)
    assert def_full_10["valid"] and def_rph_skel["valid"] and def_1d_skel["valid"]
    assert def_full_10["log_factor"] > def_rph_skel["log_factor"] + 0.015
    assert def_rph_skel["log_factor"] > def_1d_skel["log_factor"] + 0.06
    assert def_1d_skel["log_factor"] > 0.10


def test_prereaction_well_8d_path_climb_dome_bound_and_anchored_pipeline(tmp_path):
    pot = complex_suite()["prereaction_well_8d"]
    run = abstract_rate_run(pot, 220.0, 24)
    out = tmp_path / "well_8d"
    spec = {
        "output": {"directory": str(out), "reuse": False},
        "path": {"source": "mep", "step": 4.0, "instanton_beads": 48},
        "subspace": {"dimension": [1, 2], "model_2d": "rph", "environment": "adiabatic", "anchor": "skeleton"},
        "quantum": {"spacing_2d": [5.0, 4.0], "truncation_kt": 24.0, "truncation_min": 0.04},
        "rpmd": {"nbeads": 24, "windows": 12, "walkers": 12, "burn_in": 20, "sweeps": 40, "trajectories": 48, "seed": 7},
    }
    report = pipeline.run_subspace_pipeline(spec, run=run)
    r_term = report["path"]["reactant_termination"]
    assert len(r_term["minima"]) >= 1
    assert r_term["status"] == "coordinate_target_reached"
    assert r_term["accepted"] == "reactant_reference"
    assert report["instanton_anchor"]["mode"] == "skeleton"
    assert report["instanton_anchor"]["full_defect"]["valid"] is True

    d1 = report["instanton_anchor"]["dimensions"]["1"]
    d2 = report["instanton_anchor"]["dimensions"]["2"]
    assert d1["valid"] and d1["log_shift"] > 0.025
    # Underlying core of prereaction_well_8d is 2D planar, so 2D RPH skeleton matches full 8D defect closely.
    assert d2["valid"] and abs(d2["log_shift"]) < 0.025


def test_submerged_complex_6d_and_l_shaped_valley_saddle_localization():
    sub6 = complex_suite()["submerged_complex_6d"]
    run = abstract_rate_run(sub6, 200.0, 32)
    path, info = build_path(run)
    assert len(info["reactant_termination"]["minima"]) >= 1
    assert info["reactant_termination"]["accepted"] == "reactant_reference"
    assert info["product_termination"]["accepted"] == "product_well"

    opts = {**QUANTUM, "environment": "adiabatic", "path_info": info, "validity": {}, "hessian_displacement": 5e-3}
    m1, _, _ = pipeline.path_model(run, path, opts)
    _, _, rph_info, rph_rpmd = pipeline.rph_model(run, path, {**opts, "spacing_2d": [4.0, 3.5]})
    assert rph_info["captured_curvature"] > 0.95

    full_def = pipeline._full_instanton_defect(run, path, {"path": {"instanton_beads": 64}, "_path_info": info}, 64)
    skel_1d = subspace_instanton_defect(m1, run.beta, use_skeleton=True, nbeads=128)
    skel_2d = subspace_instanton_defect(rph_rpmd, run.beta, use_skeleton=True, nbeads=128)
    assert full_def["valid"] and skel_1d["valid"] and skel_2d["valid"]
    # Submerged 2D core + orthogonal bath: 2D RPH skeleton defect matches 6D full defect.
    assert skel_2d["log_factor"] == pytest.approx(full_def["log_factor"], abs=2.5e-2)
    assert full_def["log_factor"] > skel_1d["log_factor"] + 0.05

    # Synthetic L-shaped 2D grid where a deep complex leg at large q has lower global min than the (0, 0) saddle channel.
    s_axis = np.linspace(-60.0, 60.0, 121)
    q_axis = np.linspace(-18.0, 18.0, 73)
    S, Q = np.meshgrid(s_axis, q_axis, indexing="ij")
    v_channel = 0.015 / np.cosh(S / 22.0) ** 2 + 0.5 * (0.008**2) * Q**2
    v_leg = -0.025 * np.exp(-((S + 2.0) / 10.0) ** 2 - ((Q - 13.0) / 2.5) ** 2)
    l_model = SubspaceModel((s_axis, q_axis), v_channel + v_leg, (1.0, 1.0), 0.0, "asymptotic", -45.0)
    guess = _initial_subspace_saddle_guess(l_model)
    assert abs(guess[0]) <= 2.0
    assert abs(guess[1]) <= 1.5
    l_def = subspace_instanton_defect(l_model, run.beta, nbeads=128)
    assert l_def["valid"]
    assert abs(l_def["saddle"][1]) < 1.0


def test_leps_ion_complex_9d_end_to_end_cartesian_pipeline(tmp_path):
    leps = complex_suite()["leps_ion_complex_9d"]
    run = molecular_leps_rate_run(leps, 200.0, 24)
    base_spec = {
        "path": {"source": "mep", "step": 3.5, "instanton_beads": 48},
        "subspace": {"dimension": [1, 2], "model_2d": "rph", "environment": "adiabatic"},
        "quantum": {"spacing_2d": [6.0, 4.5], "truncation_kt": 24.0, "truncation_min": 0.04},
        "rpmd": {"nbeads": 24, "windows": 12, "walkers": 12, "burn_in": 20, "sweeps": 40, "trajectories": 48, "seed": 13},
    }
    rep_none = pipeline.run_subspace_pipeline(
        {**base_spec, "output": {"directory": str(tmp_path / "none"), "reuse": False},
         "subspace": {**base_spec["subspace"], "anchor": "none"}},
        run=run,
    )
    rep_skel = pipeline.run_subspace_pipeline(
        {**base_spec, "output": {"directory": str(tmp_path / "skel"), "reuse": False},
         "subspace": {**base_spec["subspace"], "anchor": "skeleton"}},
        run=run,
    )
    assert len(rep_none["path"]["reactant_termination"]["minima"]) >= 1
    assert rep_none["path"]["reactant_termination"]["accepted"] == "reactant_reference"
    assert rep_none["corrections"]["1"]["exact_general_rate_claim"] is False
    assert "instanton_anchor" not in rep_none
    assert rep_skel["instanton_anchor"]["mode"] == "skeleton"
    assert rep_skel["instanton_anchor"]["full_defect"]["valid"] is True
    assert rep_skel["corrections"]["1"]["exact_general_rate_claim"] is False
    assert rep_skel["corrections"]["2"]["exact_general_rate_claim"] is False

    d1_anch = rep_skel["instanton_anchor"]["dimensions"]["1"]
    d2_anch = rep_skel["instanton_anchor"]["dimensions"]["2"]
    assert d1_anch["valid"] and d2_anch["valid"]
    # Collinear 3-atom + ion MEP is curved, so 1D skeleton misses corner cutting
    # whereas 2D RPH captures the collinear corner-cutting plane to < 0.005.
    assert abs(d2_anch["log_shift"]) < 0.005
    assert d1_anch["log_shift"] > abs(d2_anch["log_shift"]) + 0.025


def test_complex_pes_merged_benchmark_and_report():
    from benchmarks.subspace_complex.report import (
        SUBSPACE_MODELS,
        format_complex_report,
        summarize_complex_metrics,
    )
    from benchmarks.subspace_complex.run_shard import CASE_SPECS

    json_path = Path(__file__).resolve().parents[1] / "benchmarks" / "subspace_complex" / "results" / "complex_pes.json"
    assert json_path.is_file()
    data = json.loads(json_path.read_text())
    assert data["schema"] == "pyrpmd.benchmark.subspace_complex"
    assert data["exact_general_rate_claim"] is False
    expected_keys = [spec["key"] for spec in CASE_SPECS]
    assert list(data["cases"].keys()) == expected_keys

    for key in expected_keys:
        case = data["cases"][key]
        assert case["exact_general_rate_claim"] is False
        assert case["full_instanton"]["valid"] is True
        assert case["full_instanton"]["below_crossover"] is True
        assert case["full_instanton"]["phase_mode_relative_residual"] < 5e-4
        for m in SUBSPACE_MODELS:
            assert m in case["subspace"]
            assert case["subspace"][m]["quantum"]["rate"] > 0.0
            assert case["subspace"][m]["rpmd"]["rate"] > 0.0
        # 2D RPH skeleton defect must always be valid below crossover.
        assert case["subspace"]["rph"]["anchor_valid"] is True
        assert case["subspace"]["rph"]["skeleton_defect_valid"] is True

    # 9D 3-atom Cartesian LEPS + ion complex: full_instanton_defect matches standalone optimize_instanton to < 1e-4.
    for mol_key in ("leps_ion_complex_9d_220K", "leps_ion_complex_9d_180K"):
        chk = data["cases"][mol_key]["full_instanton"]["crosscheck"]
        assert abs(chk["log_difference"]) < 1e-4
        assert abs(data["cases"][mol_key]["subspace"]["rph"]["anchor_log_shift"]) < 5e-3
        assert data["cases"][mol_key]["subspace"]["1d"]["anchor_log_shift"] > 0.02

    # Submerged complex 6D must flag barrier_submerged = True and improve 1D error via skeleton anchoring.
    sub_case = data["cases"]["submerged_complex_6d_200K"]
    assert sub_case["barrier_submerged"] is True
    assert abs(sub_case["subspace"]["1d"]["anchored_skeleton_error_log"]) < abs(
        sub_case["subspace"]["1d"]["transfer_error_log"]
    )

    metrics = summarize_complex_metrics(data)
    assert len(metrics["exact_reference_cases"]) == 5
    assert len(metrics["molecular_cartesian_cases"]) == 2
    rendered = format_complex_report(data)
    assert "twisted_10d_220K" in rendered
    assert "leps_ion_complex_9d_180K" in rendered

