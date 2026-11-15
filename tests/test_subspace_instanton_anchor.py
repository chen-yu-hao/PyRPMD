"""Tests for steepest-descent instanton defects, bare skeletons and subspace anchoring."""

from __future__ import annotations

import ast
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

import pyrpmd
from pyrpmd.analysis import (
    SubspaceCorrection,
    SubspaceModel,
    apply_subspace_correction,
    instanton_anchor,
    model_from_function,
    project_linear_subspace,
    project_path_subspace,
    subspace_instanton_defect,
)
from pyrpmd.analysis.subspace_rph import develop_path, project_rph_values, rph_bath
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError
from pyrpmd.input_subspace import rate_subspace_options
from pyrpmd.instanton import (
    InstantonError,
    crossover_temperature,
    instanton_seed,
    optimize_instanton,
    steepest_descent_defect,
)
from pyrpmd.potentials.base import PESResult
from pyrpmd.workflows import rate_subspace as stage

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class _Eckart1DPotential:
    """1D Cartesian Eckart barrier along atom 0 x, harmonic confinement on y, z."""

    def __init__(self, v0: float, width: float, w_perp: float = 0.02):
        self.v0 = float(v0)
        self.width = float(width)
        self.w_perp = float(w_perp)

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        x = q[:, 0, 0]
        yz = q[:, 0, 1:]
        ch = np.cosh(x / self.width)
        sh = np.sinh(x / self.width)
        energy = self.v0 / ch**2 + 0.5 * self.w_perp**2 * np.sum(yz**2, axis=1)
        forces = np.zeros_like(q)
        forces[:, 0, 0] = 2.0 * self.v0 * sh / (self.width * ch**3)
        forces[:, 0, 1:] = -self.w_perp**2 * yz
        return PESResult(energy=energy, forces=forces)


def test_steepest_descent_defect_matches_subspace_solver_and_universal_u_scaling():
    assert pyrpmd.steepest_descent_defect is steepest_descent_defect

    # Above crossover: result=None gives factor=1, log_factor=0 and validates temperature inputs.
    with pytest.raises(ValueError, match="beta or temperature_kelvin"):
        steepest_descent_defect(None)
    with pytest.raises(ValueError, match="same temperature"):
        steepest_descent_defect(None, beta=1000.0, temperature_kelvin=500.0)
    above = steepest_descent_defect(None, beta=800.0)
    assert above["valid"] is True
    assert above["below_crossover"] is False
    assert above["factor"] == 1.0 and above["log_factor"] == 0.0

    # Universal Eckart scaling: fixed omega_b and u = beta * omega_b = 8.5 (> 2*pi), two different v = V0 / omega_b.
    mass = 1836.0
    omega_b = 0.006  # hartree
    u = 8.5
    beta = u / omega_b
    temp_k = 1.0 / (KB_AU * beta)
    defects = []
    for v_param in (1.6, 3.2):
        v0 = v_param * omega_b
        width = math.sqrt(2.0 * v0 / (mass * omega_b**2))
        pot = _Eckart1DPotential(v0=v0, width=width, w_perp=0.02 * math.sqrt(mass))
        active = np.array([[True, False, False]])
        ts = np.zeros((1, 3))
        tc = crossover_temperature(pot, ts, [mass], active_mask=active)
        assert math.isclose(tc, omega_b / (2.0 * math.pi * KB_AU), rel_tol=1e-5)
        seed = instanton_seed(ts, np.array([[1.0, 0.0, 0.0]]), 64, 0.4 * width)
        inst = optimize_instanton(pot, [mass], temp_k, seed, active_mask=active, transition_state=ts)
        defect = steepest_descent_defect(inst, beta=beta)
        assert defect["valid"] is True and defect["below_crossover"] is True
        defects.append(defect)

        # Compare against subspace_instanton_defect on the same non-unit-mass 1D SubspaceModel.
        axis = np.linspace(-8.0 * width, 8.0 * width, 401)
        sub = SubspaceModel((axis,), v0 / np.cosh(axis / width) ** 2, [mass], 0.0, reactant_reference=-5.0 * width)
        sub_def = subspace_instanton_defect(sub, beta, nbeads=64)
        assert sub_def["valid"] is True
        assert math.isclose(defect["eta1"], sub_def["eta1"], rel_tol=2e-4)
        assert math.isclose(defect["log_factor"], sub_def["log_factor"], abs_tol=2e-4)

    # A_Eck(u) depends on u only, independent of v = V0 / omega_b.
    assert math.isclose(defects[0]["log_factor"], defects[1]["log_factor"], rel_tol=5e-4)

    # Unconverged InstantonResult is rejected cleanly.
    bad = optimize_instanton(
        _Eckart1DPotential(1.6 * omega_b, math.sqrt(3.2 / (mass * omega_b))),
        [mass], temp_k,
        instanton_seed(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), 16, 0.05),
        active_mask=np.array([[True, False, False]]),
        max_iterations=1,
        raise_on_failure=False,
    )
    if not bad.converged:
        with pytest.raises(InstantonError):
            steepest_descent_defect(bad)
        recorded = steepest_descent_defect(bad, raise_on_invalid=False)
        assert recorded["valid"] is False and np.isnan(recorded["factor"])


def test_subspace_skeleton_preserves_geometry_and_sct_coordinate_without_dressing():
    # Pure frequency-coupled 2D model (M1-type): V(x, y) = V0 sech^2(x/a) + 0.5 * w(x)^2 * y^2
    v0, a = 0.02, 1.2
    w0, dw = 0.015, -0.006

    def w_perp(x):
        return w0 + dw / np.cosh(x / a) ** 2

    def pot(xy):
        x, y = float(xy[0]), float(xy[1])
        return v0 / math.cosh(x / a) ** 2 + 0.5 * w_perp(x) ** 2 * y**2

    def grad(xy):
        x, y = float(xy[0]), float(xy[1])
        ch = math.cosh(x / a)
        sh = math.sinh(x / a)
        w = w_perp(x)
        dw_dx = -2.0 * dw * sh / (a * ch**3)
        dv_dx = -2.0 * v0 * sh / (a * ch**3) + w * dw_dx * y**2
        dv_dy = w**2 * y
        return np.array([dv_dx, dv_dy], dtype=float)

    axis = np.linspace(-6.0, 6.0, 121)
    beta = 400.0
    lin_fe = project_linear_subspace(
        pot, grad, [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [axis],
        environment="free_energy", beta=beta, dividing_surface=0.0, reactant_reference=-4.0,
    )
    lin_rel = project_linear_subspace(
        pot, grad, [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [axis],
        environment="relaxed", dividing_surface=0.0, reactant_reference=-4.0,
    )
    assert lin_fe.describe()["has_bare_skeleton"] is True
    assert lin_rel.describe()["has_bare_skeleton"] is False
    assert lin_rel.skeleton() is lin_rel

    skel = lin_fe.skeleton()
    assert skel.describe()["has_bare_skeleton"] is False
    assert np.allclose(skel.values, lin_rel.values, atol=1e-13)
    assert skel.dividing_surface == lin_fe.dividing_surface
    assert skel.reactant_reference == lin_fe.reactant_reference

    # Curved 2D path with curvature='small' (SCT): skeleton keeps SCT-transformed tau grid and bare V(tau).
    s_pts = np.linspace(-5.0, 5.0, 101)
    curved_path = np.column_stack([s_pts, 0.6 * np.tanh(s_pts / 1.5)])

    def curved_pot(xy):
        x, y = float(xy[0]), float(xy[1])
        y0 = 0.6 * math.tanh(x / 1.5)
        return v0 / math.cosh(x / a) ** 2 + 0.5 * 0.02**2 * (y - y0) ** 2

    def curved_grad(xy):
        x, y = float(xy[0]), float(xy[1])
        ch_x = math.cosh(x / a)
        sh_x = math.sinh(x / a)
        y0 = 0.6 * math.tanh(x / 1.5)
        dy0 = 0.4 / math.cosh(x / 1.5) ** 2
        dv_dy = 0.02**2 * (y - y0)
        dv_dx = -2.0 * v0 * sh_x / (a * ch_x**3) - dv_dy * dy0
        return np.array([dv_dx, dv_dy], dtype=float)

    path_axis = np.linspace(0.2, 9.8, 121)
    with pytest.warns(RuntimeWarning, match="clipped abar"):
        sct_fe = project_path_subspace(
            curved_pot, curved_grad, curved_path, [1.0, 1.0], path_axis,
            environment="free_energy", beta=beta, curvature="small",
        )
    with pytest.warns(RuntimeWarning, match="clipped abar"):
        sct_frozen = project_path_subspace(
            curved_pot, curved_grad, curved_path, [1.0, 1.0], path_axis,
            environment="frozen", curvature="small", dividing_surface=float(path_axis[len(path_axis) // 2]),
        )
    sct_skel = sct_fe.skeleton()
    assert np.allclose(sct_skel.axes[0], sct_fe.axes[0])
    assert sct_skel.dividing_surface == sct_fe.dividing_surface
    assert not np.allclose(sct_skel.values, sct_fe.values)
    assert np.allclose(sct_skel.values, sct_frozen.values, atol=1e-12)

    # 2D RPH project_rph_values(..., return_bare=True) separates bath dressing from bare potential.
    dev = develop_path(curved_path, [1.0, 1.0], rigid_body=False, extend=2.0)
    dev.orient(float(dev.s[len(dev.s) // 2]), [1.0, 0.0])
    bath = {"arc": dev.s, "energy": np.full_like(dev.s, 0.005)}
    gx = np.linspace(-3.0, 3.0, 31)
    gy = np.linspace(-2.0, 2.0, 25)
    only_dressed = project_rph_values(curved_pot, dev, (gx, gy), bath, arc_range=(-3.0, 3.0),
                                      half_width=1.5, wall_energy=0.5)
    dressed, bare = project_rph_values(curved_pot, dev, (gx, gy), bath, arc_range=(-3.0, 3.0),
                                       half_width=1.5, wall_energy=0.5, return_bare=True)
    assert np.allclose(dressed, only_dressed)
    interior = bare < 0.49
    assert np.any(interior)
    assert np.allclose(dressed[interior] - bare[interior], 0.005, atol=1e-12)


def test_subspace_instanton_defect_2d_and_invalid_ridge_detection():
    # Separable 2D barrier: 2D instanton defect must equal the 1D barrier defect.
    v0, a, wy = 0.5, 1.0, 1.3
    omega_b = math.sqrt(2.0 * v0) / a
    beta_above = 0.8 * (2.0 * math.pi / omega_b)
    beta_below = 1.6 * (2.0 * math.pi / omega_b)

    x = np.linspace(-8.0, 8.0, 161)
    y = np.linspace(-5.0, 5.0, 101)
    m1 = model_from_function(lambda p: v0 / np.cosh(p[..., 0] / a) ** 2, [x], dividing_surface=0.0,
                             reactant_reference=-5.0)
    m2 = model_from_function(
        lambda p: v0 / np.cosh(p[..., 0] / a) ** 2 + 0.5 * wy**2 * p[..., 1] ** 2,
        [x, y], dividing_surface=0.0, reactant_reference=-5.0,
    )
    # Gridded spline 2D model without analytic callable.
    m2_grid = SubspaceModel(m2.axes, m2.values, [1.0, 1.0], 0.0, reactant_reference=-5.0)

    assert subspace_instanton_defect(m2_grid, beta_above)["factor"] == 1.0
    d1 = subspace_instanton_defect(m1, beta_below, nbeads=64)
    d2 = subspace_instanton_defect(m2, beta_below, nbeads=64)
    d2_grid = subspace_instanton_defect(m2_grid, beta_below, nbeads=64)
    assert d1["valid"] and d2["valid"] and d2_grid["valid"]
    assert math.isclose(d1["log_factor"], d2["log_factor"], rel_tol=1e-3)
    assert math.isclose(d1["log_factor"], d2_grid["log_factor"], rel_tol=2e-3)

    # 2D second-order ridge (two negative modes at (0, 0)): rejected with valid=False.
    ridge = SubspaceModel(
        (x, y),
        v0 / np.cosh(x[:, None] / a) ** 2 - 0.2 * y[None, :] ** 2,
        [1.0, 1.0],
        0.0,
        reactant_reference=-5.0,
    )
    bad = subspace_instanton_defect(ridge, beta_below, saddle=[0.0, 0.0], nbeads=32)
    assert bad["valid"] is False
    assert "negative mode" in bad["failure_reason"]
    with pytest.raises(InstantonError):
        subspace_instanton_defect(ridge, beta_below, saddle=[0.0, 0.0], nbeads=32, raise_on_invalid=True)


def test_instanton_anchor_respects_first_order_theorem_and_flags_crossover_kink_mismatch():
    from benchmarks.subspace_instanton.models import family
    from benchmarks.subspace_instanton.reductions import minimum_energy_path, reductions

    qm = json.loads((ROOT / "benchmarks" / "subspace_instanton" / "results" / "qm.json").read_text())

    # 1) M1 at 0.65 T_c: pure transverse frequency coupling -> skeleton shift == 0, naive shift != 0.
    m1 = family()["M1"]
    e_m1 = qm["models"]["M1"]["temperatures"]["0.65"]
    beta_m1 = float(e_m1["beta"])
    red_m1, _ = reductions(m1, beta_m1, minimum_energy_path(m1))
    x_fe, v_fe = red_m1["fe"]
    _, v_mep = red_m1["mep"]
    sub_m1 = SubspaceModel((x_fe,), v_fe, [1.0], 0.0, reactant_reference=-8.0, bare_values=v_mep)
    base_corr = SubspaceCorrection(beta_m1, 1, 1.45, 0.02, 1.45e-4, 1.0e-4, 64, sub_m1.describe())
    full_m1 = {"beta": beta_m1, "below_crossover": True, "log_factor": float(e_m1["ln_eps_sd"]),
               "factor": math.exp(float(e_m1["ln_eps_sd"])), "valid": True}

    anchored_m1 = instanton_anchor(base_corr, full_m1, sub_m1, naive_defect=sub_m1, nbeads=128)
    assert anchored_m1["valid"] is True
    assert abs(anchored_m1["log_shift"]) < 5e-4
    assert abs(anchored_m1["naive_log_shift"]) > 0.05
    assert math.isclose(anchored_m1["anchored_factor"], base_corr.factor, rel_tol=1e-3)

    # 2) M4 bare reduction at 0.65 T_c: slow bath shifts instanton -> nonzero skeleton shift, clean serialization.
    m4 = family()["M4"]
    e_m4 = qm["models"]["M4"]["temperatures"]["0.65"]
    beta_m4 = float(e_m4["beta"])
    red_m4, _ = reductions(m4, beta_m4, minimum_energy_path(m4))
    x_b, v_b = red_m4["bare"]
    sub_b = SubspaceModel((x_b,), v_b, [1.0], 0.0, reactant_reference=-8.0)
    corr_b = SubspaceCorrection(beta_m4, 1, 1.60, 0.03, 1.6e-4, 1.0e-4, 64, sub_b.describe())
    full_m4 = {"beta": beta_m4, "below_crossover": True, "log_factor": float(e_m4["ln_eps_sd"]),
               "factor": math.exp(float(e_m4["ln_eps_sd"])), "valid": True}
    anchored_b = instanton_anchor(corr_b, full_m4, sub_b, nbeads=128)
    assert anchored_b["valid"] is True
    assert anchored_b["log_shift"] < -0.15
    ac = anchored_b["anchored_correction"]
    assert ac.exact_general_rate_claim if hasattr(ac, "exact_general_rate_claim") else ac.to_dict()["exact_general_rate_claim"] is False
    reloaded = SubspaceCorrection.from_dict(ac.to_dict())
    assert math.isclose(reloaded.factor, ac.factor, rel_tol=1e-12)
    assert "instanton-anchored" in ac.to_dict()["approximation"]

    applied_direct = apply_subspace_correction(2.0e-12, corr_b, standard_error=1.0e-13,
                                               anchor_log_shift=anchored_b["log_shift"])
    applied_obj = apply_subspace_correction(2.0e-12, ac, standard_error=1.0e-13)
    assert math.isclose(applied_direct["corrected_rate"], applied_obj["corrected_rate"], rel_tol=1e-12)
    assert math.isclose(applied_direct["corrected_rate_standard_error"],
                        applied_obj["corrected_rate_standard_error"], rel_tol=1e-12)

    # 3) Between-crossovers rejection: M4 at T = T_c(H) is above full crossover (below=False)
    # while bare barrier (higher omega_b) is already below its own crossover (below=True)!
    e_m4_tc = qm["models"]["M4"]["temperatures"]["1.0"]
    beta_tc = float(e_m4_tc["beta"])
    corr_tc = SubspaceCorrection(beta_tc, 1, 1.25, 0.02, 1.25e-3, 1.0e-3, 64, sub_b.describe())
    full_tc = {"beta": beta_tc, "below_crossover": False, "log_factor": 0.0, "factor": 1.0, "valid": True}
    between = instanton_anchor(corr_tc, full_tc, sub_b, nbeads=128)
    assert between["between_crossovers"] is True
    assert between["valid"] is False
    assert not between["anchored_correction"].valid
    overridden = instanton_anchor(corr_tc, full_tc, sub_b, nbeads=128, allow_crossover_mismatch=True)
    assert overridden["valid"] is True and overridden["warnings"]


def test_input_schema_and_workflow_forwarding(tmp_path, monkeypatch):
    opts = rate_subspace_options({"task": "rate", "rate": {"subspace_correction": {"enabled": True}}})
    assert opts["anchor"] == "none"
    opts_skel = rate_subspace_options({
        "task": "rate",
        "rate": {"subspace_correction": {"enabled": True, "anchor": "skeleton", "model_2d": "plane"}},
    })
    assert opts_skel["anchor"] == "skeleton"
    assert opts_skel["model_2d"] == "plane"
    with pytest.raises(InputError, match="anchor"):
        rate_subspace_options({"task": "rate", "rate": {"subspace_correction": {"enabled": True, "anchor": "naive"}}})

    # Verify rate_subspace forwards both model_2d and anchor into specification["subspace"].
    captured = []
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: None)
    monkeypatch.setattr(stage, "_source_identity", lambda *a: "id-1")
    monkeypatch.setattr(
        "pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline",
        lambda spec, **kw: captured.append(spec) or {
            "status": "complete", "dimensions_computed": [1], "applied_dimension": 1,
            "corrections": {"1": {"factor": 1.2}},
        },
    )
    cfg = {
        "task": "rate",
        "rate": {"subspace_correction": {"enabled": True, "mode": "factor_only", "anchor": "skeleton", "model_2d": "plane"}},
    }
    stage.postprocess_rate_subspace(cfg, tmp_path)
    assert captured[0]["subspace"]["anchor"] == "skeleton"
    assert captured[0]["subspace"]["model_2d"] == "plane"


def test_pipeline_skeleton_anchor_shifts_factor_and_invalidates_unanchored_cache(tmp_path, monkeypatch):
    from pyrpmd.analysis import subspace_pipeline as pipe

    # Pipeline anchor="skeleton" shifts C_S by exp(A_H - A_{S^0}) and invalidates unanchored cache reuse.
    x = np.linspace(-24.0, 24.0, 241)
    v_bare = 0.02 / np.cosh(x) ** 2
    v_dressed = v_bare + 0.004 / np.cosh(x) ** 2
    model = SubspaceModel((x,), v_dressed, [1.0], 0.0, reactant_reference=-8.0, bare_values=v_bare,
                          provenance={"validity_policy": {"outside_grid": "allow"}})
    omega_b = math.sqrt(0.04)
    beta = 1.5 * (2.0 * math.pi / omega_b)

    fake_run = SimpleNamespace(
        root=tmp_path,
        provider=SimpleNamespace(close=lambda: None),
        system=SimpleNamespace(geometry=np.zeros((1, 3)), masses=np.array([1.0])),
        beta=beta,
        nbeads=16,
        xi_star=1.0,
        energy=lambda p: 0.0,
    )
    monkeypatch.setattr(pipe, "load_rate_run", lambda *a: fake_run)
    monkeypatch.setattr(pipe, "build_path", lambda *a: (SimpleNamespace(curvature_width=lambda *a, **k: {"maximum": 0.1}),
                                                        None, np.zeros((8, 3)), {"source": "mep"}))
    monkeypatch.setattr(pipe, "_static_curvature_width", lambda *a: {"maximum": 0.1})
    monkeypatch.setattr(pipe, "path_model", lambda *a, **k: (model, {}, {}))

    skel_def = subspace_instanton_defect(model, beta, use_skeleton=True, nbeads=128)
    shift = 0.15
    monkeypatch.setattr(
        pipe, "_full_instanton_defect",
        lambda *a, **k: {
            "regime": "below_crossover", "below_crossover": True, "beta": beta,
            "crossover_beta": skel_def["crossover_beta"],
            "log_factor": skel_def["log_factor"] + shift,
            "factor": math.exp(skel_def["log_factor"] + shift),
            "valid": True,
        },
    )

    base_req = {
        "source": {"rate_run": str(tmp_path)},
        "subspace": {"dimension": 1, "anchor": "none"},
        "rpmd": {"windows": 12, "walkers": 4, "burn_in": 10, "sweeps": 10, "trajectories": 16, "seed": 3},
        "output": {"directory": str(tmp_path / "out"), "reuse": True},
    }
    rep_none = pipe.run_subspace_pipeline(base_req, apply=False, reuse_key="k1")
    assert "instanton_anchor" not in rep_none

    anch_req = dict(base_req, subspace={"dimension": 1, "anchor": "skeleton"})
    rep_anch = pipe.run_subspace_pipeline(anch_req, apply=False, reuse_key="k1")
    assert "instanton_anchor" in rep_anch
    assert rep_anch["instanton_anchor"]["dimensions"]["1"]["valid"] is True
    assert math.isclose(
        rep_anch["corrections"]["1"]["factor"],
        rep_none["corrections"]["1"]["factor"] * math.exp(shift),
        rel_tol=1e-6,
    )
    # Reusing the anchored record preserves its anchor summary without recomputing subspace_correction.
    monkeypatch.setattr(pipe, "subspace_correction", lambda *a, **k: pytest.fail("recomputed anchored cache"))
    rep_reused = pipe.run_subspace_pipeline(anch_req, apply=False, reuse_key="k1")
    assert rep_reused["instanton_anchor"]["dimensions"]["1"]["valid"] is True
    assert math.isclose(rep_reused["corrections"]["1"]["factor"], rep_anch["corrections"]["1"]["factor"], rel_tol=1e-12)

    legacy_stub = SimpleNamespace(dimension=2, model={"construction": {"environment": "adiabatic", "model_2d": "rph"}})
    assert pipe._saved_model_matches_options(legacy_stub, 2, {"environment": "adiabatic"})
    assert not pipe._saved_model_matches_options(legacy_stub, 2, {"environment": "adiabatic", "anchor": "skeleton"})


def _sphinx_setting(name: str):
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Sphinx setting {name!r} is missing")


def test_subspace_docs_are_included_and_free_of_broken_relative_validation_links():
    name = "subspace-instanton-anchor.md"
    assert (DOCS / name).is_file()
    assert any(fnmatchcase(name, pattern) for pattern in _sphinx_setting("include_patterns"))
    assert not any(fnmatchcase(name, pattern) for pattern in _sphinx_setting("exclude_patterns"))

    index = (DOCS / "index.md").read_text(encoding="utf-8")
    trees = re.findall(r"```\{toctree\}\n(.*?)```", index, flags=re.DOTALL)
    assert any("subspace-instanton-anchor" in block for block in trees)

    source = (DOCS / name).read_text(encoding="utf-8")
    assert "Status: draft" not in source
    assert "Not yet tested" not in source
    assert "\\[" not in source and "\\]" not in source
    equations = re.findall(r"^\$\$\n(.+?)^\$\$$", source, flags=re.MULTILINE | re.DOTALL)
    assert len(equations) >= 8

    # Historical validation directories outside include_patterns must use full GitHub URLs in manual chapters.
    included_validation = set(_sphinx_setting("include_patterns"))
    for doc_name in ("subspace-correction.md", "subspace-transfer-theory.md", "subspace-instanton-anchor.md"):
        text = (DOCS / doc_name).read_text(encoding="utf-8")
        for match in re.finditer(r"\]\((validation/[^)#]+)", text):
            rel = match.group(1)
            assert any(fnmatchcase(rel, pat) for pat in included_validation), (
                f"{doc_name} contains unindexed relative link {rel!r}; use a full GitHub URL"
            )


def test_subspace_instanton_merged_rpmd_and_report_metrics():
    from benchmarks.subspace_instanton.report import format_report, summarize_metrics

    rpmd_path = ROOT / "benchmarks" / "subspace_instanton" / "results" / "rpmd.json"
    assert rpmd_path.is_file()
    rpmd = json.loads(rpmd_path.read_text(encoding="utf-8"))
    assert set(rpmd["models"]) == {"M1", "M2", "M3", "M4"}

    metrics = summarize_metrics(rpmd)
    overall = metrics["overall_below_tc"]
    assert overall["count"] == 36
    assert overall["rms_skeleton"] < overall["rms_unanchored"] - 0.02
    assert overall["rms_skeleton"] < overall["rms_naive"] - 0.015
    assert overall["max_skeleton"] < 0.55 * overall["max_unanchored"]

    m1_fe = metrics["m1_fe_below_tc"]
    assert m1_fe["count"] == 3
    assert m1_fe["rms_skeleton"] == pytest.approx(m1_fe["rms_unanchored"], abs=1e-3)
    assert m1_fe["rms_naive"] > 3.0 * m1_fe["rms_skeleton"]

    bare = metrics["by_reduction_below_tc"]["bare"]
    assert bare["rms_skeleton"] < 0.35 * bare["rms_unanchored"]

    md = format_report(rpmd)
    assert "## 1." in md and "## 2." in md and "## 3." in md
