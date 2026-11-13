"""Validity gates of the automatic path subspace (issue #182).

A path cut short by a step budget, a transverse Hessian that is not bound,
and ring polymers that sampled a gridded ``V_S`` outside its grid each used
to end in a positive, usable-looking ``C_S``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis.subspace_correction import (SubspaceCorrection, _assemble_correction,
                                                 apply_subspace_correction, nested_subspace_sensitivity,
                                                 subspace_correction, validity_gates)
from pyrpmd.analysis.subspace_model import (SubspaceModel, assemble_reaction_path, project_path_subspace,
                                            relaxed_path)
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options


# --- 1. path continuation ---------------------------------------------------

def tail_potential(x):
    """A valley along x with a slowly decaying r^-4 tail, harmonic across."""
    return -1e-2 / (1 + x[0] ** 2) ** 2 + 0.1 * x[1] ** 2


def tail_gradient(x):
    return np.array([4e-2 * x[0] / (1 + x[0] ** 2) ** 3, 0.2 * x[1]])


def test_step_budget_ends_a_path_short_of_its_asymptote_and_says_so():
    options = dict(step=0.25, rigid_body=False, return_info=True)
    short, budget = relaxed_path(tail_potential, tail_gradient, [0.5, 0.1], [1.0, 0.0], [1.0, 1.0],
                                 max_steps=20, **options)
    assert budget["status"] == "max_steps_exhausted" and not budget["converged"]
    assert budget["steps"] == 20
    # The tail still changes by four orders of magnitude more than the tolerance.
    assert budget["tail_energy_range"] > 1e3 * budget["asymptote_tolerance"]

    full, free = relaxed_path(tail_potential, tail_gradient, [0.5, 0.1], [1.0, 0.0], [1.0, 1.0], **options)
    assert free["status"] == "asymptote_converged" and free["converged"]
    assert free["max_steps"] is None and free["steps"] > budget["steps"]
    assert free["tail_energy_range"] < free["asymptote_tolerance"]
    # Scientific consequence: the budgeted path stops well above the asymptote.
    assert tail_potential(short[-1]) < tail_potential(full[-1]) - 5e-5
    assert abs(tail_potential(full[-1])) < 2e-5
    # Default return type is unchanged.
    points = relaxed_path(tail_potential, tail_gradient, [0.5, 0.1], [1.0, 0.0], [1.0, 1.0],
                          step=0.25, rigid_body=False)
    np.testing.assert_array_equal(points, full)


def test_halved_steps_cannot_fake_a_flat_tail():
    """A constant slope on a tight bend: short steps make ten points look flat."""
    radius, k, slope = 10.0, 0.5, 0.01

    def potential(x):
        r, theta = np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])
        return 0.5 * k * (r - radius) ** 2 - slope * theta

    def gradient(x):
        r = np.hypot(x[0], x[1])
        return k * (r - radius) * x / r - slope * np.array([-x[1], x[0]]) / r ** 2

    tolerance = 2e-3
    points, info = relaxed_path(potential, gradient, [radius, 0.0], [0.0, 1.0], [1.0, 1.0], step=0.5,
                                max_angle_degrees=1.0, asymptote_tolerance=tolerance, rigid_body=False,
                                max_steps=150, return_info=True)
    energies = np.array([potential(p) for p in points])
    # The point-count criterion is satisfied although the slope never changed ...
    assert np.ptp(energies[-10:]) < tolerance
    # ... while over a fixed arc length the energy is still falling.
    assert info["tail_energy_range"] > tolerance
    assert info["status"] == "max_steps_exhausted"


def test_a_coordinate_that_never_flattens_terminates_without_a_budget():
    wall = lambda x: 0.5 * (x[0] ** 2 + x[1] ** 2)
    points, info = relaxed_path(wall, lambda x: np.array(x, dtype=float), [0.5, 0.0], [1.0, 0.0], [1.0, 1.0],
                                step=0.25, rigid_body=False, patience=5, return_info=True)
    assert info["status"] == "asymptote_not_approached" and not info["converged"]
    assert info["steps"] < 200


def test_non_finite_potential_is_a_termination_status():
    def potential(x):
        return float("nan") if x[0] > 2.0 else tail_potential(x)

    _, info = relaxed_path(potential, tail_gradient, [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], step=0.25,
                           rigid_body=False, return_info=True)
    assert info["status"] == "potential_failure"


def test_assembled_path_reports_both_terminations():
    def potential(x):  # symmetric barrier at 0 with tails on both sides
        return 1e-2 / (1 + x[0] ** 2) ** 2 + 0.1 * x[1] ** 2

    def gradient(x):
        return np.array([-4e-2 * x[0] / (1 + x[0] ** 2) ** 3, 0.2 * x[1]])

    core = np.column_stack([np.linspace(-0.6, 0.6, 5), np.zeros(5)])
    path, info = assemble_reaction_path(core, potential, gradient, [1.0, 1.0], step=0.25, rigid_body=False)
    assert info["asymptotes_converged"]
    assert info["reactant_termination"]["status"] == info["product_termination"]["status"] == "asymptote_converged"
    assert len(path) == info["core_points"] + info["reactant_continuation_points"] + info["product_continuation_points"]

    _, cut = assemble_reaction_path(core, potential, gradient, [1.0, 1.0], step=0.25, rigid_body=False,
                                    max_steps=15)
    assert not cut["asymptotes_converged"]
    assert cut["reactant_continuation_points"] == cut["product_continuation_points"] == 15
    gates = validity_gates({"path_termination": {"reactant": cut["reactant_termination"],
                                                 "product": cut["product_termination"]}})
    assert not gates["valid"] and "max_steps_exhausted" in gates["failures"][0]
    allowed = validity_gates({"path_termination": {"reactant": cut["reactant_termination"],
                                                   "product": cut["product_termination"]}},
                             policy={"unconverged_path": "allow"})
    assert allowed["valid"] and allowed["overrides"] == ["path"]
    # A model with no termination record cannot pass the path gate.
    assert not validity_gates({"path_termination": {"reactant": None, "product": None}})["valid"]


# --- 2. transverse Hessian --------------------------------------------------

def saddle(a, b):
    """V = -a x^2/2 + b y^2/2: reaction mode along x, bound bath along y."""
    return (lambda x: -0.5 * a * x[0] ** 2 + 0.5 * b * x[1] ** 2,
            lambda x: np.array([-a * x[0], b * x[1]]))


def tilted_path(angle_degrees, points=9):
    angle = np.radians(angle_degrees)
    s = np.linspace(-1.0, 1.0, points)
    return np.column_stack([s * np.cos(angle), s * np.sin(angle)])


def test_tilted_tangent_leaks_the_reaction_mode_into_the_bath():
    potential, gradient = saddle(4e-4, 1e-4)  # tan^2(theta) > b/a makes the normal plane unstable
    axis = np.linspace(0.0, 2.0, 9)
    aligned = project_path_subspace(potential, gradient, tilted_path(0.0), [1.0, 1.0], axis, dividing_surface=1.0)
    assert aligned.provenance["bath_stable"] and aligned.provenance["significant_unstable_points"] == 0

    tilted = project_path_subspace(potential, gradient, tilted_path(40.0), [1.0, 1.0], axis, dividing_surface=1.0)
    info = tilted.provenance
    assert not info["bath_stable"] and info["significant_unstable_points"] == 9
    assert info["unstable_characters"] == ["tangent_misaligned"]
    record = info["significant_unstable"][0]
    assert record["unstable_vibrational_modes"] == 1
    assert record["tangent_to_lowest_mode_degrees"] == pytest.approx(40.0, abs=1e-3)
    expected = -4e-4 * np.sin(np.radians(40.0)) ** 2 + 1e-4 * np.cos(np.radians(40.0)) ** 2
    assert record["curvature"] == pytest.approx(expected, rel=1e-5)
    # The dropped zero-point term is what the old model silently omitted.
    assert info["dropped_zero_point_bound"] == pytest.approx(0.5 * np.sqrt(-expected), rel=1e-5)
    gates = validity_gates(info)
    assert not gates["valid"] and "tangent_misaligned" in gates["failures"][0]
    assert validity_gates(info, policy={"unstable_bath": "allow"})["valid"]
    with pytest.raises(ValueError, match="unstable transverse Hessian"):
        project_path_subspace(potential, gradient, tilted_path(40.0), [1.0, 1.0], axis, dividing_surface=1.0,
                              unstable_bath="error")


def test_a_ridge_is_distinguished_from_a_tilted_tangent():
    potential = lambda x: -2e-4 * x[0] ** 2 - 2e-4 * x[1] ** 2 + 5e-5 * x[2] ** 2
    gradient = lambda x: np.array([-4e-4 * x[0], -4e-4 * x[1], 1e-4 * x[2]])
    path = np.column_stack([np.linspace(-1.0, 1.0, 9), np.zeros((9, 2))])
    model = project_path_subspace(potential, gradient, path, [1.0, 1.0, 1.0], np.linspace(0.0, 2.0, 9),
                                  dividing_surface=1.0)
    assert model.provenance["unstable_characters"] == ["ridge"]
    assert model.provenance["significant_unstable"][0]["unstable_vibrational_modes"] == 2


def test_soft_modes_are_counted_but_do_not_fail_the_bath():
    """A free relative rotation at the asymptote has zero curvature of either sign."""
    soft = -(1e-4) ** 2  # 22i cm^-1, below the 50 cm^-1 soft-mode limit
    potential = lambda x: 1e-3 / np.cosh(x[0]) ** 2 + 0.5 * soft * x[1] ** 2
    gradient = lambda x: np.array([-2e-3 * np.tanh(x[0]) / np.cosh(x[0]) ** 2, soft * x[1]])
    path = np.column_stack([np.linspace(-4.0, 4.0, 17), np.zeros(17)])
    model = project_path_subspace(potential, gradient, path, [1.0, 1.0], np.linspace(0.0, 8.0, 17))
    info = model.provenance
    assert info["bath_stable"] and info["soft_complement_points"] == 17
    assert info["unstable_complement_points"] == 17  # the raw count is still recorded
    assert validity_gates(info)["valid"]
    strict = project_path_subspace(potential, gradient, path, [1.0, 1.0], np.linspace(0.0, 8.0, 17),
                                   soft_mode_frequency=0.0)
    assert not strict.provenance["bath_stable"]


def test_hessian_step_follows_the_lightest_atom():
    from pyrpmd.analysis.subspace_pipeline import _hessian_step

    masses = np.repeat([1837.0, 25500.0], 3)
    step = _hessian_step({}, masses)
    # 5e-3 bohr for hydrogen, where the old fixed step of 1e-3 was 2e-5 bohr.
    assert step / np.sqrt(1837.0) == pytest.approx(5e-3)
    assert _hessian_step({"hessian_displacement": 1e-2}, masses) == pytest.approx(2 * step)


# --- 3. grid exits ----------------------------------------------------------

RPMD = dict(windows=12, walkers=8, burn_in=40, sweeps=60, trajectories=64, seed=1)


def barrier(ramp=0.0):
    axis = np.linspace(-8.0, 8.0, 161)
    values = 1.0 / np.cosh(axis) ** 2 + ramp * np.maximum(0.0, -(axis + 5.0))
    return SubspaceModel((axis,), values, [1.0], 0.0)


def test_committed_exits_through_a_flat_end_are_not_contamination():
    result = ring_polymer_rate(barrier(), 8.0, 4, **RPMD)
    exits = result.diagnostics["grid_exit"]
    # Most recrossing trajectories leave the grid: the all-stage fraction is large ...
    assert exits["recrossing_exited_fraction"] > 0.5
    assert result.diagnostics["spline_outside_fraction"] > 0.01
    # ... but no estimator sample saw the continuation and nobody came back.
    assert exits["ti_sample_outside_fraction"] == 0.0 and exits["surface_sample_outside_fraction"] == 0.0
    assert exits["recrossing_returned_fraction"] == 0.0 and exits["recrossing_uphill_exit_fraction"] == 0.0
    assert exits["estimator_outside_fraction"] == 0.0
    stages = exits["stages"]
    assert set(stages) == {"initialization", "hmc_burn_in", "hmc_production", "surface_sampling", "recrossing"}
    assert stages["hmc_production"]["outside"] == 0 and stages["recrossing"]["outside"] > 0
    assert sum(s["evaluations"] for s in stages.values()) > 0
    assert validity_gates({}, result.diagnostics)["valid"]


def test_a_grid_that_ends_on_a_slope_invalidates_the_correction(tmp_path):
    model = barrier(ramp=0.15)  # V_S still rises towards the reactant edge
    correction = subspace_correction(model, 8.0, 4, rpmd_options=RPMD)
    exits = correction.rpmd.diagnostics["grid_exit"]
    assert exits["recrossing_uphill_exit_fraction"] > 0.1
    assert exits["estimator_outside_fraction"] > 0.1
    assert not correction.valid and "outside its grid" in correction.invalid_reasons[0]
    record = correction.to_dict()
    # The ratio exists but is not a usable factor; the component results are kept.
    assert record["factor"] is None and record["log_factor"] is None and record["valid"] is False
    assert record["diagnostic_factor"] > 0 and record["quantum"]["rate"] > 0 and record["rpmd"]["rate"] > 0
    assert record["validity"]["gates"][0]["gate"] == "grid"
    with pytest.raises(ValueError, match="validity gates"):
        apply_subspace_correction(1.0, correction)
    restored = SubspaceCorrection.read(correction.write(tmp_path / "C_S_1d.json"))
    assert not restored.valid and restored.factor == pytest.approx(correction.factor)
    with pytest.raises(ValueError, match="validity gates"):
        nested_subspace_sensitivity([restored, restored])

    model.provenance["validity_policy"] = {"outside_grid": "allow"}
    overridden = subspace_correction(model, 8.0, 4, rpmd_options=RPMD)
    assert overridden.valid and overridden.validity["overrides"] == ["grid"]
    assert overridden.to_dict()["factor"] == pytest.approx(correction.factor)


def test_segmented_recrossing_reproduces_one_propagation():
    from pyrpmd.analysis.subspace_native import make_ring

    model = barrier()
    for backend in ("python", "auto"):
        ring = make_ring(model, 8.0, 4, backend, 1)
        rng = np.random.default_rng(3)
        modes = ring.modes(rng.normal(scale=0.3, size=(6, 4, 1)))
        momenta = ring.sample_momenta(rng, 6, False)
        force = -ring.modes(ring.potential(modes)[1])
        propagator = ring.free_propagator(0.05)
        whole, side = ring.propagate(modes, momenta, force, 0.05, propagator, False, 30, surface=0.0)
        state, parts = (modes, momenta, force), []
        for steps in (7, 11, 12):
            trial, part = ring.propagate(*state, 0.05, propagator, False, steps, surface=0.0)
            state = trial[:3]
            parts.append(part if not parts else part[1:])
        np.testing.assert_array_equal(np.concatenate(parts), side)
        for a, b in zip(state, whole[:3]):
            np.testing.assert_array_equal(a, b)


def test_a_ratio_that_cannot_be_formed_keeps_its_components():
    model = barrier()
    quantum = SimpleNamespace(rate=0.0, method="dense", diagnostics={}, to_dict=lambda: {"rate": 0.0})
    rpmd = ring_polymer_rate(model, 8.0, 4, **RPMD)
    correction = _assemble_correction(model, 8.0, 4, quantum, rpmd)
    assert not correction.valid and np.isnan(correction.factor)
    assert "must be positive" in correction.invalid_reasons[0]
    record = correction.to_dict()
    assert record["factor"] is None and record["diagnostic_factor"] is None
    assert record["rpmd"]["rate"] == pytest.approx(rpmd.rate) and record["quantum"]["rate"] == 0.0


# --- pipeline ---------------------------------------------------------------

def pipeline_spec(monkeypatch, directory, *, termination, model=None, plane=None, dimension=1,
                  plane_quantum=None, **spec):
    """A pipeline whose path and models are supplied; the gates are real."""
    from pyrpmd.analysis import subspace_pipeline as pipeline

    built = {}

    def path_model(run, path, options, path_source="centroid"):
        base = model if model is not None else barrier()
        provenance = dict(base.provenance, **pipeline._model_provenance(options["path_info"], options["validity"]))
        built[1] = SubspaceModel(base.axes, base.values, base.masses, base.dividing_surface, base.reactant,
                                 base.reactant_reference, provenance)
        return built[1], {}, {"dimension": 1}

    def plane_model(run, path, direction, options):
        built[2] = plane
        return plane, dict(plane_quantum or {}), {"dimension": 2}, None

    run = SimpleNamespace(root=directory, beta=8.0, nbeads=4, xi_star=1.0, provider=SimpleNamespace())
    mean = SimpleNamespace(curvature_width=lambda *args, **kwargs: {"maximum": 0.0})
    info = {"source": "centroid", "reactant_termination": termination, "product_termination": CONVERGED,
            "asymptotes_converged": bool(termination["converged"])}
    monkeypatch.setattr(pipeline, "pipeline_identity", lambda *args: "fixture-identity")
    monkeypatch.setattr(pipeline, "load_rate_run", lambda *args: run)
    monkeypatch.setattr(pipeline, "build_path", lambda *args: (mean, None, np.zeros((4, 2)), info))
    monkeypatch.setattr(pipeline, "_static_curvature_width",
                        lambda *args: {"maximum": 0.0, "candidate_direction": [0.0, 1.0]})
    monkeypatch.setattr(pipeline, "path_model", path_model)
    monkeypatch.setattr(pipeline, "plane_model", plane_model)
    monkeypatch.setattr(pipeline, "correct_rate_run",
                        lambda root, correction, **kwargs: {"factor": correction.factor})
    return dict(source={"rate_run": str(directory)}, subspace={"dimension": dimension}, rpmd=RPMD,
                output={"directory": str(directory)}, **spec), built


CONVERGED = {"status": "asymptote_converged", "converged": True, "steps": 40, "arc_length": 400.0,
             "tail_energy_range": 5e-7, "asymptote_tolerance": 1e-6}
EXHAUSTED = {"status": "max_steps_exhausted", "converged": False, "steps": 600, "arc_length": 93.0,
             "tail_energy_range": 6.3e-4, "asymptote_tolerance": 1e-6}


def test_pipeline_applies_a_factor_when_every_gate_passes(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED)
    report = run_subspace_pipeline(spec)
    assert report["status"] == "complete" and report["validity"]["valid"]
    assert report["applied_dimension"] == 1 and report["corrected_rate"]["factor"] > 0
    assert report["corrections"]["1"]["factor"] == pytest.approx(report["corrected_rate"]["factor"])
    assert [g["gate"] for g in report["validity"]["dimensions"]["1"]["gates"]] == ["path", "grid"]
    assert report["options"]["path"]["max_steps"] is None


def test_pipeline_refuses_a_path_that_exhausted_its_budget(tmp_path, monkeypatch):
    """Issue #182: 600/600 steps with a tail 600 times above the tolerance."""
    import json
    from pyrpmd.analysis import run_subspace_pipeline
    from pyrpmd.analysis import subspace_pipeline as pipeline
    from pyrpmd.analysis.subspace_correction import CORRECTED_SCHEMA

    stale = tmp_path / "rate_quantum_corrected.json"
    stale.write_text(json.dumps({"schema": CORRECTED_SCHEMA}))
    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=EXHAUSTED)
    monkeypatch.setattr(pipeline, "subspace_correction", lambda **case: pytest.fail("solved a rejected model"))
    report = run_subspace_pipeline(spec)
    assert report["status"] == "invalid_model" and not report["validity"]["valid"]
    assert report["applied_dimension"] is None and report["corrected_rate"] is None
    assert report["dimensions_computed"] == [] and report["validity"]["rejected_before_solving"] == [1]
    assert "max_steps_exhausted" in report["validity"]["failures"]["1"][0]
    assert report["path"]["reactant_termination"]["tail_energy_range"] == pytest.approx(6.3e-4)
    assert not stale.exists() and not (tmp_path / "C_S_1d.json").exists()
    saved = json.loads((tmp_path / "subspace_pipeline.json").read_text())
    assert saved["status"] == "invalid_model"


def test_diagnose_mode_keeps_component_rates_without_a_usable_factor(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=EXHAUSTED, validity={"on_invalid": "diagnose"})
    report = run_subspace_pipeline(spec)
    record = report["corrections"]["1"]
    assert report["status"] == "invalid_model" and report["corrected_rate"] is None
    assert record["factor"] is None and record["diagnostic_factor"] > 0
    assert record["quantum"]["rate"] > 0 and record["rpmd"]["rate"] > 0

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=EXHAUSTED,
                            validity={"unconverged_path": "allow"})
    allowed = run_subspace_pipeline(spec)
    assert allowed["status"] == "complete" and allowed["applied_dimension"] == 1
    assert allowed["validity"]["dimensions"]["1"]["overrides"] == ["path"]


def test_unstable_bath_escalates_the_automatic_dimension(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline

    potential, gradient = saddle(4e-4, 1e-4)
    unstable = project_path_subspace(potential, gradient, tilted_path(40.0), [1.0, 1.0],
                                     np.linspace(0.0, 2.0, 9), dividing_surface=1.0)
    y = np.linspace(-2.0, 2.0, 9)
    base = barrier()
    plane = SubspaceModel((base.axes[0], y), base.values[:, None] + 0.5 * y[None, :] ** 2, [1.0, 1.0], 0.0)
    spec, built = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, model=unstable, plane=plane,
                                dimension="auto")
    report = run_subspace_pipeline(spec)
    assert report["dimension_escalation"]["to"] == 2
    assert "unstable transverse Hessian" in report["dimension_escalation"]["reason"]
    assert report["validity"]["rejected_before_solving"] == [1] and report["dimensions_computed"] == [2]
    assert report["applied_dimension"] == 2 and report["applied_dimension_reason"] == "only valid dimension"
    assert report["models"]["1"]["validity"]["gates"][1]["detail"]["unstable_characters"] == ["tangent_misaligned"]


# --- input -------------------------------------------------------------------

def options(**table):
    return rate_subspace_options({"task": "rate", "rate": {"subspace_correction": table}})


def test_input_accepts_the_validity_table_and_an_unbounded_path():
    resolved = options(validity={"unstable_bath": "allow", "max_outside_fraction": 0.05, "on_invalid": "diagnose"},
                       path={"asymptote_patience": 20}, hessian_displacement=0.01)
    assert resolved["validity"] == {"unstable_bath": "allow", "max_outside_fraction": 0.05,
                                    "on_invalid": "diagnose"}
    assert "max_steps" not in resolved["path"] and resolved["path"]["asymptote_patience"] == 20


@pytest.mark.parametrize("table, message", [
    ({"validity": {"unstable_bath": "ignore"}}, "reject or allow"),
    ({"validity": {"on_invalid": "continue"}}, "stop or diagnose"),
    ({"validity": {"max_outside_fraction": 1.5}}, r"\[0, 1\]"),
    ({"validity": {"strict": True}}, "Unknown"),
    ({"path": {"asymptote_patience": 0}}, "asymptote_patience"),
    ({"hessian_displacement": 0.0}, "hessian_displacement"),
])
def test_input_rejects_bad_validity_options(table, message):
    with pytest.raises(InputError, match=message):
        options(**table)
