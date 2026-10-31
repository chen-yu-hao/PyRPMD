"""Dimensional and checkpoint compatibility gates for unified FES workflows."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from pyrpmd.analysis.fes import reconstruct_fes
from pyrpmd.analysis.fes_result import FESResult
from pyrpmd.system import System
from pyrpmd.workflows.common import (
    _fingerprint, _legacy_fingerprint, fingerprint_matches, validate_cv_dimension,
)
from pyrpmd.workflows.fes import run_fes, reanalyze_fes


def system():
    return System(("H",), np.array([2.0]), np.array([[0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("method,maximum", [("umbrella", 3), ("metadynamics", 3), ("opes", 2)])
def test_common_dimension_policy(method, maximum):
    for dimension in range(1, maximum + 1):
        cv = SimpleNamespace(dimension=dimension)
        assert validate_cv_dimension(cv, method) is cv
    for dimension in (0, maximum + 1):
        with pytest.raises(ValueError, match=f"{method} FES supports 1–{maximum} CV dimensions; got {dimension}"):
            validate_cv_dimension(SimpleNamespace(dimension=dimension), method)


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_fingerprint_aliases_resume_exact_legacy_science(method):
    historical = {"task": "fes" if method == "umbrella" else method,
                  "beta": 2.0, "beads": 4, "seed": 91,
                  method: {"cvs": ["x"]},
                  "analysis": {"measure": "cv_density"},
                  "sampling": {"production_steps": 100}}
    canonical = {**historical, "task": "fes", "fes": {"method": method}}
    stream = {"replica": 3}
    old = _legacy_fingerprint(historical, system(), stream)
    new = _fingerprint(canonical, system(), stream)
    assert old != new
    assert new == _fingerprint(historical, system(), stream)
    assert fingerprint_matches(old, canonical, system(), stream)
    assert fingerprint_matches(new, historical, system(), stream)
    for key, value in (("seed", 92), ("beta", 2.1), ("beads", 8)):
        changed = {**canonical, key: value}
        assert not fingerprint_matches(old, changed, system(), stream)
        assert not fingerprint_matches(new, changed, system(), stream)
    assert not fingerprint_matches(old, canonical, system(), {"replica": 4})
    changed = copy.deepcopy(canonical)
    changed["fes"]["method"] = "opes" if method != "opes" else "umbrella"
    assert _fingerprint(changed, system(), stream) != new


def test_non_fes_checkpoint_identity_is_unchanged():
    config = {"task": "rate", "beta": 2.0, "beads": 4, "seed": 2}
    assert _fingerprint(config, system(), {}) == _legacy_fingerprint(config, system(), {})


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_direct_workflows_without_task_still_fingerprint_selected_method(method):
    direct = {"beta": 2.0, "beads": 2, method: {"cvs": ["x"]}}
    resolved = {**direct, "task": "fes", "fes": {"method": method}}
    stream = {"replica": 1}
    assert _fingerprint(direct, system(), stream) == _fingerprint(resolved, system(), stream)
    assert fingerprint_matches(_legacy_fingerprint(direct, system(), stream),
                               direct, system(), stream)


def test_unresolved_method_options_cannot_disappear_from_fingerprint():
    with pytest.raises(ValueError, match="resolved configuration"):
        _fingerprint({"task": "fes", "fes": {"method": "umbrella",
                                              "umbrella": {"stiffness": 2.0}}},
                     system(), {})


def test_three_dimensional_adaptive_coverage_counts_all_cells():
    from pyrpmd.workflows.adaptive_umbrella import adaptive_options, _coverage_diagnostic

    config = {"task": "fes", "adaptive_umbrella": {
        "enabled": False, "coverage_bins": [2, 2, 2],
        "coverage_ranges": [[-1.0, 1.0]] * 3,
        "min_bin_effective_samples": 1.0, "target_coverage": 1.0,
    }}
    options = adaptive_options(config)
    corners = np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    report, _, effective, supported = _coverage_diagnostic(
        corners, np.full(8, 1 / 8), np.ones(8), (None,) * 3, options)
    assert report["target_met"] and report["fraction"] == 1.0
    np.testing.assert_array_equal(effective, np.ones((2, 2, 2)))
    assert np.all(supported)
    config["adaptive_umbrella"]["coverage_bins"] = [101, 101, 101]
    with pytest.raises(ValueError, match="one million cells"):
        adaptive_options(config)


def test_three_dimensional_mbar_recovers_integrated_harmonic_density():
    """Check against independent Gaussian cell integrals, including bin volume."""
    beta, physical_k, umbrella_k = 2.0, 0.5, 1.0
    axes = [-0.7, 0.7]
    centers = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(81731)
    scale = np.sqrt(1 / (beta * (physical_k + umbrella_k)))
    samples = [rng.normal(center * umbrella_k / (physical_k + umbrella_k), scale,
                          size=(10000, 3)) for center in centers]
    result = reconstruct_fes(samples, centers, np.eye(3) * umbrella_k, beta,
                             bins=[5, 5, 5], ranges=[[-1.0, 1.0]] * 3,
                             bootstrap=0, min_count=20)
    assert result["connected"] and not np.any(result["mask"])
    widths = [np.diff(edge) for edge in result["edges"]]
    probabilities = [np.diff(ndtr(edge * np.sqrt(beta * physical_k))) for edge in result["edges"]]
    density = np.einsum("i,j,k->ijk", *(p / w for p, w in zip(probabilities, widths)))
    reference = -np.log(density) / beta
    reference -= reference.min()
    difference = result["free_energy"] - reference
    # Offsets are arbitrary; compare their common mean gauge.
    difference -= difference.mean()
    assert np.sqrt(np.mean(difference**2)) < 0.04
    assert np.max(np.abs(difference)) < 0.12


def test_three_dimensional_workflow_roundtrip_reanalysis_and_resume(tmp_path):
    config = {
        "task": "fes", "fes": {"method": "umbrella"},
        "beta": 2.0, "beads": 2, "dt_au": 0.02, "seed": 71, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y", "z")],
        "umbrella": {"grid": [{"start": -0.05, "stop": 0.05, "count": 2}] * 3,
                     "stiffness": [0.4, 0.4, 0.4]},
        "sampling": {"equilibration_steps": 8, "production_steps": 80,
                     "sample_interval": 2, "andersen_interval": 7},
        "analysis": {"bins": [3, 4, 5], "bootstrap": 3, "seed": 9},
        "output": {"checkpoint_steps": 13},
    }
    sampled = run_fes(config, system(), tmp_path, return_data=True)
    surface = FESResult.read(tmp_path)
    assert surface.free_energy.shape == (3, 4, 5)
    assert surface.grid.shape == (3, 4, 5, 3)
    assert surface.method == "umbrella" and surface.uncertainty_kind == "block_bootstrap"
    assert sampled["summary"]["dimension"] == 3
    assert len(sampled["streams"]) == 8
    for name in ("fes.npz", "fes.csv", "fes.json", "summary.json"):
        assert (tmp_path / name).is_file()
    for other in (reanalyze_fes(config, system(), tmp_path, return_data=True),
                  run_fes(config, system(), tmp_path, return_data=True)):
        np.testing.assert_array_equal(other["data"]["free_energy"], sampled["data"]["free_energy"])
        np.testing.assert_array_equal(other["data"]["standard_error"], sampled["data"]["standard_error"])


def test_three_dimensional_preflight_preserves_every_axis_without_a_potential(monkeypatch):
    """The merged early-validation route must keep the third umbrella axis."""
    from pathlib import Path
    from pyrpmd.input import JobSpec
    from pyrpmd.workflows.fes import validate_fes_request

    def unavailable(*args, **kwargs):
        raise AssertionError("static FES validation must not construct a potential")

    monkeypatch.setattr("pyrpmd.potentials.make_potential", unavailable)
    example = Path(__file__).resolve().parents[1] / "examples/fes/harmonic3d.inp"
    config, sys = JobSpec.from_file(example).resolve()
    cv, _, centers, stiffness = validate_fes_request(config, sys)
    assert cv.dimension == 3 and centers.shape == (8, 3) and stiffness.shape == (3, 3)
    for changes, message in (
        ({"ranges": [[-2, 2], [-2, 2], [2, -2]]}, "increasing"),
        ({"ranges": [[-2, 2]] * 4}, "pair per CV"),
        ({"bins": [4, 4]}, "per CV"),
        ({"bins": [4, 4, [0.0, 1.0, 1.0]]}, "edges"),
    ):
        invalid = copy.deepcopy(config)
        invalid["analysis"].update(changes)
        with pytest.raises(ValueError, match=message):
            validate_fes_request(invalid, sys)
