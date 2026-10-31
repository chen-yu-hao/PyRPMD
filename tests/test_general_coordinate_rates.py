"""Independent nonlinear metric, coordinate-invariance and workflow checks."""

import copy
import json

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from pyrpmd.analysis.recrossing import summarize_recrossing
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import ATOMIC_TIME_S
from pyrpmd.cv import Cartesian, CVResult
from pyrpmd.system import System
from pyrpmd.workflows import site_rate as site


def configuration():
    return {
        "task": "site_rate", "beta": 3., "beads": 2, "dt_au": .03,
        "seed": 37, "native": False,
        "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"loop_backend": "python"},
        "sampling": {"equilibration_steps": 200, "production_steps": 2000,
                     "sample_interval": 2, "andersen_interval": 10, "replicas": 1},
        "cvs": [{"kind": "bond_difference", "name": "s", "atoms": [0, 1, 2]}],
        "umbrella": {"cvs": ["s"], "centers": [-.8, -.4, 0., .4], "stiffness": 8.},
        "analysis": {"bins": 10, "ranges": [[-1.25, 1.25]], "bootstrap": 0},
        "site_rate": {"cv": "s", "basin": [-.75, 0.], "dividing_surface": 0., "direction": 1},
        "recrossing": {"parent_replicas": 2, "parent_samples": 4,
                       "equilibration_steps": 5, "parent_interval_steps": 3,
                       "child_pairs": 2, "child_steps": 40,
                       "plateau_fraction": .5, "bootstrap": 16, "constraint_tolerance": 1e-11},
        "output": {"checkpoint_steps": 200},
    }


def system():
    return System(("H", "H", "H"), [2., 2., 2.], [[-1., 0., 0.], [0., 0., 0.], [.6, .8, 0.]])


def uniform_histogram(edges):
    edges = np.asarray(edges)
    return {"edges": [edges], "density": np.ones(len(edges) - 1),
            "probability_mass": np.diff(edges), "mask": np.zeros(len(edges) - 1, bool),
            "connected": True, "measure": "cv_density"}


def test_variable_metric_ellipse_matches_independent_surface_quadrature():
    # s=x^2/a^2+y^2/b^2: delta(s-1) dxdy = a*b/2 dtheta.
    # Its conditional density is uniform in theta; constrained parents have
    # density proportional to sqrt(z(theta)). Sample that density by inversion.
    theta = np.linspace(0., 2 * np.pi, 40001)
    z = 4 * (np.cos(theta)**2 / 4 + np.sin(theta)**2 / .25)
    cdf = cumulative_trapezoid(np.sqrt(z), theta, initial=0.)
    cdf /= cdf[-1]
    parent_theta = np.interp((np.arange(50000) + .5) / 50000, cdf, theta)
    parent_z = 4 * (np.cos(parent_theta)**2 / 4 + np.sin(parent_theta)**2 / .25)
    result = site.estimate_site_qtst(uniform_histogram(np.linspace(0., 1.2, 121)),
                                   3., None, [.1, 1.], 1., constrained_mass_metrics=parent_z)
    expected_speed = trapezoid(np.sqrt(z), theta) / (2 * np.pi * np.sqrt(6 * np.pi))
    assert result["flux_prefactor_au"] == pytest.approx(expected_speed, rel=2e-8)
    assert result["k_qtst_au"] == pytest.approx(expected_speed / .9, rel=2e-8)
    assert result["mass_metric"] is None
    # The tempting arithmetic mean in the constrained ensemble is biased.
    wrong_speed = np.sqrt(parent_z.mean() / (6 * np.pi))
    assert abs(wrong_speed / expected_speed - 1) > .1


def test_nonlinear_monotone_coordinate_change_preserves_qtst_in_fine_bin_limit():
    edges = np.linspace(0., 1.2, 2401)
    base = uniform_histogram(edges)
    metrics = np.array([1., 3., 7., 2., 4.])
    old = site.estimate_site_qtst(base, 3., None, [.1, 1.], 1., constrained_mass_metrics=metrics)
    # eta=f(s)=s+s^2, f'(s*)=3. Exact transformed bin masses include Jacobian.
    mapped = edges + edges**2
    transformed = {**base, "edges": [mapped], "density": np.diff(edges) / np.diff(mapped)}
    new = site.estimate_site_qtst(transformed, 3., None, [.11, 2.], 2., constrained_mass_metrics=9 * metrics)
    assert new["k_qtst_au"] == pytest.approx(old["k_qtst_au"], rel=1e-7)


@pytest.mark.parametrize("metrics", [[], [0.], [-1.], [np.inf], [np.nan], [[1., 2.]]])
def test_qtst_rejects_singular_or_invalid_parent_metrics(metrics):
    with pytest.raises(ValueError, match="constrained mass metrics"):
        site.estimate_site_qtst(uniform_histogram(np.linspace(-1., 1., 21)), 3., None,
                                [-.5, .5], .5, constrained_mass_metrics=metrics)


def test_bond_difference_and_distance_are_validated_with_general_metric():
    config, atoms = configuration(), system()
    metadata = site.validate_site_rate(config, atoms)
    assert metadata["mass_metric"] is None
    assert metadata["flux_prefactor_au"] is None
    assert metadata["metric_estimator"] == "constrained_parent_inverse_sqrt"
    cv = site._setup(config, atoms)[0]
    z1 = site._mass_metric(cv, atoms.geometry, atoms.masses)
    q = atoms.geometry.copy()
    q[2] = [-.6, .8, 0.]
    assert site._mass_metric(cv, q, atoms.masses) != pytest.approx(z1)
    config["cvs"][0] = {"name": "s", "kind": "distance", "atoms": [0, 1]}
    config["site_rate"].update(basin=[.5, 1.], dividing_surface=1.)
    assert site.validate_site_rate(config, atoms)["reference_mass_metric"] == pytest.approx(1.)
    atoms.geometry[1] = atoms.geometry[0]
    with pytest.raises(ValueError, match="coincident"):
        site.validate_site_rate(config, atoms)


class CentroidPlugin:
    estimator = "centroid"

    def value_and_jacobian(self, q):
        x = q[:, 0, 0].mean()
        jac = np.zeros((1,) + q.shape)
        jac[0, :, 0, 0] = 3 * x**2 / len(q)
        return CVResult(np.array([x**3]), jac)


def test_custom_centroid_contract_and_singular_gradient_guards():
    plugin = CentroidPlugin()
    q = np.array([[.5, 0., 0.]])
    site._validate_centroid_contract(plugin, q, 4)
    assert site._mass_metric(plugin, q, [2.]) == pytest.approx(.75**2 / 2)
    with pytest.raises(ValueError, match="singular"):
        site._mass_metric(plugin, np.zeros_like(q), [2.])

    class BeadAveragePlugin(CentroidPlugin):
        def value_and_jacobian(self, q):
            jac = np.zeros((1,) + q.shape)
            jac[0, :, 0, 0] = 3 * q[:, 0, 0]**2 / len(q)
            return CVResult(np.array([np.mean(q[:, 0, 0]**3)]), jac)

    with pytest.raises(ValueError, match="centroid value/Jacobian contract"):
        site._validate_centroid_contract(BeadAveragePlugin(), q, 4)


def test_class_level_cv_override_does_not_claim_a_constant_metric(monkeypatch):
    config, atoms = configuration(), system()
    config["cvs"] = [{"kind": "cartesian", "name": "s", "atom": 0, "axis": 0}]

    def cubic(self, q):
        x = q[self.atom, self.axis]
        gradient = np.zeros_like(q)
        gradient[self.atom, self.axis] = 3 * x**2
        return float(x**3), gradient

    monkeypatch.setattr(Cartesian, "centroid_value_gradient", cubic)
    metadata = site.validate_site_rate(config, atoms)
    assert metadata["mass_metric"] is None
    assert metadata["reference_mass_metric"] == pytest.approx(4.5)


def test_nonlinear_parent_and_weighted_child_restart(tmp_path):
    config, atoms = configuration(), system()
    task = {"config": config, "system": atoms, "chain": 0, "output": str(tmp_path),
            "initial_q": np.repeat(atoms.geometry[None], config["beads"], axis=0)}
    parent = site._parent_task(task)
    cv = site._setup(config, atoms)[0]
    for q in parent["snapshots"]:
        assert abs(site._centroid_value_gradient(cv, q.mean(axis=0))[0]) < 1e-11
    replay = site._parent_task(task)
    np.testing.assert_array_equal(parent["snapshots"], replay["snapshots"])
    child_task = {**task, "q": parent["snapshots"][0], "parent": 0, "pair": 0}
    child = site._child_task(child_task)
    with load_checkpoint(child["checkpoint"]) as saved:
        expected = abs(float(saved["velocity"])) / np.sqrt(child["mass_metric"])
        np.testing.assert_allclose(child["numerator"], saved["numerator"].sum(axis=0) / np.sqrt(child["mass_metric"]))
    assert child["denominator"] == pytest.approx(expected, rel=1e-15)
    replay_child = site._child_task(child_task)
    np.testing.assert_array_equal(child["numerator"], replay_child["numerator"])


def test_joint_metric_and_transmission_bootstrap_preserves_exact_cancellation(tmp_path):
    config = configuration()
    data = uniform_histogram(np.linspace(-1., 1., 41))
    pmf = site.estimate_site_qtst(data, 3., 1., [-.6, .6], .6)
    density_ratio = pmf["surface_density"] / pmf["basin_probability"]
    pmf.update(preparation={"direction": 1}, warnings=[], _bootstrap_qtst_s=[],
               metric_estimator="constrained_parent_inverse_sqrt",
               _bootstrap_density_ratio=[density_ratio] * 12,
               _density_uncertainty_available=True,
               pmf_uncertainty={"available": False, "reasons": []})
    children = []
    for chain in range(2):
        for parent, weight in enumerate([.3, 1.4, .6, 2.1]):
            children.append({"chain": chain, "parent": parent, "pair": 0,
                             "surface": .6, "direction": 1,
                             "numerator": np.full(40, .5 * weight), "denominator": 1.,
                             "mass_metric": 1 / weight**2, "configuration_weight": weight,
                             "endpoint_energy_error_kbt": 0.})
    result = site._aggregate(children, pmf, config, tmp_path)
    expected = .5 * density_ratio / (np.sqrt(6 * np.pi) * ATOMIC_TIME_S)
    assert result["rate_estimate_s"] == pytest.approx(expected, rel=1e-14)
    assert result["rate_uncertainty"]["available"]
    assert result["rate_standard_error_s"] < expected * 1e-14
    assert result["pmf"]["pmf_uncertainty"]["standard_error_s"] > expected * .05
    with load_checkpoint(tmp_path / "recrossing.npz") as saved:
        np.testing.assert_allclose(saved["bootstrap_rate_s"], expected, rtol=1e-14)


def test_metric_records_reject_inconsistent_parents_and_extra_observables():
    children = [{"chain": 0, "parent": 0, "pair": 0, "numerator": np.ones(4),
                 "denominator": 1., "mass_metric": 4., "configuration_weight": .5}]
    corrupted = copy.deepcopy(children)
    corrupted[0]["configuration_weight"] = 1.
    with pytest.raises(ValueError, match="inverse-metric"):
        site._metric_reweighted_children(corrupted)
    with pytest.raises(ValueError, match="match the recrossing parents"):
        summarize_recrossing(children, child_steps=4, plateau_fraction=.5,
                              plateau_tolerance=.1, bootstrap=2, rng=np.random.default_rng(1),
                              extra_parent_observables={(1, 0): .5})


def test_complete_general_coordinate_workflow_and_resume(tmp_path):
    config, atoms = configuration(), system()
    result = site.run_site_rate(config, atoms, tmp_path)
    assert result["pmf"]["ready"]
    assert result["pmf"]["surface_metric_status"] == "estimated"
    assert result["k_qtst_s"] > 0
    assert np.isfinite(result["rate_estimate_s"])
    assert result["parent_configurations"] == 8
    assert result["child_pairs"] == 16
    assert not result["converged"]
    assert not result["rate_uncertainty"]["available"]  # Bootstrap deliberately disabled.
    with load_checkpoint(tmp_path / "site_qtst.npz") as saved:
        assert saved["parent_inverse_sqrt_metric"].shape == (8,)
    replay = site.run_site_rate(config, atoms, tmp_path)
    assert replay["rate_estimate_s"] == result["rate_estimate_s"]
    assert replay["k_qtst_s"] == result["k_qtst_s"]
    json.loads((tmp_path / "summary.json").read_text())
