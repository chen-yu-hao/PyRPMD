"""Path-source selection and normal-cut diagnostics for subspace pipelines."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
import pyrpmd.analysis.subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_model import project_path_subspace
from pyrpmd.instanton import crossover_temperature
from pyrpmd.potentials.base import PESResult


class _MeanPath:
    def __init__(self):
        self.xi = np.arange(7.0)
        self.orientation_spread = np.zeros(7)
        self.bead_covariance = np.array([
            np.diag([0.0, 0.25, 0.0, 0.0, 0.0, 0.0]) for _ in self.xi
        ])
        x = np.linspace(-1.0, 1.0, len(self.xi))
        y = 0.2 * x * x
        self._centroid = np.zeros((len(x), 2, 3))
        self._centroid[:, 0, 0] = x
        self._centroid[:, 1, 0] = x + 1.0
        self._centroid[:, :, 1] = y[:, None]

    @property
    def mean_centroid(self):
        return self._centroid

    def locked(self, factor=2.0):
        return np.ones(len(self.xi), dtype=bool)

    def path(self, mask):
        return self._centroid[np.asarray(mask, dtype=bool)]


class _Run:
    root = "."
    beta = 1.0
    nbeads = 8
    xi_star = 0.0
    provider = None
    system = SimpleNamespace(masses=np.array([1.0, 1.0]), geometry=np.zeros((2, 3)))
    cartesian_masses = np.repeat(system.masses, 3)

    def energy(self, x):
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    def gradient(self, x):
        return 2.0 * np.asarray(x, dtype=float).ravel()


class _EckartPotential:
    """Small analytic barrier used for a real below-crossover pipeline run."""

    def __init__(self, height=0.01, inverse_width=0.5):
        self.height = float(height)
        self.inverse_width = float(inverse_width)

    def evaluate(self, q):
        x = np.asarray(q, dtype=float)[:, 0, 0]
        t = np.tanh(self.inverse_width * x)
        energy = self.height * (1.0 - t * t)
        forces = np.zeros_like(q, dtype=float)
        forces[:, 0, 0] = (self.inverse_width * (1.0 - t * t)
                           * (2.0 * self.height * t))
        return PESResult(energy, forces)


class _InstantonRun:
    def __init__(self, temperature, nbeads=32):
        self.provider = _EckartPotential()
        self.system = SimpleNamespace(
            masses=np.array([1822.0]),
            geometry=np.zeros((1, 3)),
        )
        self.beta = 1.0 / (KB_AU * float(temperature))
        self.nbeads = int(nbeads)
        self.cartesian_masses = np.repeat(self.system.masses, 3)


def test_build_path_sources_record_selection_and_bead_rms(monkeypatch):
    monkeypatch.setattr(pipeline, "mean_path_from_rate_run", lambda root: _MeanPath())
    monkeypatch.setattr(
        pipeline,
        "assemble_reaction_path",
        lambda core, *args, **kwargs: (np.asarray(core), {"assembled": True}),
    )
    run = _Run()

    _, _, centroid, centroid_info = pipeline.build_path(run, {"source": "centroid", "smoothing": None})
    _, _, weighted, weighted_info = pipeline.build_path(run, {"source": "bead_weighted", "smoothing": None})
    _, _, least, least_info = pipeline.build_path(
        run, {"source": "least_action", "smoothing": None, "least_action_points": 5}
    )

    assert centroid_info["source"] == "centroid"
    assert weighted_info["source"] == "bead_weighted"
    assert weighted_info["bead_weighted"]["bead_rms_range"][1] == 0.5
    # Only the transverse component of the y covariance contributes away
    # from the central point, where the tangent is exactly along x.
    displacement = (weighted - centroid).reshape(len(weighted), -1)
    tangent, _ = pipeline._path_tangent_and_curvature(centroid, run.cartesian_masses)
    expected_variance = 0.25 * (1.0 - tangent[:, 1] ** 2)
    np.testing.assert_allclose(np.sum(displacement ** 2, axis=1), expected_variance)
    np.testing.assert_allclose(np.sum(displacement * tangent, axis=1), 0.0, atol=1e-14)
    assert least_info["source"] == "least_action"
    assert len(least_info["least_action"]["scan"]) == 5
    assert 0.0 <= least_info["least_action"]["selected_alpha"] <= 1.0


def test_instanton_falls_back_at_or_above_crossover(monkeypatch):
    import pyrpmd.instanton as instanton

    monkeypatch.setattr(instanton, "crossover_temperature", lambda *args, **kwargs: 100.0)
    def fail_if_called(*args, **kwargs):
        raise AssertionError("optimizer must not run above crossover")
    monkeypatch.setattr(instanton, "optimize_instanton", fail_if_called)
    core = _MeanPath().mean_centroid
    result, info = pipeline._instanton_core(_Run(), core, {})
    assert np.array_equal(result, core)
    assert info["fallback_reason"] == "temperature_at_or_above_crossover"


def test_instanton_records_converged_orbit_below_crossover(monkeypatch):
    """A below-crossover orbit reports optimizer and physical-mode diagnostics."""
    run = _Run()
    core = _MeanPath().mean_centroid

    # _Run.beta=1 is a high-temperature atomic-unit value, so make the
    # synthetic crossover much higher than the resulting temperature.  This
    # exercises the optimizer branch without requiring a PES or a long solve.
    monkeypatch.setattr(pipeline, "crossover_temperature", lambda *args, **kwargs: 1.0e9)
    orbit = np.zeros((8, 2, 3), dtype=float)
    orbit[:, 0, 0] = [-1.0, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, -1.0]
    orbit[:, 1, 0] = orbit[:, 0, 0] + 1.0
    fake_result = SimpleNamespace(
        converged=True,
        gradient_max=1.0e-10,
        negative_modes=1,
        path_bohr=orbit,
        message="converged",
    )
    calls = {}

    def fake_optimizer(*args, **kwargs):
        calls["temperature"] = args[2]
        calls["seed_shape"] = np.asarray(args[3]).shape
        calls["max_iterations"] = kwargs["max_iterations"]
        return fake_result

    monkeypatch.setattr(pipeline, "optimize_instanton", fake_optimizer)
    folded, info = pipeline._instanton_core(run, core, {"instanton_max_iterations": 17})

    assert calls["seed_shape"] == (8, 2, 3)
    assert calls["max_iterations"] == 17
    assert calls["temperature"] < 1.0e9
    assert folded.shape == (4, 2, 3)
    assert info["converged"] is True
    assert info["gradient_max"] == 1.0e-10
    assert info["negative_modes"] == 1
    assert info["physical_negative_modes"] == 1
    assert info["bead_number"] == 8
    assert info["folded_beads"] == 4
    assert info["fallback"] is None
    assert info["physical_converged"] is False


def test_instanton_runs_real_below_crossover_orbit_and_records_diagnostics():
    """The pipeline must exercise a real optimizer below ``T_c``.

    The analytic Eckart barrier keeps this integration test independent of the
    restored BKMP2 data while still checking the production optimizer branch,
    its physical negative-mode count, and the full/folded bead convention.
    """
    potential = _EckartPotential()
    masses = np.array([1822.0])
    transition_state = np.zeros((1, 3))
    crossover = crossover_temperature(potential, transition_state, masses)
    run = _InstantonRun(0.5 * crossover, nbeads=32)
    core = np.zeros((7, 1, 3), dtype=float)
    core[:, 0, 0] = np.linspace(-2.0, 2.0, len(core))

    folded, info = pipeline._instanton_core(
        run,
        core,
        {"instanton_max_iterations": 200},
    )

    assert info["temperature_kelvin"] < info["crossover_temperature_kelvin"]
    assert info["converged"] is True
    assert info["gradient_max"] < 1e-8
    assert info["negative_modes"] == 1
    assert info["physical_negative_modes"] == 1
    assert info["bead_number"] == 32
    assert info["folded_beads"] == 16
    assert info["fallback"] is None
    assert info["physical_converged"] is False
    assert folded.shape == (16, 1, 3)
    assert np.ptp(folded[:, 0, 0]) > 1.0


def test_high_temperature_sources_keep_comparable_path_diagnostics(monkeypatch):
    """At the 600 K validation point, sources remain comparable and explicit.

    ``least_action`` includes the centroid member (alpha=0), so its selected
    WKB action cannot be worse than the centroid baseline.  ``bead_weighted``
    is intentionally distinct, but must still produce finite diagnostics.  The
    expensive full curvature-family Delta matrix remains a campaign check.
    """
    monkeypatch.setattr(
        pipeline,
        "assemble_reaction_path",
        lambda core, *args, **kwargs: (np.asarray(core), {"assembled": True}),
    )
    run = _Run()
    run.beta = 1.0 / (KB_AU * 600.0)
    mean = _MeanPath()
    monkeypatch.setattr(pipeline, "mean_path_from_rate_run", lambda root: mean)

    _, _, centroid, centroid_info = pipeline.build_path(
        run, {"source": "centroid", "smoothing": None}
    )
    _, _, weighted, weighted_info = pipeline.build_path(
        run, {"source": "bead_weighted", "smoothing": None}
    )
    _, _, least, least_info = pipeline.build_path(
        run, {"source": "least_action", "smoothing": None, "least_action_points": 9}
    )

    centroid_action = pipeline._wkb_action(centroid, run)[0]
    weighted_action = pipeline._wkb_action(weighted, run)[0]
    assert centroid_info["source"] == "centroid"
    assert weighted_info["source"] == "bead_weighted"
    assert least_info["source"] == "least_action"
    assert np.isfinite(weighted_action)
    assert least_info["least_action"]["selected_action_hbar"] <= centroid_action + 1e-12
    assert not np.allclose(weighted, centroid)
    assert not np.allclose(least, centroid)


def test_path_projection_records_unstable_normal_hessian():
    def potential(x):
        return 0.5 * x[0] ** 2 - 0.5 * x[1] ** 2 + 0.25 * (x[0] - 0.5) ** 4

    def gradient(x):
        return np.array([x[0] + (x[0] - 0.5) ** 3, -x[1]])

    path = np.column_stack([np.linspace(-1.0, 1.0, 9), np.zeros(9)])
    axis = np.linspace(0.0, 2.0, 17)
    model = project_path_subspace(potential, gradient, path, [1.0, 1.0], axis,
                                  dividing_surface=1.0)
    assert model.provenance["unstable_complement_points"] > 0
    assert not model.provenance["normal_hessian_positive"]
    assert model.provenance["unstable_complement_indices"]
    # The complete potential along the displaced path is retained in V_S;
    # only the non-positive normal zero-point contribution is dropped.
    assert np.isfinite(model.values).all()


def test_bead_weighted_diagonalizes_transverse_covariance():
    mean = _MeanPath()
    mean._centroid[:, :, 1] = 0.0
    tangent = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    normal = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    other_normal = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mixed = (10.0 * tangent + normal) / np.sqrt(101.0)
    covariance = 100.0 * np.outer(mixed, mixed) + 4.0 * np.outer(other_normal, other_normal)
    mean.bead_covariance[:] = covariance
    shifted, info = pipeline._bead_weighted_core(mean, mean.locked(), [1.0, 1.0])
    displacement = (shifted - mean.mean_centroid).reshape(len(mean.xi), -1)
    np.testing.assert_allclose(displacement, np.tile(2.0 * other_normal, (len(mean.xi), 1)), atol=1e-12)
    np.testing.assert_allclose(info["principal_variance_range"], [4.0, 4.0])


def test_least_action_diagnostics_describe_smoothed_selected_path(monkeypatch):
    mean = _MeanPath()
    monkeypatch.setattr(pipeline, "mean_path_from_rate_run", lambda root: mean)
    monkeypatch.setattr(pipeline, "assemble_reaction_path",
                        lambda core, *args, **kwargs: (np.asarray(core), {}))
    run = _Run()
    _, _, centroid, _ = pipeline.build_path(run, {"source": "centroid", "smoothing": 1.5})
    _, _, selected, info = pipeline.build_path(run, {"source": "least_action", "smoothing": 1.5})
    scan = info["least_action"]
    np.testing.assert_allclose(scan["scan"][0]["action_hbar"], pipeline._wkb_action(centroid, run)[0])
    np.testing.assert_allclose(scan["selected_action_hbar"], pipeline._wkb_action(selected, run)[0])
    assert scan["selected_action_hbar"] <= scan["scan"][0]["action_hbar"]


def test_instanton_records_actual_even_seed_beads(monkeypatch):
    monkeypatch.setattr(pipeline, "crossover_temperature", lambda *args: 1.0e9)
    def optimizer(provider, masses, temperature, seed, **kwargs):
        return SimpleNamespace(converged=True, gradient_max=0.0, negative_modes=1,
                               path_bohr=seed, message="converged")
    monkeypatch.setattr(pipeline, "optimize_instanton", optimizer)
    for requested, actual in [(1, 8), (9, 10)]:
        folded, info = pipeline._instanton_core(
            _Run(), _MeanPath().mean_centroid, {"instanton_beads": requested})
        assert info["requested_bead_number"] == requested
        assert info["bead_number"] == actual
        assert info["folded_beads"] == len(folded) == actual // 2


@pytest.mark.parametrize("zero_point", ["harmonic", "rph", "dvr"])
def test_path_source_provenance_survives_sct_projection(zero_point):
    radius, omega = 10.0, 0.2
    theta = np.linspace(-1.0, 1.0, 101)
    path = radius * np.column_stack([np.cos(theta), np.sin(theta)])
    def potential(x):
        return 0.5 * omega**2 * (np.linalg.norm(x) - radius)**2
    def gradient(x):
        r = np.linalg.norm(x)
        return omega**2 * (r - radius) * x / r
    model = project_path_subspace(potential, gradient, path, [1.0, 1.0],
                                  np.linspace(0.0, 20.0, 81), dividing_surface=10.0,
                                  curvature="small", zero_point=zero_point, dvr_half_width=8.0,
                                  provenance={"path_source": "least_action"})
    assert model.provenance["path_source"] == "least_action"
    assert model.provenance["curvature"] == "small"
    assert model.provenance["zero_point"] == zero_point
    assert model.provenance["tau_length"] < model.provenance["sigma_length"]
    assert model.provenance["normal_hessian_positive"]
    assert model.provenance["unstable_complement_indices"] == []
