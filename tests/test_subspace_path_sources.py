"""Path-source selection and normal-cut diagnostics for subspace pipelines."""

from types import SimpleNamespace

import numpy as np

import pyrpmd.analysis.subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_model import project_path_subspace


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
    assert weighted_info["bead_weighted"]["bead_rms_range"] == [0.5, 0.5]
    # The largest bead mode is the y direction, so the weighted path leaves
    # the centroid by exactly one bead RMS in mass-weighted coordinates.
    displacement = (weighted - centroid).reshape(len(weighted), -1)
    assert np.max(np.linalg.norm(displacement, axis=1)) == 0.5
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
