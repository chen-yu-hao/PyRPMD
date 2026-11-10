"""The dimension-2 pipeline must use the requested Cartesian channel model."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_channels import SubspaceChannels
from pyrpmd.analysis.subspace_correction import SubspaceCorrection
from pyrpmd.analysis.subspace_model import model_from_function
from pyrpmd.system import System


@pytest.fixture
def channel_pipeline(tmp_path, monkeypatch):
    run = SimpleNamespace(
        root=tmp_path, beta=100., nbeads=8, xi_star=1., reaction={},
        config={"task": "rate", "beta": 100., "beads": 8,
                "potential": {"provider": "free"}},
        system=System(("H",), np.array([1837.]), np.zeros((1, 3))),
    )
    path = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    mean = SimpleNamespace(curvature_width=lambda *a, **k: {"maximum": 1.})
    monkeypatch.setattr(pipeline, "load_rate_run", lambda _: run)
    monkeypatch.setattr(pipeline, "build_path", lambda *a: (mean, None, path, {}))
    monkeypatch.setattr(pipeline, "_static_curvature_width",
                        lambda *a: {"maximum": 1., "candidate_direction": [0., 1., 0.]})
    potential = lambda points: .1 / np.cosh(points[..., 0])**2 + .5 * points[..., 1]**2
    axes = [np.linspace(-3., 3., 9), np.linspace(-2., 2., 9)]
    model = model_from_function(potential, axes, masses=[1., 2.],
                                dividing_surface=0., reactant_reference=-1.5)
    fine = model_from_function(potential, [np.linspace(a[0], a[-1], 17) for a in axes],
                               masses=model.masses, dividing_surface=0., reactant_reference=-1.5)
    quantum = {"normalization": "slice", "dense_limit": 80, "truncation_energy": .8}
    monkeypatch.setattr(pipeline, "plane_model", lambda *a: (model, quantum, {}, fine))
    monkeypatch.setattr(pipeline, "correct_rate_run", lambda *a, **k: {})
    calls = []

    def correction(self, beta, nbeads, **kwargs):
        calls.append((self, beta, nbeads, kwargs))
        return SubspaceCorrection(beta, 2, 2., .1, 2., 1., nbeads, model.describe())

    monkeypatch.setattr(SubspaceChannels, "correction", correction)
    monkeypatch.setattr(pipeline, "subspace_correction",
                        lambda *a, **k: pytest.fail("requested channels bypassed the channel solver"))
    spec = {"source": {"rate_run": str(tmp_path)},
            "subspace": {"dimension": 2, "channels": 4},
            "rpmd": {"nbeads": 16, "walkers": 6},
            "output": {"directory": str(tmp_path / "correction"), "reuse": True}}
    return spec, run, model, fine, quantum, calls


def test_dimension_two_pipeline_forwards_channels_and_same_hamiltonian_options(channel_pipeline):
    spec, run, model, fine, quantum, calls = channel_pipeline
    report = pipeline.run_subspace_pipeline(spec)
    assert len(calls) == 1
    projection, beta, beads, kwargs = calls[0]
    assert projection.source_model is model
    assert projection.n_channels == 4
    assert (beta, beads) == (run.beta, 16)
    assert kwargs["rpmd_model"] is fine
    assert kwargs["quantum_options"] is quantum
    assert kwargs["rpmd_options"]["walkers"] == 6
    assert "nbeads" not in kwargs["rpmd_options"]
    assert report["models"]["2"]["channels"]["channels"] == 4
    assert report["dimensions_computed"] == [2]
    assert report["physical_converged"] is False
    saved = SubspaceCorrection.read(run.root / "correction" / "C_S_2d.json")
    assert saved.diagnostics["pipeline_identity"] == report["pipeline_identity"]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 10])
def test_invalid_channel_count_is_rejected_before_rate_solver(channel_pipeline, count):
    spec, run, model, fine, quantum, calls = channel_pipeline
    spec["subspace"]["channels"] = count
    with pytest.raises(ValueError, match="n_channels must be an integer"):
        pipeline.run_subspace_pipeline(spec)
    assert calls == []
    assert not (run.root / "correction" / "C_S_2d.json").exists()


def test_changed_channel_count_cannot_reuse_same_temperature_factor(channel_pipeline):
    spec, run, model, fine, quantum, calls = channel_pipeline
    first = pipeline.run_subspace_pipeline(spec)
    second = pipeline.run_subspace_pipeline(spec)
    assert len(calls) == 1
    assert second["pipeline_identity"] == first["pipeline_identity"]
    spec["subspace"]["channels"] = 6
    third = pipeline.run_subspace_pipeline(spec)
    assert len(calls) == 2
    assert calls[-1][0].n_channels == 6
    assert third["pipeline_identity"] != first["pipeline_identity"]
