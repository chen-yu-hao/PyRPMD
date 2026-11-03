"""Real ML batches through public PIMD orchestration and checkpoint resume."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mace.modules")
pytest.importorskip("vesin_torch")

from examples.potentials._random_mace import make_random_mace
from pyrpmd import run
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import JobSpec
from pyrpmd.potentials.batched_ml import MACEBatchedPotential
from pyrpmd.workflows.materials import _read_stream


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("batch_workflow_model") / "random.model"
    torch.save(make_random_mace(), path)
    return path


def _job(path, periodic):
    return {
        "task": "pimd", "temperature": "100 K", "beads": 8, "seed": 137,
        "system": {
            "symbols": ["O", "D", "T"],
            "geometry": [[0., 0., 0.], [0.95, 0., 0.], [-0.3, 0.9, 0.]],
            "geometry_unit": "angstrom", "cell": (np.eye(3) * 7).tolist(),
            "cell_unit": "angstrom", "pbc": [periodic] * 3,
        },
        "potential": {
            "provider": "mace-batched", "model": str(path),
            "device": "cpu", "dtype": "float64", "virial": True,
        },
        "dynamics": {"loop_backend": "python", "native": False,
                     "timestep": "0.1 fs", "integrator": "cayley-bcb"},
        "sampling": {
            "replicas": 3, "equilibration_steps": 0, "production_steps": 6,
            "sample_interval": 1, "thermostat": "pile-l",
            "gamma_centroid": "0.01 fs^-1",
        },
        "output": {"checkpoint_steps": 1},
        "parallel": {"trajectory_batch_size": 1, "bead_workers": 1},
    }


def _assert_streams_equal(config, system, expected, actual):
    for replica in range(3):
        a = _read_stream(config, system, expected, replica)
        b = _read_stream(config, system, actual, replica)
        for field in ("q", "p"):
            np.testing.assert_allclose(getattr(b["state"], field), getattr(a["state"], field),
                                       rtol=1e-10, atol=2e-12)
        for field in ("step", "time"):
            assert getattr(b["state"], field) == getattr(a["state"], field)
        assert b["rng"] == a["rng"]
        np.testing.assert_allclose(b["observables"], a["observables"],
                                   rtol=1e-10, atol=2e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_public_pimd_batches_isotopes_and_resumes_regrouped(model_path, periodic,
                                                          tmp_path, monkeypatch):
    config = _job(model_path, periodic)
    resolved, system = JobSpec.from_mapping(config).resolve()
    assert system.masses[1] < system.masses[2] < system.masses[0]
    reference = tmp_path / "reference"
    run(config, output=reference)

    frames = []
    real_evaluate = MACEBatchedPotential._evaluate_tensor

    def counted(provider, q):
        frames.append(len(q))
        return real_evaluate(provider, q)

    batched = copy.deepcopy(config)
    batched["parallel"]["trajectory_batch_size"] = 2
    with monkeypatch.context() as patch:
        patch.setattr(MACEBatchedPotential, "_evaluate_tensor", counted)
        run(batched, output=tmp_path / "batch")
    assert 16 in frames and set(frames) <= {8, 16}
    _assert_streams_equal(resolved, system, reference, tmp_path / "batch")

    real_step = Dynamics.sample_step

    def interrupt(dynamics, state, *args, **kwargs):
        result = real_step(dynamics, state, *args, **kwargs)
        if state.step == 2:
            raise RuntimeError("interrupt scalar ML trajectory")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "sample_step", interrupt)
        with pytest.raises(RuntimeError, match="interrupt scalar ML trajectory"):
            run(config, output=tmp_path / "resumed")
    batched["parallel"]["trajectory_batch_size"] = 3
    frames.clear()
    with monkeypatch.context() as patch:
        patch.setattr(MACEBatchedPotential, "_evaluate_tensor", counted)
        run(batched, output=tmp_path / "resumed", resume=True)
    assert 24 in frames and set(frames) <= {8, 16, 24}
    _assert_streams_equal(resolved, system, reference, tmp_path / "resumed")
