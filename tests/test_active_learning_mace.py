"""Optional real-MACE CPU training, force units and model-state round trips."""
import json

import numpy as np
import pytest

from pyrpmd.active_learning_mace import MACEModelCodec, MACETrainer
from pyrpmd.potentials.base import PESResult, check_potential
from pyrpmd.potentials.lifecycle import LabelledDataset


@pytest.mark.parametrize("option", [
    {"device": "cuda"}, {"dtype": "float32"}, {"epochs": True},
    {"epochs": 0}, {"learning_rate": 0.0}, {"force_weight": float("nan")},
    {"seeds": (1, 1)}, {"seeds": (1,)}, {"seeds": (-1, 3)},
    {"fine_tune": "yes"}, {"atomic_energies": [float("inf")]},
    {"atomic_energies": [1, 2]}, {"max_ell": 0}, {"num_threads": 0},
])
def test_mace_reference_options_fail_explicitly_without_optional_imports(option):
    with pytest.raises(ValueError):
        MACETrainer(["H", "H"], **option)


def bond_dataset():
    q = np.zeros((7, 2, 3))
    q[:, 1, 0] = np.linspace(1.2, 2.1, len(q))
    displacement = q[:, 1, 0] - 1.5
    energy = .02 * displacement**2
    forces = np.zeros_like(q)
    forces[:, 0, 0] = .04 * displacement
    forces[:, 1, 0] = -.04 * displacement
    return LabelledDataset(q, energy, forces,
        reference_identity={"provider": "analytic_harmonic_bond", "k": .04, "r0": 1.5})


@pytest.fixture(scope="module")
def trained():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    trainer = MACETrainer(["H", "H"], epochs=24, learning_rate=.003,
        channels=4, max_ell=1, radial_channels=8, seeds=(37, 53))
    data = bond_dataset()
    models = trainer.initial_models()
    previous = [MACEModelCodec().encode(model) for model in models]
    trained_models = trainer(data, models)
    return torch, trainer, data, models, previous, trained_models


def test_real_mace_training_reduces_independent_analytic_error(trained):
    _, _, data, models, _, trained_models = trained
    def loss(model):
        result = model.evaluate(data.positions)
        return np.mean((result.energy-data.energy)**2) + np.mean((result.forces-data.forces)**2)
    before = np.mean([loss(model) for model in models])
    after = np.mean([loss(model) for model in trained_models])
    assert after < before * .5
    assert all(model.provenance["dataset_sha256"] == data.sha256 for model in trained_models)
    for model in trained_models:
        result = check_potential(model, data.positions[[1, 4]], epsilon=2e-5)
        assert result["max_absolute_force_error"] < 1e-8
        assert model.evaluate(data.positions).forces.dtype == np.float64


def test_mace_training_leaves_previous_models_unchanged(trained):
    _, _, _, models, previous, _ = trained
    for model, saved in zip(models, previous):
        now = MACEModelCodec().encode(model)
        assert set(now) == set(saved)
        for key in saved:
            np.testing.assert_array_equal(now[key], saved[key])


def test_mace_codec_exact_roundtrip_and_strict_payload(trained):
    _, _, data, _, _, models = trained
    codec = MACEModelCodec()
    encoded = codec.encode(models[0])
    restored = codec.decode(encoded)
    np.testing.assert_array_equal(restored.evaluate(data.positions).energy, models[0].evaluate(data.positions).energy)
    np.testing.assert_array_equal(restored.evaluate(data.positions).forces, models[0].evaluate(data.positions).forces)
    for key, value in codec.encode(restored).items():
        np.testing.assert_array_equal(value, encoded[key])
    changed = dict(encoded)
    first_float = next(key for key, value in encoded.items() if value.dtype.kind == "f")
    changed[first_float] = changed[first_float].astype(np.float32)
    with pytest.raises(ValueError, match="dtype"):
        codec.decode(changed)
    changed = dict(encoded)
    metadata = json.loads(str(changed["metadata"]))
    metadata["versions"]["mace-torch"] = "different"
    changed["metadata"] = np.asarray(json.dumps(metadata))
    with pytest.raises(ValueError, match="versions"):
        codec.decode(changed)


def test_mace_replay_is_deterministic_and_restores_global_rng_and_dtype(trained):
    torch, _, data, models, _, _ = trained
    trainer = MACETrainer(["H", "H"], epochs=2, learning_rate=.003,
        channels=4, max_ell=1, radial_channels=8, seeds=(37, 53))
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    first = trainer(data, models)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert dtype == torch.get_default_dtype() and threads == torch.get_num_threads()
    for left, right in zip(numpy_rng, np.random.get_state()):
        np.testing.assert_array_equal(left, right)
    replay = trainer(data, models)
    for left, right in zip(first, replay):
        a, b = MACEModelCodec().encode(left), MACEModelCodec().encode(right)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_mace_finetuning_and_identity_are_explicit(trained):
    _, trainer, data, _, _, models = trained
    trainer2 = MACETrainer(["H", "H"], epochs=1, learning_rate=.003,
        channels=4, max_ell=1, radial_channels=8, seeds=(37, 53))
    next_models = trainer2(data, models)
    assert trainer2.identity != trainer.identity
    assert not np.array_equal(next_models[0].evaluate(data.positions).energy, models[0].evaluate(data.positions).energy)
    incompatible = MACETrainer(["H", "H"], channels=8, epochs=1, seeds=(37, 53))
    with pytest.raises(ValueError, match="architectures"):
        incompatible(data, models)
    failed = LabelledDataset(data.positions[:1], [np.nan], np.full((1, 2, 3), np.nan),
        status=[1], messages=["reference failed"], reference_identity=data.reference_identity)
    with pytest.raises(ValueError, match="successful"):
        trainer2(failed, models)
    with pytest.raises(ValueError, match="atom count"):
        models[0].evaluate(np.zeros((2, 3, 3)))


def test_mace_loop_resume_before_training_preserves_all_hashes(trained, tmp_path):
    from pyrpmd.active_learning import ActiveLearningLoop, SamplingBatch

    _, _, data, models, _, _ = trained
    trainer = MACETrainer(["H", "H"], epochs=1, learning_rate=.003,
        channels=4, max_ell=1, radial_channels=8, seeds=(37, 53))

    class Reference:
        def evaluate(self, q):
            vector = q[:, 1] - q[:, 0]
            distance = np.linalg.norm(vector, axis=1)
            energy = .02 * (distance - 1.5)**2
            f1 = -.04 * (distance - 1.5)[:, None] * vector / distance[:, None]
            return PESResult(energy, np.stack((-f1, f1), axis=1))

    def sampler(committee, rng, state):
        return SamplingBatch(data.positions, {"round": 1})

    options = dict(reference=Reference(), reference_identity=data.reference_identity,
        sampler=sampler, sampler_identity={"sampler": "fixed_analytic_bond_grid.v1"},
        trainer=trainer, trainer_identity=trainer.identity, model_codec=MACEModelCodec(),
        force_threshold=1e-12, max_rounds=1, max_labels=2,
        labels_per_round=2, label_batch_size=1, seed=91)
    full = ActiveLearningLoop(tmp_path / "full", models, **options)
    expected = full.run()
    interrupted = ActiveLearningLoop(tmp_path / "resumed", models, **options)
    interrupted.run(max_steps=3)
    assert interrupted.phase == "train"
    interrupted.close()
    resumed = ActiveLearningLoop(tmp_path / "resumed", **options)
    actual = resumed.run()
    assert actual == expected
    assert actual["rounds_completed"] == 1
    assert len(actual["model_versions"]) == 2
    assert actual["model_versions"][0]["sha256"] != actual["model_versions"][1]["sha256"]
    assert resumed.dataset.sha256 == full.dataset.sha256
