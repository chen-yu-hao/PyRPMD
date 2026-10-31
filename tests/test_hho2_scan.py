"""HHO2 scan mechanics on an explicitly synthetic, nonreactive harmonic model.

These tests validate input identity, units, resuming and MBAR plumbing. They
are not a chemical PES validation or an H + HO2 PMF calculation.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU, AU_ENERGY_KCAL_MOL, KB_AU
from pyrpmd.potentials import FreePotential, check_potential
from pyrpmd.potentials.python import import_object


MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/hho2/scan.py")
plan = import_object(MODULE + ":plan")
scan = __import__(plan.__module__, fromlist=["unused"])


@pytest.fixture
def synthetic_model(tmp_path):
    source = tmp_path / "synthetic_harmonic.py"
    source.write_text(
        '"""Synthetic mechanics test only; this is not a reactive HHOO PES."""\n'
        'from pathlib import Path\n'
        'from pyrpmd.potentials import HarmonicPotential\n'
        'def build(data_file, center):\n'
        '    return HarmonicPotential(k=float(Path(data_file).read_text()), center=center)\n'
    )
    coefficients = tmp_path / "synthetic_k.txt"
    coefficients.write_text("0.005\n")
    raw = {
        "atomic_numbers": [1, 1, 8, 8], "multiplicity": 3,
        "reference": "SYNTHETIC harmonic plumbing test; NOT a reactive H+HO2 model",
        "artifacts": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in (source, coefficients)],
        "potential": {"provider": "python", "factory": source.name + ":build",
                      "params": {"data_file": coefficients.name,
                                 "center": scan.seed_geometry("reactant").tolist()}},
    }
    path = tmp_path / "synthetic_manifest.json"
    path.write_text(json.dumps(raw))
    return path, raw, scan.load_model(path)


def _tiny_plan():
    settings = plan("smoke", beads=2, dt_fs=0.02)
    geometry = scan.seed_geometry("reactant") / ANGSTROM_TO_AU
    settings.update(
        centers_angstrom=[[float(np.linalg.norm(geometry[0]-geometry[1])), 1.33]],
        seed_families=["reactant"], repeats_per_family=1, streams=1,
        equilibration_steps=4, production_steps=16, sample_interval=2,
        andersen_interval=3, checkpoint_steps=5, bins=[3, 3], bootstrap=0,
        min_count=1, min_weighted_bin_ess=1.,
        planned_samples_per_stream=8, planned_total_samples=8,
        dense_reduced_potential_bytes_lower_bound=64, aggregate_bead_steps=40,
    )
    return settings


def _run(tmp_path, model):
    settings = _tiny_plan()
    output = tmp_path / "synthetic_run"
    scan.write_json(output / "plan.json", settings)
    scan.write_json(output / "model.json", model)
    path = Path(scan._stream((settings, model, str(output), 0, "reactant", 0)))
    return settings, output, path


def _payload(path):
    with load_checkpoint(path, require_integrity=True) as saved:
        return {key: saved[key].copy() for key in saved.files}


def test_plan_units_atom_order_and_seed_com():
    settings = plan()
    assert settings["atomic_order"] == ["H0_incoming", "H1_in_HO2", "O0", "O1"]
    assert settings["streams"] == 11 * 10 * 5
    assert settings["planned_samples_per_stream"] == 2000
    assert settings["planned_total_samples"] == 1_100_000
    assert settings["dense_reduced_potential_bytes_lower_bound"] == 4_840_000_000
    assert plan("extended")["dense_reduced_potential_bytes_lower_bound"] == 233_928_000_000
    assert settings["beta"] == pytest.approx(1 / (KB_AU * 1500))
    assert settings["wall"]["retained_in_target_pmf"]
    for family in scan.SEED_NAMES:
        q = scan.seed_geometry(family)
        np.testing.assert_allclose(np.average(q, axis=0, weights=scan.MASSES), 0, atol=1e-15)
    q = scan.seed_geometry("H2_O2") / ANGSTROM_TO_AU
    assert np.linalg.norm(q[0]-q[1]) == pytest.approx(0.741)
    assert np.linalg.norm(q[2]-q[3]) == pytest.approx(1.208)


def test_common_pair_wall_units_gradient_and_rigid_symmetries():
    potential = scan.ConfinedPotential(FreePotential(), wall_angstrom=5.,
                                      k_kcal_mol_angstrom2=20.)
    q = np.array([[[0, 0, 0], [6., 0, 0], [3., 1., 0], [3., -1., 0]]]) * ANGSTROM_TO_AU
    # Only HH exceeds 5 A, by exactly 1 A: 1/2 * 20 * 1^2 kcal/mol.
    result = potential.evaluate(q)
    assert result.energy[0] * AU_ENERGY_KCAL_MOL == pytest.approx(10.)
    assert check_potential(potential, q)["max_absolute_force_error"] < 1e-10
    np.testing.assert_allclose(result.forces.sum(axis=1), 0, atol=1e-15)
    rotated = q[..., [2, 0, 1]] + [3., -7., 0.2]
    np.testing.assert_allclose(potential.evaluate(rotated).energy, result.energy, atol=1e-15)
    np.testing.assert_allclose(potential.evaluate(q[:, [1, 0, 3, 2]]).energy, result.energy, atol=1e-15)


def test_manifest_pins_real_entry_and_resolves_artifact_parameters(synthetic_model):
    path, raw, model = synthetic_model
    assert model["potential"]["factory"].startswith(str(path.parent))
    assert Path(model["potential"]["params"]["data_file"]).is_absolute()
    assert model["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # Pinning an unrelated data artifact must not stand in for pinning the
    # executable adapter that actually evaluates the model.
    raw["artifacts"] = raw["artifacts"][1:]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="entry file"):
        scan.load_model(path)


def test_manifest_rejects_wrong_atom_order_or_changed_source(synthetic_model):
    path, raw, model = synthetic_model
    altered = copy.deepcopy(raw)
    altered["atomic_numbers"] = [1, 8, 8, 1]
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="HHOO"):
        scan.load_model(path)
    path.write_text(json.dumps(raw))
    (path.parent / raw["artifacts"][0]["path"]).write_text("# changed\n")
    with pytest.raises(ValueError, match="checksum"):
        scan.load_model(path)


@pytest.mark.skipif(
    not importlib.util.find_spec("matplotlib"),
    reason="optional matplotlib is required for the rendered scan figure",
)
def test_actual_synthetic_stream_analysis_and_masked_channel_fractions(tmp_path, synthetic_model):
    _, _, model = synthetic_model
    settings, output, path = _run(tmp_path, model)
    saved = _payload(path)
    assert saved["q"].shape == (2, 4, 3)
    assert saved["samples"].shape == (8, 2)
    assert int(saved["elapsed"]) == 20
    summary = scan.analyze(output)
    assert summary["samples"] == 8
    assert summary["overlap_connected"]
    assert summary["physical_converged"] is False
    assert "SYNTHETIC" in summary["model_reference"]
    assert summary["supported_bins"] == 1
    assert not summary["projected_connections"]
    counts = summary["per_stream_channel_diagnostics"][0]["channel_counts"]
    assert sum(counts.values()) == 8
    assert summary["per_stream_channel_diagnostics"][0]["initial_family"] == "reactant"
    with np.load(output / "fes.npz", allow_pickle=False) as result:
        mask = result["mask"]
        assert np.isnan(result["free_energy_kcal_mol"][mask]).all()
        assert np.isnan(result["channel_fractions"][:, mask]).all()
        np.testing.assert_allclose(result["channel_fractions"][:, ~mask].sum(0), 1)
        assert "reactant_H0+H1O2" in result["channel_names"]
        assert "H+HO2" in result["basin_channel_names"]
    assert (output / "pmf.png").stat().st_size > 0


def test_interrupted_synthetic_stream_resumes_bitwise(tmp_path, synthetic_model, monkeypatch):
    _, _, model = synthetic_model
    settings, output, path = _run(tmp_path / "complete", model)
    uninterrupted = _payload(path)
    interrupted = tmp_path / "interrupted"
    task = (settings, model, str(interrupted), 0, "reactant", 0)
    original = scan.Dynamics.step

    def stop_at_eleven(self, state, bias=None):
        if state.step == 11:
            raise RuntimeError("synthetic injected interruption")
        return original(self, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(scan.Dynamics, "step", stop_at_eleven)
        with pytest.raises(RuntimeError, match="injected interruption"):
            scan._stream(task)
    partial = scan._stream_path(interrupted, 0, "reactant", 0)
    assert int(_payload(partial)["elapsed"]) == 10
    resumed = _payload(Path(scan._stream(task)))
    for key in ("q", "p", "rng", "time", "samples", "centroids", "physical_energies",
                "wall_energies", "ring_radii"):
        np.testing.assert_array_equal(resumed[key], uninterrupted[key])


def test_analysis_rejects_mixed_identity_and_checkpoint_clock(tmp_path, synthetic_model):
    _, _, model = synthetic_model
    settings, output, path = _run(tmp_path, model)
    saved = _payload(path)
    changed = dict(saved, fingerprint=np.asarray("another-model-or-plan"))
    atomic_savez(path, kind="arrays", **changed)
    with pytest.raises(ValueError, match="identity changed"):
        scan.analyze(output)
    changed = dict(saved, elapsed=np.asarray(19))
    atomic_savez(path, kind="arrays", **changed)
    with pytest.raises(ValueError, match="clock"):
        scan.analyze(output)
    # A correctly counted but unplanned filename is not a complete dataset.
    atomic_savez(path, kind="arrays", **saved)
    path.rename(path.with_name("w9999_reactant_000.npz"))
    with pytest.raises(ValueError, match="planned streams"):
        scan.analyze(output)


def test_resume_rejects_changed_sampling_identity(tmp_path, synthetic_model):
    _, _, model = synthetic_model
    settings, output, path = _run(tmp_path, model)
    changed = copy.deepcopy(settings)
    changed["dt_fs"] *= 2
    with pytest.raises(ValueError, match="identity changed"):
        scan._stream((changed, model, str(output), 0, "reactant", 0))


def test_plan_command_cannot_overwrite_a_different_existing_run(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    existing = plan("smoke")
    scan.write_json(path, existing)
    monkeypatch.setattr(scan.sys, "argv", ["scan.py", "plan", "--output", str(tmp_path),
                                          "--profile", "pilot"])
    with pytest.raises(ValueError, match="plan differs"):
        scan.main()
    assert json.loads(path.read_text()) == existing


def test_symmetric_entrance_identities_form_one_chemical_basin():
    channels = ["reactant_H0+H1O2", "exchanged_H1+H0O2", "OH+OH"]
    # Neither H identity alone reaches 80%, while their chemical entrance
    # has 90% support. A lower-energy masked bin must remain unavailable.
    fractions = np.array([[[.45, .49]], [[.45, .49]], [[.10, .02]]])
    original = fractions.copy()
    free = np.array([[2., 0.]])
    supported = np.array([[True, False]])
    basins, grouped = scan._projected_basin_seeds(channels, fractions, free, supported)
    assert basins == {"H+HO2": (0, 0)}
    np.testing.assert_allclose(grouped["H+HO2"], [[.9, .98]])
    assert not any(name in basins for name in scan.ENTRANCE_IDENTITIES)
    np.testing.assert_array_equal(fractions, original)
