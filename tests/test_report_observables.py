"""Numerical/unit/qualification checks for the saved-observable report adapters."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.report_observables import collect_observables
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU


def collect(path, summary, config=None, **kwargs):
    return collect_observables(path, summary, config or {}, prefix="run 1", label="Test", **kwargs)


def plot(result, suffix):
    return next(p for p in result["plots"] if p["id"].endswith("-" + suffix))


def scattering_summary(**changes):
    result = {
        "task": "drpmd", "trajectories": 10,
        "counts": {"product": 3, "nonreactive": 5, "unresolved": 2, "failed": 0},
        "sampling": "thermal_flux", "all_trajectories_resolved": False,
        "channels": {"exchange": {"count": 3, "event_fraction": .3,
                                    "rate_cm3_molecule_s": None,
                                    "observed_lower_cm3_molecule_s": 3e-12,
                                    "possible_upper_cm3_molecule_s": 5e-12,
                                    "parent_cluster_bootstrap95": None}},
    }
    result.update(changes)
    return result


def test_outcomes_preserve_unknowns_and_do_not_promote_bounds(tmp_path):
    result = collect(tmp_path, scattering_summary())
    assert plot(result, "outcomes")["y"] == [3, 5, 2, 0]
    assert result["metrics"]["outcomes"]["fractions"]["product"] == .3
    assert not any(p["id"].endswith("channel-estimates") for p in result["plots"])
    channel = result["metrics"]["channels"]["exchange"]
    assert channel["rate_cm3_molecule_s"] is None
    assert channel["observed_lower_cm3_molecule_s"] == 3e-12
    assert any(f["code"] == "unresolved_scattering" for f in result["findings"])


def test_supported_channel_interval_is_not_mislabelled_as_se(tmp_path):
    summary = scattering_summary(
        counts={"product": 3, "nonreactive": 7, "unresolved": 0, "failed": 0},
        all_trajectories_resolved=True,
        channels={"exchange": {"count": 3, "event_fraction": .3,
                                "rate_cm3_molecule_s": 3e-12,
                                "parent_cluster_bootstrap95": [1e-12, 7e-12]}})
    result = collect(tmp_path, summary)
    assert plot(result, "channel-estimates")["y"] == [3e-12]
    assert np.asarray(plot(result, "channel-estimates")["yerr"]).ravel() == pytest.approx([2e-12, 4e-12])
    assert result["metrics"]["channels"]["exchange"]["parent_cluster_bootstrap95"] == [1e-12, 7e-12]


def test_global_withholding_wins_over_inconsistent_nonnull_rate(tmp_path):
    summary = scattering_summary()
    summary["channels"]["exchange"]["rate_cm3_molecule_s"] = 9e-12
    result = collect(tmp_path, summary)
    assert not any(p["id"].endswith("channel-estimates") for p in result["plots"])


def test_fixed_energy_uses_cross_section_not_thermal_rate(tmp_path):
    summary = scattering_summary(
        counts={"product": 3, "nonreactive": 7, "unresolved": 0, "failed": 0},
        all_trajectories_resolved=True, sampling="fixed_energy",
        channels={"exchange": {"count": 3, "event_fraction": .3,
                                "rate_cm3_molecule_s": None, "cross_section_bohr2": 2.5}})
    result = collect(tmp_path, summary)
    assert plot(result, "channel-estimates")["ylabel"] == "Cross section (bohr²)"
    assert plot(result, "channel-estimates")["y"] == [2.5]
    assert plot(result, "channel-estimates")["yerr"] == [[None], [None]]


@pytest.mark.parametrize("interval_key,method", [
    ("iid_wilson95_bohr2", "iid_wilson"),
    ("parent_cluster_bootstrap95_bohr2", "parent_cluster_percentile_bootstrap"),
])
def test_selected_state_cross_section_intervals_keep_producer_meaning(tmp_path, interval_key, method):
    entry = {"count": 3, "event_fraction": .3, "cross_section_bohr2": 2.5,
             "observed_lower_bohr2": 2.5, "possible_upper_bohr2": 2.5,
             interval_key: [1., 6.], "uncertainty": {"method": method, "confidence_level": .95}}
    summary = scattering_summary(
        counts={"product": 3, "nonreactive": 7, "unresolved": 0, "failed": 0},
        all_trajectories_resolved=True, sampling="fixed_energy", channels={"exchange": entry},
        fixed_energy_scattering={"initialization_mode": "independent", "channels": {"exchange": entry}})
    result = collect(tmp_path, summary)
    figure = plot(result, "channel-estimates")
    assert figure["yerr"] == [[1.5], [3.5]]
    assert "95%" in figure["caption"] and "Wilson" in figure["caption"]
    assert result["metrics"]["channels"]["exchange"]["uncertainty"]["method"] == method
    assert result["metrics"]["fixed_energy_scattering"] == summary["fixed_energy_scattering"]


def test_unresolved_fixed_energy_outcome_bounds_survive_reporting(tmp_path):
    summary = scattering_summary(sampling="fixed_energy", channels={"exchange": {
        "cross_section_bohr2": None, "observed_lower_bohr2": 1.2, "possible_upper_bohr2": 2.8,
        "uncertainty": {"method": "unavailable", "reason": "missing_outcomes"}}})
    result = collect(tmp_path, summary)
    entry = result["metrics"]["channels"]["exchange"]
    assert (entry["observed_lower_bohr2"], entry["possible_upper_bohr2"]) == (1.2, 2.8)
    assert entry["uncertainty"]["reason"] == "missing_outcomes"
    assert not any(p["id"].endswith("channel-estimates") for p in result["plots"])


def test_signed_electronic_values_are_not_clipped(tmp_path):
    summary = scattering_summary(nonadiabatic={
        "state_labels": ["A", "B"], "estimator": "bead_mean_MMST_signed",
        "units": "cm^3 molecule^-1 s^-1", "final_estimates_available": False,
        "channels": {"total_reaction": {"signed_observed_contributions": [-1e-12, 4e-12],
                                          "rate_cm3_molecule_s": None}},
    })
    result = collect(tmp_path, summary)
    assert plot(result, "electronic-contributions")["y"] == [-1e-12, 4e-12]
    assert result["metrics"]["electronic"]["final_estimates_available"] is False


def test_scattering_energy_is_already_per_bead_and_zero_independent(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    path = folder / "trajectory_0000.npz"
    np.savez(folder / "trajectory_0000.checkpoint.npz", state=[0])
    np.savez(path, time_au=[0, FS_TO_AU, 2 * FS_TO_AU], energy_per_bead=[-10, -9.999, -10.002])
    result = collect(tmp_path, scattering_summary(), {"beads": 32})
    line = plot(result, "scattering-energy")["series"][0]
    assert line["x"] == pytest.approx([0, 1, 2])
    assert line["y"] == pytest.approx(np.array([0, .001, -.002]) / KCAL_MOL_TO_AU)
    assert result["metrics"]["scattering_energy"][0]["maximum_saved_frame_energy_error_hartree_per_bead"] == pytest.approx(.002)
    assert result["sources"] == [path]
    assert not any("checkpoint" in s["figure"] for s in result["skipped"])


def materials_file(path, **overrides):
    (path / "samples").mkdir()
    data = dict(times=[0, FS_TO_AU], observables=[[1, 2, 3, 4, 10], [2, 3, 4, 5, 10.8]],
                radii_gyration_squared=[[1, 9], [4, 16]],
                metadata=json.dumps({"energy_unit": "hartree", "time_unit": "atomic_time"}))
    data.update(overrides)
    np.savez(path / "samples" / "materials_0000.npz", **data)


def test_materials_nve_normalizes_total_hamiltonian_by_beads(tmp_path):
    materials_file(tmp_path)
    result = collect(tmp_path, {"task": "diffusion"}, {"beads": 8, "beta": 2}, energy_unit="hartree")
    assert plot(result, "materials-energy")["series"][0]["y"] == pytest.approx([0, .1])
    assert result["metrics"]["nve_energy"][0]["maximum_saved_frame_error_kbt_per_bead"] == pytest.approx(.2)
    assert plot(result, "gyration")["series"][0]["y"] == pytest.approx(np.sqrt([5, 10]) * BOHR_ANGSTROM)


def test_pimd_fluctuations_are_not_treated_as_conservation_failures(tmp_path):
    materials_file(tmp_path)
    result = collect(tmp_path, {"task": "pimd"}, {"beads": 8}, energy_unit="hartree")
    assert "nve_energy" not in result["metrics"]
    assert plot(result, "materials-energy")["series"][0]["y"] == [1, 2]
    assert "not a physical dynamics clock" in plot(result, "materials-energy")["caption"]


def test_electronic_friction_energy_exchange_is_not_an_nve_error(tmp_path):
    materials_file(tmp_path)
    result = collect(tmp_path, {"task": "diffusion", "phase": "centroid_ef_rpmd_production"},
                     {"beads": 8, "beta": 2, "electronic_friction": {"provider": "constant"}}, energy_unit="hartree")
    assert "nve_energy" not in result["metrics"]
    assert "electronic_bath_energy_exchange" in result["metrics"]
    assert "not conserved" in plot(result, "materials-energy")["caption"]
    assert plot(result, "materials-energy")["series"][0]["y"] == pytest.approx([0, .1])


def materials_stream(path):
    from pyrpmd.trajectory import TrajectoryStream

    directory = path / "samples/materials_0000"
    schema = {"sample_steps": {"dtype": "<i8", "shape": []},
              "observables": {"dtype": "<f8", "shape": [5]},
              "radii_gyration_squared": {"dtype": "<f8", "shape": [2]}}
    stream = TrajectoryStream(directory, schema=schema, chunk_frames=1)
    for index, (obs, radius) in enumerate((([1, 2, 3, 4, 10], [1, 9]), ([2, 3, 4, 5, 10.8], [4, 16]))):
        stream.append({"sample_steps": 10 + index, "observables": obs, "radii_gyration_squared": radius})
    descriptor = stream.prepare_commit()
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"format": "PyRPMD.materials-trajectory-stream.v1", "trajectory": descriptor,
        "metadata": {"energy_unit": "hartree", "time_unit": "atomic_time", "equilibration_steps": 10,
                     "production_timestep_au": FS_TO_AU}}))
    return stream, descriptor, manifest


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
def test_default_materials_stream_matches_npz_and_ignores_uncommitted_tail(tmp_path, task):
    legacy, streamed = tmp_path / "legacy", tmp_path / "streamed"
    legacy.mkdir()
    materials_file(legacy)
    stream, descriptor, manifest = materials_stream(streamed)
    # A chunk written after the exported checkpoint is not committed history.
    stream.append({"sample_steps": 12, "observables": [99] * 5, "radii_gyration_squared": [99, 99]})
    extra = stream.prepare_commit()["head"]
    summary, config = {"task": task}, {"beads": 8, "beta": 2}
    before = collect(legacy, summary, config, energy_unit="hartree")
    after = collect(streamed, summary, config, energy_unit="hartree")
    for suffix in ("materials-energy", "gyration"):
        assert plot(after, suffix) == plot(before, suffix)
    assert len(after["source_records"]) == 3  # manifest and two committed chunks
    assert all(extra not in item["path"] for item in after["source_records"])
    assert str(manifest) in {item["path"] for item in after["source_records"]}
    (manifest.parent / (descriptor["head"] + ".npz")).write_bytes(b"corrupt committed chunk")
    rejected = collect(streamed, summary, config)
    assert not any(p["id"].endswith("materials-energy") for p in rejected["plots"])
    assert any("digest mismatch" in item["reason"] for item in rejected["skipped"])


def test_diffusion_units_and_withheld_fit(tmp_path):
    summary = {"task": "diffusion", "diffusion": {
        "lag_time_fs": [0, 1, 2], "msd_au": [0, 2, 4], "vacf_au": [1, .5, 0],
        "vacf_integral_m2_s": [0, 1e-9, 2e-9], "diffusion_m2_s": None,
        "status": "not_diffusive", "raw_fit": {"diffusion_m2_s": 99},
    }}
    result = collect(tmp_path, summary)
    assert plot(result, "msd")["series"][0]["y"] == pytest.approx(np.array([0, 2, 4]) * BOHR_ANGSTROM**2)
    assert plot(result, "vacf")["series"][0]["y"] == pytest.approx(np.array([1, .5, 0]) * (BOHR_ANGSTROM * FS_TO_AU)**2)
    assert result["metrics"]["diffusion"]["diffusion_m2_s"] is None
    assert any(f["code"] == "diffusion_withheld" for f in result["findings"])


def test_pcet_log_domain_rates_survive_underflow(tmp_path):
    summary = {"task": "pcet", "potential_reference": "SHE", "rates": [
        {"temperature_K": 300, "potential_V": .1, "isotope": "H", "reduction_rate_s": 0,
         "log_reduction_rate_s": -1000, "oxidation_rate_s": 10},
    ], "kinetic_isotope_effects": [
        {"temperature_K": 300, "potential_V": .1, "reduction_KIE": 2, "oxidation_KIE": 3},
    ]}
    result = collect(tmp_path, summary)
    lines = plot(result, "pcet-rates")["series"]
    assert lines[0]["y"] == pytest.approx([-1000 / np.log(10)])
    assert lines[1]["y"] == [1]
    assert plot(result, "pcet-kie")["series"][0]["y"] == [2]


def test_malformed_npz_is_skipped_and_symlink_does_not_escape(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    (folder / "bad.npz").write_text("broken zip")
    external = tmp_path.parent / (tmp_path.name + "_outside.npz")
    np.savez(external, time_au=[0, 1], energy_per_bead=[0, 1])
    (folder / "escape.npz").symlink_to(external)
    result = collect(tmp_path, scattering_summary())
    assert len(result["skipped"]) >= 2
    assert not any(p["id"].endswith("scattering-energy") for p in result["plots"])
    assert any("outside" in r["reason"] for r in result["skipped"])


def test_mismatched_diffusion_arrays_do_not_render(tmp_path):
    result = collect(tmp_path, {"task": "diffusion", "diffusion": {
        "lag_time_fs": [0, 1, 2], "msd_au": [0, 1],
    }})
    assert not result["plots"]
    assert any(f["code"] == "malformed_diffusion" for f in result["findings"])


def test_unknown_task_has_no_invented_plots_and_invalid_unit_rejected(tmp_path):
    assert not collect(tmp_path, {"task": "unknown"})["plots"]
    with pytest.raises(ValueError):
        collect(tmp_path, {"task": "pimd"}, energy_unit="fs")


def test_corrupted_integrity_envelope_cannot_supply_plots(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    path = folder / "trajectory-0000.npz"
    atomic_savez(path, time_au=[0, 1], energy_per_bead=[1., 2.])
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["energy_per_bead"][1] = 42
    np.savez(path, **payload)
    result = collect(tmp_path, scattering_summary())
    assert not any(p["id"].endswith("scattering-energy") for p in result["plots"])
    assert any("integrity check failed" in s["reason"] for s in result["skipped"])
    assert not result["source_records"]


def test_verified_source_snapshot_is_returned(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    path = folder / "trajectory-0000.npz"
    atomic_savez(path, time_au=[0, 1], energy_per_bead=[1., 2.])
    result = collect(tmp_path, scattering_summary())
    record, = result["source_records"]
    assert record["path"] == str(path)
    assert record["integrity"] == "verified"
    assert record["size"] == path.stat().st_size
    assert len(record["sha256"]) == 64


def test_source_change_during_read_is_not_hashed_as_original(tmp_path, monkeypatch):
    import pyrpmd.analysis.report_observables as module
    folder = tmp_path / "trajectories"
    folder.mkdir()
    path = folder / "trajectory-0000.npz"
    atomic_savez(path, time_au=[0, 1], energy_per_bead=[1., 2.])

    @contextmanager
    def changing_reader(source):
        with load_checkpoint(source) as archive:
            yield archive
        atomic_savez(source, time_au=[0, 1], energy_per_bead=[1., 99.])

    monkeypatch.setattr(module, "load_checkpoint", changing_reader)
    result = collect(tmp_path, scattering_summary())
    assert not any(p["id"].endswith("scattering-energy") for p in result["plots"])
    assert any("changed while reading" in s["reason"] for s in result["skipped"])
    assert not result["source_records"]


def test_centroid_pair_distances_convert_bohr_and_use_explicit_reaction_pairs(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    centroid = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 3., 0.]],
                         [[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]]])
    atomic_savez(folder / "trajectory-0000.npz", time_au=[0, FS_TO_AU],
                 energy_per_bead=[0., 0.], centroid=centroid)
    result = collect(tmp_path, scattering_summary(), {"reaction": {"forming_bonds": [[0, 1]]}})
    figure = plot(result, "centroid-distances-1")
    assert len(figure["series"]) == 1
    assert figure["series"][0]["x"] == pytest.approx([0, 1])
    assert figure["series"][0]["y"] == pytest.approx(np.array([1, 2]) * BOHR_ANGSTROM)
    assert "Atoms 1–2 (forming)" == figure["series"][0]["label"]
    assert figure["ylabel"] == "Centroid pair distance (Å)"


def test_small_system_all_pair_fallback_does_not_infer_bonds(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    centroid = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]])
    atomic_savez(folder / "trajectory-0000.npz", time_au=[0], energy_per_bead=[0.], centroid=centroid)
    result = collect(tmp_path, scattering_summary())
    figure = plot(result, "centroid-distances-1")
    assert len(figure["series"]) == 3
    assert [s["y"][0] for s in figure["series"]] == pytest.approx(np.array([1, 2, np.sqrt(5)]) * BOHR_ANGSTROM)
    assert "do not infer chemical bonds" in figure["caption"]


@pytest.mark.parametrize("centroid", [None, np.full((2, 2, 3), np.nan)])
def test_missing_or_nonfinite_centroid_skips_distance_without_losing_energy(tmp_path, centroid):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    arrays = {"time_au": [0, FS_TO_AU], "energy_per_bead": [0., 0.]}
    if centroid is not None:
        arrays["centroid"] = centroid
    atomic_savez(folder / "trajectory-0000.npz", **arrays)
    result = collect(tmp_path, scattering_summary())
    assert not any(p["id"].endswith("centroid-distances-1") for p in result["plots"])
    assert plot(result, "scattering-energy")
    assert any(s["figure"].startswith("centroid-distances") for s in result["skipped"])


def test_large_or_periodic_system_does_not_invent_distance_selection(tmp_path):
    folder = tmp_path / "trajectories"
    folder.mkdir()
    atomic_savez(folder / "trajectory-0000.npz", time_au=[0], energy_per_bead=[0.], centroid=np.zeros((1, 7, 3)))
    result = collect(tmp_path, scattering_summary())
    assert any("More than six atoms" in s["reason"] for s in result["skipped"])
    result = collect(tmp_path, scattering_summary(), {"reaction": {"breaking_bonds": [[0, 1]]}, "system": {"pbc": [True, False, False]}})
    assert any("nonperiodic" in s["reason"] for s in result["skipped"])
