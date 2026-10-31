"""Saved-data contracts, provenance and scientific-status regression tests."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.rate_io import postprocess_rate
from pyrpmd.cli import main
from pyrpmd.constants import AU_RATE_TO_CM3_S
from pyrpmd.checkpoint import CheckpointError, atomic_savez


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _saved_rate(tmp_path, *, published=False):
    beta, rinf, mass, barrier = 200.0, 10.0, 1200.0, .015
    prefactor = 4 * np.pi * rinf**2 / np.sqrt(2 * np.pi * beta * mass)
    qtst = prefactor * np.exp(-beta * barrier)
    pmf = {
        "sampling_measure": "rate_flux", "method": "mbar", "ready": True,
        "xi_star": 1.0, "free_energy_barrier_hartree": barrier,
        "reactant_prefactor_au": prefactor, "k_qtst_au": qtst,
        "rinf_bohr": rinf, "reduced_mass_au": mass,
        "symmetry_factor": 1.0, "electronic_weight": 1.0,
    }
    estimate = qtst * .4 * AU_RATE_TO_CM3_S
    _dump(tmp_path / "rate_pmf.json", pmf)
    _dump(tmp_path / "input/resolved.json", {"task": "rate", "beta": beta})
    _dump(tmp_path / "summary.json", {
        "task": "rate", "status": "plateau_estimate" if published else "pilot_requires_convergence",
        "pmf": pmf, "rate_estimate_cm3_s": estimate,
        "rate_cm3_s": estimate if published else None,
        "plateau_detected": True, "maximum_endpoint_energy_error_kbt": .01 if published else .2,
        "converged": False, "warnings": [],
    })
    np.savez(tmp_path / "rate_pmf.npz", sampling_measure="rate_flux", method="mbar",
             ready=True, xi_star=1.0, xi=[0.0, 1.0], free_energy=[0.0, barrier], mask=[False, False])
    np.savez(tmp_path / "recrossing.npz", time_au=np.arange(1, 61),
             kappa=np.full(60, .4), parent_numerator=np.full((4, 60), .4),
             parent_denominator=np.ones(4), parent_ids=[[0, 0], [0, 1], [1, 0], [1, 1]])
    return estimate


def _saved_fes(tmp_path):
    energy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    np.savez(tmp_path / "fes.npz", free_energy=energy, mask=np.zeros_like(energy, dtype=bool),
             edges_0=[-.5, .5, 1.5, 2.5], edges_1=[0., 1., 3.], measure="rate_flux")
    _dump(tmp_path / "summary.json", {
        "task": "fes", "dimension": 2, "units": ["dimensionless", "bohr"],
        "energy_unit": "hartree", "measure": "rate_flux", "flux_axis": 0,
        "overlap_connected": True, "status": "sampled", "converged": False,
    })
    return {
        "method": "flux_slice_2d", "beta": 1., "surface": 1., "reference": 0.,
        "reference_rate_au": 7., "reference_definition": "Test entrance with known flux",
    }


@pytest.mark.parametrize("sealed", [False, True])
def test_saved_fes_postprocessing_accepts_legacy_and_checked_archives(tmp_path, sealed):
    spec = _saved_fes(tmp_path)
    path = tmp_path / "fes.npz"
    if sealed:
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved.items())
        atomic_savez(path, **arrays)
    result = postprocess_rate(tmp_path, spec=spec, bootstrap=0)
    assert result["static"]["k_tst_au"] == pytest.approx(7 * np.exp(-1))
    assert result["rate_cm3_s"] is None


def test_saved_fes_postprocessing_rejects_failed_array_integrity_before_publication(tmp_path):
    spec = _saved_fes(tmp_path)
    path = tmp_path / "fes.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved.items())
    atomic_savez(path, **arrays)
    postprocess_rate(tmp_path, spec=spec, bootstrap=0)
    previous = (tmp_path / "rate_postprocess.json").read_bytes()
    with np.load(path, allow_pickle=False) as saved:
        damaged = dict(saved.items())
    damaged["free_energy"][1] += .25
    np.savez(path, **damaged)
    with pytest.raises(CheckpointError, match="integrity check failed for array 'free_energy'"):
        postprocess_rate(tmp_path, spec=spec, bootstrap=0)
    assert (tmp_path / "rate_postprocess.json").read_bytes() == previous


@pytest.mark.parametrize("filename,member", [
    ("rate_pmf.npz", "standard_error"), ("recrossing.npz", "bootstrap_kappa"),
])
def test_saved_rate_postprocessing_checks_even_unconsumed_members(tmp_path, filename, member):
    expected = _saved_rate(tmp_path)
    path = tmp_path / filename
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved.items())
    arrays[member] = np.array([.1, .2])
    atomic_savez(path, **arrays)
    result = postprocess_rate(tmp_path, bootstrap=20)
    assert result["rate_estimate_cm3_s"] == pytest.approx(expected)
    with np.load(path, allow_pickle=False) as saved:
        damaged = dict(saved.items())
    damaged[member][0] += .5
    np.savez(path, **damaged)
    with pytest.raises(CheckpointError, match="integrity check failed"):
        postprocess_rate(tmp_path, bootstrap=20)


def test_saved_rate_is_recomputed_without_upgrading_failed_energy_check(tmp_path):
    expected = _saved_rate(tmp_path)
    before = (tmp_path / "summary.json").read_bytes()
    result = postprocess_rate(tmp_path, bootstrap=20)
    assert result["rate_estimate_cm3_s"] == pytest.approx(expected)
    assert result["rate_cm3_s"] is None
    assert result["status"] == "source_validation_incomplete"
    assert result["converged"] is False
    assert (tmp_path / "summary.json").read_bytes() == before
    assert len(result["source_files"]) >= 4
    assert all(len(item["sha256"]) == 64 for item in result["source_files"])
    assert json.loads((tmp_path / "rate_postprocess.json").read_text())["schema"] == "pyrpmd.rate_postprocess"


def test_saved_valid_plateau_remains_an_estimate_not_converged(tmp_path):
    expected = _saved_rate(tmp_path, published=True)
    result = postprocess_rate(tmp_path, bootstrap=20)
    assert result["rate_cm3_s"] == pytest.approx(expected)
    assert result["converged"] is False
    assert result["channels"]["branching_fractions"] == pytest.approx([1.0])


def test_mixed_pmf_and_recrossing_outputs_are_rejected(tmp_path):
    _saved_rate(tmp_path)
    path = tmp_path / "rate_pmf.json"
    pmf = json.loads(path.read_text())
    pmf["xi_star"] = .95
    _dump(path, pmf)
    with pytest.raises(ValueError, match="surfaces disagree"):
        postprocess_rate(tmp_path)


def test_saved_kappa_must_match_parent_signed_flux(tmp_path):
    _saved_rate(tmp_path)
    path = tmp_path / "recrossing.npz"
    with np.load(path) as data:
        arrays = dict(data)
    arrays["kappa"] = np.ones(60)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="signed parent flux"):
        postprocess_rate(tmp_path)


def test_postprocessor_never_overwrites_saved_source_files(tmp_path):
    _saved_rate(tmp_path)
    before = (tmp_path / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        postprocess_rate(tmp_path, output=tmp_path / "summary.json")
    assert (tmp_path / "summary.json").read_bytes() == before


def test_two_dimensional_flux_cli_saves_static_only_result(tmp_path, capsys):
    spec = _saved_fes(tmp_path)
    _dump(tmp_path / "spec.json", spec)
    assert main(["rate-postprocess", str(tmp_path), "--spec", str(tmp_path / "spec.json"), "--bootstrap", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["static"]["k_tst_au"] == pytest.approx(7 * np.exp(-1))
    assert output["rate_cm3_s"] is None
    assert output["channels"] is None
    assert output["converged"] is False


def test_fes_metadata_and_requested_temperature_must_match(tmp_path):
    spec = _saved_fes(tmp_path)
    _dump(tmp_path / "input/resolved.json", {"beta": 2.})
    with pytest.raises(ValueError, match="temperature"):
        postprocess_rate(tmp_path, spec=spec)


def test_npz_measure_cannot_override_metadata(tmp_path):
    spec = _saved_fes(tmp_path)
    meta = json.loads((tmp_path / "summary.json").read_text())
    meta["measure"] = "cv_density"
    _dump(tmp_path / "summary.json", meta)
    with pytest.raises(ValueError, match="measures disagree"):
        postprocess_rate(tmp_path, spec=spec)


def test_unknown_or_inapplicable_spec_keys_fail_closed(tmp_path):
    spec = _saved_fes(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        postprocess_rate(tmp_path, spec={**spec, "referense": 0.})
    with pytest.raises(ValueError, match="already contain"):
        postprocess_rate(tmp_path, spec={**spec, "reference_normal_speed": 1.})


def test_channel_archive_requires_matching_surface_identity(tmp_path):
    spec = _saved_fes(tmp_path)
    np.savez(tmp_path / "channels.npz", schema="pyrpmd.channel_flux", schema_version=1,
             estimator="signed_flux_side", time_unit="atomic", surface_identity="different")
    with pytest.raises(ValueError, match="different dividing surface"):
        postprocess_rate(tmp_path, spec={**spec, "channel_flux": "channels.npz", "surface_identity": "requested"})


@pytest.mark.parametrize("key,value", [
    ("beta", 999.), ("energy_unit", "kcal/mol"),
    ("cv_units", ["dimensionless", "angstrom"]), ("flux_axis", 1),
])
def test_embedded_fes_metadata_must_match_the_saved_run(tmp_path, key, value):
    spec = _saved_fes(tmp_path)
    path = tmp_path / "fes.npz"
    with np.load(path) as data:
        arrays = dict(data)
    arrays[key] = np.asarray(value)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="disagree"):
        postprocess_rate(tmp_path, spec=spec)


@pytest.mark.parametrize("filename", ["fes.csv", "rate_pmf.csv", "recrossing.csv", "checkpoints/child.json"])
def test_existing_scientific_outputs_are_protected_from_overwrite(tmp_path, filename):
    _saved_rate(tmp_path)
    path = tmp_path / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text("original scientific data")
    with pytest.raises(ValueError, match="overwrite"):
        postprocess_rate(tmp_path, output=path)
    assert path.read_text() == "original scientific data"


@pytest.mark.parametrize("change", ["barrier", "mask"])
def test_ready_json_cannot_hide_an_invalid_pmf_array(tmp_path, change):
    _saved_rate(tmp_path, published=True)
    path = tmp_path / "rate_pmf.npz"
    with np.load(path) as data:
        arrays = dict(data)
    if change == "barrier":
        arrays["free_energy"] = np.array([0., 999.])
    else:
        arrays["mask"] = np.array([True, True])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="PMF"):
        postprocess_rate(tmp_path)


@pytest.mark.parametrize("bootstrap", [-1, 1, True, 2.5])
def test_invalid_bootstrap_is_rejected_before_reading_files(tmp_path, bootstrap):
    with pytest.raises(ValueError, match="bootstrap"):
        postprocess_rate(tmp_path, bootstrap=bootstrap)


@pytest.mark.parametrize("electronic,barrier,reason", [
    (0.0, 0.015, "zero_electronic_weight"),
    (1.0, 10.0, "floating_point_underflow"),
])
@pytest.mark.parametrize("with_recrossing", [False, True])
def test_saved_zero_qtst_is_preserved_as_a_diagnostic(
    tmp_path, electronic, barrier, reason, with_recrossing,
):
    # These are supported by the existing gas-rate producer. A finite zero
    # must not be confused with a corrupt negative/nonfinite static rate.
    _saved_rate(tmp_path, published=True)
    pmf_path = tmp_path / "rate_pmf.json"
    pmf = json.loads(pmf_path.read_text())
    pmf.update(electronic_weight=electronic,
               free_energy_barrier_hartree=barrier, k_qtst_au=0.0)
    _dump(pmf_path, pmf)
    with np.load(tmp_path / "rate_pmf.npz") as data:
        arrays = dict(data)
    arrays["free_energy"] = np.array([0., barrier])
    np.savez(tmp_path / "rate_pmf.npz", **arrays)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary.update(pmf=pmf, rate_estimate_cm3_s=0.0, rate_cm3_s=0.0)
    _dump(tmp_path / "summary.json", summary)
    if not with_recrossing:
        (tmp_path / "recrossing.npz").unlink()
    result = postprocess_rate(tmp_path, bootstrap=0)
    assert result["static"]["k_tst_au"] == 0.0
    assert result["static"]["k_tst_cm3_s"] == 0.0
    assert result["static"]["zero_rate_reason"] == reason
    assert result["static"]["underflow"] is (electronic > 0)
    assert result["converged"] is False
    assert result["rate_cm3_s"] is None
    if with_recrossing:
        assert result["rate_estimate_cm3_s"] == 0.0
        assert "zero_static_rate" in result["channels"]["reasons"]
        assert result["channels"]["branching_fractions"] is None
    else:
        assert result["status"] == "static_only"
        assert result["channels"] is None


def test_zero_saved_qtst_does_not_hide_inconsistent_positive_pmf(tmp_path):
    _saved_rate(tmp_path)
    path = tmp_path / "rate_pmf.json"
    pmf = json.loads(path.read_text())
    pmf["k_qtst_au"] = 0.0
    _dump(path, pmf)
    with pytest.raises(ValueError, match="Recomputed QTST"):
        postprocess_rate(tmp_path, bootstrap=0)
