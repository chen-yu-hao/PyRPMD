"""Conservative ingestion of actual thermal rate artifact layouts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.io import file_sha256
from pyrpmd.precision_artifacts import read_multiple_rate_diagnostics, read_rate_diagnostics
from pyrpmd.rate_artifacts import EXTENSION_FORMAT, RECEIPT_FORMAT, generation_name


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def saved_rate(path, *, seed=1, parent_chains=2):
    path.mkdir(parents=True)
    pmf = {
        "method": "mbar", "sampling_measure": "rate_flux", "ready": True,
        "free_energy_barrier_hartree": .03, "xi_star": .9,
        "barrier_standard_error_hartree": .001, "overlap_connected": True,
        "effective_samples_per_stream": [100., 120.],
        "uncertainty": {"available": True, "bootstrap_rejected": 0},
        "bootstrap": {"block_plan": {"status": "finite_block_estimate"}},
    }
    support = {
        "scope": "independent_parent_chains" if parent_chains > 1 else "conditional_on_single_chain_mixing",
        "independent_parent_chains": parent_chains,
        "effective_parent_blocks": {str(i): 2. for i in range(parent_chains)},
        "available": True,
    }
    summary = {
        "task": "rate", "status": "plateau_estimate", "rate_cm3_s": 1e-12,
        "rate_estimate_cm3_s": 1e-12, "rate_standard_error_cm3_s": 1e-13,
        "conditional_rate_standard_error_cm3_s": 5e-14,
        "rate_uncertainty": {"available": True, "recrossing_support": support,
                             "combination": "independent PMF and parent bootstrap draws",
                             "independence_assumption": "not established automatically"},
        "kappa": .5, "kappa_standard_error": .04, "plateau_detected": True,
        "plateau_diagnostics": {"passed": True, "scope": "selected_finite_time_tail"},
        "parent_configurations": 4, "child_pairs": 40, "child_time_fs": 20.,
        "energy_diagnostic_child_pairs": 40, "maximum_endpoint_energy_error_kbt": .02,
        "pmf": pmf,
    }
    write_json(path / "summary.json", summary)
    write_json(path / "rate_pmf.json", pmf)
    write_json(path / "manifest.json", {"physics_fingerprint": "physical-identity"})
    write_json(path / "input/resolved.json", {"task": "rate", "seed": seed, "beta": 1000.})
    np.savez(path / "rate_pmf.npz", method="mbar", ready=True,
             sampling_measure="rate_flux", overlap=np.array([[.9, .1], [.1, .9]]))
    ids = np.array([[i % parent_chains, i // parent_chains] for i in range(4)])
    np.savez(path / "recrossing.npz", parent_ids=ids,
             parent_numerator=np.full((4, 20), .5), parent_denominator=np.ones(4),
             kappa=np.full(20, .5), time_au=np.arange(1, 21),
             standard_error=np.full(20, .04))
    write_json(path / "umbrella/summary.json", {"overlap_connected": True})
    np.savez(path / "umbrella/fes.npz", free_energy=[0., .03])
    return summary


def publish_extension(path, summary, steps=200):
    relative = generation_name(steps)
    generation = path / relative
    generation.mkdir(parents=True)
    for name in ("rate_pmf.json", "rate_pmf.npz", "recrossing.npz",
                 "umbrella/summary.json", "umbrella/fes.npz"):
        target = generation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path / name, target)
    summary = dict(summary, effective_production_steps=steps)
    write_json(generation / "summary.json", summary)
    write_json(generation / "generation.json", {"production_steps": steps})
    names = ["summary.json", "generation.json", "rate_pmf.json", "rate_pmf.npz",
             "recrossing.npz", "umbrella/summary.json", "umbrella/fes.npz"]
    receipt = {"format": RECEIPT_FORMAT, "production_steps": steps, "generation": relative,
               "physics_fingerprint": "physical-identity",
               "artifacts": {name: file_sha256(generation / name) for name in names}}
    write_json(generation / "receipt.json", receipt)
    summary["rate_extension"] = {"format": EXTENSION_FORMAT, "production_steps": steps,
                                  "generation": relative,
                                  "receipt_sha256": file_sha256(generation / "receipt.json")}
    write_json(path / "results/summary.json", summary)
    return generation


def test_reads_real_layout_without_promoting_legacy_uncertainty(tmp_path):
    saved_rate(tmp_path / "rate")
    result = read_rate_diagnostics(tmp_path / "rate")
    assert result["pmf"]["minimum_effective_samples"] == 100.
    assert result["recrossing"]["independent_parent_chains"] == 2
    assert result["recrossing"]["parent_configurations"] == 4
    assert result["recrossing"]["child_pairs"] == 40
    assert result["recrossing"]["variance_components"] is None
    assert result["energy"]["within_workflow_tolerance"] is True
    assert result["provenance"]["phase"] == "unknown"
    assert result["provenance"]["validated_for_acceptance"] is False
    assert result["uncertainty"]["raw_rate_standard_error_cm3_s"] == 1e-13
    assert result["uncertainty"]["validated_total_rate_error"] is None
    assert result["uncertainty"]["producer_metadata"]["independence_assumption"] == "not established automatically"
    json.dumps(result, allow_nan=False)


def test_children_and_parent_configurations_never_become_independent_chains(tmp_path):
    saved_rate(tmp_path / "rate", parent_chains=1)
    result = read_rate_diagnostics(tmp_path / "rate")
    assert result["recrossing"]["independent_parent_chains"] == 1
    assert result["recrossing"]["uncertainty_scope"] == "conditional_on_single_chain_mixing"
    summary = json.loads((tmp_path / "rate/summary.json").read_text())
    del summary["rate_uncertainty"]
    write_json(tmp_path / "rate/summary.json", summary)
    assert read_rate_diagnostics(tmp_path / "rate")["recrossing"]["independent_parent_chains"] is None


def test_missing_and_nonfinite_error_are_not_zero_or_conditional_substitutes(tmp_path):
    path = tmp_path / "rate"
    summary = saved_rate(path)
    summary["rate_standard_error_cm3_s"] = float("nan")
    summary["pmf"]["barrier_standard_error_hartree"] = None
    summary["pmf"]["supported_draw_barrier_standard_error_hartree"] = .001
    summary["pmf"]["uncertainty"] = {"available": False, "bootstrap_rejected": 1}
    write_json(path / "summary.json", summary)
    write_json(path / "rate_pmf.json", summary["pmf"])
    result = read_rate_diagnostics(path)
    assert result["uncertainty"]["raw_rate_standard_error_cm3_s"] is None
    assert result["pmf"]["barrier_standard_error_hartree"] is None
    assert result["pmf"]["uncertainty"]["bootstrap_rejected"] == 1
    json.dumps(result, allow_nan=False)


def test_pending_generation_ignored_and_published_generation_authenticated(tmp_path):
    path = tmp_path / "rate"
    summary = saved_rate(path)
    original = read_rate_diagnostics(path)
    write_json(path / generation_name(300) / "summary.json", {"task": "rate", "rate_cm3_s": 9e-12})
    assert read_rate_diagnostics(path)["source_ids"] == original["source_ids"]
    generation = publish_extension(path, summary)
    active = read_rate_diagnostics(path)
    assert active["published_generation"] == generation_name(200)
    assert active["source_ids"] != original["source_ids"]
    assert active["source_family_id"] == original["source_family_id"]
    (generation / "rate_pmf.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum differs"):
        read_rate_diagnostics(path)


def test_duplicate_copied_runs_rejected(tmp_path):
    saved_rate(tmp_path / "rate")
    shutil.copytree(tmp_path / "rate", tmp_path / "copy")
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        read_multiple_rate_diagnostics([tmp_path / "rate", tmp_path / "copy"])


def test_overlapping_generations_rejected_but_distinct_seeds_allowed(tmp_path):
    summary = saved_rate(tmp_path / "rate")
    shutil.copytree(tmp_path / "rate", tmp_path / "extended")
    publish_extension(tmp_path / "extended", summary)
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        read_multiple_rate_diagnostics([tmp_path / "rate", tmp_path / "extended"])
    saved_rate(tmp_path / "independent", seed=2)
    assert len(read_multiple_rate_diagnostics([tmp_path / "rate", tmp_path / "independent"])) == 2


@pytest.mark.parametrize("kind", ["trpmd", "site_rate"])
def test_scope_excludes_nonthermal_rpmd(kind, tmp_path):
    path = tmp_path / "rate"
    summary = saved_rate(path)
    summary.update({"dynamics": "trpmd"} if kind == "trpmd" else {"task": "site_rate"})
    write_json(path / "summary.json", summary)
    with pytest.raises(ValueError):
        read_rate_diagnostics(path)


def test_inconsistent_fluxes_rejected(tmp_path):
    path = tmp_path / "rate"
    saved_rate(path)
    with np.load(path / "recrossing.npz") as data:
        arrays = dict(data.items())
    arrays["kappa"] = np.full(20, .9)
    np.savez(path / "recrossing.npz", **arrays)
    with pytest.raises(ValueError, match="curve disagrees"):
        read_rate_diagnostics(path)


def test_symlink_escape_and_object_arrays_rejected(tmp_path):
    path = tmp_path / "rate"
    saved_rate(path)
    (path / "rate_pmf.npz").unlink()
    np.savez(tmp_path / "outside.npz", overlap=np.eye(2))
    (path / "rate_pmf.npz").symlink_to(tmp_path / "outside.npz")
    with pytest.raises(ValueError, match="inside its run"):
        read_rate_diagnostics(path)
    (path / "rate_pmf.npz").unlink()
    np.savez(path / "rate_pmf.npz", unsafe=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        read_rate_diagnostics(path)


def test_production_claim_requires_explicit_frozen_independent_metadata(tmp_path):
    path = tmp_path / "rate"
    summary = saved_rate(path)
    summary["precision_provenance"] = {"format": "PyRPMD.precision.production.v1", "phase": "production"}
    write_json(path / "summary.json", summary)
    assert read_rate_diagnostics(path)["provenance"]["phase"] == "unknown"
    summary["precision_provenance"].update(epoch_id="epoch1", design_id="design1",
        frozen_before_sampling=True, independent_from_pilot=True, physics_fingerprint="physical-identity")
    write_json(path / "summary.json", summary)
    provenance = read_rate_diagnostics(path)["provenance"]
    assert provenance["phase"] == "production"
    assert provenance["validated_for_acceptance"] is False


def test_incomplete_energy_diagnostics_remain_unresolved(tmp_path):
    path = tmp_path / "rate"
    summary = saved_rate(path)
    summary["energy_diagnostic_child_pairs"] = 39
    write_json(path / "summary.json", summary)
    energy = read_rate_diagnostics(path)["energy"]
    assert energy["complete"] is False
    assert energy["within_workflow_tolerance"] is None
