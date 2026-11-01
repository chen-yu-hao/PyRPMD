"""Keep classical and quantum reference selection distinct in raw-CV replay."""
import json

import h5py
import numpy as np
import pytest

from benchmarks.published_rates.diagnose_pilot_pmf import diagnose, pack


def _saved_run(tmp_path, *, beads=1, temperature=300.):
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "umbrella").mkdir()
    (run / "input/resolved.json").write_text(json.dumps(
        {"beta": 1000., "temperature_K": temperature, "beads": beads}))
    (run / "rate_pmf.json").write_text(json.dumps(
        {"xi_star": 1., "reference": 0., "free_energy_barrier_hartree": .02,
         "reactant_prefactor_au": .1}))
    (run / "summary.json").write_text(json.dumps(
        {"kappa": .9, "rate_estimate_cm3_s": 1e-18}))
    with h5py.File(run / "umbrella/fes.h5", "w") as saved:
        group = saved.create_group("streams/0")
        group.create_dataset("samples", data=np.linspace(.95, 1.05, 10)[:, None])
        group.create_dataset("center", data=[1.])
        group.create_dataset("stiffness", data=[[30.]])
        group.attrs.update(replica=0, window=0)
        saved.create_dataset("edges_0", data=np.linspace(0, 1.1, 12))
        saved.create_dataset("effective_samples_per_stream", data=[4.])
        saved.create_dataset("mask", data=np.zeros(11, dtype=bool))
        saved.attrs["measure"] = "rate_flux"
    return run


@pytest.mark.parametrize("beads,method,value", [(1, "Classical", 1.56e-18),
                                              (128, "RPMD", 2.51e-16)])
def test_reference_matches_nuclear_model(tmp_path, beads, method, value):
    path = pack("control", _saved_run(tmp_path, beads=beads), tmp_path / "packed")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
    assert metadata["reference"]["method"] == method
    assert metadata["reference"]["qtst_rate_cm3_molecule_s"] == value


def test_quantum_reference_rejected_for_classical_samples(tmp_path):
    with pytest.raises(ValueError, match="method and temperature"):
        pack("classical", _saved_run(tmp_path), tmp_path / "packed",
             reference_id="h_h2_bkmp2_rpmd_300k")


def test_reference_temperature_must_match(tmp_path):
    with pytest.raises(ValueError, match="method and temperature"):
        pack("wrongtemperature", _saved_run(tmp_path, temperature=600.), tmp_path / "packed")


def test_replay_detects_altered_sample_before_estimation(tmp_path):
    path = pack("control", _saved_run(tmp_path), tmp_path / "packed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["samples"][0] += .001
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="integrity mismatch: samples"):
        diagnose(path)
