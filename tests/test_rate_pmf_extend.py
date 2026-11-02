"""Production extensions use the independent PMF stage's clock and budget."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import extend_run, resume_run, run_resolved
from pyrpmd.workflows import fes, rate_extend


SHORT = 31
LONG = 74


def configuration(production=SHORT):
    return {
        "task": "rate", "temperature": 1 / (5 * KB_AU), "beads": 2,
        "seed": 17, "native": False,
        "system": {"symbols": ["H", "H", "H"], "masses": [2., 2., 2.],
                   "geometry": [[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]],
                   "geometry_unit": "bohr", "fragments": [[0], [1, 2]]},
        "potential": {"provider": "free"},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 4.},
        "dynamics": {"integrator": "cayley-bcb", "timestep": .003,
                     "block_steps": 11, "loop_backend": "python"},
        # Deliberately larger than the requested extension. It is not the PMF
        # budget, and these equilibration steps are not the PMF clock either.
        "sampling": {"equilibration_steps": 17, "production_steps": 200,
                     "interval_steps": 3, "andersen_interval": 7},
        "analysis": {"bins": 12, "bootstrap": 0},
        "parallel": {"backend": "serial"},
        "output": {"checkpoint_steps": 13},
        "recrossing": {"dividing_surface": 1., "parent_replicas": 1, "parent_samples": 2,
                       "equilibration_steps": 5, "parent_interval_steps": 3,
                       "child_pairs": 1, "child_steps": 10, "bootstrap": 0},
        "rate": {"pmf": {
            "method": "umbrella", "estimator": "ui",
            "dynamics": {"integrator": "normal-mode-bab", "timestep": .006},
            "sampling": {"thermostat": "pile-l", "gamma_centroid": 1.,
                         "equilibration_steps": 7, "production_steps": production,
                         "interval_steps": 3},
            "umbrella": {"centers": np.linspace(0, 1, 5).tolist(), "kappa": 10.},
        }},
    }


def checkpoint_arrays(path):
    with load_checkpoint(path) as saved:
        arrays = {key: np.array(saved[key], copy=True) for key in
                  ("q", "p", "step", "time", "rng", "elapsed")}
        if "sample_stream" in saved:
            arrays.update(fes.read_fes_sample_history(
                saved, dimension=1, chunk_directory=Path(path).with_suffix(".chunks")))
        else:
            arrays.update(values=np.array(saved["values"]), energies=np.array(saved["energies"]))
    return arrays


def test_independent_pmf_extension_matches_continuous_sampling_and_keeps_parent_clock(tmp_path):
    short = tmp_path / "short"
    config, system = JobSpec.from_mapping(configuration()).resolve()
    initial = run_resolved(config, system, output=short)
    assert initial["pmf"]["ready"]
    original_input = (short / "input/resolved.json").read_bytes()
    continuous = tmp_path / "continuous"
    reference_config, reference_system = JobSpec.from_mapping(configuration(LONG)).resolve()
    expected = run_resolved(reference_config, reference_system, output=continuous)
    result = extend_run(short, production_steps=LONG)
    assert result["original_production_steps"] == SHORT
    assert result["effective_production_steps"] == LONG
    assert (short / "input/resolved.json").read_bytes() == original_input
    for reference in (continuous / "umbrella/checkpoints").glob("window_*.npz"):
        actual = checkpoint_arrays(short / "umbrella/checkpoints" / reference.name)
        wanted = checkpoint_arrays(reference)
        for key in ("q", "p", "step", "time", "rng", "elapsed", "values", "energies"):
            np.testing.assert_array_equal(actual[key], wanted[key], err_msg=key)
        assert int(actual["elapsed"]) == 7 + LONG
        assert float(actual["time"]) == pytest.approx((7 + LONG) * .006)
        assert len(actual["values"]) == LONG // 3
    generation = short / result["rate_extension"]["generation"]
    with load_checkpoint(generation / "checkpoints/parent_0000.npz") as saved:
        assert float(saved["time"]) == pytest.approx(int(saved["step"]) * .003)
    with np.load(generation / "rate_pmf.npz", allow_pickle=False) as actual, np.load(
        continuous / "rate_pmf.npz", allow_pickle=False,
    ) as wanted:
        np.testing.assert_array_equal(actual["free_energy"], wanted["free_energy"])
    assert result["kappa"] == expected["kappa"]
    assert result["rate_estimate_cm3_s"] == expected["rate_estimate_cm3_s"]
    assert resume_run(short)["rate_extension"] == result["rate_extension"]
    assert extend_run(short, production_steps=LONG)["rate_extension"] == result["rate_extension"]


@pytest.mark.parametrize("method", ["metadynamics", "opes", "external", "legacy_external", "adaptive"])
def test_unsupported_extension_rejected_before_authentication_or_ledger_write(tmp_path, monkeypatch, method):
    config = {"task": "rate", "analysis": {"bootstrap": 0}}
    if method == "external":
        config["rate"] = {"pmf": {"method": "external", "directory": "missing-source"}}
    elif method == "legacy_external":
        config["rate"] = {"pmf_source": "missing-source"}
    elif method == "adaptive":
        config["rate"] = {"pmf": {"method": "umbrella", "adaptive_umbrella": {"enabled": True}}}
    else:
        config["rate"] = {"pmf": {"method": method, method: {
            "freeze_after_steps": 10, "freeze_equilibration_steps": 5,
        }}}
    (tmp_path / "input").mkdir()
    input_file = tmp_path / "input/resolved.json"
    input_file.write_text(json.dumps(config))
    before = input_file.read_bytes()
    monkeypatch.setattr(rate_extend, "_authenticate", lambda *args: pytest.fail("unsupported method reached authentication"))
    with pytest.raises(InputError, match="fixed windows|fixed umbrella|external PMF"):
        rate_extend._prepare_extension(tmp_path, LONG)
    assert input_file.read_bytes() == before
    assert not (tmp_path / "results/extensions.json").exists()
    assert not (tmp_path / "extensions").exists()


def test_independent_pmf_original_budget_controls_extension_lower_bound(tmp_path):
    config, system = JobSpec.from_mapping(configuration()).resolve()
    run_resolved(config, system, output=tmp_path)
    with pytest.raises((InputError, RuntimeError), match="exceed the original"):
        extend_run(tmp_path, production_steps=SHORT)
    assert not (tmp_path / "results/extensions.json").exists()
