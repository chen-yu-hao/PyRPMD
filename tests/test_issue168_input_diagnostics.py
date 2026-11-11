"""Dimensional warnings are actionable without changing atomic-unit inputs."""
import json

import numpy as np
import pytest

from pyrpmd.input import JobSpec, InputError
from pyrpmd.cli import main
from pyrpmd.input_diagnostics import input_diagnostics
from pyrpmd.runner import run
from pyrpmd.system import System


def job():
    return {
        "task": "pimd", "temperature": 300, "beads": 1,
        "system": {"symbols": ["H", "H"], "geometry": [[0, 0, 0], [2, 0, 0]], "geometry_unit": "bohr"},
        "potential": {"provider": "free"}, "dynamics": {"timestep": 0.5},
        "sampling": {"equilibration_steps": 0, "production_steps": 4, "sample_interval": 1},
        "output": {"checkpoint_steps": 2},
    }


def test_numeric_quantities_warn_with_paths_and_conversion_without_reinterpreting(tmp_path):
    raw = job()
    resolved, _ = JobSpec.from_mapping(raw).resolve()
    assert resolved["dt_au"] == 0.5
    assert any("dynamics.timestep" in note and "0.012094 fs" in note for note in resolved["_warnings"])
    assert any("temperature = 300" in note and "K" in note for note in resolved["_warnings"])
    assert not any("beads" in note or "sample_interval" in note for note in resolved["_warnings"])
    summary = run(raw, output=tmp_path / "run")
    assert summary["input_warnings"] == resolved["_warnings"]
    saved = json.loads((tmp_path / "run/results/summary.json").read_text())
    assert saved["input_warnings"] == summary["input_warnings"]


def test_unit_strings_and_quantity_pairs_do_not_warn():
    raw = job()
    raw["temperature"] = "300 K"
    raw["dynamics"]["timestep"] = [0.5, "autime"]
    resolved, _ = JobSpec.from_mapping(raw).resolve()
    assert not any("bare numbers" in note or "interpreted in" in note for note in resolved["_warnings"])


@pytest.mark.parametrize("explicit", [False, True])
def test_noncommensurate_duration_names_both_fields_and_default(explicit):
    raw = job()
    raw["output"].pop("checkpoint_steps")
    if explicit:
        raw["output"]["checkpoint_interval"] = "1 fs"
    with pytest.raises(InputError) as caught:
        JobSpec.from_mapping(raw).resolve()
    text = str(caught.value)
    assert "output.checkpoint_interval" in text and "dynamics.timestep = 0.5" in text
    assert "default '10 fs'" in text if not explicit else "'1 fs'" in text
    assert "integer multiple" in text


@pytest.mark.parametrize("symbols,charge,multiplicity,warn", [
    (["H", "H", "H"], 0, 1, True), (["H", "H", "H"], 0, 2, False),
    (["D", "T"], 0, 1, False), (["H", "H"], 1, 1, True),
    (["He"], 0, 4, True),
])
def test_electron_spin_parity_is_a_frontend_warning(symbols, charge, multiplicity, warn):
    system = System(symbols, np.ones(len(symbols)), np.zeros((len(symbols), 3)), charge=charge, multiplicity=multiplicity)
    notes = input_diagnostics({"system": {"multiplicity": multiplicity}}, {}, system)
    assert any("electron/spin parity" in note for note in notes) is warn


def test_cabi_mismatch_names_both_declared_values():
    from pyrpmd.potentials.cabi import CABIPotential
    from pyrpmd.potentials import PotentialError
    provider = object.__new__(CABIPotential)
    provider.natoms = 3
    provider.metadata = {"atomic_numbers": [1, 1, 1], "state": {"charge": 0, "multiplicity": 2}}
    system = System(["H"] * 3, [1] * 3, np.zeros((3, 3)))
    with pytest.raises(PotentialError, match="system declares 1, provider declares 2"):
        provider.bind_system(system)


def test_site_rate_withheld_summary_publishes_both_unit_spellings(tmp_path):
    from pyrpmd.workflows.site_rate import _aggregate
    config = {"dt_au": 0.1, "sampling": {},
              "recrossing": {"child_steps": 2, "equilibration_steps": 0, "parent_interval_steps": 1}}
    pmf = {"ready": False, "status": "unsupported_pmf", "k_qtst_s": None, "warnings": []}
    result = _aggregate([], pmf, config, tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == result
    for key in ("rate_s", "rate_estimate_s", "rate_standard_error_s", "k_qtst_s"):
        assert saved[key] is None and saved[key + "_inv"] is None
    assert saved["pmf"]["k_qtst_s_inv"] is None
