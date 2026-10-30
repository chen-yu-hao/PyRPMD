"""NRPMD integration with periodic workflows and portable run identity."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec, validate_nonadiabatic
from pyrpmd.io import fingerprint
from pyrpmd.potentials import PotentialError, make_potential
from pyrpmd.runner import run, run_resolved
from pyrpmd.system import System
from pyrpmd.workflows.drpmd import _physics_hash, run_drpmd
from test_nrpmd_input import job


@pytest.mark.parametrize("boundary", ["periodic", "fixed"])
def test_nrpmd_public_and_resolved_entries_reject_surface_boundaries_before_output(boundary, tmp_path):
    raw = job()
    config, system = JobSpec.from_mapping(raw).resolve()
    if boundary == "periodic":
        raw["system"].update(cell=np.eye(3).tolist(), pbc=[True, True, True])
        changed = replace(system, cell=np.eye(3), pbc=(True, True, True))
    else:
        raw["system"]["fixed_atoms"] = [0]
        changed = replace(system, fixed_atoms=(0,))
    for entry, args in ((run, (raw,)), (run_resolved, (config, changed))):
        destination = tmp_path / entry.__name__
        with pytest.raises(InputError, match="periodic|mobile|fixed"):
            entry(*args, output=destination)
        assert not destination.exists()
    with pytest.raises(ValueError, match="periodic|fixed"):
        run_drpmd(config, changed, tmp_path / "direct")
    assert not (tmp_path / "direct").exists()


def test_matrix_provider_cannot_ignore_periodic_system_before_importing_factory():
    system = System(("H",), [1837.0], [[0, 0, 0]], cell=np.eye(3))
    with pytest.raises(PotentialError, match="cell/PBC contract"):
        make_potential({"provider": "diabatic_python", "factory": "module_that_must_not_import:make"}, system)


@pytest.mark.parametrize("selector", ["provider", "type"])
def test_resolved_matrix_provider_requires_nonadiabatic_before_output(selector, tmp_path):
    config, system = JobSpec.from_mapping(job()).resolve()
    config.pop("nonadiabatic")
    config["potential"][selector] = config["potential"].pop("provider")
    with pytest.raises(InputError, match=r"requires \[nonadiabatic\]"):
        run_resolved(config, system, output=tmp_path / "runner")
    assert not (tmp_path / "runner").exists()
    with pytest.raises(InputError, match=r"requires \[nonadiabatic\]"):
        run_drpmd(config, system, tmp_path / "workflow")
    assert not (tmp_path / "workflow").exists()


@pytest.mark.parametrize("task", ["pcet", "pimd", "diffusion", "site_rate", "surface_rate", "network", "metadynamics"])
def test_nonadiabatic_section_cannot_be_silently_ignored_by_other_workflows(task):
    config = job()
    config["task"] = task
    with pytest.raises(InputError, match="state-selected drpmd"):
        validate_nonadiabatic(config)


@pytest.mark.parametrize("changed", [
    {"initial_state": 1}, {"state_labels": ["lower", "upper"]},
    {"asymptotic_coupling_tolerance": 1e-6},
])
def test_electronic_setup_is_in_manifest_and_drpmd_stream_physics(changed):
    config, system = JobSpec.from_mapping(job()).resolve()
    updated = copy.deepcopy(config)
    updated["nonadiabatic"].update(changed)
    assert fingerprint(updated, system) != fingerprint(config, system)
    assert _physics_hash(updated, system, 2) != _physics_hash(config, system, 2)


def test_raw_system_cell_uses_the_same_default_pbc_in_preflight(tmp_path):
    config, system = JobSpec.from_mapping(job()).resolve()
    boundary = system.to_dict()
    boundary["cell"] = np.eye(3).tolist()
    assert "pbc" not in boundary
    with pytest.raises(InputError, match="nonperiodic"):
        run_resolved(config, boundary, output=tmp_path / "runner")
    assert not (tmp_path / "runner").exists()
    with pytest.raises(ValueError, match="periodic"):
        run_drpmd(config, boundary, tmp_path / "workflow")
    assert not (tmp_path / "workflow").exists()
