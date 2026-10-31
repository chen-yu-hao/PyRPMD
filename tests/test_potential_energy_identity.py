"""Never concatenate historical ASE energies with force-consistent streams."""

import copy
import hashlib
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.io import fingerprint, jsonable, write_json
from pyrpmd.potentials import energy_convention_identity
from pyrpmd.propagation import scientific_config
from pyrpmd.reactions import path_identity
from pyrpmd.system import System
from pyrpmd.workflows.common import _legacy_fingerprint, fingerprint_matches
from pyrpmd.workflows.fes import _fingerprint


def _system():
    return System(("H",), np.array([1837.]), np.zeros((1, 3)))


def _legacy_digest(config, system, *, stream=None):
    # The exact historical serialization, independently of the new convention
    # selector. These inputs have no implicit periodic CV geometry convention.
    omitted = {"parallel", "output", "resume", "_resume", "_input_path",
               "_base_directory", "_warnings"}
    if stream is None:
        value = {"config": {k: v for k, v in config.items() if k not in omitted},
                 "system": system.to_dict()}
        raw = json.dumps(jsonable(value), sort_keys=True, allow_nan=False)
    else:
        physics = {k: v for k, v in scientific_config(config).items()
                   if k not in omitted | {"analysis"}}
        value = {"schema": 1, "physics": physics,
                 "sampling_measure": config.get("analysis", {}).get("measure", "cv_density"),
                 "system": system.to_dict(), "stream": stream}
        if "flux_axis" in config.get("analysis", {}):
            value["sampling_flux_axis"] = config["analysis"]["flux_axis"]
        raw = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("provider", ["ase", "mace", "nequip", "fairchem"])
@pytest.mark.parametrize("selector", ["provider", "type"])
def test_ase_run_stream_and_network_path_identities_change_without_loading_models(provider, selector):
    config = {"task": "pimd", "beads": 2, "beta": 100.,
              "potential": {selector: provider, "factory": "not_imported:factory"}}
    system = _system()
    original = copy.deepcopy(config)
    assert energy_convention_identity(config) == "ase-force-consistent-v1"
    assert fingerprint(config, system) != _legacy_digest(config, system)
    assert _fingerprint(config, system, {"replica": 0}) != _legacy_digest(config, system, stream={"replica": 0})
    old_path = {"format": "PyRPMD.network.path.v1", "entrance": "a",
                "transition_state": "ts", "channel": "b", "config": config,
                "system": system.to_dict()}
    old_path_digest = hashlib.sha256(json.dumps(jsonable(old_path), sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert path_identity(config, system, entrance="a", transition_state="ts", channel="b") != old_path_digest
    assert config == original


@pytest.mark.parametrize("provider", ["builtin", "free", "python", "torch", "deepmd",
                                      "cabi", "native", "tblite", "xtb", "pyscf", "ipi", "socket", "rpmdrate"])
def test_non_ase_historical_identities_are_unchanged(provider):
    config = {"task": "pimd", "beads": 2, "beta": 100.,
              "potential": {"provider": provider,
                            "params": {"unrelated": {"provider": "ase"}}}}
    system = _system()
    assert energy_convention_identity(config) is None
    assert fingerprint(config, system) == _legacy_digest(config, system)
    assert _fingerprint(config, system, {"replica": 0}) == _legacy_digest(config, system, stream={"replica": 0})


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
@pytest.mark.parametrize("provider", ["ase", "mace", "nequip", "fairchem", "builtin", "python"])
def test_fes_alias_migration_preserves_the_selected_energy_convention(method, provider):
    config = {"task": "fes", "fes": {"method": method}, "beads": 2, "beta": 100.,
              "potential": {"provider": provider, "factory": "not_imported:factory"}}
    historical = copy.deepcopy(config)
    historical.pop("fes")
    historical["task"] = "fes" if method == "umbrella" else method
    system, stream = _system(), {"replica": 0}
    current = _fingerprint(config, system, stream)
    assert current == _fingerprint(historical, system, stream)
    assert fingerprint_matches(current, historical, system, stream)
    # The current main branch still writes schema 1, including its selected
    # energy convention. Migrating its task spelling must remain possible.
    assert fingerprint_matches(_legacy_fingerprint(historical, system, stream),
                               config, system, stream)

    # Both historical serializations before the ASE energy change are exact:
    # schema 1 used aliases, and schema 2 wrapped a canonical task='fes' hash.
    old_schema1 = _legacy_digest(historical, system, stream=stream)
    old_canonical = copy.deepcopy(config)
    old_canonical.pop("fes")
    old_schema2 = hashlib.sha256(json.dumps({
        "schema": 2, "method": method,
        "physics_fingerprint": _legacy_digest(old_canonical, system, stream=stream),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    unchanged_energy = energy_convention_identity(config) is None
    assert fingerprint_matches(old_schema1, config, system, stream) == unchanged_energy
    assert fingerprint_matches(old_schema2, config, system, stream) == unchanged_energy
    assert (current == old_schema2) == unchanged_energy


def test_resolved_network_uses_effective_path_potentials():
    config = {"task": "network", "potential": {}, "network": {"jobs": [
        {"config": {"task": "rate", "potential": {"provider": "builtin"}}},
        {"config": {"task": "rate", "potential": {"provider": "ase"}}},
    ]}}
    assert energy_convention_identity(config) == "ase-force-consistent-v1"
    assert fingerprint(config, _system()) != _legacy_digest(config, _system())
    config["network"]["jobs"].pop()
    assert energy_convention_identity(config) is None
    assert fingerprint(config, _system()) == _legacy_digest(config, _system())


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
@pytest.mark.parametrize("schema_version", [1, 2])
def test_fes_task_migration_cannot_accept_old_ase_energy_convention(tmp_path, method, schema_version):
    from pyrpmd.input import InputError
    from pyrpmd.runner import _initialize_output
    from pyrpmd.workflows.common import fingerprint_matches

    system = _system()
    legacy = {
        "task": "fes" if method == "umbrella" else method,
        "beads": 2, "beta": 100.,
        "potential": {"provider": "ase", "factory": "not_imported:factory"},
    }
    current = {**legacy, "task": "fes", "fes": {"method": method}}
    saved_config = legacy if schema_version == 1 else scientific_config(legacy)
    manifest = {
        "schema": "pyrpmd.run", "schema_version": schema_version,
        "physics_fingerprint": _legacy_digest(saved_config, system), "files": {},
    }
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "input/resolved.json", legacy)
    write_json(tmp_path / "input/system.json", system.to_dict())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    message = ("Legacy saved input does not match" if schema_version == 1
               else "Cannot resume with changed scientific settings")
    with pytest.raises(InputError, match=message):
        _initialize_output(current, system, tmp_path, resume=True)
    assert {path: path.read_bytes() for path in before} == before
    stream = {"replica": 0}
    assert not fingerprint_matches(
        _legacy_digest(legacy, system, stream=stream), current, system, stream,
    )


@pytest.fixture(params=["pimd", "diffusion"])
def ase_run(tmp_path, request):
    pytest.importorskip("ase")
    from pyrpmd.runner import run

    raw = {
        "task": request.param, "temperature": "300 K", "beads": 2, "seed": 612,
        "native": False,
        "system": {"symbols": ["Cu", "Cu"], "geometry": [[0., 0., 0.], [2.6, 0., 0.]],
                   "geometry_unit": "angstrom", "cell": np.eye(3) * 10.,
                   "cell_unit": "angstrom", "pbc": [True] * 3},
        "potential": {"provider": "ase", "factory": "ase.calculators.emt:EMT"},
        "dynamics": {"timestep": "0.05 fs", "loop_backend": "python"},
        "sampling": {"equilibration_steps": 2, "production_steps": 6,
                     "sample_interval": 1, "replicas": 1, "thermostat": "pile-l"},
        "output": {"trajectory_storage": "memory", "checkpoint_steps": 2},
    }
    if request.param == "diffusion":
        raw["diffusion"] = {"atom_indices": [0, 1], "axes": [0, 1, 2], "bootstrap": 0}
    directory = tmp_path / "ase-run"
    result = run(raw, output=directory)
    config = json.loads((directory / "input/resolved.json").read_text())
    system = System.from_dict(json.loads((directory / "input/system.json").read_text()))
    return directory, config, system, result


def test_current_ase_run_resumes_exactly(ase_run):
    from pyrpmd.runner import resume_run

    directory, config, system, first = ase_run
    checkpoint = directory / "checkpoints/materials_0000.npz"
    original = checkpoint.read_bytes()
    repeated = resume_run(directory, execution={"block_steps": 3})
    key = "diffusion" if config["task"] == "diffusion" else "estimators"
    assert repeated[key] == first[key]
    assert checkpoint.read_bytes() == original


def test_legacy_ase_manifest_is_rejected_before_constructing_a_pes(ase_run, monkeypatch):
    from pyrpmd.potentials.optional import ASEPotential
    from pyrpmd.runner import analyze_run, resume_run

    directory, config, system, _ = ase_run
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["physics_fingerprint"] = _legacy_digest(scientific_config(config), system)
    manifest_path.write_text(json.dumps(manifest))
    before = manifest_path.read_bytes()
    monkeypatch.setattr(ASEPotential, "__init__", lambda *a, **k: pytest.fail("old run constructed a PES"))
    with pytest.raises(RuntimeError, match="InputError: Cannot resume with changed scientific settings"):
        resume_run(directory)
    assert manifest_path.read_bytes() == before
    assert analyze_run(directory) == json.loads((directory / "results/summary.json").read_text())


def test_legacy_ase_stream_is_rejected_even_with_current_manifest(ase_run, monkeypatch):
    from pyrpmd.potentials.optional import ASEPotential
    from pyrpmd.runner import analyze_run, resume_run

    directory, config, system, _ = ase_run
    checkpoint = directory / "checkpoints/materials_0000.npz"
    with load_checkpoint(checkpoint, require_integrity=True) as saved:
        payload = {name: np.array(saved[name], copy=True) for name in saved.files}
        kind = saved.envelope["kind"]
    physics = copy.deepcopy(config)
    physics.pop("diffusion", None)
    payload["fingerprint"] = _legacy_digest(physics, system, stream={"materials_replica": 0})
    atomic_savez(checkpoint, kind=kind, **payload)
    original = checkpoint.read_bytes()
    monkeypatch.setattr(ASEPotential, "__init__", lambda *a, **k: pytest.fail("old stream constructed a PES"))
    with pytest.raises(ValueError, match="materials checkpoint physics differs"):
        resume_run(directory)
    assert checkpoint.read_bytes() == original
    if config["task"] == "diffusion":
        with pytest.raises(RuntimeError, match="materials checkpoint physics differs"):
            analyze_run(directory, recompute=True)
    assert analyze_run(directory) == json.loads((directory / "results/summary.json").read_text())
