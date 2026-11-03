"""One explicit PIMD route without changing historical scientific identities."""
from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from pyrpmd.input import JobSpec
from pyrpmd.io import fingerprint
from pyrpmd.propagation import scientific_config


def _job(task="pimd", engine=None):
    # This fixture and the literal fingerprints below were recorded before
    # T16b. Keep identity coverage independent of optional Torch availability.
    data = {
        "task": task, "temperature": "300 K", "beads": 4, "seed": 17,
        "system": {"symbols": ["H", "He"], "masses": [2., 5.],
                   "geometry": [[.1, -.2, .3], [.4, .5, -.1]],
                   "geometry_unit": "bohr"},
        "potential": ({"provider": "harmonic", "params": {"k": .2}}
                      if task == "pimd" else {"provider": "free"}),
        "dynamics": {"timestep": .04, "integrator": "cayley-bcb"},
        "sampling": {"thermostat": "pile-l", "integrator": "bcocb", "gamma_centroid": .7,
                     "equilibration_steps": 3, "production_steps": 24,
                     "sample_interval": 2, "replicas": 2},
        "output": {"checkpoint_steps": 4, "chunk_frames": 3},
    }
    if engine is not None:
        data["sampling"]["engine"] = engine
    if task == "diffusion":
        data["diffusion"] = {"atom_indices": [0], "axes": [0, 1, 2],
                             "bootstrap": 0, "max_lag": 3}
    return data


@pytest.mark.parametrize("task,engine,expected", [
    ("pimd", None, "e9a18c8abc32a747a9b1269f6550c5c8a7f0b06fef12e62c4734caffd2bf5a02"),
    ("diffusion", None, "d1527ad6d2a20effe17f7800146c8c576dd752d2a3c74728052603910c8a7fff"),
    ("pimd", "legacy", "b9c6efa25cbedea5e3dba26db900c35fcdb5266ec737758133be3a6ef526b159"),
    ("pimd", "native", "7ce7e8de1b9c0aa0cdc416150572ce43514f4f839077ee8ba1389aae01c40838"),
])
def test_historical_scientific_identities_survive_shared_dispatch(task, engine, expected):
    from pyrpmd.workflows.materials import resolve_materials
    from pyrpmd.workflows.pimd import select_engine

    raw = _job(task, engine)
    before = copy.deepcopy(raw)
    config, system = JobSpec.from_mapping(raw).resolve()
    resolved_before = copy.deepcopy(config)
    assert select_engine(config).name == (engine or "legacy")
    resolve_materials(config, system)
    assert fingerprint(scientific_config(config), system) == expected
    assert raw == before
    assert config == resolved_before
    assert ("engine" in config["sampling"]) == (engine is not None)
    assert "device" not in config["sampling"]
    assert "normal_mode_layout" not in config["sampling"]


@pytest.mark.parametrize("config", [{}, {"sampling": {}}, {"sampling": {"engine": "legacy"}}])
def test_selecting_default_does_not_insert_scientific_defaults(config):
    from pyrpmd.workflows.pimd import select_engine

    before = copy.deepcopy(config)
    assert select_engine(config).name == "legacy"
    assert config == before


@pytest.mark.parametrize("value", [None, True, 1, "auto", "Legacy", "", [], {}])
def test_invalid_engine_is_rejected_before_workflow_io(tmp_path, monkeypatch, value):
    from pyrpmd import potentials
    from pyrpmd.runner import run_resolved
    from pyrpmd.workflows.materials import resolve_materials, run_materials
    from pyrpmd.workflows.pimd import select_engine

    config, system = JobSpec.from_mapping(_job()).resolve()
    config["sampling"]["engine"] = value
    output = tmp_path / "not-created"

    def unexpected(*args, **kwargs):
        raise AssertionError("an invalid engine constructed a potential")

    monkeypatch.setattr(potentials, "make_potential", unexpected)
    for name, operation in (
        ("select_engine", lambda: select_engine(config)),
        ("resolve_materials", lambda: resolve_materials(config, system)),
        ("run_materials", lambda: run_materials(config, system, output)),
        ("run_resolved", lambda: run_resolved(config, system, output=output)),
    ):
        with pytest.raises(ValueError, match=r"sampling\.engine"):
            operation()
        assert not output.exists(), name


@pytest.mark.parametrize("sampling", [None, [], "native", 1])
def test_selector_rejects_non_table_sampling(sampling):
    from pyrpmd.workflows.pimd import select_engine

    with pytest.raises(ValueError, match="sampling must be a table"):
        select_engine({"sampling": sampling})


def test_selecting_routes_does_not_import_execution_backends():
    # A clean interpreter is essential: unrelated tests may already have
    # imported an optional backend. Selection itself must remain available
    # on hosts that can only inspect or prepare a saved run.
    code = """
import sys
from pyrpmd.workflows.pimd import select_engine

for name in ("legacy", "native", "torch"):
    route = select_engine({"sampling": {"engine": name}})
    assert route.name == name
for name in ("torch", "pyrpmd._thermal", "pyrpmd.workflows.native_pimd",
             "pyrpmd.workflows.torch_materials"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("engine,module,resolver", [
    ("native", "native_pimd", "resolve_native_pimd"),
    ("torch", "torch_materials", "resolve_torch_materials"),
])
def test_selected_backend_failure_never_falls_back(monkeypatch, engine, module, resolver):
    from importlib import import_module
    from pyrpmd.workflows import materials

    config, system = JobSpec.from_mapping(_job()).resolve()
    config["sampling"]["engine"] = engine
    selected = import_module("pyrpmd.workflows." + module)

    def unavailable(*args, **kwargs):
        raise RuntimeError("selected execution backend unavailable")

    def wrong_backend(*args, **kwargs):
        raise AssertionError("explicit engine silently fell back to legacy")

    monkeypatch.setattr(selected, resolver, unavailable)
    monkeypatch.setattr(materials, "_resolve_legacy_materials", wrong_backend)
    with pytest.raises(RuntimeError, match="selected execution backend unavailable"):
        materials.resolve_materials(config, system)


def test_native_route_rejects_diffusion_reanalysis_before_io(tmp_path):
    from pyrpmd.workflows.materials import reanalyze_diffusion

    config, system = JobSpec.from_mapping(_job("diffusion")).resolve()
    config["sampling"]["engine"] = "native"
    with pytest.raises(ValueError, match="native|diffusion"):
        reanalyze_diffusion(config, system, tmp_path / "absent")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["resolve", "run", "check"])
def test_cached_route_cannot_execute_under_a_changed_engine_identity(tmp_path, monkeypatch, operation):
    from pyrpmd.workflows import native_pimd
    from pyrpmd.workflows.pimd import select_engine

    config, system = JobSpec.from_mapping(_job(engine="native")).resolve()
    route = select_engine(config)
    config["sampling"]["engine"] = "legacy"
    output = tmp_path / "not-created"

    def unexpected(*args, **kwargs):
        raise AssertionError("a stale route reached the wrong numerical backend")

    monkeypatch.setattr(native_pimd, "resolve_native_pimd", unexpected)
    monkeypatch.setattr(native_pimd, "run_native_pimd", unexpected)
    monkeypatch.setattr(native_pimd, "_check_native_pimd", unexpected)
    args = {"resolve": (config, system), "run": (config, system, output),
            "check": (config, system, object())}[operation]
    with pytest.raises(ValueError, match=r"sampling\.engine.*selected"):
        getattr(route, operation)(*args)
    assert not output.exists()


def test_native_route_check_validates_controls_before_constructing_engine(monkeypatch):
    from pyrpmd.workflows import native_pimd
    from pyrpmd.workflows.pimd import select_engine

    config, system = JobSpec.from_mapping(_job(engine="native")).resolve()
    config["sampling"]["thermostat"] = "none"

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid native controls constructed an engine")

    monkeypatch.setattr(native_pimd, "_engine", unexpected)
    with pytest.raises(ValueError, match="sampling thermostat"):
        select_engine(config).check(config, system, object())
