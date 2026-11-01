"""Input/CLI coverage of the actual split-update bridge using a mock kernel.

This checks the PLUMED command protocol and file/clock ownership. It does not
claim that an installed libplumedKernel was exercised by these tests.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.io import referenced_files
from pyrpmd.workflows.plumed import resolve_plumed_job, run_plumed


class CommandKernel:
    def __init__(self, *, fail_update=None, api=10, stop_after=None):
        self.values = {}
        self.updates = []
        self.calculations = []
        self.fail_update = fail_update
        self.api = api
        self.stop_after = stop_after
        self.finalized = 0
        self.directory = Path.cwd()

    def cmd(self, name, value=None):
        if name == "getApiVersion":
            value[0] = self.api
        elif name == "isEnergyNeeded":
            value[0] = 0
        elif name == "performCalcNoUpdate":
            x = self.values["setPositions"] * self.values["setMDLengthUnits"]
            force = -3.7 * x * self.values["setMDLengthUnits"] / self.values["setMDEnergyUnits"]
            self.values["setForces"][:] += force
            self.bias = 1.85 * np.sum(x * x) / self.values["setMDEnergyUnits"]
            self.calculations.append((self.values["setStep"], len(self.updates)))
        elif name == "getBias":
            value[0] = self.bias
        elif name == "update":
            self.updates.append(self.values["setStep"])
            with Path("COLVAR").open("a") as stream:
                stream.write(str(self.updates[-1]) + "\n")
            if len(self.updates) == self.fail_update:
                raise RuntimeError("injected partial PLUMED update")
            if len(self.updates) == self.stop_after:
                self.values["setStopFlag"][0] = 1
        elif name.startswith("set") or name == "readInputLines":
            self.values[name] = value

    def finalize(self):
        self.finalized += 1


def raw_input(tmp_path, *, inline=False):
    script = "x: POSITION ATOM=1\nb: RESTRAINT ARG=x.x AT=0 KAPPA=3.7\nPRINT ARG=x.x FILE=COLVAR STRIDE=1\n"
    (tmp_path / "bias.dat").write_text(script)
    return {
        "task": "plumed", "temperature": 1 / (3.1 * KB_AU), "beads": 4,
        "native": False, "seed": 13,
        "system": {"symbols": ["H"], "masses": [2.], "geometry": [[.1, .2, -.1]]},
        "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"timestep": .02},
        "sampling": {"equilibration_steps": 2, "production_steps": 8, "sample_interval": 2},
        "plumed": {"input": script} if inline else {"file": "bias.dat"},
        "output": {"checkpoint_steps": 3},
    }


def install_mock(monkeypatch, **kwargs):
    kernels = []
    def factory(**options):
        kernel = CommandKernel(**kwargs)
        kernel.options = options
        kernels.append(kernel)
        return kernel
    monkeypatch.setitem(sys.modules, "plumed", SimpleNamespace(Plumed=factory))
    return kernels


@pytest.mark.parametrize("inline", [False, True])
def test_plumed_input_paths_units_and_default_destination(tmp_path, inline):
    data = raw_input(tmp_path, inline=inline)
    kernel = tmp_path / "libplumedKernel.so"
    kernel.write_bytes(b"test path only")
    data["plumed"]["kernel"] = kernel.name
    config, system = resolve_plumed_job(data, base_directory=tmp_path)
    assert config["task"] == "plumed"
    assert config["dt_au"] == .02
    assert config["beta"] == pytest.approx(3.1)
    assert config["output"]["directory"] == str(tmp_path / "runs" / "plumed")
    assert config["plumed"]["kernel"] == str(kernel)
    if not inline:
        assert config["plumed"]["file"] == str(tmp_path / "bias.dat")
        assert str(tmp_path / "bias.dat") in referenced_files(config)
    assert str(kernel) in referenced_files(config)


def test_input_workflow_updates_exactly_once_and_owns_relative_files(tmp_path, monkeypatch):
    kernels = install_mock(monkeypatch)
    data = raw_input(tmp_path)
    config, system = resolve_plumed_job(data, base_directory=tmp_path)
    original_directory = Path.cwd()
    result = run_plumed(config, system, tmp_path / "run")
    assert Path.cwd() == original_directory
    assert result["status"] == "complete"
    assert result["steps"] == 10
    assert result["samples"] == 4
    assert not result["restart_supported"]
    kernel = kernels[0]
    assert kernel.directory == tmp_path / "run" / "plumed"
    assert kernel.updates == list(range(1, 11))
    assert kernel.finalized == 1
    assert len(kernel.calculations) == 2 * 10 + 4
    assert kernel.values["setTimestep"] == .02
    assert not (tmp_path / "COLVAR").exists()
    assert (tmp_path / "run" / "plumed" / "COLVAR").read_text().splitlines() == [str(i) for i in range(1, 11)]
    with load_checkpoint(result["md_snapshot"], require_integrity=True) as saved:
        assert saved["step"] == 10
        assert float(saved["time"]) == pytest.approx(.2)
        assert not bool(saved["kernel_state_saved"])
        np.testing.assert_array_equal(saved["sample_steps"], [4, 6, 8, 10])
        assert saved["positions"].shape == (4, 4, 1, 3)
        assert np.isfinite(saved["physical_bias"]).all()


@pytest.mark.parametrize("kernel_options,accepted", [({"fail_update": 3}, 2), ({"api": 7}, 0)])
def test_failure_keeps_last_accepted_md_snapshot_and_restores_cwd(tmp_path, monkeypatch, kernel_options, accepted):
    kernels = install_mock(monkeypatch, **kernel_options)
    config, system = resolve_plumed_job(raw_input(tmp_path), base_directory=tmp_path)
    original_directory = Path.cwd()
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        run_plumed(config, system, out)
    assert Path.cwd() == original_directory
    with load_checkpoint(out / "checkpoints" / "plumed-md-state.npz", require_integrity=True) as saved:
        assert int(saved["step"]) == accepted
        assert float(saved["time"]) == pytest.approx(accepted * .02)
        assert not bool(saved["kernel_state_saved"])
    diagnostic = json.loads((out / "results" / "plumed-failure.json").read_text())
    assert diagnostic["status"] == "failed"
    assert diagnostic["last_accepted_step"] == accepted
    assert not diagnostic["restart_supported"]
    assert kernels[0].finalized == 1


def test_plumed_stop_flag_ends_at_a_complete_step(tmp_path, monkeypatch):
    kernels = install_mock(monkeypatch, stop_after=5)
    config, system = resolve_plumed_job(raw_input(tmp_path), base_directory=tmp_path)
    result = run_plumed(config, system, tmp_path / "stop")
    assert result["status"] == "plumed_stop"
    assert result["steps"] == 5
    assert kernels[0].updates == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("change", [
    {"sampling": {"timestep": .01}}, {"sampling": {"replicas": 2}},
    {"path_integral": {}}, {"active_learning": {}}, {"barostat": {}},
    {"umbrella": {}}, {"metadynamics": {}}, {"analysis": {}},
    {"output": {"trajectory_storage": "stream"}}, {"plumed": {"file": True}},
    {"plumed": {"kernel": "missing-kernel.so"}}, {"plumed": {"charges": [True]}},
    {"sampling": {"parents": 2}}, {"plumed": {"input": "duplicate source"}},
    {"potential": {"misspelled": 1}},
])
def test_plumed_preflight_rejects_ignored_or_invalid_controls(tmp_path, change):
    data = raw_input(tmp_path)
    for key, value in change.items():
        if key in data and isinstance(data[key], dict):
            data[key].update(value)
        else:
            data[key] = value
    with pytest.raises(InputError):
        resolve_plumed_job(data, base_directory=tmp_path)


def test_plumed_cli_preflight_run_and_resume_rejection(tmp_path, monkeypatch, capsys):
    from pyrpmd.cli import main
    kernels = install_mock(monkeypatch)
    data = raw_input(tmp_path)
    path = tmp_path / "job.py"
    path.write_text("def build():\n    return " + repr(data) + "\n")
    output = tmp_path / "cli-run"
    assert main(["validate", str(path), "--check-pes", "--output", str(output)]) == 0
    capsys.readouterr()
    assert not kernels  # Read/validation does not initialize or update PLUMED.
    assert main(["run", str(path), "--output", str(output)]) == 0
    capsys.readouterr()
    before = (output / "plumed" / "COLVAR").read_bytes()
    assert main(["resume", str(output)]) != 0
    captured = capsys.readouterr()
    assert "resume is unsupported" in captured.err
    assert (output / "plumed" / "COLVAR").read_bytes() == before
    assert len(kernels) == 1
