"""File-restart protocol tests with independent Gaussian history evaluation.

These tests exercise the bridge protocol, not an installed libplumedKernel.
The optional live test in test_plumed_live_workflow.py checks that distinction.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError
from pyrpmd.io import fingerprint
from pyrpmd.plumed import file_restart_plan
from pyrpmd.workflows.plumed import resolve_plumed_job, run_plumed


SCRIPT = ("UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic\n"
          "x: POSITION ATOM=1\n"
          "m: METAD ARG=x.x SIGMA=0.07 HEIGHT=0.03 PACE=2 FILE=HILLS\n"
          "PRINT ARG=x.x,m.bias STRIDE=1 FILE=COLVAR\n")


class FileKernel:
    """Independent, file-backed Gaussian bias with PLUMED's split clock."""

    def __init__(self, *, fail_step=None, stop_step=None, well_tempered=False):
        self.values, self.hills, self.updates = {}, [], []
        self.fail_step, self.stop_step = fail_step, stop_step
        self.well_tempered = well_tempered
        self.finalized = 0
        # Matches the actual PLUMED 2.9 MetaD::update first-update rule, even
        # with setRestart. A restarted kernel must first replay the saved step.
        self.first_update = True

    def cmd(self, name, value=None):
        if name == "getApiVersion":
            value[0] = 10
        elif name == "readInputLines":
            self.values[name] = value
            assert "FLUSH STRIDE=1" in value
            assert value.count("FMT=%25.17g") == 2
            if self.values["setRestart"]:
                self.hills = [tuple(map(float, line.split())) for line in Path("HILLS").read_text().splitlines()
                              if line and not line.startswith("#")]
            else:
                Path("HILLS").write_text("#! FIELDS time x height\n")
                Path("COLVAR").write_text("#! FIELDS time x bias\n")
        elif name == "isEnergyNeeded":
            value[0] = 0
        elif name == "performCalcNoUpdate":
            self.x = float(self.values["setPositions"][0, 0])
            self.bias, force = 0., 0.
            for _, center, height in self.hills:
                delta = self.x - center
                hill = height * np.exp(-.5 * (delta / .07)**2)
                self.bias += hill
                force += hill * delta / .07**2
            self.values["setForces"][0, 0] += force
        elif name == "getBias":
            value[0] = self.bias
        elif name == "update":
            step = self.values["setStep"]
            self.updates.append(step)
            time = step * self.values["setTimestep"]
            if step % 2 == 0 and not self.first_update:
                height = .03 * (np.exp(-self.bias / .5) if self.well_tempered else 1.)
                hill = (time, self.x, height)
                self.hills.append(hill)
                with Path("HILLS").open("a") as stream:
                    stream.write(" ".join(format(number, ".17g") for number in hill) + "\n")
            with Path("COLVAR").open("a") as stream:
                stream.write(" ".join(format(number, ".17g") for number in (time, self.x, self.bias)) + "\n")
            self.first_update = False
            if step == self.fail_step:
                raise RuntimeError("injected failure after writing history")
            if step == self.stop_step:
                self.values["setStopFlag"][0] = 1
        elif name.startswith("set"):
            self.values[name] = value

    def finalize(self):
        self.finalized += 1


def install_kernel(monkeypatch, **options):
    kernels = []

    def factory(**kwargs):
        instance = FileKernel(**options)
        kernels.append(instance)
        return instance

    monkeypatch.setitem(sys.modules, "plumed", SimpleNamespace(Plumed=factory))
    return kernels


def data():
    return {
        "task": "plumed", "temperature": 1 / (3 * KB_AU), "beads": 4,
        "native": False, "seed": 51,
        "system": {"symbols": ["H"], "masses": [2.], "geometry": [[.1, .2, -.1]]},
        "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"timestep": .02},
        "sampling": {"equilibration_steps": 2, "production_steps": 12, "sample_interval": 2},
        "plumed": {"input": SCRIPT, "restart": "files"},
        "output": {"checkpoint_steps": 3},
    }


def read_snapshot(output):
    with load_checkpoint(output / "checkpoints" / "plumed-md-state.npz", require_integrity=True) as saved:
        return {name: saved[name] for name in saved.files}


@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
@pytest.mark.parametrize("interruption", ["failure", "stop"])
@pytest.mark.parametrize("well_tempered", [False, True])
def test_restart_matches_uninterrupted_state_rng_samples_and_history(tmp_path, monkeypatch, thermostat,
                                                                    interruption, well_tempered):
    raw = data()
    raw["sampling"]["thermostat"] = thermostat
    if thermostat == "pile-l":
        raw["sampling"]["integrator"] = "bcocb"
    if well_tempered:
        raw["plumed"]["input"] = SCRIPT.replace("PACE=2", "PACE=2 BIASFACTOR=4 TEMP=300")
    config, system = resolve_plumed_job(raw, base_directory=tmp_path)
    install_kernel(monkeypatch, well_tempered=well_tempered)
    full = tmp_path / "full"
    run_plumed(config, system, full)
    split = tmp_path / "split"
    options = {"fail_step": 8} if interruption == "failure" else {"stop_step": 7}
    first = install_kernel(monkeypatch, well_tempered=well_tempered, **options)
    if interruption == "failure":
        with pytest.raises(RuntimeError, match="after writing history"):
            run_plumed(config, system, split)
        assert int(read_snapshot(split)["step"]) == 6
        failure = json.loads((split / "results" / "plumed-failure.json").read_text())
        assert failure["last_accepted_step"] == 7
        assert failure["checkpoint_step"] == 6
        assert failure["restart_supported"]
    else:
        assert run_plumed(config, system, split)["status"] == "plumed_stop"
        assert int(read_snapshot(split)["step"]) == 7
    assert first[0].finalized == 1
    # Restoration must use embedded checkpoint bytes, not mutable external files.
    (split / "plumed" / "HILLS").write_text("partial or corrupted post-checkpoint append\n")
    (split / "plumed" / "COLVAR").unlink()
    second = install_kernel(monkeypatch, well_tempered=well_tempered)
    result = run_plumed({**config, "resume": True}, system, split)
    assert result["restart_supported"] and result["steps"] == 14
    assert second[0].values["setRestart"] == 1
    assert second[0].updates[:2] == ([6, 7] if interruption == "failure" else [7, 8])
    a, b = read_snapshot(full), read_snapshot(split)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    for name in ("HILLS", "COLVAR"):
        assert (full / "plumed" / name).read_bytes() == (split / "plumed" / name).read_bytes()
    assert len(second[0].hills) == 7  # No duplicated or missing deposition.
    before = (split / "plumed" / "HILLS").read_bytes()
    run_plumed({**config, "resume": True}, system, split)
    assert (split / "plumed" / "HILLS").read_bytes() == before


def test_first_update_failure_restarts_cleanly_from_step_zero(tmp_path, monkeypatch):
    config, system = resolve_plumed_job(data(), base_directory=tmp_path)
    install_kernel(monkeypatch, fail_step=1)
    with pytest.raises(RuntimeError):
        run_plumed(config, system, tmp_path / "run")
    assert int(read_snapshot(tmp_path / "run")["step"]) == 0
    kernels = install_kernel(monkeypatch)
    run_plumed({**config, "resume": True}, system, tmp_path / "run")
    assert kernels[0].values["setRestart"] == 0
    assert kernels[0].updates == list(range(1, 15))


@pytest.mark.parametrize("suffix", [
    "ADAPTIVE=DIFF", "GRID_MIN=-1 GRID_MAX=1", "WALKERS_MPI", "NLIST",
    "CALC_WORK", "CALC_RCT", "ACCELERATION", "FREQUENCY_ADAPTIVE",
    "RESTART=YES", "RESTART=NO", "FMT=%8.3f", "FILE=../HILLS",
])
def test_stateful_or_inconsistent_metad_options_fail_preflight(suffix, tmp_path):
    raw = data()
    script = SCRIPT.replace("FILE=HILLS", suffix if suffix.startswith("FILE=") else "FILE=HILLS " + suffix)
    raw["plumed"]["input"] = script
    with pytest.raises(InputError):
        resolve_plumed_job(raw, base_directory=tmp_path)


@pytest.mark.parametrize("old,new", [
    ("SIGMA=0.07", "SIGMA=nan"), ("SIGMA=0.07", "SIGMA=-0.1"),
    ("HEIGHT=0.03", "HEIGHT=inf"), ("HEIGHT=0.03", "HEIGHT=-0.1"),
    ("PACE=2", "PACE=0"), ("PACE=2", "PACE=2.5"),
    ("PACE=2", "PACE=2 BIASFACTOR=1"), ("PACE=2", "PACE=2 TEMP=0"),
    ("FILE=HILLS", "FILE='HILLS#comment'"),
])
def test_invalid_fixed_hill_values_fail_before_kernel_construction(old, new):
    with pytest.raises(ValueError):
        file_restart_plan(SCRIPT.replace(old, new))


@pytest.mark.parametrize("action", [
    "RESTART", "RESTART NO", "INCLUDE FILE=extra.dat", "LOAD FILE=plugin.so",
    "h: HISTOGRAM ARG=x.x GRID_MIN=-1 GRID_MAX=1 GRID_BIN=20 BANDWIDTH=0.1",
    "a: AVERAGE ARG=x.x", "t: TIME", "m: OPES_METAD ARG=x.x PACE=2 BARRIER=10",
    "g: GROUP NDX_FILE=index.ndx NDX_GROUP=atoms", "ENERGY",
])
def test_unverified_actions_and_external_dependencies_are_rejected(action):
    with pytest.raises(ValueError):
        file_restart_plan(SCRIPT + action + "\n")


def test_multiline_fixed_hills_are_normalized_and_full_precision(tmp_path):
    program = ("d: DISTANCE ATOMS=1,2 # comment\n"
               "m: METAD ...\n ARG=d SIGMA=0.1 HEIGHT=0.01\n PACE=5\n ... METAD\n"
               "PRINT ARG=d,m.bias FILE=history/COLVAR\nFLUSH STRIDE=10\n")
    plan = file_restart_plan(program)
    assert plan.files == ("HILLS", "history/COLVAR")
    assert "m: METAD ARG=d SIGMA=0.1 HEIGHT=0.01 PACE=5 FILE=HILLS FMT=%25.17g" in plan.input_lines
    assert plan.input_lines.count("FLUSH") == 1
    assert plan.input_lines.endswith("FLUSH STRIDE=1\n")


@pytest.mark.parametrize("fmt,accepted", [
    ("%25.17g", True), ("%30.17g", True), ("%26.17e", True),
    # Real PLUMED 2.9 METAD writes these without separators between fields.
    ("%.17g", False), ("%.17e", False), ("%24.17g", False), ("%25.17e", False),
    ("%25.16g", False), ("%14.9f", False),
])
def test_restart_formats_must_keep_hills_fields_separated(fmt, accepted):
    program = f"d: DISTANCE ATOMS=1,2\nm: METAD ARG=d SIGMA=0.1 HEIGHT=0.01 PACE=5 FMT={fmt}\n"
    if accepted:
        assert f"FMT={fmt}" in file_restart_plan(program).input_lines
    else:
        with pytest.raises(ValueError, match="padded round-trip FMT"):
            file_restart_plan(program)


@pytest.mark.parametrize("invalid", [True, False, "auto", 1, None])
def test_invalid_restart_setting_is_not_silently_ignored(invalid, tmp_path):
    raw = data()
    raw["plumed"]["restart"] = invalid
    with pytest.raises(InputError, match="plumed.restart"):
        resolve_plumed_job(raw, base_directory=tmp_path)


def test_restart_is_opt_in_and_changes_only_new_run_identity(tmp_path):
    raw = data()
    enabled, system = resolve_plumed_job(raw, base_directory=tmp_path)
    del raw["plumed"]["restart"]
    legacy, _ = resolve_plumed_job(raw, base_directory=tmp_path)
    assert "restart" not in legacy["plumed"]
    assert fingerprint(enabled, system) != fingerprint(legacy, system)


@pytest.mark.parametrize("change", ["input", "temperature", "seed", "dt", "sample_interval"])
def test_changed_scientific_input_is_rejected_before_files_or_kernel(tmp_path, monkeypatch, change):
    config, system = resolve_plumed_job(data(), base_directory=tmp_path)
    kernels = install_kernel(monkeypatch)
    output = tmp_path / "run"
    run_plumed(config, system, output)
    config = deepcopy(config)
    if change == "input":
        config["plumed"]["input"] += "# changed input\n"
    elif change == "temperature":
        config["beta"] *= 2
    elif change == "dt":
        config["dt_au"] *= 2
    elif change == "sample_interval":
        config["sampling"]["sample_interval"] += 1
    else:
        config["seed"] += 1
    before = (output / "plumed" / "HILLS").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        run_plumed({**config, "resume": True}, system, output)
    assert len(kernels) == 1
    assert (output / "plumed" / "HILLS").read_bytes() == before


@pytest.mark.parametrize("corrupt", ["step", "sample_steps", "rng", "file_manifest", "q", "file_0"])
def test_invalid_checkpoint_is_rejected_before_any_output_changes(tmp_path, monkeypatch, corrupt):
    config, system = resolve_plumed_job(data(), base_directory=tmp_path)
    kernels = install_kernel(monkeypatch)
    output = tmp_path / "run"
    run_plumed(config, system, output)
    payload = read_snapshot(output)
    if corrupt == "step":
        payload[corrupt] = np.asarray(-1)
    elif corrupt == "sample_steps":
        payload[corrupt][0] += 1
    elif corrupt == "rng":
        payload[corrupt] = "{}"
    elif corrupt == "file_manifest":
        records = json.loads(str(payload[corrupt]))
        records[0]["path"] = "../elsewhere"
        payload[corrupt] = json.dumps(records)
    elif corrupt == "q":
        payload[corrupt][0, 0, 0] = np.nan
    else:
        payload[corrupt] = payload[corrupt].astype(np.int64)
    atomic_savez(output / "checkpoints" / "plumed-md-state.npz", kind="PyRPMD.plumed-file-state.v1", **payload)
    before = (output / "plumed" / "HILLS").read_bytes()
    with pytest.raises(ValueError):
        run_plumed({**config, "resume": True}, system, output)
    assert len(kernels) == 1
    assert (output / "plumed" / "HILLS").read_bytes() == before


def test_restart_output_symlink_is_rejected_before_file_restoration(tmp_path, monkeypatch):
    config, system = resolve_plumed_job(data(), base_directory=tmp_path)
    install_kernel(monkeypatch)
    output = tmp_path / "run"
    run_plumed(config, system, output)
    target = tmp_path / "external"
    target.write_text("untouched\n")
    path = output / "plumed" / "HILLS"
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("platform does not permit symlinks")
    with pytest.raises(ValueError, match="symlinks"):
        run_plumed({**config, "resume": True}, system, output)
    assert target.read_text() == "untouched\n"


def test_kernel_that_changes_bias_during_restore_clock_is_rejected(tmp_path, monkeypatch):
    config, system = resolve_plumed_job(data(), base_directory=tmp_path)
    install_kernel(monkeypatch, stop_step=6)
    output = tmp_path / "run"
    run_plumed(config, system, output)
    checkpoint = output / "checkpoints" / "plumed-md-state.npz"
    before = checkpoint.read_bytes()

    class IncompatibleKernel(FileKernel):
        def cmd(self, name, value=None):
            super().cmd(name, value)
            if name == "readInputLines" and self.values["setRestart"]:
                self.first_update = False  # Would deposit the saved hill twice.

    monkeypatch.setitem(sys.modules, "plumed", SimpleNamespace(Plumed=IncompatibleKernel))
    with pytest.raises(RuntimeError, match="replay changed its bias"):
        run_plumed({**config, "resume": True}, system, output)
    assert checkpoint.read_bytes() == before


def test_cli_resume_uses_verified_file_checkpoint(tmp_path, monkeypatch, capsys):
    from pyrpmd.cli import main

    job = tmp_path / "job.py"
    job.write_text("def build():\n    return " + repr(data()) + "\n")
    output = tmp_path / "cli"
    install_kernel(monkeypatch, fail_step=8)
    assert main(["run", str(job), "--output", str(output)]) != 0
    capsys.readouterr()
    install_kernel(monkeypatch)
    assert main(["resume", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 14
    assert int(read_snapshot(output)["step"]) == 14
