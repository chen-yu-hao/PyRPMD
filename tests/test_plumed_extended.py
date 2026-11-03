"""Independent references for the extended PLUMED workflow."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError
from pyrpmd.plumed_extended import (
    ExtendedPlumedBias,
    configure_multiple_walkers,
    plumed_atom_map,
    validate_managed_output_paths,
)
from pyrpmd.workflows.plumed import resolve_plumed_job, run_plumed


class QuadraticKernel:
    """Analytic PLUMED-like kernel, W=k*sum(x^2)/2 in bridge units."""

    def __init__(self, k=2.5, stop_after=None):
        self.k = k
        self.stop_after = stop_after
        self.values = {}
        self.commands = []
        self.updates = 0
        self.finalized = 0

    def cmd(self, name, value=None):
        self.commands.append((name, value))
        if name == "getApiVersion":
            value[0] = 10
        elif name == "isEnergyNeeded":
            value[0] = 0
        elif name == "performCalcNoUpdate":
            positions = self.values["setPositions"]
            self.values["setForces"][:] = -self.k * positions
            self.bias = 0.5 * self.k * float(np.sum(positions * positions))
        elif name == "getBias":
            value[0] = self.bias
        elif name == "update":
            self.updates += 1
            with Path("STATE").open("a", encoding="utf-8") as stream:
                stream.write(f"{int(self.values['setStep'])}\n")
            if self.stop_after is not None and self.updates == self.stop_after:
                self.values["setStopFlag"][0] = 1
        elif name.startswith("set") or name == "readInputLines":
            self.values[name] = value

    def finalize(self):
        self.finalized += 1


def test_bead_mode_matches_analytic_force_and_finite_difference():
    kernel = QuadraticKernel(k=1.7)
    masses = np.array([2.0, 3.0])
    q = np.array(
        [
            [[0.2, -0.1, 0.3], [0.4, 0.0, -0.2]],
            [[-0.3, 0.5, 0.1], [0.2, -0.4, 0.6]],
            [[0.1, 0.2, -0.5], [-0.2, 0.3, 0.4]],
        ]
    )
    with ExtendedPlumedBias(
        "x: POSITION ATOM=1\n",
        masses,
        beta=4.0,
        nbeads=3,
        dt=0.02,
        mode="beads",
        _kernel=kernel,
    ) as bridge:
        result = bridge.evaluate(q)
        expected_energy = 0.5 * kernel.k * np.sum(q * q)
        assert result.energy == pytest.approx(expected_energy)
        assert result.physical_energy == pytest.approx(expected_energy)
        np.testing.assert_allclose(result.forces, -kernel.k * q)
        np.testing.assert_allclose(
            result.centroid_forces, np.sum(-kernel.k * q, axis=0)
        )
        assert kernel.values["setNatoms"] == 6
        np.testing.assert_array_equal(
            kernel.values["setMasses"], np.tile(masses, 3)
        )

        epsilon = 1.0e-6
        index = (1, 0, 2)
        plus, minus = q.copy(), q.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        derivative = (
            bridge.evaluate(plus).energy - bridge.evaluate(minus).energy
        ) / (2 * epsilon)
        assert result.forces[index] == pytest.approx(
            -derivative, rel=2e-9, abs=2e-9
        )


def test_centroid_mode_retains_sum_beads_hamiltonian_scaling():
    kernel = QuadraticKernel(k=1.25)
    q = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    with ExtendedPlumedBias(
        "x: POSITION ATOM=1\n",
        [1.0],
        3.0,
        2,
        0.1,
        mode="centroid",
        _kernel=kernel,
    ) as bridge:
        result = bridge.evaluate(q)
    physical = 0.5 * kernel.k
    assert result.physical_energy == pytest.approx(physical)
    assert result.energy == pytest.approx(2 * physical)
    np.testing.assert_allclose(result.forces[:, 0, 0], [-kernel.k, -kernel.k])


def test_bead_atom_map_is_bead_major_and_one_based():
    assert plumed_atom_map(2, 2, "beads") == [
        {"plumed_atom": 1, "source": "bead", "bead": 0, "atom": 0},
        {"plumed_atom": 2, "source": "bead", "bead": 0, "atom": 1},
        {"plumed_atom": 3, "source": "bead", "bead": 1, "atom": 0},
        {"plumed_atom": 4, "source": "bead", "bead": 1, "atom": 1},
    ]


def test_multiple_walker_injection_handles_metad_opes_and_ves():
    script = (
        "d: DISTANCE ATOMS=1,2\n"
        "m: METAD ...\n"
        "  ARG=d SIGMA=0.1 HEIGHT=0.01 PACE=5\n"
        "... METAD\n"
        "o: OPT_AVERAGED_SGD BIAS=v STRIDE=5 STEPSIZE=0.1\n"
    )
    files = configure_multiple_walkers(
        script, walkers="files", count=4, walker_id=2, read_stride=7
    )
    assert (
        "WALKERS_N=4 WALKERS_ID=2 WALKERS_DIR=../shared "
        "WALKERS_RSTRIDE=7" in files
    )
    mpi = configure_multiple_walkers(
        script, walkers="mpi", count=4, walker_id=2
    )
    assert "WALKERS_MPI" in mpi
    assert "MULTIPLE_WALKERS" in mpi
    with pytest.raises(ValueError, match="owned"):
        configure_multiple_walkers(
            script.replace("PACE=5", "PACE=5 WALKERS_ID=0"),
            walkers="files",
            count=4,
            walker_id=2,
        )


def test_multi_sim_communicator_is_bound_before_kernel_init():
    kernel = QuadraticKernel()
    replica_comm = object()
    multi_comm = object()
    with ExtendedPlumedBias(
        "x: POSITION ATOM=1\n",
        [1.0],
        2.0,
        1,
        0.1,
        replica_comm=replica_comm,
        multi_sim_comm=multi_comm,
        _kernel=kernel,
    ):
        pass
    names = [name for name, _ in kernel.commands]
    assert names.index("setMPIComm") < names.index("init")
    assert names.index("setMPImultiSimComm") < names.index("init")
    assert kernel.values["setMPIComm"] is replica_comm
    assert kernel.values["setMPImultiSimComm"] is multi_comm


def test_managed_paths_reject_escape_and_windows_paths():
    for value in ("../HILLS", "/tmp/HILLS", "C:\\tmp\\HILLS"):
        with pytest.raises(ValueError, match="normalized relative path"):
            validate_managed_output_paths(f"PRINT ARG=x FILE={value}\n")


def raw_input(*, mode="centroid", replicas=1, restart=False, walkers=None):
    script = (
        "x: POSITION ATOM=1\n"
        "b: RESTRAINT ARG=x.x AT=0 KAPPA=1.5\n"
        "PRINT ARG=x.x,b.bias FILE=STATE STRIDE=1\n"
    )
    plumed = {"input": script, "mode": mode}
    if restart:
        plumed["restart"] = "files"
    if walkers is not None:
        plumed["walkers"] = walkers
    return {
        "task": "plumed",
        "temperature": 1 / (3.1 * KB_AU),
        "beads": 2,
        "native": False,
        "seed": 19,
        "system": {
            "symbols": ["H"],
            "masses": [2.0],
            "geometry": [[0.1, 0.2, -0.1]],
        },
        "potential": {"provider": "harmonic", "params": {"k": 0.2}},
        "dynamics": {"timestep": 0.02},
        "sampling": {
            "replicas": replicas,
            "equilibration_steps": 1,
            "production_steps": 3,
            "sample_interval": 1,
        },
        "plumed": plumed,
        "output": {"checkpoint_steps": 2},
    }


def install_mock(monkeypatch, stops):
    kernels = []
    stops = list(stops)

    def factory(**options):
        kernel = QuadraticKernel(stop_after=stops.pop(0) if stops else None)
        kernel.options = options
        kernels.append(kernel)
        return kernel

    monkeypatch.setitem(sys.modules, "plumed", SimpleNamespace(Plumed=factory))
    return kernels


def checkpoint_payload(path):
    with load_checkpoint(path, require_integrity=True) as saved:
        return {name: saved[name] for name in saved.files}


def test_independent_replicas_get_private_kernels_and_directories(tmp_path, monkeypatch):
    kernels = install_mock(monkeypatch, [None, None])
    config, system = resolve_plumed_job(
        raw_input(replicas=2), base_directory=tmp_path
    )
    result = run_plumed(config, system, tmp_path / "run")
    assert result["replicas"] == 2
    assert result["walkers"] == "independent"
    assert [item["replica"] for item in result["replica_results"]] == [0, 1]
    assert len(kernels) == 2
    for replica in range(2):
        directory = tmp_path / "run" / "plumed" / f"replica-{replica:04d}"
        assert directory.is_dir()
        assert (directory / "STATE").read_text().splitlines() == [
            "1",
            "2",
            "3",
            "4",
        ]


def test_bead_workflow_writes_explicit_atom_map(tmp_path, monkeypatch):
    install_mock(monkeypatch, [None])
    config, system = resolve_plumed_job(
        raw_input(mode="beads"), base_directory=tmp_path
    )
    result = run_plumed(config, system, tmp_path / "beads")
    assert result["mode"] == "beads"
    mapping = json.loads(
        (tmp_path / "beads" / "results" / "plumed-atom-map.json").read_text()
    )
    assert mapping["ordering"] == "bead-major"
    assert [entry["plumed_atom"] for entry in mapping["map"]] == [1, 2]


def test_restart_matches_uninterrupted_rng_trajectory_and_files(tmp_path, monkeypatch):
    kernels = install_mock(monkeypatch, [None, 2, None])
    reference_config, system = resolve_plumed_job(
        raw_input(restart=True), base_directory=tmp_path
    )
    reference = tmp_path / "reference"
    run_plumed(reference_config, system, reference)

    resumed_config, resumed_system = resolve_plumed_job(
        raw_input(restart=True), base_directory=tmp_path
    )
    resumed = tmp_path / "resumed"
    partial = run_plumed(resumed_config, resumed_system, resumed)
    assert partial["status"] == "plumed_stop"
    assert partial["replica_results"][0]["steps"] == 2
    resumed_config["resume"] = True
    completed = run_plumed(resumed_config, resumed_system, resumed)
    assert completed["status"] == "complete"

    reference_checkpoint = checkpoint_payload(
        reference / "checkpoints" / "plumed-replica-0000.npz"
    )
    resumed_checkpoint = checkpoint_payload(
        resumed / "checkpoints" / "plumed-replica-0000.npz"
    )
    for name in (
        "q",
        "p",
        "step",
        "time",
        "rng",
        "sample_steps",
        "positions",
        "physical_bias",
        "file_manifest",
    ):
        np.testing.assert_array_equal(
            resumed_checkpoint[name], reference_checkpoint[name]
        )
    assert (
        resumed / "plumed" / "replica-0000" / "STATE"
    ).read_bytes() == (
        reference / "plumed" / "replica-0000" / "STATE"
    ).read_bytes()
    assert len(kernels) == 3


@pytest.mark.parametrize("damage", ["changed", "missing", "extra"])
def test_resume_rejects_changed_missing_or_untracked_plumed_state(
    tmp_path, monkeypatch, damage
):
    kernels = install_mock(monkeypatch, [2])
    config, system = resolve_plumed_job(
        raw_input(restart=True), base_directory=tmp_path
    )
    output = tmp_path / damage
    run_plumed(config, system, output)
    state = output / "plumed" / "replica-0000" / "STATE"
    if damage == "changed":
        state.write_bytes(b"tampered\n")
    elif damage == "missing":
        state.unlink()
    else:
        (state.parent / "UNTRACKED").write_text("unexpected")
    config["resume"] = True
    with pytest.raises((ValueError, RuntimeError), match="PLUMED"):
        run_plumed(config, system, output)
    # Resume rejected before constructing another kernel or consuming its RNG.
    assert len(kernels) == 1


def test_file_walkers_require_matching_parallel_workers(tmp_path):
    data = raw_input(replicas=2, walkers="files")
    data["plumed"]["input"] = (
        "x: POSITION ATOM=1\n"
        "m: METAD ARG=x.x SIGMA=0.1 HEIGHT=0.01 PACE=1 FILE=HILLS\n"
    )
    with pytest.raises(InputError, match="process or MPI"):
        resolve_plumed_job(data, base_directory=tmp_path)
    data["parallel"] = {"backend": "process", "trajectory_workers": 1}
    with pytest.raises(InputError, match="trajectory_workers=replicas"):
        resolve_plumed_job(data, base_directory=tmp_path)


def test_process_file_walkers_share_one_ordered_history(tmp_path, monkeypatch):
    module = tmp_path / "plumed.py"
    module.write_text(
        """
from pathlib import Path
import re
import numpy as np

class Plumed:
    def __init__(self, **options):
        self.values = {}
        self.walker = None
    def cmd(self, name, value=None):
        if name == 'getApiVersion':
            value[0] = 10
        elif name == 'isEnergyNeeded':
            value[0] = 0
        elif name == 'performCalcNoUpdate':
            x = self.values['setPositions']
            self.values['setForces'][:] = -0.1*x
            self.bias = 0.05*float(np.sum(x*x))
        elif name == 'getBias':
            value[0] = self.bias
        elif name == 'readInputLines':
            self.values[name] = value
            match = re.search(r'WALKERS_ID=(\\d+)', value)
            self.walker = int(match.group(1))
        elif name == 'update':
            step = int(self.values['setStep'])
            with Path('STATE').open('a') as stream:
                stream.write(f'{step}\\n')
            shared = Path('../shared/HILLS')
            shared.parent.mkdir(parents=True, exist_ok=True)
            with shared.open('a') as stream:
                stream.write(f'{self.walker}:{step}\\n')
        elif name.startswith('set'):
            self.values[name] = value
    def finalize(self):
        pass
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "plumed", importlib.import_module("plumed"))
    data = raw_input(replicas=2, restart=True, walkers="files")
    data["plumed"]["input"] = (
        "x: POSITION ATOM=1\n"
        "m: METAD ARG=x.x SIGMA=0.1 HEIGHT=0.01 PACE=1 FILE=HILLS\n"
    )
    data["parallel"] = {
        "backend": "process",
        "trajectory_workers": 2,
        "scheduling": "static",
    }
    config, system = resolve_plumed_job(data, base_directory=tmp_path)
    result = run_plumed(config, system, tmp_path / "walkers")
    assert result["status"] == "complete"
    assert (
        tmp_path / "walkers" / "plumed" / "shared" / "HILLS"
    ).read_text().splitlines() == [
        "0:1",
        "1:1",
        "0:2",
        "1:2",
        "0:3",
        "1:3",
        "0:4",
        "1:4",
    ]


def test_four_rank_mpi_multi_sim_smoke(tmp_path):
    import os
    import shutil
    import subprocess

    mpi4py = pytest.importorskip("mpi4py")
    if mpi4py.MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("nested MPI launch is unsupported")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher is unavailable")
    environment = os.environ.copy()
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    subprocess.run(
        [
            launcher,
            "-n",
            "4",
            sys.executable,
            str(Path(__file__).with_name("mpi_plumed_smoke.py")),
            str(tmp_path / "mpi-plumed"),
        ],
        check=True,
        timeout=180,
        env=environment,
    )
