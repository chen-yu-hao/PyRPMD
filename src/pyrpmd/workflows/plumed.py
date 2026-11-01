"""Input-file access to the existing explicitly clocked PLUMED bridge."""
from __future__ import annotations

from contextlib import chdir
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez
from ..dynamics import Dynamics, State
from ..plumed import PlumedBias
from ..potentials import make_potential
from ..propagation import loop_options
from ..io import write_json


FIELDS = {"file", "input", "kernel", "charges"}


def resolve_plumed_job(data, *, base_directory=None, input_path=None):
    from ..input import JobSpec, InputError, _path
    base = Path(base_directory or ".").resolve()
    if not isinstance(data, Mapping) or data.get("task") != "plumed":
        raise InputError("resolve_plumed_job requires task='plumed'")
    raw = deepcopy(dict(data))
    table = raw.pop("plumed", None)
    if not isinstance(table, dict) or set(table) - FIELDS:
        raise InputError("[plumed] must contain file or input, optional kernel and charges")
    if ("file" in table) == ("input" in table):
        raise InputError("choose exactly one of plumed.file or plumed.input")
    forbidden = {
        "path_integral", "active_learning", "barostat", "electronic_friction",
        "reaction", "collision", "outcomes", "cvs", "umbrella", "analysis", "rate",
        "recrossing", "network", "metadynamics", "opes", "adaptive_umbrella", "pcet",
        "diffusion", "site_rate", "surface", "nonadiabatic", "replica_exchange", "initial_ensemble",
    }
    if forbidden.intersection(raw):
        raise InputError("PLUMED workflow does not support sections: "
                         + ", ".join(sorted(forbidden.intersection(raw))))
    if "file" in table:
        if not isinstance(table["file"], str) or not table["file"]:
            raise InputError("plumed.file must be a nonempty file path")
        table["file"] = _path(base, table["file"])
        if not Path(table["file"]).is_file():
            raise InputError("PLUMED input file is missing")
        text = Path(table["file"]).read_text(encoding="utf-8")
    else:
        text = table["input"]
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise InputError("PLUMED input must be nonempty text")
    if "kernel" in table:
        if not isinstance(table["kernel"], str) or not table["kernel"]:
            raise InputError("plumed.kernel must be a nonempty library path")
        table["kernel"] = _path(base, table["kernel"])
        if not Path(table["kernel"]).is_file():
            raise InputError("PLUMED kernel library is missing")
    output = raw.setdefault("output", {})
    if not isinstance(output, Mapping):
        raise InputError("output must be a table")
    unsupported_output = {"trajectory_stride", "trajectory_storage", "chunk_frames", "diagnostics", "store_configurations"}
    if unsupported_output.intersection(output):
        raise InputError("PLUMED workflow does not support output controls: "
                         + ", ".join(sorted(unsupported_output.intersection(output))))
    output.setdefault("directory", "runs/plumed")
    raw["task"] = "pimd"
    resolved, system = JobSpec(raw, base, input_path).resolve()
    parallel = resolved["parallel"]
    if (parallel["backend"] != "serial" or parallel["mode"] != "serial"
            or parallel["trajectory_workers"] != 1 or parallel["bead_workers"] != 1
            or parallel.get("pes_ranks", 1) != 1):
        raise InputError("PLUMED input workflow requires serial execution")
    if (resolved["sampling"].get("replicas", 1) != 1 or system.fixed_atoms
            or (any(system.pbc) and not all(system.pbc))):
        raise InputError("PLUMED workflow requires one all-mobile trajectory and all/no PBC")
    if resolved["sampling"].get("engine", "legacy") != "legacy" or resolved.get("dynamics", {}).get("loop_backend") == "native":
        raise InputError("PLUMED requires Python step orchestration")
    if resolved["sampling"]["dt_au"] != resolved["dt_au"]:
        raise InputError("PLUMED requires sampling.timestep equal to dynamics.timestep for its fixed update clock")
    if resolved["sampling"]["thermostat"] not in ("andersen", "pile-l"):
        raise InputError("PLUMED workflow supports only Andersen or PILE-L sampling")
    if {"parents", "thermostat_params"}.intersection(data.get("sampling", {})):
        raise InputError("PLUMED workflow does not use parents or NHC thermostat parameters")
    if "charges" in table:
        charges = np.asarray(table["charges"])
        if charges.dtype.kind not in "fiu" or charges.shape != (system.natoms,) or not np.isfinite(charges).all():
            raise InputError("plumed.charges must be a finite real vector of length N")
    resolved.update(task="plumed", plumed=table, _input_path=input_path)
    resolved["_warnings"].append("PLUMED owns its history/files; automatic kernel restart is not supported. Relative PLUMED FILE paths use RUN/plumed.")
    return resolved, system


def run_plumed(config, system, output):
    if config.get("resume"):
        raise ValueError("PLUMED kernel history cannot be restored from an MD-only checkpoint; automatic resume is unsupported")
    output = Path(output).resolve()
    work = output / "plumed"
    work.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    table, sampling = config["plumed"], config["sampling"]
    script = Path(table["file"]).read_text(encoding="utf-8") if "file" in table else table["input"]
    provider = make_potential(config["potential"], system)
    steps, positions, biases = [], [], []
    snapshot = output / "checkpoints" / "plumed-md-state.npz"
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=config.get("seed", 0), native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            **loop_options(config))
        state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
        total = sampling["equilibration_steps"] + sampling["production_steps"]
        def save():
            atomic_savez(snapshot, kind="PyRPMD.plumed-md-state.v1", q=state.q, p=state.p,
                step=state.step, time=state.time, rng=json.dumps(dynamics.rng.bit_generator.state),
                kernel_state_saved=False, sample_steps=np.asarray(steps, dtype=np.int64),
                positions=np.asarray(positions).reshape(-1, config["beads"], system.natoms, 3),
                physical_bias=np.asarray(biases))
        save()
        try:
            with chdir(work), PlumedBias(script, system.masses, config["beta"], config["beads"], config["dt_au"],
                    cell=system.cell, pbc=system.pbc, charges=table.get("charges"),
                    kernel_path=table.get("kernel"), log_file=str(output / "logs" / "plumed.log")) as bridge:
                while state.step < total and not bridge.stop_requested:
                    bridge.step(dynamics, state, thermostat=sampling["thermostat"],
                        gamma_centroid=sampling.get("gamma_centroid", 1.0),
                        andersen_interval=sampling["andersen_interval"])
                    production = state.step - sampling["equilibration_steps"]
                    if production > 0 and production % sampling["sample_interval"] == 0:
                        bias = bridge.evaluate(state.q).physical_energy
                        steps.append(state.step)
                        positions.append(state.q.copy())
                        biases.append(bias)
                    if state.step % config["output"]["checkpoint_steps"] == 0:
                        save()
                stopped = bridge.stop_requested
        except BaseException as error:
            write_json(output / "results" / "plumed-failure.json", {
                "task": "plumed", "status": "failed", "last_accepted_step": state.step,
                "error_type": type(error).__name__, "error": str(error),
                "md_snapshot": str(snapshot), "kernel_state_saved": False,
                "restart_supported": False,
            })
            raise
        finally:
            save()
        return {"task": "plumed", "status": "plumed_stop" if stopped else "complete",
            "steps": state.step, "samples": len(steps), "restart_supported": False,
            "md_snapshot": str(snapshot), "plumed_directory": str(work),
            "biased_dynamics": True, "physical_rate": None}
    finally:
        provider.close()
