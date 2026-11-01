"""Run explicit H3 classical/bead/timestep controls with fixed physical times.

This reuses the published-rate benchmark and production workflow. It never
labels a numerical control as a converged literature reproduction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.published_rates.h3 import configuration
from pyrpmd.constants import FS_TO_AU
from pyrpmd.io import write_json
from pyrpmd.runner import run_resolved


def _steps(duration, timestep):
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    value = round(duration/timestep)
    if abs(value*timestep-duration) > 1e-9:
        raise ValueError("each physical duration must be an integer multiple of the timestep")
    return value


def resolved_control(output, library, *, beads=1, timestep_fs=.1, seed=831072,
                     equilibration_fs=500., production_fs=1000., workers=3):
    if isinstance(beads, bool) or not isinstance(beads, int) or beads < 1:
        raise ValueError("beads must be a positive integer")
    if not np.isfinite(timestep_fs) or not 0 < timestep_fs <= .1:
        raise ValueError("timestep must be positive and <= 0.1 fs")
    config, system = configuration(output, library, profile="audit", seed=seed, workers=workers)
    config["beads"] = beads
    config["dt_au"] = timestep_fs*FS_TO_AU
    config["dynamics"]["timestep"] = f"{timestep_fs} fs"
    config["sampling"].update(
        dt_au=config["dt_au"],
        equilibration=f"{equilibration_fs} fs",
        production=f"{production_fs} fs",
        equilibration_steps=_steps(equilibration_fs, timestep_fs),
        production_steps=_steps(production_fs, timestep_fs),
        sample_interval=_steps(.5, timestep_fs),
        interval_steps=_steps(.5, timestep_fs),
        andersen_interval=_steps(5., timestep_fs),
    )
    if config["sampling"]["production_steps"] < 2*config["sampling"]["sample_interval"]:
        raise ValueError("at least 1 fs production required")
    config["recrossing"].update(
        equilibration_steps=_steps(1000., timestep_fs),
        parent_interval_steps=_steps(20., timestep_fs),
        child_steps=_steps(100., timestep_fs),
    )
    config["output"]["checkpoint_steps"] = _steps(50., timestep_fs)
    return config, system


def source_snapshot():
    files = subprocess.check_output(["git", "ls-files", "-z", "src", "native"], cwd=ROOT).decode().split("\0")
    files += ["benchmarks/published_rates/h3.py", "benchmarks/h3/pes.py",
              "benchmarks/h3/sampling_control.py"]
    result = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in sorted(set(files)) if name and (ROOT/name).is_file()}
    extension = Path(importlib.import_module("pyrpmd._native").__file__)
    result["loaded_native_extension:"+extension.name] = hashlib.sha256(extension.read_bytes()).hexdigest()
    return result


def run(output, library, **options):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config, system = resolved_control(output, library, **options)
    before = source_snapshot()
    write_json(output/"settings.json", {"config": config, "system": system.to_dict(),
                                        "control": options, "source_at_start": before})
    started = time.perf_counter()
    print(json.dumps({"event": "start", "control": options}), flush=True)
    result = run_resolved(config, system)
    after = source_snapshot()
    report = {
        "schema": "pyrpmd.h3.sampling-control.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control": options, "physical_converged": False, "literature_reproduced": False,
        "classical_control": config["beads"] == 1,
        "rate_estimate_cm3_molecule_s": result.get("rate_estimate_cm3_s"),
        "qtst_cm3_molecule_s": result.get("pmf", {}).get("k_qtst_cm3_s"),
        "pmf_barrier_hartree": result.get("pmf", {}).get("free_energy_barrier_hartree"),
        "transmission_coefficient": result.get("kappa"),
        "workflow_status": result.get("status"),
        "elapsed_seconds": time.perf_counter()-started,
        "source_at_start": before, "source_at_end": after,
        "sources_unchanged_during_run": before == after,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "scope": "controlled sampling/parameter investigation; a plateau does not establish equilibrium or numerical convergence",
    }
    write_json(output/"control-report.json", report)
    print(json.dumps({key: report[key] for key in ("elapsed_seconds", "qtst_cm3_molecule_s",
          "pmf_barrier_hartree", "transmission_coefficient", "rate_estimate_cm3_molecule_s",
          "sources_unchanged_during_run")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--timestep-fs", type=float, default=.1)
    parser.add_argument("--equilibration-fs", type=float, default=500.)
    parser.add_argument("--production-fs", type=float, default=1000.)
    parser.add_argument("--seed", type=int, default=831072)
    parser.add_argument("--workers", type=int, default=3)
    args = vars(parser.parse_args())
    output, library = args.pop("output"), args.pop("library")
    run(output, library, **args)


if __name__ == "__main__":
    main()
