"""Create real historical checkpoints, then verify current exact continuation.

Run both commands with the same interpreter and installed optional dependencies:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/old/checkout/src \
      python cross_version_resume.py prepare /scratch/t16b-cross-version
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/new/checkout/src \
      python cross_version_resume.py verify /scratch/t16b-cross-version \
      --report cross_version_resume.json

``prepare`` refuses an existing destination. It creates uninterrupted reference
runs and interrupted public runs using only the selected historical source.
``verify`` resumes those actual files using the selected current source. The
legacy cases additionally undergo memory -> stream -> memory migration while
partial. NPZ container envelopes and content-addressed trajectory identifiers
may change; every physical/RNG/clock payload and every sample must match exactly.

This is a tiny deterministic implementation check, not a convergence campaign.
Torch CPU and the compiled native extension are required; no case is skipped.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import numpy as np


CASES = (
    ("legacy-pile-nvt", "legacy", "pile-l", False),
    ("legacy-andersen-nvt", "legacy", "andersen", False),
    ("legacy-bzp-npt", "legacy", "pile-l", True),
    ("native-pile-nvt", "native", "pile-l", False),
    ("native-nhc-nvt", "native", "nhc", False),
    ("native-bzp-npt", "native", "pile-l", True),
    ("torch-cpu-pile-nvt", "torch", "pile-l", False),
)


class CommittedInterruption(RuntimeError):
    """The harness stops at a known durable prefix, never edits saved state."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source():
    import pyrpmd

    package = Path(pyrpmd.__file__).resolve().parent
    repository = package.parent.parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                         text=True, capture_output=True, check=True)
    files = ("checkpoint.py", "native_ensemble.py", "pimd_checkpoint.py", "runner.py",
             "workflows/materials.py", "workflows/native_pimd.py",
             "workflows/pimd.py", "workflows/torch_materials.py")
    return {"commit": git.stdout.strip(), "package": str(package),
            "python": platform.python_version(), "numpy": np.__version__,
            "source_sha256": {name: _sha(package / name) for name in files
                              if (package / name).is_file()}}


def _job(engine, thermostat, pressure):
    job = {
        "task": "pimd", "temperature": "300 K", "beads": 4, "seed": 431,
        "system": {"symbols": ["H", "He"], "masses": [2., 5.],
                   "geometry": [[.1, -.2, .3], [.4, .5, -.1]], "geometry_unit": "bohr"},
        "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"timestep": .04, "integrator": "cayley-bcb",
                     "loop_backend": {"legacy": "python", "native": "native", "torch": "auto"}[engine],
                     "block_steps": 5},
        "sampling": {"engine": engine, "thermostat": thermostat,
                     "equilibration_steps": 3, "production_steps": 16,
                     "sample_interval": 2, "replicas": 2 if engine == "torch" else 1},
        "output": {"checkpoint_steps": 4, "chunk_frames": 3,
                   "trajectory_storage": "memory" if engine == "legacy" else "stream"},
    }
    if thermostat == "pile-l":
        job["sampling"]["gamma_centroid"] = .7
    if thermostat == "andersen":
        job["sampling"]["andersen_interval"] = 3
    if thermostat == "nhc":
        job["sampling"]["thermostat_params"] = {"tau": "1 fs", "chain_length": 3, "substeps": 2}
    if engine == "legacy":
        job["native"] = False
    if engine == "torch":
        job["sampling"].update(device="cpu", normal_mode_layout="rfft")
    if pressure:
        job["potential"] = {"provider": "free"}
        job["system"].update(cell=(6 * np.eye(3)).tolist(), cell_unit="bohr", pbc=[True] * 3)
        job["barostat"] = ({"method": "bzp", "pressure": "1 bar", "time_constant": "10 fs",
                            "friction": "0.01 fs^-1"} if engine == "legacy" else
                           {"mode": "bzp", "pressure": "1 bar", "tau": "10 fs", "gamma": "0.01 fs^-1"})
    return job


def _saved_input(directory):
    from pyrpmd.runner import _read_identity

    config = json.loads((directory / "input/resolved.json").read_text())
    return config, _read_identity(config, directory)


def _capture(directory, engine):
    """Separate scientific payload/samples from container/trajectory addressing."""
    from pyrpmd.checkpoint import load_checkpoint
    from pyrpmd.trajectory import materialize_stream

    config, system = _saved_input(directory)
    samples = {}
    if engine == "native":
        journal_path = directory / "samples/native_pimd_0000/journal.json"
        journal = json.loads(journal_path.read_text())
        checkpoint = journal_path.parent / journal["blocks"][-1]["directory"] / "state.npz"
        records = [np.load(journal_path.parent / block["directory"] / "samples.npy", allow_pickle=False)
                   for block in journal["blocks"] if block["samples_sha256"] is not None]
        samples["samples"] = np.concatenate(records) if records else np.empty((0,))
        step = journal["blocks"][-1]["end"]
        omitted = set()
    elif engine == "legacy":
        from pyrpmd.workflows import materials

        options = materials.resolve_materials(config, system)
        checkpoint = materials._checkpoint_path(directory, 0)
        saved = materials._read_stream(config, system, directory, 0, options, materialize=True)
        schema = materials._trajectory_schema(system, options)
        samples = {"samples_" + name: saved[name] for name in schema}
        step = saved["elapsed"]
        omitted = set(schema) | {"trajectory"}
    else:
        checkpoint = directory / "checkpoints/materials_torch.npz"
        with load_checkpoint(checkpoint) as data:
            descriptors = json.loads(str(data["trajectories"]))
            step = int(data["step"])
        for replica, descriptor in enumerate(descriptors):
            arrays = materialize_stream(directory / "samples" / f"materials_{replica:04d}", descriptor)
            samples.update({f"samples_{replica}_{name}": value for name, value in arrays.items()})
        omitted = {"trajectories"}
    with load_checkpoint(checkpoint) as data:
        arrays = {"state_" + name: data[name] for name in data.files if name not in omitted}
        integrity = {"schema": data.envelope["schema"], "integrity": data.envelope["integrity"]}
    arrays.update(samples)
    return arrays, {"checkpoint": str(checkpoint.relative_to(directory)), "step": step,
                    "container": integrity, "checkpoint_sha256": _sha(checkpoint)}


def _equal(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError(f"payload fields differ: {set(actual) ^ set(expected)}")
    for name, value in actual.items():
        reference = expected[name]
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise AssertionError(f"dtype/shape differs: {name}")
        if value.tobytes(order="C") != reference.tobytes(order="C"):
            raise AssertionError(f"payload/sample bytes differ: {name}")


def _interrupted_run(job, output, engine):
    from pyrpmd.runner import run

    if engine == "legacy":
        from pyrpmd.workflows import materials as module
        original, hook = module.atomic_savez, "atomic_savez"

        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            if int(kwargs.get("elapsed", -1)) == 8:
                raise CommittedInterruption("legacy prefix committed at step 8")
    elif engine == "native":
        from pyrpmd.workflows import native_pimd as module
        original, hook = module.write_json, "write_json"

        def interrupt(path, value):
            # Fail BEFORE publishing the next journal. _commit rolls back the
            # attempted block, leaving the genuinely committed step-7 files.
            if Path(path).name == "journal.json" and value["blocks"][-1]["end"] == 11:
                raise CommittedInterruption("native prefix remains committed at step 7")
            return original(path, value)
    else:
        from pyrpmd.workflows import torch_materials as module
        original, hook = module._save, "_save"

        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            if args[3].step == 8:
                raise CommittedInterruption("Torch batch committed at step 8")
    try:
        with patch.object(module, hook, interrupt):
            run(copy.deepcopy(job), output=output)
    except RuntimeError as error:
        # root_call intentionally broadcasts a normalized RuntimeError for
        # Torch's root-owned batch. Accept only our exact injected failure.
        if isinstance(error, CommittedInterruption) or str(error).startswith(
                "Root stage failed: CommittedInterruption:"):
            return
        raise
    raise AssertionError(f"{engine} did not reach its planned interruption")


def prepare(directory):
    from pyrpmd.runner import run

    directory.mkdir(parents=True, exist_ok=False)
    result = {"schema": "t16b.cross-version-resume.v1", "source": _source(), "cases": []}
    for name, engine, thermostat, pressure in CASES:
        case = directory / name
        job = _job(engine, thermostat, pressure)
        run(copy.deepcopy(job), output=case / "reference")
        expected, reference = _capture(case / "reference", engine)
        np.savez(case / "expected.npz", **expected)
        _interrupted_run(job, case / "partial", engine)
        _, partial = _capture(case / "partial", engine)
        if partial["step"] != (7 if engine == "native" else 8):
            raise AssertionError(f"unexpected committed prefix: {name}")
        if engine == "native" and partial["container"]["schema"] != "legacy.npz":
            raise AssertionError("prepare must use the pre-T16b native checkpoint writer")
        result["cases"].append({"name": name, "engine": engine, "reference": reference,
                                "partial": partial, "expected_sha256": _sha(case / "expected.npz")})
        print(f"prepared {name}: real historical checkpoint at step {partial['step']}", flush=True)
    (directory / "preparation.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def verify(directory):
    from pyrpmd.pimd_checkpoint import migrate_legacy_checkpoint
    from pyrpmd.runner import resume_run

    prepared = json.loads((directory / "preparation.json").read_text())
    result = {"schema": prepared["schema"], "baseline_source": prepared["source"],
              "current_source": _source(), "cases": [],
              "comparison": "identical dtype, shape and C-order bytes for every retained state and sample array",
              "physical_converged": False}
    for info in prepared["cases"]:
        name, engine = info["name"], info["engine"]
        case, output = directory / name, directory / name / "partial"
        if _sha(case / "expected.npz") != info["expected_sha256"]:
            raise AssertionError(f"baseline reference changed: {name}")
        before, partial = _capture(output, engine)
        if partial != info["partial"]:
            raise AssertionError(f"historical partial checkpoint changed before verification: {name}")
        migration = False
        if engine == "legacy":
            config, system = _saved_input(output)
            input_files = [output / "manifest.json", output / "input/resolved.json", output / "input/system.json"]
            input_bytes = {path: path.read_bytes() for path in input_files}
            migrate_legacy_checkpoint(config, system, output, storage="stream")
            _equal(_capture(output, engine)[0], before)
            migrate_legacy_checkpoint(config, system, output, storage="memory")
            _equal(_capture(output, engine)[0], before)
            if any(path.read_bytes() != value for path, value in input_bytes.items()):
                raise AssertionError("offline migration changed saved inputs or run manifest")
            migration = True
        summary = resume_run(output)
        actual, final = _capture(output, engine)
        with np.load(case / "expected.npz", allow_pickle=False) as data:
            _equal(actual, {name: data[name] for name in data.files})
        if final["step"] != 19 or summary["status"] != "complete":
            raise AssertionError(f"resume did not finish: {name}")
        result["cases"].append({"name": name, "engine": engine, "passed": True,
                                "partial_step": partial["step"], "final_step": final["step"],
                                "matched_arrays": len(actual), "storage_roundtrip": migration,
                                "source_container": partial["container"], "final_container": final["container"]})
        print(f"verified {name}: {len(actual)} state/sample arrays exactly equal", flush=True)
    result["passed"] = len(result["cases"]) == len(CASES)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    result = prepare(directory) if args.mode == "prepare" else verify(directory)
    report = args.report or directory / ("verification.json" if args.mode == "verify" else "preparation.json")
    report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"{args.mode} complete: {report}", flush=True)


if __name__ == "__main__":
    main()
