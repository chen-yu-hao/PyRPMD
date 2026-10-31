"""Connect offline fitted models to the existing native batched ONNX bridge.

The exporter is an explicit trusted script from PyRPMD's ONNX integration. It
runs in a separate process, so jax2onnx instrumentation cannot affect fitting.
No alternative ONNX graph writer or dynamics kernel is introduced here.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .jaxpip_data import checked_json, member, output_lock, write_checked
from .jaxpip_fit import load_fit
from .reference import atomic_json, file_hash


def load_export(directory):
    directory = Path(directory).resolve()
    manifest = checked_json(directory/"export.json", "pyrpmd.jaxpip-export/1")
    if not {"model.onnx", "export-report.json", "model.json"}.issubset(manifest["artifacts"]):
        raise ValueError("missing native-export artifacts")
    for name, digest in manifest["artifacts"].items():
        if file_hash(member(directory, name)) != digest:
            raise ValueError(f"export artifact checksum mismatch: {name}")
    fitted = load_fit(directory)
    if manifest["fit_manifest_sha256"] != file_hash(directory/"model.json") or manifest["contract"] != fitted["contract"]:
        raise ValueError("export and fitted-model contracts disagree")
    return manifest, fitted


def export_jaxpip(directory, exporter, output, *, timeout=600):
    """Export fitted linear/PIP-NN models using the validated PR #34 exporter.

    Export tolerances are the exporter's numerical parity checks, not chemical
    test-set accuracy. Failed exporters leave logs but no completed manifest.
    """
    directory, output, exporter = map(lambda p: Path(p).resolve(), (directory, output, exporter))
    fitted = load_fit(directory)
    if type(timeout) is not int or timeout < 1:
        raise ValueError("timeout must be a positive integer number of seconds")
    if not exporter.is_file() or output == directory or directory.is_relative_to(output):
        raise ValueError("supply a trusted exporter file and a separate output directory")
    with output_lock(output):
        if any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("export output must be empty; existing results are never overwritten")
        contract, settings = fitted["contract"], fitted["settings"]
        exporter_sha = file_hash(exporter)
        source_sha = file_hash(directory/"model.json")
        with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as td:
            stage = Path(td)
            for name in fitted["artifacts"]:
                shutil.copyfile(member(directory, name), stage/name)
            shutil.copyfile(directory/"model.json", stage/"model.json")
            load_fit(stage)
            command = [sys.executable, str(exporter), "--basis", str(stage/fitted["basis"]),
                       "--checkpoint", str(stage/"model.eqx"), "--geometries", str(stage/"probe.npy"),
                       "--output", str(stage/"model.onnx"), "--kind", settings["kind"],
                       "--atomic-numbers", *map(str, contract["atomic_numbers"]),
                       "--charge", str(contract["charge"]), "--multiplicity", str(contract["multiplicity"]),
                       "--position-unit", settings["position_unit"], "--energy-unit", settings["energy_unit"]]
            with (output/"export.stdout.log").open("w") as stdout, (output/"export.stderr.log").open("w") as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            if process.returncode:
                raise RuntimeError(f"JaxPIP ONNX exporter failed ({process.returncode}); see export logs")
            if file_hash(exporter) != exporter_sha or file_hash(directory/"model.json") != source_sha:
                raise ValueError("exporter or source manifest changed during export")
            report = json.loads((output/"export.stdout.log").read_text())
            metadata = report["metadata"]
            sources = metadata.get("source_sha256", {})
            expected = {"basis": file_hash(stage/fitted["basis"]), "checkpoint": file_hash(stage/"model.eqx")}
            if (report.get("sha256") != file_hash(stage/"model.onnx") or sources != expected
                    or metadata.get("atomic_numbers") != contract["atomic_numbers"]
                    or metadata.get("state") != {"charge": contract["charge"], "multiplicity": contract["multiplicity"]}
                    or metadata.get("position_unit") != settings["position_unit"]
                    or metadata.get("energy_unit") != settings["energy_unit"]
                    or report.get("batches_checked") != [1, 3, 32]):
                raise ValueError("export report does not match the fitted model contract")
            report["model"] = "model.onnx"  # avoid a stale staging path in portable reports
            atomic_json(stage/"export-report.json", report)
            for p in stage.iterdir():
                shutil.copyfile(p, output/p.name)
        artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != ".lock"}
        manifest = {"schema": "pyrpmd.jaxpip-export/1", "artifacts": artifacts,
                    "fit_manifest_sha256": source_sha, "exporter_sha256": exporter_sha,
                    "contract": contract, "physical_validation": False}
        write_checked(output/"export.json", manifest)
        return manifest


def native_potential_spec(directory, library, *, device="cpu", intra_op_threads=1,
                          allow_unvalidated=False):
    """Return the standard C ABI provider mapping without importing JAX/PySCF."""
    directory, library = Path(directory).resolve(), Path(library).resolve()
    _, fitted = load_export(directory)
    if type(allow_unvalidated) is not bool:
        raise ValueError("allow_unvalidated must be Boolean")
    if not fitted["within_requested_test_bounds"] and not allow_unvalidated:
        raise ValueError("no passing explicit test-set bounds; allow_unvalidated=True is required for pilots")
    if not library.is_file():
        raise FileNotFoundError(library)
    if type(intra_op_threads) is not int or intra_op_threads < 1:
        raise ValueError("intra_op_threads must be a positive integer")
    if not isinstance(device, str) or not (device == "cpu" or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("device must be cpu or cuda:<nonnegative index>")
    return {"provider": "cabi", "library": str(library),
            "params": {"model": str(directory/"model.onnx"), "device": device,
                       "intra_op_threads": intra_op_threads}}


def run_jaxpip_job(directory, library, job, *, output, device="cpu",
                   allow_unvalidated=False, resume=False):
    """Inject the fitted native PES into an existing validated JobSpec."""
    from ..input import JobSpec
    from ..runner import run
    spec = JobSpec.from_file(job) if isinstance(job, (str, Path)) else JobSpec.from_mapping(job)
    spec.data["potential"] = native_potential_spec(directory, library, device=device,
                                                   allow_unvalidated=allow_unvalidated)
    return run(spec, output=output, resume=resume)
