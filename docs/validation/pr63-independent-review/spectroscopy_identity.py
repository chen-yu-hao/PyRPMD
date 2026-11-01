"""Execute the same offline-analysis witnesses against a selected PYTHONPATH.

Run once with the reviewed baseline source, once with the corrected source.
This is a provenance/data-preservation experiment, not a molecular spectrum.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

import pyrpmd.workflows.spectroscopy as workflow
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "source_sha256": hashlib.sha256(Path(workflow.__file__).read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observable": "unnormalized_relative_ir",
    }
    with tempfile.TemporaryDirectory(prefix="spectroscopy-identity-") as directory:
        root = Path(directory)
        model = root / "properties.py"
        model.write_text('''import numpy as np
def build():
    def evaluate(q):
        mu=q.sum(axis=1)
        return {"dipole":mu,"polarizability":np.einsum("bi,bj->bij",mu,mu)+np.eye(3)}
    return evaluate
''')
        config = {
            "task": "spectroscopy", "beta": 3.1, "beads": 4, "seed": 57, "dt_au": .01,
            "potential": {"provider": "harmonic", "params": {"k": .3}},
            "properties": {"provider": "python", "factory": str(model) + ":build"},
            "spectroscopy": {"method": "rpmd", "spectra": ["ir", "raman"], "normalize": False},
            "sampling": {"equilibration_steps": 10, "production_steps": 80,
                         "sample_interval": 2, "replicas": 2, "gamma_centroid": .7},
            "output": {"checkpoint_steps": 20},
        }
        system = System(("H",), np.array([2.]), np.array([[.2, -.1, .3]]))
        output = root / "run"
        workflow.run_spectroscopy(config, system, output)
        with load_checkpoint(output / "results/ir-spectrum.npz") as data:
            before = data["unnormalized_relative_ir"]
        changed = copy.deepcopy(config)
        changed["beta"] *= 2
        try:
            workflow.reanalyze_spectroscopy(changed, system, output)
        except ValueError as error:
            report["changed_temperature"] = {"rejected": True, "error": str(error)}
        else:
            with load_checkpoint(output / "results/ir-spectrum.npz") as data:
                after = data["unnormalized_relative_ir"]
            mask = np.abs(before) > np.max(np.abs(before)) * 1e-10
            report["changed_temperature"] = {
                "rejected": False,
                "intensity_ratio_min": float(np.min(after[mask] / before[mask])),
                "intensity_ratio_max": float(np.max(after[mask] / before[mask])),
            }
        trace = output / "results/spectroscopy-traces.npz"
        prior = trace.read_bytes()
        config["spectroscopy"]["spectra"] = ["ir"]
        workflow.reanalyze_spectroscopy(config, system, output)
        with load_checkpoint(trace) as data:
            report["ir_only_reanalysis"] = {
                "raw_bytes_preserved": trace.read_bytes() == prior,
                "polarizability_retained": "polarizability" in data,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
