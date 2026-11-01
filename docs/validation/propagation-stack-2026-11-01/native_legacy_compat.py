"""Reproduce the legacy/current NativeEnsemble wrapper compatibility check.

Both wrappers deliberately use the currently built _thermal extension. This
checks constructor normalization, fingerprints and checkpoint compatibility;
it is not an independent comparison against an old compiled native engine.
Run from the repository with its virtualenv; the result JSON is written beside this file.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = "2f907ba96d22cb134198715e398b1d9f8d83d929"
sys.path.insert(0, str(REPO / "src"))
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd import _thermal

source = subprocess.check_output(
    ["git", "show", f"{BASE}:src/pyrpmd/native_ensemble.py"], cwd=REPO, text=True)
legacy = types.ModuleType("pyrpmd._legacy_native_ensemble_compat")
legacy.__package__ = "pyrpmd"
legacy.__file__ = f"{BASE}:src/pyrpmd/native_ensemble.py"
sys.modules[legacy.__name__] = legacy
exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)


def thermostat(mode, beads):
    if "gle" not in mode and mode != "piglet":
        return mode
    spec = {"A": [[1., 2.], [-2., 1.]], "C": np.eye(2).tolist()}
    if mode == "piglet":
        spec.update(A=np.repeat(np.array(spec["A"])[None], beads, axis=0).tolist(),
                    beta=2., beads=beads,
                    provenance="Compatibility control: canonical identity, not a calibrated PIGLET set")
    return {"mode": mode, "gle": spec}


def configuration(beads=3, **kwargs):
    options = dict(q=np.random.default_rng(8).normal(scale=.1, size=(beads, 2, 3)),
                   masses=[1., 2.], beta=2., dt=.025, seed=41,
                   potential={"kind": "harmonic", "k": .7})
    options.update(kwargs)
    return options


cases = []


def compare(label, options):
    old, new = legacy.NativeEnsemble(**options), NativeEnsemble(**options)
    assert old.fingerprint == new.fingerprint, label
    assert old.state()["digest"] == new.state()["digest"], label
    old_records = old.run(9, stride=2, record_centroids=True, record_moments=True)
    new_records = new.run(9, stride=2, record_centroids=True, record_moments=True)
    for key in ("samples", "centroids", "velocities", "radii_gyration_squared"):
        np.testing.assert_array_equal(old_records[key], new_records[key], err_msg=label)
    assert old.state()["digest"] == new.state()["digest"], label
    cases.append({"case": label, "fingerprint": new.fingerprint,
                  "terminal_digest": new.state()["digest"],
                  "fingerprints_equal": True, "initial_checkpoints_equal": True,
                  "terminal_checkpoints_equal": True, "observation_arrays_equal": True})


for fuse in (False, True):
    for mode in ("none", "andersen", "langevin", "pile-l", "pile-g", "svr",
                 "gle", "nm-gle", "nm-gle-g", "piglet", "trpmd", "ffl"):
        compare(f"thermostat={mode},fuse_modes={fuse}",
                configuration(thermostat=thermostat(mode, 3), fuse_modes=fuse))
    for mode in ("bzp", "mttk", "rgb", "scr"):
        barostat = {"mode": mode, "tau": 1.5}
        if mode == "rgb":
            barostat.update(stress=np.eye(3).tolist(), reference_cell=(np.eye(3)*2).tolist())
        else:
            barostat["pressure"] = .7
        if mode == "scr":
            barostat["compressibility"] = .3
        if mode in ("rgb", "mttk"):
            barostat["cell_mask"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        compare(f"barostat={mode},fuse_modes={fuse}",
                configuration(beads=1 if mode == "scr" else 3, barostat=barostat,
                              cell=np.eye(3)*2, potential={"kind": "free"},
                              thermostat="pile-l", fuse_modes=fuse))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


result = {
    "base_revision": BASE,
    "comparison": "main/current Python NativeEnsemble wrappers against the same current _thermal extension",
    "scope": "constructor schema/fingerprint and complete checkpoint/observation compatibility; not an independently built old native engine",
    "cases": cases, "passed": len(cases), "steps_per_case": 9,
    "rng_abi": _thermal.rng_abi,
    "current_native_ensemble_sha256": sha256(REPO / "src/pyrpmd/native_ensemble.py"),
    "current_thermal_binary_sha256": sha256(_thermal.__file__),
    "current_thermal_header_sha256": sha256(REPO / "native/thermal.hpp"),
    "current_thermal_binding_sha256": sha256(REPO / "native/thermal.cpp"),
}
destination = ROOT / "native_legacy_compat_results.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(f"{len(cases)} wrapper/schema/fingerprint/checkpoint/observation comparisons passed")
print(destination)
