"""Compare MBAR algebra on identical saved H3 samples, not physical convergence.

The stride limits numerical test cost; it is not asserted to decorrelate the
trajectory. No asymptotic PyMBAR uncertainty is used. The independent library
receives an extra unsampled target state, so its public W_nk column supplies
the target weights without using PyRPMD's normalization implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from pyrpmd.analysis.fes import solve_mbar, umbrella_reduced_potentials
from pyrpmd.workflows.fes import jsonable


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def check(run, *, stride=10, reference_pymbar=False):
    run = Path(run).resolve()
    if not (run / "umbrella").is_dir() and (run / "run/umbrella").is_dir():
        run = run / "run"
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    config_path = run / "input/resolved.json"
    h5_path = run / "umbrella/fes.h5"
    config = json.loads(config_path.read_text())
    arrays, centers, stiffness = [], [], []
    with h5py.File(h5_path, "r") as h:
        if h.attrs["measure"] != "rate_flux":
            raise ValueError("Expected saved rate_flux umbrella samples")
        for key in sorted(h["streams"], key=int):
            group = h["streams"][key]
            values = np.asarray(group["samples"])[::stride]
            if values.ndim != 2 or values.shape[1] != 1 or len(values) < 2:
                raise ValueError("Need at least two scalar-CV samples per stream")
            arrays.append(values)
            centers.append(np.asarray(group["center"]))
            stiffness.append(np.asarray(group["stiffness"]))
    values = np.concatenate(arrays)
    counts = np.asarray([len(a) for a in arrays])
    u = umbrella_reduced_potentials(
        values, np.asarray(centers), np.asarray(stiffness), config["beta"]
    )
    started = time.perf_counter()
    current = solve_mbar(u, counts, tolerance=1e-9)
    current_seconds = time.perf_counter() - started
    started = time.perf_counter()
    legacy = solve_mbar(
        u, counts, tolerance=1e-9, method="legacy", combine_identical_states=False
    )
    legacy_seconds = time.perf_counter() - started
    comparison = {
        "maximum_target_weight_difference": float(
            np.max(np.abs(current.weights - legacy.weights))
        ),
        "target_weight_l1_difference": float(
            np.sum(np.abs(current.weights - legacy.weights))
        ),
        "maximum_gauge_free_energy_difference": float(
            np.max(np.abs(current.free_energies - legacy.free_energies))
        ),
    }
    gates = {
        "auto_residual_below_1e_9": bool(current.converged and current.residual < 1e-9),
        "legacy_residual_below_1e_9": bool(legacy.converged and legacy.residual < 1e-9),
        "target_weights_match_legacy": comparison["target_weight_l1_difference"] < 1e-6,
        "gauge_free_energies_match_legacy": comparison[
            "maximum_gauge_free_energy_difference"
        ]
        < 1e-6,
        "overlap_components_match_legacy": bool(
            np.array_equal(current.components, legacy.components)
        ),
    }
    reference = None
    if reference_pymbar:
        from pymbar import MBAR
        import pymbar

        started = time.perf_counter()
        fit = MBAR(
            np.vstack((u, np.zeros((1, u.shape[1])))),
            np.r_[counts, 0],
            relative_tolerance=1e-11,
            maximum_iterations=10000,
            solver_protocol="robust",
        )
        weights = np.asarray(fit.W_nk)[:, -1]
        free = np.asarray(fit.f_k)[:-1].copy()
        free -= free[0]
        reference = {
            "version": pymbar.__version__,
            "seconds": time.perf_counter() - started,
            "target_weight_source": "public W_nk column of the explicit unsampled zero-potential target",
            "maximum_target_weight_difference": float(
                np.max(np.abs(current.weights - weights))
            ),
            "target_weight_l1_difference": float(
                np.sum(np.abs(current.weights - weights))
            ),
            "maximum_gauge_free_energy_difference": float(
                np.max(np.abs(current.free_energies - free))
            ),
            "asymptotic_uncertainty_used": False,
        }
        gates["target_weights_match_independent_pymbar"] = (
            reference["target_weight_l1_difference"] < 1e-6
        )
        gates["gauge_free_energies_match_independent_pymbar"] = (
            reference["maximum_gauge_free_energy_difference"] < 1e-6
        )
    return {
        "schema": "pyrpmd.h3.mbar-solver-comparison.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "streams": len(arrays),
        "samples": len(values),
        "stride": stride,
        "source_hdf5_sha256": checksum(h5_path),
        "source_config_sha256": checksum(config_path),
        "auto": {
            "seconds": current_seconds,
            "residual": current.residual,
            "solver": current.diagnostics,
        },
        "legacy": {
            "seconds": legacy_seconds,
            "residual": legacy.residual,
            "solver": legacy.diagnostics,
        },
        "legacy_seconds_over_auto_seconds": legacy_seconds / current_seconds,
        "same_data_comparison": comparison,
        "independent_pymbar": reference,
        "scope": "Same finite-data nonlinear equations only; no claim of physical convergence or independence after striding. Timings are informational, not an acceptance gate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--reference-pymbar", action="store_true")
    args = parser.parse_args()
    result = jsonable(
        check(args.run, stride=args.stride, reference_pymbar=args.reference_pymbar)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
