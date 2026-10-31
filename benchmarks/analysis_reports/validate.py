"""Reproduce short dynamics-to-report checks with numerical plot-data oracles.

These deliberately small harmonic/free-potential jobs validate software contracts.
They are not chemical-rate benchmarks or evidence of sampling convergence.
Run from a source checkout with ``python -m benchmarks.analysis_reports.validate``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd.analysis.reporting import generate_report
from pyrpmd.constants import AU_TIME_FS, BOHR_ANGSTROM, KB_AU
from pyrpmd.runner import run_resolved
from pyrpmd.system import System
from pyrpmd.units import unit_factor


def short_fes_config(dimension=1, *, adaptive=False):
    """A deterministic, explicitly artificial atomic-unit mechanics example."""
    config = {
        "task": "fes", "beta": 1.0, "temperature_K": 1 / KB_AU,
        "beads": 2, "dt_au": 0.03, "native": False, "seed": 91,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 10, "production_steps": 40,
                     "sample_interval": 1, "andersen_interval": 10, "replicas": 1},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0,
                 "axis": axis, "unit": "bohr"} for axis in ("x", "y")[:dimension]],
        "umbrella": {
            "centers": [[-1.0], [1.0]] if dimension == 1
            else [[-.3, -.3], [-.3, .3], [.3, -.3], [.3, .3]],
            "stiffness": 30.0 if dimension == 1 else 1.0,
        },
        "analysis": {"bins": [6] if dimension == 1 else [5, 7], "bootstrap": 0},
        "output": {"checkpoint_steps": 25},
    }
    if dimension == 2:
        # Wide bounds retain unsupported bins, making the masking check non-vacuous.
        config["analysis"]["ranges"] = [[-3.0, 3.0], [-2.5, 2.5]]
    if adaptive:
        config["adaptive_umbrella"] = {
            "enabled": True, "pilot_steps": 150, "pilot_equilibration_steps": 30,
            "max_rounds": 4, "max_windows": 16,
            "target_overlap": 0.001, "min_effective_samples": 1.0,
        }
    return config


def short_meta_config():
    config = short_fes_config(2)
    config.pop("umbrella")
    config.update(task="metadynamics", beta=2.0, temperature_K=1 / (2 * KB_AU),
                  beads=3, dt_au=0.02, seed=321)
    config["sampling"] = {"equilibration_steps": 3, "production_steps": 24,
                          "sample_interval": 2, "andersen_interval": 4, "replicas": 2}
    config["metadynamics"] = {"mode": "well_tempered", "sigma": [0.3, 0.3],
                              "height": 0.05, "pace": 3, "bias_factor": 5.0}
    config["output"]["checkpoint_steps"] = 5
    return config


def short_rate_config():
    # The same free-potential mechanics construction used by the rate workflow
    # tests: it exercises flux PMF and paired children, without a chemical PES.
    return {
        "task": "rate", "beta": 5.0, "temperature_K": 1 / (5 * KB_AU),
        "beads": 2, "dt_au": .003, "seed": 17, "native": False,
        "potential": {"provider": "free"},
        "sampling": {"equilibration_steps": 50, "production_steps": 200,
                     "sample_interval": 2, "andersen_interval": 20},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 4.0},
        "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.0},
        "analysis": {"bins": 30, "bootstrap": 0},
        "output": {"checkpoint_steps": 50},
        "recrossing": {"dividing_surface": 1.0, "parent_replicas": 1,
                       "parent_samples": 2, "equilibration_steps": 20,
                       "parent_interval_steps": 5, "child_pairs": 1,
                       "child_steps": 20, "bootstrap": 0},
    }


def create_cases(output, *, include_incomplete=True):
    """Run real short trajectories and retain all inputs and raw output files."""
    output = Path(output)
    atom = System(("H",), np.array([1.0]), np.zeros((1, 3)))
    exchange = System(("H", "H", "H"), np.array([2., 2., 2.]),
                      np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
                      ((0,), (1, 2)))
    specifications = [
        ("adaptive-1d", short_fes_config(adaptive=True), atom),
        ("fes-2d", short_fes_config(2), atom),
        ("metadynamics-2d", short_meta_config(), atom),
        ("toy-rate", short_rate_config(), exchange),
    ]
    cases = {}
    for name, config, system in specifications:
        path = output / name
        summary = run_resolved(config, system, output=path)
        cases[name] = {"path": path, "summary": summary}
    if include_incomplete:
        config = short_fes_config(adaptive=True)
        config["adaptive_umbrella"]["max_rounds"] = 1
        path = output / "adaptive-incomplete"
        try:
            run_resolved(config, atom, output=path)
        except RuntimeError as exc:
            if "production was not started" not in str(exc):
                raise
        else:
            raise AssertionError("The deliberately exhausted adaptive pilot unexpectedly completed")
        summary = json.loads((path / "summary.json").read_text())
        assert summary["production_started"] is False
        cases["adaptive-incomplete"] = {"path": path, "summary": summary}
    return cases


def snapshot_files(directory):
    """Hash every existing source file, including files not used by reporting."""
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def figure(report, run_index, suffix):
    identifier = f"run-{run_index:03d}-{suffix}"
    return next(item for item in report["figures"] if item["id"] == identifier)


def read_figure_csv(report, run_index, suffix):
    item = figure(report, run_index, suffix)
    with (Path(report["output"]) / item["data"]).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_fes_data(case, report, run_index):
    """Compare exported cells with raw archive arrays, not renderer internals."""
    factor = 1 / unit_factor(report["settings"]["energy_unit"], "energy")
    rows = read_figure_csv(report, run_index, "fes")
    with np.load(case["path"] / "fes.npz", allow_pickle=False) as archive:
        free_energy = archive["free_energy"]
        mask = archive["mask"] | ~np.isfinite(free_energy)
        centers = [(archive[f"edges_{axis}"][:-1] + archive[f"edges_{axis}"][1:]) / 2
                   for axis in range(free_energy.ndim)]
        assert len(rows) == free_energy.size
        if free_energy.ndim == 1:
            np.testing.assert_allclose([float(row["x"]) for row in rows], centers[0] * BOHR_ANGSTROM, atol=1e-14)
            values = np.array([float(row["y"]) if row["y"] else np.nan for row in rows])
            np.testing.assert_allclose(values, np.where(mask, np.nan, free_energy) * factor, equal_nan=True)
        else:
            for row, (i, j) in zip(rows, np.ndindex(free_energy.shape)):
                np.testing.assert_allclose([float(row["x"]), float(row["y"])],
                                           [centers[0][i] * BOHR_ANGSTROM, centers[1][j] * BOHR_ANGSTROM], atol=1e-14)
                assert bool(int(row["masked"])) == mask[i, j]
                if mask[i, j]:
                    assert row["z"] == "", "Unsupported FES bins must remain empty in reusable CSV data"
                else:
                    np.testing.assert_allclose(float(row["z"]), free_energy[i, j] * factor)
        if "overlap" in archive:
            overlap = archive["overlap"]
            overlap_rows = read_figure_csv(report, run_index, "fes-overlap")
            assert len(overlap_rows) == overlap.size
            for row in overlap_rows:
                # Horizontal coordinate is stream j, vertical coordinate is i.
                i, j = int(float(row["y"])), int(float(row["x"]))
                np.testing.assert_allclose(float(row["z"]), overlap[i, j])
    return {"shape": list(free_energy.shape), "supported_bins": int((~mask).sum()),
            "masked_bins": int(mask.sum()), "csv_matches_archive": True}


def validate_report(cases, output, *, formats=("png", "pdf", "svg"), style="nature"):
    """Generate one report and independently verify its numerical exports."""
    before = {name: snapshot_files(case["path"]) for name, case in cases.items()}
    report = generate_report([case["path"] for case in cases.values()], output=output,
                             labels=list(cases), formats=formats, dpi=300,
                             energy_unit="eV", style=style)
    details = {}
    for index, (name, case) in enumerate(cases.items(), 1):
        assert snapshot_files(case["path"]) == before[name], f"Reporting modified source files for {name}"
        saved = report["runs"][index - 1]
        assert saved["converged"] == case["summary"].get("converged")
        if name in ("adaptive-1d", "fes-2d", "metadynamics-2d"):
            details[name] = verify_fes_data(case, report, index)
        if name == "adaptive-1d":
            plan = json.loads((case["path"] / "adaptive-plan.json").read_text())
            assert plan["rounds"][0]["inserted"]
            rows = read_figure_csv(report, index, "sampling-adaptive-windows")
            np.testing.assert_array_equal([int(float(row["y"])) for row in rows],
                                          [len(row["centers"]) for row in plan["rounds"]])
            windows = saved["sampling_diagnostics"]["sampling"]["windows"]
            assert sum(window["samples"] for window in windows) == case["summary"]["samples"]
            assert plan["pilot_samples_used_in_production"] == 0
            assert not any("/adaptation/" in source["path"] for source in report["sources"])
            details[name].update(windows=len(windows), production_samples=case["summary"]["samples"],
                                 pilot_rounds=len(plan["rounds"]), pilot_samples_used_in_production=0)
        elif name == "metadynamics-2d":
            hill_rows = read_figure_csv(report, index, "sampling-hill-height")
            assert len(hill_rows) == sum(case["summary"]["hills_per_replica"])
            histories = sorted((case["path"] / "replicas").glob("*/HILLS.csv"))
            expected = []
            for history in histories:
                with history.open(newline="") as stream:
                    expected.extend((float(row["time_au"]) * AU_TIME_FS,
                                     float(row["height_hartree"]) / unit_factor("eV", "energy"))
                                    for row in csv.DictReader(stream))
            np.testing.assert_allclose([(float(row["x"]), float(row["y"])) for row in hill_rows], expected)
            assert not any(item["id"] == f"run-{index:03d}-fes-error" for item in report["figures"])
            details[name].update(hills_per_replica=case["summary"]["hills_per_replica"],
                                 biased_clock_matches_saved_hills=True)
        elif name == "toy-rate":
            rows = read_figure_csv(report, index, "recrossing")
            with np.load(case["path"] / "recrossing.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose([float(row["x"]) for row in rows], archive["time_au"] * AU_TIME_FS)
                np.testing.assert_allclose([float(row["y"]) if row["y"] else np.nan for row in rows],
                                           archive["kappa"], equal_nan=True)
            assert saved["metrics"]["rate"] == case["summary"].get("rate_cm3_s")
            details[name] = {"child_trajectories": case["summary"]["child_trajectories"],
                             "recrossing_csv_matches_archive": True,
                             "source_rate_acceptance_preserved": True}
        elif name == "adaptive-incomplete":
            assert any(item["id"] == f"run-{index:03d}-sampling-adaptive-windows" for item in report["figures"])
            assert not any(item["id"] == f"run-{index:03d}-fes" for item in report["figures"])
            details[name] = {"production_started": False, "pilot_diagnostics_available": True,
                             "final_fes_not_fabricated": True}
    for item in report["figures"]:
        for fmt in formats:
            path = Path(report["output"]) / item["files"][fmt]
            assert path.stat().st_size > 100, f"Missing or empty {fmt} export: {path}"
    return report, {
        "schema": "pyrpmd.analysis-validation", "schema_version": 1,
        "passed": True, "physical_convergence_claimed": False,
        "scope": "Short deterministic artificial-potential workflow, file-integrity and numerical plot-data checks",
        "source_files_unchanged": True, "source_file_counts": {name: len(files) for name, files in before.items()},
        "figures": len(report["figures"]), "formats": list(formats), "style": style, "cases": details,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--style", choices=("jctc", "nature"), default="nature")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")
    started = time.perf_counter()
    cases = create_cases(args.output / "runs")
    _, evidence = validate_report(cases, args.output / "report", formats=args.formats, style=args.style)
    evidence.update(elapsed_seconds=time.perf_counter() - started, python=platform.python_version())
    path = args.output / "validation.json"
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
