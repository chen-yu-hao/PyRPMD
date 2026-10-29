"""Collect all required real-system benchmark artifacts into a reviewable report.

This command needs only the Python standard library. A green report requires
all three simulation reports and the independent PES/NVE evidence to exist.
It does not upgrade a finite-sampling run to a converged physical rate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def compact(value, key=""):
    if key in ("final_states", "classifier_contract_checks"):
        return {"entries": len(value), "see": "benchmark.json"}
    if isinstance(value, dict):
        return {
            k: compact(v, k)
            for k, v in value.items()
            if k not in ("raw_directory", "files")
        }
    if isinstance(value, list):
        if len(value) > 16:
            return {
                "entries": len(value),
                "first_entries": value[:4],
                "see": "benchmark.json",
            }
        return [compact(v) for v in value]
    return value


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    reports = {}
    for path in sorted(root.rglob("benchmark.json")):
        if "resume-evidence" in path.relative_to(root).parts:
            continue
        value = json.loads(path.read_text())
        task = value.get("task")
        if task in reports:
            raise ValueError(f"Duplicate {task} benchmark report under {root}")
        reports[task] = (path, value)
    expected = {"drpmd", "fes2d", "rate"}
    if set(reports) != expected:
        raise ValueError(f"Expected {sorted(expected)}, found {sorted(reports)}")
    for task, (_, value) in reports.items():
        if value.get("passed") is not True:
            raise ValueError(f"{task} did not pass all physical/numerical gates")
    pes = sorted(root.rglob("pes-check.json"))
    numerical = sorted(root.rglob("numerics.json"))
    if not pes or len(numerical) != 1:
        raise ValueError("Missing independent BKMP2 oracle or NVE validation report")
    for path in pes:
        if json.loads(path.read_text()).get("passed") is not True:
            raise ValueError(f"BKMP2 author-reference verification failed: {path}")
    nve = json.loads(numerical[0].read_text())
    if nve.get("status") != "passed":
        raise ValueError("NVE numerical validation did not pass")
    commit = os.environ.get("GITHUB_SHA", "local checkout; see run manifests")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    lines = [
        "# PyRPMD: physical H + H₂ validation",
        "",
        f"Commit: `{commit}`.",
        "",
        "All required stages executed on the BKMP2 H₃ potential: direct RPMD collisions, "
        "two-dimensional centroid free energy, flux-metric umbrella PMF and "
        "Bennett–Chandler recrossing. Numerical gates passed. **These are finite-sampling "
        "physical-system validation runs, not converged chemical rate constants.**",
        "",
    ]
    if repo and run_id:
        lines.extend(
            [
                f"[GitHub Actions run](https://github.com/{repo}/actions/runs/{run_id})",
                "",
            ]
        )
    lines.extend(
        [
            "## Stage checks",
            "",
            "| Stage | Temperature / K | Beads | Time step / fs | Gate result |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    names = {"drpmd": "dRPMD", "fes2d": "2D FES", "rate": "Umbrella PMF + recrossing"}
    for task in ("drpmd", "fes2d", "rate"):
        value = reports[task][1]
        settings = value.get("settings", value)
        lines.append(
            f"| {names[task]} | {settings['temperature_K']:g} | {settings['beads']} | {settings['timestep_fs']:g} | Passed |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- BKMP2 is a published reactive H₃ potential; source identity and compiler flags "
            "are recorded in each PES report. No teaching potential is substituted.",
            "- Both equivalent H₂ exchange memberships are represented explicitly. "
            "The total exchange estimator does not add an arbitrary factor of two.",
            "- Native/NumPy comparison and time-step refinement test numerical integration. "
            "They do not establish equilibrium or bead-number convergence.",
            "- The FES contains the sampled centroid-coordinate density and preserves masks "
            "where data are insufficient. It is not a contour plot of the bare PES.",
            "- Overlap, bootstrap and a recrossing time plateau are diagnostics at the stated "
            "sample size. Bead count, sampling length, parent correlation, R∞/R₀ and bmax "
            "need independent convergence studies before reporting a production rate.",
            "- The rate report compares Gaussian umbrella integration against histogram MBAR "
            "on the same samples and unchanged dividing surface. An estimator discrepancy "
            "is reported even when numerical gates pass. The MBAR rate interval is conditional "
            "on the measured transmission coefficient; it does not include its uncertainty.",
            "",
            "## Independent PES and integration checks",
            "",
            "### BKMP2 author-reference comparison",
            "",
            "```json",
            json.dumps(
                compact(json.loads(pes[0].read_text())), indent=2, ensure_ascii=False
            ),
            "```",
            "",
            "### NVE and native/NumPy comparison",
            "",
            "```json",
            json.dumps(compact(nve), indent=2, ensure_ascii=False),
            "```",
        ]
    )
    for task in ("drpmd", "fes2d", "rate"):
        path, value = reports[task]
        lines.extend(
            [
                "",
                f"## {names[task]}",
                "",
                "```json",
                json.dumps(compact(value), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
        for figure in sorted(path.parent.glob("figures/*.png")):
            relative = os.path.relpath(figure, output.parent)
            lines.extend([f"![{names[task]}]({relative})", ""])
    lines.extend(
        [
            "## Sources",
            "",
            "- Boothroyd et al., *A Refined H₃ Potential Energy Surface*, J. Chem. Phys. 104, "
            "7139–7152 (1996), https://doi.org/10.1063/1.471430.",
            "- Author code and numerical test: https://www.cita.utoronto.ca/~boothroy/bkmp2.html.",
            "- Suleimanov, Allen and Green, *RPMDrate*, Comput. Phys. Commun. 184, 833–840 "
            "(2013), https://doi.org/10.1016/j.cpc.2012.10.017.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    machine = {
        "schema": "pyrpmd.h3.validation-summary.v1",
        "commit": commit,
        "github_run_id": run_id,
        "physical_convergence_demonstrated": False,
        "stages": {task: compact(value) for task, (_, value) in reports.items()},
        "pes_check": compact(json.loads(pes[0].read_text())),
        "numerics": nve,
    }
    output.with_suffix(".json").write_text(
        json.dumps(machine, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Machine-readable evidence remains in the Actions log after artifact expiry.
    print(
        "PYRPMD_H3_REPORT " + json.dumps(machine, ensure_ascii=False, allow_nan=False)
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.directory, args.output)
