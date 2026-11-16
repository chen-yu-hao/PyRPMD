"""Markdown report and aggregate metrics for ``benchmarks/subspace_complex``.

Usage:

    python -m benchmarks.subspace_complex.report \\
        benchmarks/subspace_complex/results/complex_pes.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SUBSPACE_MODELS = ("1d", "1d_sct", "plane", "rph")


def _present(case: dict[str, Any]) -> list[str]:
    """Subspace models solved for a case, in report order."""
    return [m for m in SUBSPACE_MODELS if m in case["subspace"]]


def _label(m: str, entry: dict[str, Any]) -> str:
    """Model name, marked when a validity gate rejected it (older results carry no record)."""
    validity = entry.get("validity")
    if validity is None or validity.get("valid", True):
        return f"`{m}`"
    failed = [g for g, ok in validity.get("gates", {}).items() if not ok]
    return f"`{m}` (rejected: {', '.join(failed)})"


def _rms(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _max_abs(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else float("nan")


def summarize_complex_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate unanchored vs skeleton-anchored error metrics across exact-reference cases."""
    by_model: dict[str, dict[str, list[float]]] = {
        m: {"unanchored": [], "skeleton": []} for m in SUBSPACE_MODELS
    }
    exact_cases = []
    molecular_cases = []

    for key, case in data["cases"].items():
        if case.get("full_reference") is not None:
            exact_cases.append(key)
            for m in _present(case):
                entry = case["subspace"][m]
                by_model[m]["unanchored"].append(float(entry["transfer_error_log"]))
                if entry.get("anchored_skeleton_error_log") is not None:
                    by_model[m]["skeleton"].append(float(entry["anchored_skeleton_error_log"]))
        else:
            molecular_cases.append(key)

    summary_models = {}
    for m, bucket in by_model.items():
        summary_models[m] = {
            "count": len(bucket["unanchored"]),
            "valid_skeleton_count": len(bucket["skeleton"]),
            "rms_unanchored": _rms(bucket["unanchored"]),
            "rms_skeleton": _rms(bucket["skeleton"]),
            "max_unanchored": _max_abs(bucket["unanchored"]),
            "max_skeleton": _max_abs(bucket["skeleton"]),
        }

    return {
        "exact_reference_cases": exact_cases,
        "molecular_cartesian_cases": molecular_cases,
        "by_subspace_model": summary_models,
    }


def format_complex_report(data: dict[str, Any]) -> str:
    """Render Markdown validation tables for ``complex_pes.json``."""
    metrics = summarize_complex_metrics(data)
    lines: list[str] = []

    lines.append("## 1. High-dimensional complex PES benchmark against exact full-core quantum/RPMD rates")
    lines.append("")
    lines.append(
        "| Case | $D$ | Topology | $T$ (K) | $T/T_c$ | $\\ln C_{\\mathrm{exact}}$ | $A_H$ | "
        "Model | $\\ln C_S$ | $A_H - A_{S^0}$ | $\\Delta = \\ln(C_{\\mathrm{exact}}/C_S)$ | "
        "$\\Delta - (A_H - A_{S^0})$ (skel) |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")

    for key in metrics["exact_reference_cases"]:
        case = data["cases"][key]
        ln_c = float(case["full_reference"]["ln_C_exact"])
        a_h = float(case["full_instanton"]["log_factor"])
        for idx, m in enumerate(_present(case)):
            e = case["subspace"][m]
            se = float(e["combined_log_standard_error"])
            head = (
                f"| `{key}` | {case['dimension']} | `{case['topology']}` | "
                f"{case['temperature_K']:.0f} | {case['t_over_tc']:.2f} | "
                f"{ln_c:+.3f} | {a_h:+.3f} "
                if idx == 0
                else "|  |  |  |  |  |  |  "
            )
            shift_str = f"{e['anchor_log_shift']:+.3f}" if e.get("anchor_log_shift") is not None else "invalid"
            skel_err_str = (
                f"{e['anchored_skeleton_error_log']:+.3f} ± {se:.3f}"
                if e.get("anchored_skeleton_error_log") is not None
                else "invalid"
            )
            lines.append(
                f"{head}| {_label(m, e)} | {e['ln_C_S']:+.3f} | {shift_str} | "
                f"{e['transfer_error_log']:+.3f} ± {se:.3f} | "
                f"{skel_err_str} |"
            )

    lines.append("")
    lines.append("## 2. Aggregate log-rate residuals across high-dimensional complex PESs (6D--10D)")
    lines.append("")
    lines.append("| Subspace model | $N$ | RMS $|\\Delta|$ (unanchored) | RMS skel-anchored | Max $|\\Delta|$ | Max skel-anchored |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for m in SUBSPACE_MODELS:
        s = metrics["by_subspace_model"][m]
        if not s["count"]:
            continue
        lines.append(
            f"| `{m}` | {s['count']} | {s['rms_unanchored']:.3f} | {s['rms_skeleton']:.3f} | "
            f"{s['max_unanchored']:.3f} | {s['max_skeleton']:.3f} |"
        )

    if metrics["molecular_cartesian_cases"]:
        lines.append("")
        lines.append("## 3. 9D 3-atom Cartesian LEPS + ion--dipole complex (`rigid_body=True`, `RPMDrateCoordinate`)")
        lines.append("")
        lines.append(
            "| Case | $T$ (K) | $T/T_c$ | $A_H$ (9D) | $\\delta A_{\\mathrm{opt}}$ | "
            "Model | $\\ln C_S$ | $A_H - A_{S^0}$ | $\\ln C_S^{\\mathrm{skel}}$ | "
            "$\\ln(C_{\\mathrm{RPH}}/C_S)$ | $\\ln(C_{\\mathrm{RPH}}^{\\mathrm{skel}}/C_S^{\\mathrm{skel}})$ |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for key in metrics["molecular_cartesian_cases"]:
            case = data["cases"][key]
            a_h = float(case["full_instanton"]["log_factor"])
            chk = case["full_instanton"].get("crosscheck") or {}
            diff_opt = float(chk.get("log_difference", float("nan")))
            for idx, m in enumerate(_present(case)):
                e = case["subspace"][m]
                head = (
                    f"| `{key}` | {case['temperature_K']:.0f} | {case['t_over_tc']:.2f} | "
                    f"{a_h:+.3f} | {diff_opt:+.1e} "
                    if idx == 0
                    else "|  |  |  |  |  "
                )
                shift_str = f"{e['anchor_log_shift']:+.3f}" if e.get("anchor_log_shift") is not None else "invalid"
                ln_anch_str = f"{e['ln_C_S_anchored']:+.3f}" if e.get("ln_C_S_anchored") is not None else "invalid"
                vs_skel_str = (
                    f"{e['vs_rph_skeleton_log']:+.3f}" if e.get("vs_rph_skeleton_log") is not None else "invalid"
                )
                lines.append(
                    f"{head}| {_label(m, e)} | {e['ln_C_S']:+.3f} | {shift_str} | "
                    f"{ln_anch_str} | {e['vs_rph_unanchored_log']:+.3f} | "
                    f"{vs_skel_str} |"
                )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    sys.stdout.write(format_complex_report(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
