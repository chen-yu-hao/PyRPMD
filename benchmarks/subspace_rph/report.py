"""Tabulate the RPH 2D validation results (results/*.json) as Markdown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = {"1d": "1D", "1d_sct": "1D + SCT", "plane": "2D plane", "rph": "2D RPH"}


def load(name):
    path = HERE / "results" / name
    return json.loads(path.read_text()) if path.is_file() else None


def correction_table(results):
    lines = ["| model | T (K) | P | exact `C` | " + " | ".join(f"`C_S` {v}" for v in NAMES.values()) + " |",
             "|---|---|---|---|" + "---|" * len(NAMES)]
    for report in results:
        if report is None:
            continue
        full = report.get("full")
        exact = (f"{full['C']:.3f} ± {full['C'] * full['log_standard_error']:.3f}" if full else "(base model)")
        cells = []
        for key in NAMES:
            entry = report["subspace"].get(key)
            if entry is None:
                cells.append("–")
                continue
            text = f"{entry['C_S']:.3f} ± {entry['C_S'] * entry['log_standard_error']:.3f}"
            if "error_vs_full" in entry:
                text += f" ({100 * entry['error_vs_full']:+.0f}%)"
            cells.append(text)
        lines.append(f"| {report['model']} | {report['temperature_K']:.0f} | {report['nbeads']} | {exact} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def rate_table(results):
    lines = ["| model | T (K) | case | grid | k_QM | k_RPMD | kappa | quantum s | RPMD s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for report in results:
        if report is None:
            continue
        cases = [(NAMES[k], v) for k, v in report["subspace"].items()]
        if report.get("full"):
            cases.append(("full", report["full"]))
        for name, entry in cases:
            lines.append(f"| {report['model']} | {report['temperature_K']:.0f} | {name} | "
                         f"{'×'.join(map(str, entry['shape']))} | {entry['quantum']['rate']:.4g} | "
                         f"{entry['rpmd']['rate']:.4g} | {entry['rpmd']['transmission']:.3f} | "
                         f"{entry['quantum']['seconds']:.0f} | {entry['rpmd']['seconds']:.0f} |")
    return "\n".join(lines)


def same_slice_table(report):
    if report is None:
        return "(not run)"
    lines = ["| RPH spacing | slice X_ref | k_QM(RPH) | full spacing | k_QM(full, same slice) | ratio |",
             "|---|---|---|---|---|---|"]
    for row in report["rows"]:
        for full in row["full"]:
            lines.append(f"| {row['rph_spacing'][0]:.2f} | {row['reference']:.2f} | {row['rph_rate']:.4g} | "
                         f"{full['spacing']:.1f} | {full['rate']:.4g} | {row['rph_rate'] / full['rate']:.3f} |")
    return "\n".join(lines)


def embedding_table(base, embedded):
    if base is None or embedded is None:
        return "(not run)"
    lines = ["| case | `C_S` (3D model) | `C_S` (8D, rotated) | ratio | captured curvature 3D / 8D |",
             "|---|---|---|---|---|"]
    for key, name in NAMES.items():
        a, b = base["subspace"].get(key), embedded["subspace"].get(key)
        if a is None or b is None:
            continue
        error = np.hypot(a["log_standard_error"], b["log_standard_error"])
        captured = (f"{a['captured_curvature']:.3f} / {b['captured_curvature']:.3f}"
                    if a.get("captured_curvature") is not None else "–")
        lines.append(f"| {name} | {a['C_S']:.3f} | {b['C_S']:.3f} | {b['C_S'] / a['C_S']:.3f} ± {error:.3f} | "
                     f"{captured} |")
    return "\n".join(lines)


def main() -> int:
    runs = [load(name) for name in ("curved_2d_300K.json", "curved_2d_200K.json", "curved_3d_300K.json",
                                    "embedded_8d_300K.json")]
    sections = {
        "CORRECTIONS": correction_table(runs),
        "RATES": rate_table(runs),
        "SAME_SLICE": same_slice_table(load("same_slice_300K.json")),
        "EMBEDDING": embedding_table(runs[2], runs[3]),
    }
    template = Path(sys.argv[1]).read_text()
    for key, value in sections.items():
        template = template.replace("{{" + key + "}}", value)
    Path(sys.argv[2]).write_text(template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
