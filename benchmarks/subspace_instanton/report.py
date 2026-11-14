"""Tabulate the instanton-anchor tests (``results/*.json``) as Markdown.

    python report.py                # writes docs/validation/2026-11-14-subspace-instanton-anchor/README.md
    python report.py --check        # fails if that file is not the generated one

All quantities are logarithms of rate ratios.  For a model ``H`` and a
reduction ``S`` with skeleton ``S0`` (same path and mass, bare potential):

``ln eps``      ``ln[k_QM / k_RPMD]``, the RPMD defect (``ln C_S`` for a reduction);
``A``           ``ln eps_SD = ln[2 pi / (beta eta_1)]``, its steepest-descent part;
``transfer``    ``Delta = ln eps(H) - ln eps(S)``, the error of ``k_RPMD(H) C_S``;
``anchor``      ``A_H - A_S0`` (skeleton) and ``A_H - A_S`` (naive), the predicted ``Delta``;
``G``           ``ln[k_QM(H) / k_QM(S)]``, the error of the reduction used on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TEMPLATE = HERE / "README.template.md"
OUTPUT = HERE.parents[1] / "docs" / "validation" / "2026-11-14-subspace-instanton-anchor" / "README.md"
MODELS = {"M1": "zero-point change", "M2": "curved path", "M3": "both", "M4": "slow bath"}
SKELETON = {"mep": "mep", "fe": "mep", "fe+sct": "mep+sct", "bare": "bare"}


def load_results(directory: Path = RESULTS) -> dict:
    read = lambda path: json.loads(path.read_text())
    return {"qm": read(directory / "qm.json"), "skeleton": read(directory / "skeleton.json"),
            "rpmd": [read(path) for path in sorted(directory.glob("rpmd_*.json"))],
            "eckart": [read(path) for path in sorted(directory.glob("eckart_*.json"))]}


def adiabatic_height(record: dict, model) -> float:
    """``V_ad / (hbar lambda)``: the barrier plus the change of transverse zero-point energy."""
    return (model.V0 + 0.5 * (record["stable_frequency"] - model.omega_inf)) / record["unstable_frequency"]


def eckart_entry(results: dict, u: float, v: float, nbeads: int | None = None):
    """The Eckart run at ``(u, v)``; with several bead numbers the default one unless ``nbeads`` is given."""
    default = 32 if u <= 6.0 else (48 if u <= 9.0 else 64)
    for entry in results["eckart"]:
        if abs(entry["u"] - u) < 1e-9 and abs(entry["v"] - v) < 1e-9 and entry["nbeads"] == (nbeads or default):
            return entry
    return None


def transfer_rows(results: dict) -> list[dict]:
    """One row per model and temperature with the full defect and every reduction's transfer error."""
    sys.path.insert(0, str(HERE.parents[1]))
    from benchmarks.subspace_instanton.models import family

    rows = []
    for result in results["rpmd"]:
        name, ratio = result["model"], result["temperature_ratio"]
        record = results["qm"]["models"][name]
        entry = record["temperatures"][str(ratio)]
        skeleton = results["skeleton"]["models"][name][str(ratio)]
        full = result["full"]
        row = {"model": name, "ratio": ratio, "beta": result["beta"], "u": result["beta"] * record["unstable_frequency"],
               "v": adiabatic_height(record, family()[name]), "nbeads": full["nbeads"], "kappa": full["kappa"],
               "bead_factor": full["bead_factor"], "eps": full["eps"], "ln_eps": float(np.log(full["eps"])),
               "error": full["relative_error"], "anchor": entry["ln_eps_sd"], "reductions": {}}
        row["remainder"] = row["ln_eps"] - row["anchor"]
        for key, item in result["reductions"].items():
            reduction = entry["reductions"][key]
            ln_eps = float(np.log(item["eps"]))
            transfer = row["ln_eps"] - ln_eps
            anchors = {"anchor_skeleton": row["anchor"] - skeleton[SKELETON[key]],
                       "anchor_naive": row["anchor"] - reduction["ln_eps_sd"]}
            row["reductions"][key] = {
                "g_qm": reduction["g_qm"], "ln_eps": ln_eps, "error_reduction": item["relative_error"],
                "kappa": item["kappa"], "nbeads": item["nbeads"], "transfer": transfer,
                "error": float(np.hypot(row["error"], item["relative_error"])), **anchors,
                "residual_skeleton": transfer - anchors["anchor_skeleton"],
                "residual_naive": transfer - anchors["anchor_naive"],
                # Two SCT instantons far below the crossover are not first-order saddles (see the theory page).
                "anchor_valid": reduction.get("instanton", {}).get("negative_modes", 1) == 1}
        eckart = eckart_entry(results, row["u"], round(row["v"], 3))
        if eckart is not None:
            row["eckart"] = {"remainder": eckart["remainder"], "error": eckart["relative_error"],
                             "estimate": row["anchor"] + eckart["remainder"],
                             "difference": row["remainder"] - eckart["remainder"],
                             "difference_error": float(np.hypot(row["error"], eckart["relative_error"]))}
        rows.append(row)
    order = list(MODELS)
    return sorted(rows, key=lambda r: (order.index(r["model"]), -r["ratio"]))


def _pm(value: float, error: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f} ± {error:.{digits}f}"


def defect_table(rows) -> str:
    lines = ["| model | T/T_c | P | kappa | B_P | eps(H) | ln eps(H) | A_H | R_H = ln eps − A_H |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} ({MODELS[r['model']]}) | {r['ratio']} | {r['nbeads']} | {r['kappa']:.3f} | "
                     f"{r['bead_factor']:.4f} | {r['eps']:.3f} ± {r['eps'] * r['error']:.3f} | "
                     f"{_pm(r['ln_eps'], r['error'])} | {r['anchor']:.3f} | {_pm(r['remainder'], r['error'])} |")
    return "\n".join(lines)


def transfer_table(rows) -> str:
    lines = ["| model | T/T_c | reduction | G (QM alone) | ln C_S | Δ = ln eps(H) − ln C_S | skeleton anchor "
             "| Δ − anchor | naive anchor | Δ − anchor |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for key, item in r["reductions"].items():
            naive = (f"{item['anchor_naive']:+.3f} | {_pm(item['residual_naive'], item['error'])}"
                     if item["anchor_valid"] else "– | –")
            lines.append(f"| {r['model']} | {r['ratio']} | `{key}` | {item['g_qm']:+.3f} | "
                         f"{_pm(item['ln_eps'], item['error_reduction'])} | {_pm(item['transfer'], item['error'])} | "
                         f"{item['anchor_skeleton']:+.3f} | {_pm(item['residual_skeleton'], item['error'])} | {naive} |")
    return "\n".join(lines)


def summary(rows) -> dict:
    """Root-mean-square errors (and chi^2 per point) of the three corrections, per reduction, below T_c."""
    out = {}
    for key in ("mep", "fe", "fe+sct", "bare"):
        items = [r["reductions"][key] for r in rows if r["ratio"] < 1.0 and key in r["reductions"]]
        if not items:
            continue
        stats = {"points": len(items), "g_qm": float(np.sqrt(np.mean([i["g_qm"] ** 2 for i in items])))}
        for name, field in (("plain", "transfer"), ("skeleton", "residual_skeleton"), ("naive", "residual_naive")):
            values = np.array([i[field] for i in items])
            errors = np.array([i["error"] for i in items])
            stats[name] = {"rms": float(np.sqrt(np.mean(values**2))), "max": float(np.abs(values).max()),
                           "chi2": float(np.mean((values / errors) ** 2))}
        out[key] = stats
    return out


def summary_table(rows) -> str:
    lines = ["| reduction | points | rms G | rms Δ: `C_S` | chi² | skeleton-anchored | chi² | naive-anchored | chi² |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary(rows).items():
        lines.append(f"| `{key}` | {s['points']} | {s['g_qm']:.3f} | {s['plain']['rms']:.3f} | {s['plain']['chi2']:.1f} | "
                     f"{s['skeleton']['rms']:.3f} | {s['skeleton']['chi2']:.1f} | {s['naive']['rms']:.3f} | "
                     f"{s['naive']['chi2']:.1f} |")
    return "\n".join(lines)


def above_table(rows) -> str:
    """Transfer errors above the crossover, where every anchor is zero."""
    lines = ["| model | reduction | G (QM alone) | Δ |", "|---|---|---|---|"]
    for r in rows:
        if r["ratio"] > 1.0:
            for key, item in r["reductions"].items():
                lines.append(f"| {r['model']} | `{key}` | {item['g_qm']:+.3f} | {_pm(item['transfer'], item['error'])} |")
    return "\n".join(lines)


def eckart_table(results) -> str:
    lines = ["| u = βħω_b | v = V_0/ħω_b | P | kappa | eps | ln eps | A(u) | R = ln eps − A |",
             "|---|---|---|---|---|---|---|---|"]
    for e in sorted(results["eckart"], key=lambda e: (e["v"], e["u"], e["nbeads"])):
        lines.append(f"| {e['u']:.3f} | {e['v']:.3f} | {e['nbeads']} | {e['kappa']:.3f} | "
                     f"{e['eps']:.3f} ± {e['eps'] * e['relative_error']:.3f} | {np.log(e['eps']):+.3f} | "
                     f"{e['ln_eps_sd']:.3f} | {_pm(e['remainder'], e['relative_error'])} |")
    return "\n".join(lines)


def path_free_table(rows) -> str:
    lines = ["| model | T/T_c | u | v | ln eps(H) | A_H | R_Eck(u, v) | ln C_Eck = A_H + R_Eck | ln eps(H) − ln C_Eck |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r.get("eckart")
        if e is None:
            continue
        lines.append(f"| {r['model']} | {r['ratio']} | {r['u']:.3f} | {r['v']:.3f} | {_pm(r['ln_eps'], r['error'])} | "
                     f"{r['anchor']:.3f} | {_pm(e['remainder'], e['error'])} | {e['estimate']:+.3f} | "
                     f"{_pm(e['difference'], e['difference_error'])} |")
    return "\n".join(lines)


def path_free_summary(rows) -> str:
    values = np.array([r["eckart"]["difference"] for r in rows if "eckart" in r])
    errors = np.array([r["eckart"]["difference_error"] for r in rows if "eckart" in r])
    if not len(values):
        return "(not run)"
    return (f"rms {np.sqrt(np.mean(values**2)):.3f}, largest {np.abs(values).max():.3f}, "
            f"chi² per point {np.mean((values / errors) ** 2):.1f} ({len(values)} points)")


def render(results: dict | None = None) -> str:
    results = load_results() if results is None else results
    rows = transfer_rows(results)
    sections = {"DEFECT": defect_table(rows), "TRANSFER": transfer_table(rows), "SUMMARY": summary_table(rows),
                "ABOVE": above_table(rows), "ECKART": eckart_table(results), "PATH_FREE": path_free_table(rows),
                "PATH_FREE_SUMMARY": path_free_summary(rows)}
    text = TEMPLATE.read_text()
    for key, value in sections.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def main() -> int:
    text = render()
    if "--check" in sys.argv[1:]:
        if not OUTPUT.is_file() or OUTPUT.read_text() != text:
            print(f"{OUTPUT} is not the generated report; run report.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
