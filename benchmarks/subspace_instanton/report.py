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
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return {"qm": read(directory / "qm.json"), "skeleton": read(directory / "skeleton.json"),
            "rpmd": [read(path) for path in sorted(directory.glob("rpmd_*.json"))],
            "eckart": [read(path) for path in sorted(directory.glob("eckart_*.json"))]}


def adiabatic_height(record: dict, model) -> float:
    """``V_ad / (hbar lambda)``: the barrier plus the change of transverse zero-point energy."""
    return (model.V0 + 0.5 * (record["stable_frequency"] - model.omega_inf)) / record["unstable_frequency"]


def eckart_entry(results: dict, u: float, v: float, nbeads: int | None = None):
    """The Eckart run at ``(u, v)``; with several bead numbers the default one unless ``nbeads`` is given."""
    default = 32 if u <= 6.0 else (64 if u <= 9.0 else 128)
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
               "runs": len(full["runs"]),
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
    lines = ["| model | T/T_c | P | runs | kappa | B_P | eps(H) | ln eps(H) | A_H | R_H = ln eps − A_H |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} ({MODELS[r['model']]}) | {r['ratio']} | {r['nbeads']} | {r['runs']} | {r['kappa']:.3f} | "
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
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in sections.items():
        text = text.replace("{{" + key + "}}", value)
    return text


MODEL_NAMES = ("M1", "M2", "M3", "M4")
REDUCTION_ORDER = ("mep", "fe", "fe+sct", "bare")


def _rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _max_abs(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else float("nan")


def summarize_metrics(rpmd: dict, *, below_ratios=(0.8, 0.65, 0.55)) -> dict:
    """Compute aggregate RMS and max-abs log-rate residuals from merged RPMD results."""
    below_set = {float(r) for r in below_ratios}
    by_reduction = {key: {"unanchored": [], "skeleton": [], "naive": []} for key in REDUCTION_ORDER}
    all_below = {"unanchored": [], "skeleton": [], "naive": []}
    m1_fe = {"unanchored": [], "skeleton": [], "naive": []}
    between_crossovers = []
    eckart_all = []
    eckart_below = []

    for model in MODEL_NAMES:
        m_entry = rpmd["models"][model]
        for r_str, t_entry in m_entry["temperatures"].items():
            ratio = float(r_str)
            eck_err = float(t_entry["eckart_estimate"]["error"])
            eckart_all.append(eck_err)
            if ratio in below_set:
                eckart_below.append(eck_err)
            a_h = float(t_entry["full"]["ln_eps_sd"])
            for key, red in t_entry["reductions"].items():
                unanch = float(red["transfer_error"])
                skel = float(red["anchored_skeleton_error"])
                naive = float(red["anchored_naive_error"])
                a_s0 = float(red["ln_eps_sd_skeleton"])
                if (a_h > 0.0) != (a_s0 > 0.0):
                    between_crossovers.append({
                        "model": model,
                        "ratio": ratio,
                        "reduction": key,
                        "skeleton": red["skeleton"],
                        "transfer_error": unanch,
                        "anchored_skeleton_error": skel,
                    })
                if ratio in below_set:
                    by_reduction[key]["unanchored"].append(unanch)
                    by_reduction[key]["skeleton"].append(skel)
                    by_reduction[key]["naive"].append(naive)
                    all_below["unanchored"].append(unanch)
                    all_below["skeleton"].append(skel)
                    all_below["naive"].append(naive)
                    if model == "M1" and key == "fe":
                        m1_fe["unanchored"].append(unanch)
                        m1_fe["skeleton"].append(skel)
                        m1_fe["naive"].append(naive)

    def _pack(bucket):
        return {
            "count": len(bucket["unanchored"]),
            "rms_unanchored": _rms(bucket["unanchored"]),
            "rms_skeleton": _rms(bucket["skeleton"]),
            "rms_naive": _rms(bucket["naive"]),
            "max_unanchored": _max_abs(bucket["unanchored"]),
            "max_skeleton": _max_abs(bucket["skeleton"]),
            "max_naive": _max_abs(bucket["naive"]),
        }

    return {
        "below_ratios": [float(r) for r in below_ratios],
        "overall_below_tc": _pack(all_below),
        "by_reduction_below_tc": {k: _pack(v) for k, v in by_reduction.items()},
        "m1_fe_below_tc": _pack(m1_fe),
        "eckart": {
            "count_all": len(eckart_all),
            "rms_all": _rms(eckart_all),
            "max_all": _max_abs(eckart_all),
            "count_below_tc": len(eckart_below),
            "rms_below_tc": _rms(eckart_below),
            "max_below_tc": _max_abs(eckart_below),
        },
        "between_crossovers": between_crossovers,
    }


def _fmt_signed(val: float, digits: int = 3) -> str:
    return "nan" if np.isnan(val) else f"{val:+.{digits}f}"


def format_report(rpmd: dict, *, transfer_ratios=(1.0, 0.8, 0.65, 0.55)) -> str:
    """Render Markdown summary tables for full defects, Eckart estimates, and anchored transfers."""
    metrics = summarize_metrics(rpmd)
    lines: list[str] = []
    lines.append("## 1. Full 2D RPMD defect, steepest-descent factor and matched Eckart estimate")
    lines.append("")
    lines.append(
        "| Model | $T/T_c$ | $\\varepsilon(H)$ | $\\ln\\varepsilon(H)$ | $A_H$ | "
        "$R_H=\\ln\\varepsilon(H)-A_H$ | $\\ln C_{\\mathrm{Eck}}$ | $\\ln\\varepsilon(H)-\\ln C_{\\mathrm{Eck}}$ |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODEL_NAMES:
        temps = rpmd["models"][model]["temperatures"]
        for idx, ratio in enumerate(rpmd["temperature_ratios"]):
            t_entry = temps[str(ratio)]
            full = t_entry["full"]
            eck = t_entry["eckart_estimate"]
            eps = float(full["eps"])
            eps_err = eps * float(full["relative_error"])
            ln_eps = float(full["ln_eps"])
            ln_err = float(full["relative_error"])
            a_h = float(full["ln_eps_sd"])
            rem = float(full["remainder"])
            ln_eck = float(eck["ln_c_eckart"])
            eck_err = float(eck["error"])
            eck_se = float(eck["standard_error"])
            label = model if idx == 0 else ""
            lines.append(
                f"| {label} | {ratio:.2f} | {eps:.3f} ± {eps_err:.3f} | "
                f"{_fmt_signed(ln_eps)} ± {ln_err:.3f} | {a_h:.3f} | {_fmt_signed(rem)} | "
                f"{_fmt_signed(ln_eck)} | {_fmt_signed(eck_err)} ± {eck_se:.3f} |"
            )

    lines.append("")
    lines.append("## 2. Subspace transfer error and instanton-anchored residuals")
    lines.append("")
    lines.append(
        "| Model | $T/T_c$ | $S$ | $S^0$ | $\\Delta=\\ln\\varepsilon(H)-\\ln C_S$ | "
        "$A_H-A_{S^0}$ | $\\Delta-(A_H-A_{S^0})$ (skel) | $A_H-A_S$ | $\\Delta-(A_H-A_S)$ (naive) |"
    )
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model in MODEL_NAMES:
        temps = rpmd["models"][model]["temperatures"]
        first_model = True
        for ratio in transfer_ratios:
            t_entry = temps[str(ratio)]
            first_temp = True
            for key in REDUCTION_ORDER:
                if key not in t_entry["reductions"]:
                    continue
                red = t_entry["reductions"][key]
                se = float(red["transfer_standard_error"])
                m_col = model if first_model else ""
                t_col = f"{ratio:.2f}" if first_temp else ""
                first_model = False
                first_temp = False
                lines.append(
                    f"| {m_col} | {t_col} | `{key}` | `{red['skeleton']}` | "
                    f"{_fmt_signed(float(red['transfer_error']))} ± {se:.3f} | "
                    f"{_fmt_signed(float(red['delta_sd_skeleton']))} | "
                    f"{_fmt_signed(float(red['anchored_skeleton_error']))} ± {se:.3f} | "
                    f"{_fmt_signed(float(red['delta_sd_naive']))} | "
                    f"{_fmt_signed(float(red['anchored_naive_error']))} ± {se:.3f} |"
                )

    lines.append("")
    lines.append("## 3. Aggregate log-rate residuals below $T_c$ ($T/T_c \\in \\{0.8, 0.65, 0.55\\}$)")
    lines.append("")
    lines.append(
        "| Subset | $N$ | RMS $|\\Delta|$ | RMS skel | RMS naive | "
        "Max $|\\Delta|$ | Max skel | Max naive |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = [("All reductions", metrics["overall_below_tc"]), ("M1 `fe` (1st-order coupling)", metrics["m1_fe_below_tc"])]
    for key in REDUCTION_ORDER:
        rows.append((f"`{key}`", metrics["by_reduction_below_tc"][key]))
    for title, item in rows:
        lines.append(
            f"| {title} | {item['count']} | {item['rms_unanchored']:.3f} | "
            f"{item['rms_skeleton']:.3f} | {item['rms_naive']:.3f} | "
            f"{item['max_unanchored']:.3f} | {item['max_skeleton']:.3f} | {item['max_naive']:.3f} |"
        )
    eck = metrics["eckart"]
    lines.append("")
    lines.append(
        f"Matched Eckart estimate $\\ln C_{{\\mathrm{{Eck}}}}$: "
        f"RMS error {eck['rms_all']:.3f} (max {eck['max_all']:.3f}) across all {eck['count_all']} points; "
        f"RMS {eck['rms_below_tc']:.3f} (max {eck['max_below_tc']:.3f}) across {eck['count_below_tc']} points below $T_c$."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and not args_list[0].startswith("-"):
        rpmd = json.loads(Path(args_list[0]).read_text(encoding="utf-8"))
        sys.stdout.write(format_report(rpmd))
        return 0
    text = render()
    if "--check" in args_list:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT} is not the generated report; run report.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
