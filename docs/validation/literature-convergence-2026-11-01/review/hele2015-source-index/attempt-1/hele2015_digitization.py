#!/usr/bin/env python3
"""Read pinned author figure PDFs; never compute or approve a physical rate.

PyMuPDF is an optional, offline source-audit dependency. Original PDFs/TeX are
external inputs, not repository assets. The coordinate sensitivity interval is
not a paper error bar, statistical confidence interval, or acceptance margin.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SOURCES = {
    "hel15trpmd.pdf": "3a3b1c2d257d7af0f56ffb8ec195e4e6db9e8d729d57390a4bcdb62623e3ace9",
    "arxiv-1506.07460v2-source.tar.gz": "f24cd312ced5125a8f67137314b1db7d0e1c9be381d16ec0cf7ef24fd4977546",
    "t.tex": "35f742da39e3747c6cb792b2eb856ed705db0f2cd9d2e0034dfafa4a0bb69fdf",
    "fhh.pdf": "f5953a36abf65ddb6ca3d2965a6bde62c26113d8e2a76357d5598d6c43d15347",
    "dmuh.pdf": "e38db2b3844f694e9801d4110284167917952a84ba193e56dc32d7c6aebfeddf",
    "hhh.pdf": "91ef2ee5dc824cacfa7022adad9672e1e1f617b8c30408acc74f2754b9787b94",
    "ch4h.pdf": "485ae36a16a1a46b12410fdc24f796beb4af158d2c9ba8b994a1b433a1a15a95",
}
FIGURES = {
    "hhh.pdf": {"reaction": "H + H2", "figure": 4, "temperatures_K": [200, 300, 500], "beads": [128, 128, 128], "tick_count": 5},
    "dmuh.pdf": {"reaction": "D + MuH", "figure": 5, "temperatures_K": [200, 300, 500], "beads": [512, 512, 512], "tick_count": 3},
    "ch4h.pdf": {"reaction": "H + CH4", "figure": 6, "temperatures_K": [250, 300, 500], "beads": [192, 128, 64], "tick_count": 6},
    "fhh.pdf": {"reaction": "F + H2", "figure": 7, "temperatures_K": [200, 300, 500], "beads": [384, 256, 64], "tick_count": 7},
}
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def line_fit(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("At least three matched calibration points are required")
    if any(not math.isfinite(x) for x in (*xs, *ys)):
        raise ValueError("Nonfinite calibration")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denominator = math.fsum((x-mx)**2 for x in xs)
    if denominator == 0:
        raise ValueError("Degenerate axis")
    slope = math.fsum((x-mx)*(y-my) for x, y in zip(xs, ys))/denominator
    intercept = my-slope*mx
    residual = max(abs(y-(slope*x+intercept)) for x, y in zip(xs, ys))
    return {"slope": slope, "intercept": intercept, "maximum_absolute_residual": residual}


def log_axis(ys, rates):
    if any(not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("Log-axis labels must be positive")
    fit = line_fit(ys, [math.log(r) for r in rates])
    if fit["slope"] >= 0 or fit["maximum_absolute_residual"] > 0.001:
        raise ValueError("The pinned figure must have an accurately calibrated downward log axis")
    return fit


def calibrated_rate(fit, y, *, sensitivity_points=1.0):
    if not math.isfinite(y) or not math.isfinite(sensitivity_points) or sensitivity_points <= 0:
        raise ValueError("Finite coordinate and positive sensitivity are required")
    center = fit["slope"]*y+fit["intercept"]
    delta = abs(fit["slope"])*sensitivity_points+fit["maximum_absolute_residual"]
    return math.exp(center), [math.exp(center-delta), math.exp(center+delta)]


def segments(drawings, color):
    for index, drawing in enumerate(drawings):
        if not color(drawing.get("color")):
            continue
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:]
                yield {"drawing_index": index, "a": [float(a.x), float(a.y)],
                       "b": [float(b.x), float(b.y)], "width": float(drawing["width"])}


def source_figure(path, specification, fitz):
    with fitz.open(path) as document:
        if len(document) != 1:
            raise ValueError("Expected a one-page original figure")
        page = document[0]
        drawings = page.get_drawings()
        black = list(segments(drawings, lambda c: c == (0.0, 0.0, 0.0)))
        horizontal = [x for x in black if abs(x["a"][1]-x["b"][1]) < 1e-4
                      and 8.1 < abs(x["a"][0]-x["b"][0]) < 8.3]
        vertical = [x for x in black if abs(x["a"][0]-x["b"][0]) < 1e-4
                    and 8.1 < abs(x["a"][1]-x["b"][1]) < 8.3]
        spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
        labels = [s for s in spans if re.fullmatch(r"\d(?:\.\d+)?E[-+]\d+", s["text"])]
        if len(labels) != specification["tick_count"] or len(horizontal) != len(labels) or len(vertical) != 7:
            raise ValueError("Unexpected major-tick membership")
        y_ticks = []
        for label in labels:
            cy = (label["bbox"][1]+label["bbox"][3])/2
            tick = min(horizontal, key=lambda x: abs(x["a"][1]-cy))
            if abs(tick["a"][1]-cy) > 2 or not 4 < min(tick["a"][0], tick["b"][0])-label["bbox"][2] < 8:
                raise ValueError("Text label does not identify its physical tick")
            y_ticks.append({"label": label["text"], "rate_cm3_s": float(label["text"]),
                            "coordinate_y_points": tick["a"][1], "drawing_index": tick["drawing_index"]})
        if len({row["drawing_index"] for row in y_ticks}) != len(y_ticks):
            raise ValueError("Duplicate tick assignment")
        y_fit = log_axis([row["coordinate_y_points"] for row in y_ticks], [row["rate_cm3_s"] for row in y_ticks])
        vertical.sort(key=lambda x: x["a"][0])
        x_fit = line_fit([x["a"][0] for x in vertical], [i*.25 for i in range(7)])
        if x_fit["slope"] <= 0 or x_fit["maximum_absolute_residual"] > .001:
            raise ValueError("Incorrect lambda axis")
        for label in [s for s in spans if s["text"] in ("0.00", "1.50")]:
            cx = (label["bbox"][0]+label["bbox"][2])/2
            if abs(x_fit["slope"]*cx+x_fit["intercept"]-float(label["text"])) > .002:
                raise ValueError("Lambda axis endpoint text differs")
        grouped = defaultdict(list)
        for line in segments(drawings, lambda c: bool(c) and c[0] > .7 and c[1] < .3 and c[2] < .3):
            dx, dy = line["a"][0]-line["b"][0], line["a"][1]-line["b"][1]
            if not 10 < abs(dx) < 11 or abs(abs(dx)-abs(dy)) > 1e-3:
                raise ValueError("Unexpected red marker shape")
            cx, cy = (line["a"][0]+line["b"][0])/2, (line["a"][1]+line["b"][1])/2
            grouped[(round(cx, 3), round(cy, 3))].append(line)
        if len(grouped) != 19 or any(len(lines) != 2 for lines in grouped.values()):
            raise ValueError("Expected eighteen data crosses and one legend cross")
        markers, excluded = [], []
        for lines in grouped.values():
            if (lines[0]["a"][0]-lines[0]["b"][0])*(lines[0]["a"][1]-lines[0]["b"][1]) * (lines[1]["a"][0]-lines[1]["b"][0])*(lines[1]["a"][1]-lines[1]["b"][1]) >= 0:
                raise ValueError("Cross arms do not have opposite slopes")
            cx = statistics.fmean([p[0] for line in lines for p in (line["a"], line["b"])])
            cy = statistics.fmean([p[1] for line in lines for p in (line["a"], line["b"])])
            lam = x_fit["slope"]*cx+x_fit["intercept"]
            selected = min(LAMBDAS, key=lambda x: abs(x-lam))
            row = {"coordinate_x_points": cx, "coordinate_y_points": cy, "extracted_lambda": lam,
                   "drawing_indices": [x["drawing_index"] for x in lines], "stroke_width_points": lines[0]["width"]}
            if abs(lam-selected) > .001:
                excluded.append(row)
            else:
                rate, interval = calibrated_rate(y_fit, cy)
                markers.append({**row, "lambda": selected, "rate_cm3_s": rate,
                                "coordinate_sensitivity_interval_cm3_s": interval})
        if len(excluded) != 1 or len(markers) != 18:
            raise ValueError("Legend/data membership is not unambiguous")
        for lam in LAMBDAS:
            group = sorted([r for r in markers if r["lambda"] == lam], key=lambda r: -r["coordinate_y_points"])
            if len(group) != 3:
                raise ValueError("Each registered friction must contain three temperatures")
            for row, temperature, beads in zip(group, specification["temperatures_K"], specification["beads"]):
                row.update(temperature_K=temperature, beads=beads)
        markers.sort(key=lambda r: (r["temperature_K"], r["lambda"]))
        return {**specification, "original_figure": path.name, "original_figure_sha256": sha256(path),
                "log_axis_fit": y_fit, "lambda_axis_fit": x_fit, "y_ticks": y_ticks,
                "markers": markers, "excluded_legend_cross": excluded[0],
                "temperature_assignment": "bottom, middle, top correspond to the visible low, middle, high T labels",
                "coordinate_sensitivity_points": 1.0, "paper_statistical_uncertainty": None,
                "physical_comparison_eligible": False}


def journal_fh2_crosscheck(path, source, fitz):
    """Separate scaled/typeset vector geometry, with visually identified tick set."""
    with fitz.open(path) as doc:
        drawings = doc[9].get_drawings()
        axis = drawings[63]
        ys = []
        for item in axis["items"]:
            if item[0] == "l":
                a, b = item[1:]
                if abs(a.y-b.y) < 1e-4 and abs(a.x-98.8766098) < .001 and abs(b.x-102.3229904) < .001:
                    ys.append(float(a.y))
        if len(ys) != 7:
            raise ValueError("Typeset Figure 7 major ticks changed")
        fit = log_axis(ys, [x*1e-11 for x in range(2, 9)])
        rows = []
        for temperature, index in ((200, 64), (300, 67), (500, 70)):
            drawing = drawings[index]
            if len(drawing["items"]) != 2 or not drawing["color"][0] > .7:
                raise ValueError("Typeset zero-friction marker changed")
            r = drawing["rect"]
            rate, _ = calibrated_rate(fit, (r.y0+r.y1)/2)
            original = next(x for x in source["markers"] if x["lambda"] == 0 and x["temperature_K"] == temperature)
            relative = rate/original["rate_cm3_s"]-1
            if abs(relative) > .0001:
                raise ValueError("Original and typeset Figure 7 disagree")
            rows.append({"temperature_K": temperature, "journal_rate_cm3_s": rate,
                         "original_figure_rate_cm3_s": original["rate_cm3_s"], "relative_difference": relative,
                         "journal_drawing_index": index})
        return {"journal_pdf_page_zero_based": 9, "printed_page": "074107-9", "y_ticks_points": ys,
                "log_axis_fit": fit, "rows": rows, "all_relative_differences_below_0_0001": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name, digest in SOURCES.items():
        if sha256(args.source_dir/name) != digest:
            raise ValueError(f"Pinned source changed: {name}")
    if args.output.exists():
        raise ValueError("Refusing to replace a digitization record")
    import fitz
    figures = {name: source_figure(args.source_dir/name, spec, fitz) for name, spec in FIGURES.items()}
    record = {"schema": "pyrpmd.hele2015.vector-source-digitization.v1",
              "doi": "10.1063/1.4928599", "arxiv_version": "1506.07460v2",
              "source_sha256": SOURCES, "extractor_sha256": sha256(__file__), "pymupdf_version": fitz.VersionBind,
              "scope": "Author-published graph coordinates, not new molecular simulation results",
              "statistical_uncertainty_supplied_by_figure": False,
              "coordinate_sensitivity_is_confidence_interval": False,
              "coordinate_sensitivity_is_physical_tolerance": False,
              "source_PDF_or_TeX_redistributed": False, "PES_calls": 0, "random_draws": 0,
              "physical_gate_pass": False, "prior_reference_targets_changed": False,
              "figures": figures,
              "typeset_fh2_crosscheck": journal_fh2_crosscheck(args.source_dir/"hel15trpmd.pdf", figures["fhh.pdf"], fitz)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "RPMD_lambda0": {name: [{"T_K": r["temperature_K"], "rate_cm3_s": r["rate_cm3_s"]}
                                                for r in f["markers"] if r["lambda"] == 0]
                                       for name, f in figures.items()}}, indent=2))


if __name__ == "__main__":
    main()
