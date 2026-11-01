"""Independent PDFMiner/pdfplumber + stdlib check of Figure 7 only.

Read source figures as data. Do not import the producer's PyMuPDF extractor,
load any PES, draw random numbers, or calculate a new molecular rate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import pdfplumber


def record(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def regress(x, y):
    n = len(x)
    x0, y0 = math.fsum(x) / n, math.fsum(y) / n
    b = math.fsum((a-x0)*(c-y0) for a, c in zip(x, y)) / math.fsum((a-x0)**2 for a in x)
    a = y0-b*x0
    return a, b, max(abs(c-a-b*d) for d, c in zip(x, y))


def interpolate_log(y, left, right):
    fraction = (y-left[0])/(right[0]-left[0])
    return left[1] * math.exp(fraction * math.log(right[1]/left[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    producer = json.loads(args.source_record.read_text())
    inputs = [record(args.protocol), record(args.source_record), record(Path(__file__))]
    for name, expected in protocol["source_bytes"].items():
        row = record(args.source_dir/name)
        assert row["sha256"] == expected, name
        inputs.append(row)
    with pdfplumber.open(args.source_dir/"fhh.pdf") as document:
        assert len(document.pages) == 1
        page = document.pages[0]
        labels = [w for w in page.extract_words() if re.fullmatch(r"[2-8]E-11", w["text"])]
        ticks = [line for line in page.lines if line["stroking_color"] == 0.0
                 and line["height"] < 1e-8 and abs(line["width"]-8.22) < 1e-6]
        assert len(labels) == len(ticks) == 7
        calibrated = []
        for word in labels:
            cy = (word["top"]+word["bottom"])/2
            tick = min(ticks, key=lambda line: abs(line["top"]-cy))
            assert abs(tick["top"]-cy) < 1 and 4 < tick["x0"]-word["x1"] < 8
            calibrated.append((tick["top"], float(word["text"])))
        assert len({y for y, _ in calibrated}) == 7
        calibrated.sort()
        a, b, residual = regress([y for y, _ in calibrated], [math.log(k) for _, k in calibrated])
        assert b < 0 and residual < 0.0001
        # Axis labels locate zero independently of the producer's drawing IDs.
        zero_word = next(w for w in page.extract_words() if w["text"] == "0.00")
        zx = (zero_word["x0"]+zero_word["x1"])/2
        arms = defaultdict(list)
        for line in page.lines:
            if line["stroking_color"] != (1.0, 0.0, 0.0):
                continue
            cx, cy = (line["x0"]+line["x1"])/2, (line["top"]+line["bottom"])/2
            if abs(cx-zx) < 0.1:
                assert abs(line["width"]-10.44) < 1e-6 and abs(line["height"]-10.44) < 1e-6
                arms[(round(cx, 8), round(cy, 8))].append(line["pts"])
        assert len(arms) == 3 and all(len(v) == 2 for v in arms.values())
        points = sorted(arms, key=lambda p: -p[1])
        temperatures = [200, 300, 500]  # Separately verified visible labels.
        rows = []
        for temperature, (x, y) in zip(temperatures, points):
            first, second = arms[(x, y)]
            slopes = [(p[1][0]-p[0][0])*(p[1][1]-p[0][1]) for p in (first, second)]
            assert slopes[0]*slopes[1] < 0
            source = next(p for p in producer["figures"]["fhh.pdf"]["markers"]
                          if p["temperature_K"] == temperature and p["lambda"] == 0)
            k = math.exp(a+b*y)
            relative = k/source["rate_cm3_s"]-1
            assert abs(relative) < 1e-6
            adjacent = sorted(calibrated, key=lambda pair: abs(pair[0]-y))[:2]
            rows.append({"temperature_K": temperature, "lambda": 0,
                         "pdf_center_points": [x, y], "independent_graph_rate_cm3_s": k,
                         "producer_graph_rate_cm3_s": source["rate_cm3_s"],
                         "relative_reader_difference": relative,
                         "two_endpoint_log_interpolation_cm3_s": interpolate_log(y, calibrated[0], calibrated[-1]),
                         "two_nearest_tick_log_interpolation_cm3_s": interpolate_log(y, *adjacent),
                         "below_lowest_printed_major_tick": y > max(p[0] for p in calibrated),
                         "one_original_point_sensitivity_including_tick_fit_residual":
                             [math.exp(a+b*y-abs(b)-residual), math.exp(a+b*y+abs(b)+residual)]})
    tex = (args.source_dir/"t.tex").read_text()
    assert "\\date{\\today}" in tex
    assert "$\\lambda = 0$ corresponding to zero friction" in tex
    assert "Stark–Werner (SW) potential energy surface for F+H$_2$" in tex
    assert "K.~Stark and H.-J. Werner" in tex and "6515" in tex
    fh2 = next(r for r in protocol["reactions"] if r["reaction"] == "F + H2")
    assert fh2["temperatures_K"] == [200, 300, 500] and fh2["beads"] == [384, 256, 64]
    assert fh2["product_channels"] == 2 and fh2["PMF_xi_dagger"] == [.9671, .9885, .9947]
    assert fh2["child_dt_ps"] == .0001 and fh2["child_duration_ps"] == .2 and fh2["total_children"] == 500000
    for item in ["384 (200 K)", "256 (300 K)", "64 (500 K)", "0.9671 (200 K)", "0.9885 (300 K)", "0.9947 (500 K)"]:
        assert item in tex
    result = {"schema": "pyrpmd.hele2015.fh2-independent-source-check.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
              "reader": "pdfplumber / PDFMiner, separately from the producer's PyMuPDF",
              "pdfplumber_version": pdfplumber.__version__, "inputs": inputs,
              "seven_major_ticks_downward_points_and_rates": calibrated,
              "natural_log_axis": {"intercept": a, "slope": b, "maximum_absolute_residual": residual},
              "zero_friction_rows": rows,
              "maximum_absolute_relative_reader_difference": max(abs(r["relative_reader_difference"]) for r in rows),
              "visual_review_external_images": ["fhh-source-figure.png", "page-07.png"],
              "temperature_assignment_independently_visually_confirmed": True,
              "TableII_FH2_column_visually_checked": True,
              "source_only": True, "new_simulations": 0, "PES_calls": 0, "RNG_draws": 0,
              "paper_statistical_uncertainty": None, "coordinate_sensitivity_is_confidence_interval": False,
              "physical_equivalence_tolerance_created": False, "prior_targets_changed": False,
              "wall_seconds": time.monotonic()-started, "CPU_seconds": time.process_time()-cpu}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": record(args.output), "all_checks_passed": True,
                      "maximum_relative_reader_difference": result["maximum_absolute_relative_reader_difference"],
                      "rows": rows}))


if __name__ == "__main__":
    main()
