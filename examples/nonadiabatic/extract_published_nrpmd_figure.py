"""Extract numeric Fig. 1(c) data from the Richardson--Thoss (2013) PDF.

This optional provenance tool requires PyMuPDF (``pip install pymupdf``).
The repository contains the extracted facts, not the copyrighted paper/image.
Download the institutional author-deposited version separately from
https://open.fau.de/bitstreams/c56516c7-4f8a-456b-95ee-7a55a144cfd0/download
and pass its local filename. The SHA256 check fixes the supported PDF version.

The six colored traces are PDF polylines and the two DVR series are circular
vector markers. Axis calibration uses only printed tick labels and their
vector positions. No calculated correlation function enters the extraction.
Values are read at original polyline vertices / marker centers; no curve
fitting, resampling, smoothing, or correction to an independent calculation
is performed. Reported uncertainty is a conservative graphic-readout bound,
not an estimate of the authors' simulation or sampling error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PDF_SHA256 = "20bc0e0f1424aae7beb97b2718bce7a3dae0b6947e23ce8e9f6ead2004086dce"
PDF_URL = "https://open.fau.de/bitstreams/c56516c7-4f8a-456b-95ee-7a55a144cfd0/download"
DATA_BASENAME = "richardson_thoss_2013_fig1c"
COORDINATE_TOLERANCE_PT = 0.003


def _near(a, b, tolerance=COORDINATE_TOLERANCE_PT):
    return abs(a-b) <= tolerance


def _ticks(drawings, frame, axis):
    result = []
    for drawing in drawings:
        items = drawing["items"]
        if len(items) != 1 or items[0][0] != "l":
            continue
        _, a, b = items[0]
        if axis == "x":
            if (_near(a.x, b.x) and _near(a.y, frame.y1)
                    and _near(a.y-b.y, 4.0) and frame.x0-0.01 <= a.x <= frame.x1+0.01):
                result.append(a.x)
        elif (_near(a.y, b.y) and _near(a.x, frame.x0)
              and _near(b.x-a.x, 4.0) and frame.y0-0.01 <= a.y <= frame.y1+0.01):
            result.append(a.y)
    return np.asarray(sorted(set(result)))


def _calibrate(positions, values, words, frame, axis):
    values = np.asarray(values, dtype=float)
    if len(positions) != len(values):
        raise ValueError(f"unexpected number of {axis} axis ticks")
    # Endpoints fix the transformation. Interior ticks independently check it.
    slope = (values[-1]-values[0])/(positions[-1]-positions[0])
    intercept = values[0]-slope*positions[0]
    residual = float(np.max(abs(slope*positions+intercept-values)))
    if residual > 1e-4:
        raise ValueError("axis ticks are inconsistent with linear calibration")
    for position, value in zip(positions, values):
        nearby = []
        for x0, y0, x1, y1, text, *_ in words:
            center = (x0+x1)/2 if axis == "x" else (y0+y1)/2
            outside = (frame.y1 < y0 < frame.y1+5 if axis == "x"
                       else frame.x0-20 < x0 < x1 < frame.x0)
            if outside and abs(center-position) < 1.0:
                try:
                    nearby.append(float(text.replace("−", "-")))
                except ValueError:
                    pass
        if not any(_near(label, value, 1e-9) for label in nearby):
            raise ValueError(f"could not verify printed {axis} tick {value}")
    return {"pdf_positions_pt": positions.tolist(), "printed_values": values.tolist(),
            "slope": float(slope), "intercept": float(intercept),
            "maximum_interior_tick_residual": residual}


def _color_series(color):
    if color is None:
        return None
    r, g, b = color
    if b > max(r, g)+0.2:
        return "P1"
    if g > max(r, b)+0.2:
        return "P4"
    if r > max(g, b)+0.5:
        return "P8"
    return None


def extract_figure(pdf_path):
    """Return CSV row dictionaries and auditable geometric provenance."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("figure extraction requires the optional pymupdf package") from exc
    pdf_path = Path(pdf_path)
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if source_hash != PDF_SHA256:
        raise ValueError("PDF SHA256 mismatch; use the documented FAU-deposited version")
    with fitz.open(pdf_path) as document:
        page = document[4]
        drawings = page.get_drawings()
        words = page.get_text("words")
        if "FIG. 1." not in page.get_text():
            raise ValueError("published figure caption missing from expected page")
        frames = [d["rect"] for d in drawings
                  if d["type"] == "f" and d["fill"] == (1., 1., 1.)
                  and d["rect"].x0 > 400 and d["rect"].y1 < 200
                  and 120 < d["rect"].width < 125 and 60 < d["rect"].height < 70]
        frames.sort(key=lambda rect: rect.y0)
        if len(frames) != 2:
            raise ValueError("could not identify both Fig. 1(c) plotting frames")
        rows, panels = [], []
        for observable, frame, y_values in zip(("Cxx", "C11"), frames,
                                               ((2., 1., 0., -1.), (.5, .4, .3, .2, .1))):
            xticks, yticks = _ticks(drawings, frame, "x"), _ticks(drawings, frame, "y")
            # Both panels share the independently verified lower time axis.
            xcal = _calibrate(xticks, np.arange(6), words, frames[1], "x")
            ycal = _calibrate(yticks, y_values, words, frame, "y")
            traces = []
            for index, drawing in enumerate(drawings):
                rect, series = drawing["rect"], _color_series(drawing["color"])
                if (drawing["type"] == "s" and series and rect.x0 >= frame.x0-0.01
                        and rect.x1 <= frame.x1+0.01 and rect.y0 >= frame.y0-0.01
                        and rect.y1 <= frame.y1+0.01):
                    items = drawing["items"]
                    if len(items) != 20 or any(item[0] != "l" for item in items):
                        raise ValueError("unexpected published polyline geometry")
                    points = [items[0][1]]+[item[2] for item in items]
                    if any(a.x >= b.x for a, b in zip(points, points[1:])):
                        raise ValueError("nonmonotonic time coordinates in published polyline")
                    radius = drawing["width"]/2
                    traces.append({"series": series, "drawing_indices": [index],
                                   "points": points, "vertical_graphic_radius_pt": radius})
            marker_indices, marker_points, marker_radii = [], [], []
            for index, drawing in enumerate(drawings):
                rect = drawing["rect"]
                center = fitz.Point((rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2)
                if (drawing["type"] == "fs" and len(drawing["items"]) == 8
                        and drawing["fill"] is not None and max(drawing["fill"]) < .2
                        and _near(rect.width, 1.25) and _near(rect.height, 1.25)
                        and frame.x0-0.01 <= center.x <= frame.x1+0.01
                        and frame.y0 <= center.y <= frame.y1):
                    marker_indices.append(index)
                    marker_points.append(center)
                    marker_radii.append(rect.height/2+drawing["width"]/2)
            if len(marker_points) != 20 or len(traces) != 3:
                raise ValueError("incorrect published trace/marker count")
            traces.append({"series": "dvr", "drawing_indices": marker_indices,
                           "points": marker_points,
                           "vertical_graphic_radius_pt": float(max(marker_radii))})
            for trace in traces:
                uncertainty = (abs(ycal["slope"])*(trace["vertical_graphic_radius_pt"]
                               + COORDINATE_TOLERANCE_PT)+ycal["maximum_interior_tick_residual"])
                for point in trace.pop("points"):
                    time = xcal["slope"]*point.x+xcal["intercept"]
                    value = ycal["slope"]*point.y+ycal["intercept"]
                    if abs(time) < 1e-12:
                        time = 0.
                    if abs(time-5.) < 1e-12:
                        time = 5.
                    rows.append({"observable": observable, "series": trace["series"],
                                 "time": float(time), "value": float(value),
                                 "absolute_uncertainty": float(uncertainty)})
                trace["points"] = 20 if trace["series"] == "dvr" else 21
                trace["absolute_uncertainty"] = float(uncertainty)
            panels.append({"observable": observable, "frame_pdf_points": list(frame),
                           "time_calibration": xcal, "value_calibration": ycal,
                           "traces": traces})
    order = {series: i for i, series in enumerate(("dvr", "P1", "P4", "P8"))}
    rows.sort(key=lambda row: (("Cxx", "C11").index(row["observable"]),
                               order[row["series"]], row["time"]))
    provenance = {
        "schema_version": 1,
        "publication": {"authors": ["Jeremy O. Richardson", "Michael Thoss"],
                        "title": "Communication: Nonadiabatic ring-polymer molecular dynamics",
                        "journal": "The Journal of Chemical Physics 139, 031102 (2013)",
                        "doi": "10.1063/1.4816124", "figure": "1(c)",
                        "printed_page": "031102-4", "pdf_page_number": 5},
        "source": {"repository_record": "https://open.fau.de/handle/openfau/3580",
                   "pdf_url": PDF_URL, "sha256": source_hash,
                   "bytes": pdf_path.stat().st_size,
                   "repository": "Friedrich-Alexander-Universitaet Erlangen-Nuernberg",
                   "retrieved_date_UTC": "2026-11-01"},
        "method": "PDF vector polyline vertices and filled circular marker centers",
        "coordinate_system": "PDF points, origin upper left, y increases downward",
        "calibration": "printed tick values matched to vector tick locations; endpoint affine map with interior-tick residual checks",
        "series": {"dvr": "published black DVR markers", "P1": "published blue one-bead NRPMD polyline",
                   "P4": "published green four-bead NRPMD polyline", "P8": "published red eight-bead NRPMD polyline"},
        "model": {"equation": 15, "m": 1, "hbar": 1, "omega": 1, "kappa": 1,
                  "beta": 1, "alpha": 0, "Delta": 1,
                  "units": "paper reduced units", "constant_potential": False},
        "observables": {"Cxx": "Kubo nuclear-position autocorrelation, Eq. 12(a)",
                        "C11": "Kubo electronic population autocorrelation, Eq. 12(b)"},
        "uncertainty": {
            "meaning": "conservative absolute ordinate graphic-readout bound; not a statistical confidence interval or the authors' Monte Carlo uncertainty",
            "formula": "abs(value_axis_slope) * (half_line_width_or_marker_outer_radius + 0.003 PDF points) + max_interior_tick_residual",
            "coordinate_tolerance_pt": COORDINATE_TOLERANCE_PT,
            "time_coordinate_bound": "abs(time_axis_slope) * 0.003 PDF points + max_interior_tick_residual",
            "reported_simulation_uncertainty": "not available from this figure",
            "between_vertex_interpolation_uncertainty": "not quantified; use original supplied vertices/markers for comparison",
        },
        "rows": len(rows), "panels": panels,
        "numerical_reference_used_for_calibration": False,
        "original_pdf_or_image_redistributed": False,
        "parser": {"name": "PyMuPDF", "version": fitz.VersionBind},
        "extraction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return rows, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="local FAU-deposited PDF; must match documented SHA256")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data"))
    arguments = parser.parse_args(argv)
    rows, provenance = extract_figure(arguments.pdf)
    arguments.output.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output/(DATA_BASENAME+".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(v, ".12g") if isinstance(v, float) else v for k, v in row.items()})
    provenance["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (arguments.output/(DATA_BASENAME+".provenance.json")).write_text(
        json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "csv": str(csv_path),
                      "csv_sha256": provenance["csv_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
