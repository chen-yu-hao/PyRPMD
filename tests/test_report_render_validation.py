"""Regression checks for sparse uncertainty and portable publication exports."""

import csv
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrpmd.analysis.report_render import _line_plot, render_figures


def _curve():
    return {
        "id": "transmission", "kind": "line", "title": "H + H2: transmission",
        "xlabel": "Time (fs)", "ylabel": "Transmission coefficient",
        "series": [{"label": "RPMD", "x": [0, 1, 2], "y": [1, 0.7, 0.5]}],
        "caption": "Saved values only; no uncertainty was supplied.",
    }


def test_dense_curve_keeps_isolated_asymmetric_uncertainties_visible(tmp_path):
    figure = Figure()
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    x = np.arange(30.0)
    lower = np.full(30, np.nan)
    upper = np.full(30, np.nan)
    supported = [0, 4, 5, 6, 15, 29]
    lower[supported], upper[supported] = 0.1, 0.3
    _line_plot(ax, {"series": [{"x": x, "y": np.ones(30),
                                 "yerr": np.vstack((lower, upper))}]},
               tmp_path / "uncertainty.csv")

    # The adjacent points make a band; the three isolated points need bars.
    bands = ax.collections[0].get_paths()
    assert any(np.ptp(path.vertices[:, 0]) == 2 for path in bands)
    bars = ax.containers[0].lines[2][0].get_segments()
    np.testing.assert_array_equal([segment[:, 0] for segment in bars],
                                  [[0, 0], [15, 15], [29, 29]])
    np.testing.assert_allclose([segment[:, 1] for segment in bars],
                               [[0.9, 1.3]] * 3)
    with (tmp_path / "uncertainty.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["error_minus"] == "0.1" and rows[0]["error_plus"] == "0.3"
    assert rows[1]["error_minus"] == rows[1]["error_plus"] == ""


def test_one_shot_plot_iterable_is_rendered(tmp_path):
    plots = (_curve() for _ in range(1))
    manifest = render_figures(plots, tmp_path, formats="svg")
    assert len(manifest) == 1
    assert (tmp_path / manifest[0]["files"]["svg"]).is_file()
    assert (tmp_path / manifest[0]["data"]).is_file()


@pytest.mark.parametrize("style,width_mm", [("jctc", 3.33 * 25.4), ("nature", 89.0)])
def test_style_width_is_not_overridden_by_external_savefig_settings(tmp_path, style, width_mm):
    matplotlib.get_backend()
    with matplotlib.rc_context({"savefig.bbox": "tight", "savefig.transparent": True,
                                "axes.grid": True}):
        before = dict(matplotlib.rcParams)
        result = render_figures([_curve()], tmp_path, formats="png", dpi=254, style=style)
        assert dict(matplotlib.rcParams) == before
    figure = result[0]
    assert figure["style"] == style
    assert figure["width_mm"] == pytest.approx(width_mm)
    png = (tmp_path / figure["files"]["png"]).read_bytes()
    assert struct.unpack(">II", png[16:24])[0] == int(width_mm / 25.4 * 254)


def test_nature_profile_has_separate_legends_and_editable_vector_text(tmp_path, monkeypatch):
    plot = _curve()
    plot["note"] = "T = 300 K; P = 32. Source convergence remains false."
    plot["series"] = [
        {"label": f"Window {index}: replica-averaged biased coordinate distribution",
         "x": [0, 1, 2], "y": [index, index + 0.2, index + 0.4]}
        for index in range(8)
    ]
    inspected = []
    save = Figure.savefig

    def inspect(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        assert figure.get_size_inches()[0] * 25.4 == pytest.approx(183)
        data_ax = figure.axes[0]
        assert data_ax.get_legend() is None
        assert not any(line.get_visible() for line in data_ax.get_xgridlines() + data_ax.get_ygridlines())
        assert data_ax.title.get_fontsize() == 7
        assert data_ax.xaxis.label.get_fontsize() == 7
        assert all(line.get_linewidth() <= 1 for line in data_ax.lines)
        legends = [ax.get_legend() for ax in figure.axes if ax.get_legend() is not None]
        assert len(legends) == 1 and len(legends[0].get_texts()) == 8
        # Check the actual rendered layout, including long, wrapped run labels.
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        bounds = legends[0].get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= figure.bbox.x0 and bounds.x1 <= figure.bbox.x1
        assert bounds.y0 >= figure.bbox.y0 and bounds.y1 <= figure.bbox.y1
        inspected.append(True)
        return result

    monkeypatch.setattr(Figure, "savefig", inspect)
    result = render_figures([plot], tmp_path, formats=("png", "pdf", "svg"), style="nature")
    assert len(inspected) == 3
    assert result[0]["title"] == plot["title"]
    assert result[0]["caption"] == plot["caption"]
    svg = (tmp_path / result[0]["files"]["svg"]).read_text()
    assert "<text" in svg and "H + H2: transmission" in svg
    assert (tmp_path / result[0]["files"]["pdf"]).read_bytes().startswith(b"%PDF-")


def test_render_error_restores_global_style(tmp_path):
    matplotlib.get_backend()
    before = dict(matplotlib.rcParams)
    plot = _curve()
    plot["series"][0]["yerr"] = [-0.1, 0.1, 0.1]
    with pytest.raises(ValueError, match="negative"):
        render_figures([plot], tmp_path, style="nature")
    assert dict(matplotlib.rcParams) == before


def test_render_never_initializes_a_requested_gui_backend(tmp_path):
    script = """
import matplotlib
import sys
from pyrpmd.analysis.report_render import render_figures
render_figures([{'id':'headless','series':[{'x':[0,1],'y':[1,2]}]}],
               sys.argv[1], formats=('png','svg'), style='nature')
assert matplotlib.get_backend().lower() == 'tkagg'
assert 'matplotlib.pyplot' not in sys.modules
assert 'tkinter' not in sys.modules
"""
    env = dict(os.environ, MPLBACKEND="TkAgg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env.pop("DISPLAY", None)
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "report")],
                            env=env, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report" / "figures" / "headless.png").is_file()


def test_unknown_graphics_style_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="style must be jctc or nature"):
        render_figures([_curve()], tmp_path, style="unrecognized")
