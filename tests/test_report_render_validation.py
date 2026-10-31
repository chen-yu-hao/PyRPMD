"""Regression checks for sparse uncertainty and portable publication exports."""

import csv
import os
import struct
import subprocess
import sys
import warnings

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


@pytest.mark.parametrize("style", ["jctc", "nature"])
def test_publication_profiles_keep_text_visible_under_external_dark_colors(tmp_path, monkeypatch, style):
    from matplotlib.colors import to_hex
    save = Figure.savefig

    def inspect(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        ax = figure.axes[0]
        assert to_hex(figure.get_facecolor()) == to_hex(ax.get_facecolor()) == "#ffffff"
        text = [ax._left_title, ax.xaxis.label, ax.yaxis.label,
                *ax.get_xticklabels(), *ax.get_yticklabels()]
        assert all(to_hex(item.get_color()) == "#000000" for item in text)
        assert all(to_hex(spine.get_edgecolor()) == "#000000" for spine in ax.spines.values())
        return result

    monkeypatch.setattr(Figure, "savefig", inspect)
    with matplotlib.rc_context({"axes.facecolor": "black", "axes.edgecolor": "white",
                                "axes.labelcolor": "white", "axes.titlecolor": "white",
                                "text.color": "white", "xtick.color": "white", "ytick.color": "white",
                                "xtick.labelcolor": "white", "ytick.labelcolor": "white"}):
        before = dict(matplotlib.rcParams)
        render_figures([_curve()], tmp_path, formats="png", style=style)
        assert dict(matplotlib.rcParams) == before


def test_default_profile_keeps_long_legends_inside_all_export_formats(tmp_path, monkeypatch):
    plot = _curve()
    plot["note"] = "Long-label export fixture; source values remain unchanged."
    labels = [f"Window {index}: replica-averaged biased coordinate distribution"
              for index in range(8)]
    plot["series"] = [{"label": label, "x": [0, 1, 2], "y": [i, i + .2, i + .4]}
                      for i, label in enumerate(labels)]
    save = Figure.savefig
    inspected = []

    def inspect(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        assert figure.get_size_inches()[0] == pytest.approx(7.0)
        data_ax = figure.axes[0]
        assert data_ax.get_legend() is None
        legends = [ax.get_legend() for ax in figure.axes if ax.get_legend() is not None]
        assert len(legends) == 1
        assert [" ".join(text.get_text().split()) for text in legends[0].get_texts()] == labels
        assert all(text.get_fontsize() == 8 for text in legends[0].get_texts())
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = legends[0].get_window_extent(renderer)
        assert bounds.x0 >= figure.bbox.x0 and bounds.x1 <= figure.bbox.x1
        assert bounds.y0 >= figure.bbox.y0 and bounds.y1 <= data_ax.get_window_extent(renderer).y0
        inspected.append(True)
        return result

    monkeypatch.setattr(Figure, "savefig", inspect)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = render_figures([plot], tmp_path, formats=("png", "pdf", "svg"), style="jctc")
    assert not any("constrained_layout" in str(warning.message) for warning in observed)
    assert len(inspected) == 3 and result[0]["width_mm"] == pytest.approx(177.8)
    with (tmp_path / result[0]["data"]).open() as stream:
        assert set(row["series"] for row in csv.DictReader(stream)) == set(labels)


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
