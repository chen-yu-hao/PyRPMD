"""Scientific export semantics and offline report safety."""

import csv
import json
from pathlib import Path
import struct

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrpmd.analysis.report_render import (
    _heatmap_plot, _line_plot, render_figures, write_report,
)


def _figure():
    figure = Figure()
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot()


def test_heatmap_ij_orientation_and_unsupported_cells(tmp_path):
    fig, ax = _figure()
    z = np.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
    mask = np.array([[False, True, False], [False, False, False]])
    _heatmap_plot(fig, ax, {
        "xedges": [0, 2, 6], "yedges": [-3, -1, 0, 4], "z": z,
        "mask": mask, "colorbar": "Free energy / hartree",
    }, tmp_path / "grid.csv")
    shown = ax.collections[0].get_array().reshape(3, 2)
    np.testing.assert_equal(shown.data, z.T)
    np.testing.assert_equal(shown.mask, mask.T)
    with (tmp_path / "grid.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[1] == {"x": "1.0", "y": "-0.5", "z": "", "masked": "1"}
    assert rows[3] == {"x": "4.0", "y": "-2.0", "z": "21.0", "masked": "0"}


def test_missing_line_data_remain_gaps_and_errors_are_not_invented(tmp_path):
    _, ax = _figure()
    _line_plot(ax, {
        "series": [{"label": "sample", "x": [0, 1, 2, 3], "y": [1, None, 3, 4],
                    "yerr": [0.2, 0.3, None, 0.4]}],
    }, tmp_path / "curve.csv")
    assert np.isnan(ax.lines[0].get_ydata()[1])
    # Two supported errors, not four zero-filled bars or a line bridging the gap.
    assert len(ax.containers[0].lines[2][0].get_segments()) == 2
    with (tmp_path / "curve.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows[1]["y"] == ""
    assert rows[2]["error_minus"] == rows[2]["error_plus"] == ""


def test_log_curves_preserve_nonpositive_data_in_export_but_not_display(tmp_path):
    _, ax = _figure()
    ax.set_yscale("log")
    _line_plot(ax, {
        "yscale": "log", "series": [{"x": [0, 1, 2], "y": [1, -2, 3], "yerr": [2, 0.1, 0.2]}],
    }, tmp_path / "curve.csv")
    assert np.isnan(ax.lines[0].get_ydata()[1])
    assert len(ax.containers[0].lines[2][0].get_segments()) == 1
    assert ",-2.0," in (tmp_path / "curve.csv").read_text()


def test_dense_error_band_does_not_bridge_unsupported_points(tmp_path):
    _, ax = _figure()
    x = np.arange(30.0)
    y = np.ones(30)
    y[14:17] = np.nan
    _line_plot(ax, {"series": [{"x": x, "y": y, "yerr": np.full(30, 0.2)}]},
               tmp_path / "band.csv")
    paths = ax.collections[0].get_paths()
    assert len(paths) == 2
    assert paths[0].vertices[:, 0].max() == 13
    assert paths[1].vertices[:, 0].min() == 17
    assert not ax.containers


def test_all_masked_and_constant_heatmaps_have_finite_scaling(tmp_path):
    for index, values in enumerate(([[None, None]], [[3.0, 3.0]])):
        fig, ax = _figure()
        _heatmap_plot(fig, ax, {
            "xedges": [0, 1], "yedges": [0, 1, 2], "z": values, "log_norm": True,
        }, tmp_path / f"grid-{index}.csv")
        fig.canvas.draw()
        assert np.isfinite(ax.collections[0].norm.vmin)
        assert np.isfinite(ax.collections[0].norm.vmax)
        assert ax.collections[0].norm.vmin < ax.collections[0].norm.vmax


def test_render_exports_requested_formats_and_preserves_global_style(tmp_path):
    matplotlib.get_backend()  # Resolve the lazy backend before comparing state.
    before = dict(matplotlib.rcParams)
    figures = render_figures([{
        "id": "rate", "kind": "line", "title": "Rate comparison",
        "xlabel": "1000 K / T", "ylabel": "Rate / cm³ molecule⁻¹ s⁻¹",
        "series": [{"label": "RPMD", "x": [1, 2, 3], "y": [1e-10, 1e-11, 1e-12]}],
        "yscale": "log", "caption": "No uncertainties supplied.",
    }], tmp_path, formats=("png", "pdf", "svg"), dpi=300)
    assert dict(matplotlib.rcParams) == before
    manifest = figures[0]
    assert set(manifest["files"]) == {"png", "pdf", "svg"}
    png = (tmp_path / manifest["files"]["png"]).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24])[0] == 999  # 3.33 inches at 300 dpi.
    assert (tmp_path / manifest["files"]["pdf"]).read_bytes().startswith(b"%PDF-")
    assert "<svg" in (tmp_path / manifest["files"]["svg"]).read_text()
    assert (tmp_path / manifest["data"]).is_file()
    json.dumps(figures, allow_nan=False)


def test_report_escapes_every_untrusted_text_field(tmp_path):
    attack = '<script>alert("x")</script>'
    write_report({
        "title": attack,
        "runs": [{"label": attack, "path": attack, "task": "rate", "status": "pilot",
                  "converged": False, "metrics": {attack: None}, "warnings": [attack]}],
        "figures": [{"id": "safe", "title": attack, "caption": attack,
                     "files": {"png": "figures/safe.png"}, "data": "data/safe.csv"}],
        "findings": [{"level": "warning", "code": attack, "message": attack}],
        "skipped": [{"run": attack, "figure": "test", "reason": attack}],
        "settings": {attack: attack},
        "sources": [{"path": attack, "sha256": attack, "size": 20}],
        "references": [{"title": attack, "url": "javascript:alert(1)"}],
    }, tmp_path)
    html = (tmp_path / "index.html").read_text()
    markdown = (tmp_path / "report.md").read_text()
    assert "<script>" not in html
    assert "javascript:" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;script&gt;" in markdown
    assert "Unavailable" in html
    assert "Converged (saved result)" in html and ">No<" in html
    assert 'href="data/safe.csv"' in html


@pytest.mark.parametrize("path", ["../outside.png", "/tmp/outside.png", "javascript:evil()", "https://example.org/a.png"])
def test_report_rejects_nonlocal_artifact_links(tmp_path, path):
    with pytest.raises(ValueError, match="relative local"):
        write_report({"figures": [{"id": "bad", "files": {"png": path}}]}, tmp_path)


@pytest.mark.parametrize("plot", [
    {"id": "../bad"}, {"id": "bad", "kind": "unknown"},
    {"id": "bad", "series": [{"x": [0, 1], "y": [2]}]},
    {"id": "bad", "series": [{"x": [0], "y": [2], "yerr": [-1]}]},
    {"id": "bad", "kind": "heatmap", "xedges": [0, 1], "yedges": [0, 1], "z": [[1, 2]]},
])
def test_invalid_plot_specs_fail_explicitly(tmp_path, plot):
    with pytest.raises(ValueError):
        render_figures([plot], tmp_path)
