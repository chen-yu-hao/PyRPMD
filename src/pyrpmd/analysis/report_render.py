"""Optional, offline scientific figure and report rendering.

The input is a deliberately small plot specification, independent of any run
schema.  This module does not estimate errors, smooth data, execute input files,
or decide that a calculation has converged.  Matplotlib is imported only when
figures are requested; importing PyRPMD remains possible without it.
"""

from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path, PurePosixPath
import re
import textwrap
from urllib.parse import quote, urlsplit

import numpy as np


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
_STYLES = ("-", "--", "-.", ":")
_FORMAT = frozenset(("png", "pdf", "svg"))


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    return array


def _errors(value, count):
    if value is None:
        return np.full((2, count), np.nan)
    error = np.asarray(value, dtype=float)
    if error.ndim == 0:
        error = np.full((2, count), float(error))
    elif error.shape == (count,):
        error = np.vstack((error, error))
    elif error.shape != (2, count):
        raise ValueError("yerr must be scalar, length N, or shape (2, N)")
    if np.any(np.isfinite(error) & (error < 0)):
        raise ValueError("Finite uncertainty magnitudes cannot be negative")
    return error


def _csv_value(value):
    if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def _write_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_csv_value(value) for value in row])


def _line_plot(ax, plot, data_path, *, legend=True, linewidth=1.25):
    rows = []
    for index, series in enumerate(plot.get("series", [])):
        x, y = _vector(series["x"], "x"), _vector(series["y"], "y")
        if len(x) != len(y):
            raise ValueError("Line x and y must have equal lengths")
        error = _errors(series.get("yerr"), len(x))
        label = str(series.get("label", ""))
        rows.extend((label, a, b, lo, hi) for a, b, lo, hi in zip(x, y, *error))
        valid = np.isfinite(x) & np.isfinite(y)
        if plot.get("xscale") == "log":
            valid &= x > 0
        if plot.get("yscale") == "log":
            valid &= y > 0
        # Setting unsupported values to NaN preserves gaps in connected curves.
        draw_x, draw_y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
        color = series.get("color", _COLORS[index % len(_COLORS)])
        style = series.get("style")
        if style not in (None, "solid", "dashed", "points"):
            raise ValueError("Line style must be solid, dashed, or points")
        linestyle = {"solid": "-", "dashed": "--", "points": "None"}.get(
            style, _STYLES[(index // len(_COLORS)) % len(_STYLES)]
        )
        ax.plot(draw_x, draw_y, label=label or None, color=color, linestyle=linestyle,
                marker="o" if style == "points" or len(x) == 1 else None,
                markersize=3.5, linewidth=linewidth)
        supported = valid & np.all(np.isfinite(error), axis=0)
        if plot.get("yscale") == "log":
            # Do not clip error bars at zero and thereby misrepresent their size.
            supported &= y - error[0] > 0
        if np.any(supported) and len(x) >= 20 and style != "points":
            ax.fill_between(draw_x, y - error[0], y + error[1], where=supported,
                            color=color, alpha=0.18, linewidth=0, interpolate=False)
            # fill_between needs two adjacent supported points to form an area.
            # Keep isolated, genuinely available uncertainties visible as bars.
            adjacent = np.zeros(len(x), dtype=bool)
            adjacent[1:] |= supported[:-1]
            adjacent[:-1] |= supported[1:]
            bars = supported & ~adjacent
        else:
            bars = supported
        if np.any(bars):
            ax.errorbar(x[bars], y[bars], yerr=error[:, bars],
                        fmt="none", ecolor=color, elinewidth=0.75, capsize=2,
                        alpha=0.75)
    for item in plot.get("vlines", []):
        ax.axvline(float(item["x"]), color="#777777", linestyle=":", linewidth=0.9,
                   label=item.get("label"))
    for item in plot.get("hlines", []):
        ax.axhline(float(item["y"]), color="#777777", linestyle=":", linewidth=0.9,
                   label=item.get("label"))
    for item in plot.get("spans", []):
        ax.axvspan(float(item["start"]), float(item["stop"]), color="#009E73",
                   alpha=0.09, label=item.get("label"))
    handles, labels = ax.get_legend_handles_labels()
    if labels and legend:
        ax.legend(handles, labels, frameon=False, fontsize=8,
                  ncols=2 if len(labels) > 7 else 1)
    _write_csv(data_path, ["series", "x", "y", "error_minus", "error_plus"], rows)


def _heatmap_plot(fig, ax, plot, data_path):
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm, Normalize

    x, y = _vector(plot["xedges"], "xedges"), _vector(plot["yedges"], "yedges")
    if any(len(e) < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0) for e in (x, y)):
        raise ValueError("Heatmap edges must be finite and strictly increasing")
    z = np.asarray(plot["z"], dtype=float)
    if z.shape != (len(x) - 1, len(y) - 1):
        raise ValueError("Heatmap z must have shape (len(xedges)-1, len(yedges)-1)")
    supplied_mask = plot.get("mask")
    mask = np.asarray(np.zeros(z.shape, dtype=bool) if supplied_mask is None else supplied_mask, dtype=bool)
    if mask.shape != z.shape:
        raise ValueError("Heatmap mask must have the same shape as z")
    mask = mask | ~np.isfinite(z)
    if plot.get("log_norm", False):
        mask |= z <= 0
    values = z[~mask]
    if len(values):
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            if plot.get("log_norm", False):
                lower, upper = lower / 2, upper * 2
            else:
                padding = max(abs(lower) * 0.05, 1e-12)
                lower, upper = lower - padding, upper + padding
        norm = LogNorm(lower, upper) if plot.get("log_norm", False) else Normalize(lower, upper)
    else:
        norm = Normalize(0, 1)
    cmap = colormaps.get_cmap(plot.get("cmap", "viridis")).copy()
    cmap.set_bad("#D9D9D9")
    ax.set_facecolor("#D9D9D9")
    # Run arrays use indexing='ij'; Matplotlib rows correspond to the y axis.
    mesh = ax.pcolormesh(x, y, np.ma.array(z, mask=mask).T, cmap=cmap, norm=norm,
                        shading="flat", edgecolors="none", antialiased=False)
    if len(values):
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.035)
        colorbar.set_label(str(plot.get("colorbar", "")))
    else:
        ax.text(0.5, 0.5, "No supported data", ha="center", va="center",
                transform=ax.transAxes, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    xc, yc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
    rows = ((a, b, None if mask[i, j] else z[i, j], mask[i, j])
            for i, a in enumerate(xc) for j, b in enumerate(yc))
    _write_csv(data_path, ["x", "y", "z", "masked"], rows)


def _bar_plot(ax, plot, data_path):
    labels = [str(value) for value in plot["labels"]]
    y = _vector(plot["y"], "y")
    if len(labels) != len(y):
        raise ValueError("Bar labels and y must have equal lengths")
    error = _errors(plot.get("yerr"), len(y))
    x = np.arange(len(y))
    valid = np.isfinite(y)
    if plot.get("yscale") == "log":
        valid &= y > 0
    ax.bar(x[valid], y[valid], width=0.65,
           color=[_COLORS[i % len(_COLORS)] for i in x[valid]])
    supported = valid & np.all(np.isfinite(error), axis=0)
    if plot.get("yscale") == "log":
        supported &= y - error[0] > 0
    if np.any(supported):
        ax.errorbar(x[supported], y[supported], yerr=error[:, supported], fmt="none",
                    ecolor="#222222", elinewidth=0.8, capsize=3)
    ax.set_xticks(x, labels, rotation=30 if any(len(v) > 9 for v in labels) else 0,
                  ha="right" if any(len(v) > 9 for v in labels) else "center")
    _write_csv(data_path, ["label", "y", "error_minus", "error_plus"],
               ((label, value, lo, hi) for label, value, lo, hi in zip(labels, y, *error)))


def render_figures(plots, output, *, formats=("png", "pdf"), dpi=300, style="jctc"):
    """Render normalized plots and export the plotted data without resampling.

    ``yerr`` contains uncertainty magnitudes (symmetric N or asymmetric 2×N).
    Its statistical interpretation is supplied in each plot's caption.  Missing
    uncertainties are omitted, never replaced with zero.  The caller owns the
    output directory and is responsible for atomic publication of the report.
    ``plots`` may be any iterable.  ``style="nature"`` uses 89/183 mm widths,
    6–7 pt type, no background grid and a separate legend row; ``"jctc"``
    retains the default 3.33/7 inch graphics profile.  Both retain all input data,
    uncertainty meanings, full titles and run labels in the exported manifest.
    """
    try:
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Figure generation requires Matplotlib; install PyRPMD[analysis]") from exc

    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(fmt not in _FORMAT for fmt in formats):
        raise ValueError("Figure formats must be selected from png, pdf, svg")
    if isinstance(dpi, bool) or not isinstance(dpi, (int, np.integer)) or dpi <= 0:
        raise ValueError("Figure dpi must be a positive integer")
    if style not in ("jctc", "nature"):
        raise ValueError("Figure style must be jctc or nature")
    # Validation and rendering both inspect the collection; preserve one-shot
    # iterables instead of silently consuming all plots during ID validation.
    plots = list(plots)
    output = Path(output)
    figure_dir, data_dir = output / "figures", output / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(plot["id"]) for plot in plots]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) for value in ids):
        raise ValueError("Figure ids must be unique safe filename slugs")
    manifests = []
    rc_style = {
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.usetex": False, "savefig.facecolor": "white", "figure.facecolor": "white",
        "savefig.bbox": None, "savefig.transparent": False, "axes.grid": False,
    }
    if style == "nature":
        rc_style.update({
            "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.major.size": 3, "ytick.major.size": 3,
            "xtick.direction": "out", "ytick.direction": "out",
        })
    with rc_context(rc_style):
        for plot in plots:
            kind = plot.get("kind", "line")
            if kind not in ("line", "heatmap", "bar"):
                raise ValueError(f"Unsupported figure kind: {kind}")
            wide = (kind == "heatmap" or len(plot.get("series", [])) > 4
                    or len(plot.get("labels", [])) > 5
                    or any(len(str(s.get("label", ""))) > 28 for s in plot.get("series", [])))
            width = (183 if wide else 89) / 25.4 if style == "nature" else (7.0 if wide else 3.33)
            note = str(plot.get("note") or "")
            wrapped = textwrap.fill(note, width=110 if wide else 48)
            lines = len(wrapped.splitlines())
            legend_labels = []
            if style == "nature" and kind == "line":
                legend_labels = [str(item["label"]) for field in ("series", "vlines", "hlines", "spans")
                                 for item in plot.get(field, [])
                                 if item.get("label") and not str(item["label"]).startswith("_")]
            legend_columns = 2 if wide and len(legend_labels) > 1 else 1
            label_width = 58 if wide and legend_columns == 1 else 45
            legend_labels = [textwrap.fill(label, width=label_width) for label in legend_labels]
            legend_lines = max((sum(len(label.splitlines()) for label in column)
                                for column in np.array_split(legend_labels, legend_columns)), default=0)
            legend_height = 0.13 * legend_lines + (0.06 if legend_labels else 0)
            base_height = (3.7 if wide else 2.6) if style == "nature" else (4.1 if wide else 2.8)
            height = base_height + 0.12 * lines + legend_height
            fig = Figure(figsize=(width, height), layout="constrained")
            FigureCanvasAgg(fig)
            legend_ax = None
            if note or legend_labels:
                heights = [base_height]
                if legend_labels:
                    heights.append(legend_height)
                if note:
                    heights.append(0.12 * max(lines, 1))
                grid = fig.add_gridspec(len(heights), 1, height_ratios=heights)
                ax = fig.add_subplot(grid[0])
                if legend_labels:
                    legend_ax = fig.add_subplot(grid[1])
                    legend_ax.axis("off")
                if note:
                    note_ax = fig.add_subplot(grid[-1])
                    note_ax.axis("off")
                    note_ax.text(0, 1, wrapped, va="top", ha="left", fontsize=6 if style == "nature" else 8,
                                 transform=note_ax.transAxes, color="#444444")
            else:
                ax = fig.add_subplot()
            for name in ("xscale", "yscale"):
                scale = plot.get(name, "linear")
                if scale not in ("linear", "log", "symlog"):
                    raise ValueError(f"Unsupported {name}: {scale}")
                getattr(ax, "set_" + name)(scale)
            data_path = data_dir / f"{plot['id']}.csv"
            if kind == "line":
                _line_plot(ax, plot, data_path, legend=style != "nature",
                           linewidth=1.0 if style == "nature" else 1.25)
            elif kind == "heatmap":
                _heatmap_plot(fig, ax, plot, data_path)
            else:
                _bar_plot(ax, plot, data_path)
            ax.set_xlabel(str(plot.get("xlabel", "")))
            ax.set_ylabel(str(plot.get("ylabel", "")))
            ax.set_title(textwrap.fill(str(plot.get("title", "")), width=90 if wide else 46), loc="left", pad=9)
            if legend_ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                legend_ax.legend(handles, [textwrap.fill(label, width=label_width) for label in labels],
                                 frameon=False, loc="center left", ncols=legend_columns,
                                 fontsize=7, borderaxespad=0, handlelength=2.0,
                                 columnspacing=1.4, labelspacing=0.45)
            if kind != "heatmap" and style != "nature":
                ax.grid(axis="y", alpha=0.16, linewidth=0.5)
                ax.set_axisbelow(True)
            files = {}
            for fmt in formats:
                relative = f"figures/{plot['id']}.{fmt}"
                fig.savefig(output / relative, format=fmt, dpi=dpi)
                files[fmt] = relative
            fig.clear()
            manifests.append({
                "id": plot["id"], "kind": kind, "title": str(plot.get("title", "")),
                "caption": str(plot.get("caption", "")), "note": note,
                "xlabel": str(plot.get("xlabel", "")), "ylabel": str(plot.get("ylabel", "")),
                "style": style, "width_mm": width * 25.4,
                "files": files, "data": f"data/{plot['id']}.csv",
            })
    return manifests


def _value(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:.8g}" if np.isfinite(value) else "Unavailable"
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(value, ensure_ascii=False, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))
    return str(value)


def _html(value):
    return escape(_value(value), quote=True)


def _md(value):
    text = escape(_value(value), quote=False).replace("\n", " ")
    for char in ("\\", "|", "[", "]", "*", "_", "`"):
        text = text.replace(char, "\\" + char)
    return text


def _local_link(value):
    value = str(value)
    parsed = urlsplit(value)
    path = PurePosixPath(value)
    if not value or value == "." or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or path.is_absolute() or "\\" in value or any(p in ("..", ".") for p in path.parts):
        raise ValueError("Report artifact links must be relative local paths")
    return quote(value, safe="/-_.")


def _reference_link(value):
    value = str(value)
    parsed = urlsplit(value)
    return value if parsed.scheme in ("https", "http") and parsed.netloc else None


def _tables(mapping):
    if not mapping:
        return "", ""
    html = '<table><thead><tr><th scope="col">Quantity</th><th scope="col">Value</th></tr></thead><tbody>'
    markdown = "| Quantity | Value |\n| --- | --- |\n"
    for key, value in mapping.items():
        html += f'<tr><th scope="row">{_html(key)}</th><td>{_html(value)}</td></tr>'
        markdown += f"| {_md(key)} | {_md(value)} |\n"
    return html + "</tbody></table>", markdown + "\n"


def write_report(report, output):
    """Write escaped, offline HTML and Markdown beside generated figures.

    No input text is interpreted as HTML, scripts, or executable code.  Sources
    and settings are shown so the report can be traced to its original files.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    title = report.get("title", "PyRPMD analysis report")
    html = [f"<h1>{_html(title)}</h1>"]
    markdown = [f"# {_md(title)}\n"]
    stamp = f"PyRPMD {report.get('software_version', '')} · {report.get('generated_at', '')}"
    html.append(f'<p class="muted">{_html(stamp)}</p>')
    markdown.append(f"{_md(stamp)}\n")
    method = (
        "This report summarizes saved results; it does not establish convergence. "
        "Missing uncertainties are unavailable, not zero. Standard errors and confidence "
        "intervals have different meanings; each figure caption identifies the supplied "
        "quantity. Grey cells are unsupported or masked, and missing samples are never interpolated."
    )
    html.append(f'<p class="notice">{_html(method)}</p>')
    markdown.append(method + "\n")
    for number, run in enumerate(report.get("runs", []), 1):
        label = run.get("label", f"Run {number}")
        html.append(f'<section><h2>{_html(label)}</h2><p class="muted">{_html(run.get("path", ""))}</p>')
        markdown.extend([f"## {_md(label)}\n", f"Source: {_md(run.get('path', ''))}\n"])
        status = {"Task": run.get("task"), "Status": run.get("status"), "Converged (saved result)": run.get("converged")}
        for mapping in (status, run.get("metadata", {}), run.get("metrics", {})):
            ht, md = _tables(mapping)
            html.append(ht)
            markdown.append(md)
        warnings = run.get("warnings", [])
        if warnings:
            html.append("<h3>Saved warnings</h3><ul>" + "".join(f"<li>{_html(w)}</li>" for w in warnings) + "</ul>")
            markdown.append("### Saved warnings\n\n" + "\n".join(f"- {_md(w)}" for w in warnings) + "\n")
        html.append("</section>")
    findings = report.get("findings", [])
    if findings:
        html.append("<section><h2>Analysis findings</h2><ul>")
        markdown.append("## Analysis findings\n")
        for finding in findings:
            prefix = ": ".join(str(finding[k]) for k in ("run", "level", "code") if finding.get(k))
            message = f"{prefix} — {finding.get('message', '')}" if prefix else finding.get("message", "")
            html.append(f"<li>{_html(message)}</li>")
            markdown.append(f"- {_md(message)}")
        html.append("</ul></section>")
        markdown.append("")
    if report.get("comparisons"):
        html.append("<section><h2>Comparison analysis</h2>")
        markdown.append("## Comparison analysis\n")
        ht, md = _tables(report["comparisons"])
        html.append(ht + "</section>")
        markdown.append(md)
    html.append("<section><h2>Figures and numerical data</h2>")
    markdown.append("## Figures and numerical data\n")
    if not report.get("figures"):
        html.append("<p>No figures were generated from the available supported data.</p>")
        markdown.append("No figures were generated from the available supported data.\n")
    for figure in report.get("figures", []):
        html.append(f'<figure><h3>{_html(figure.get("title", figure["id"]))}</h3>')
        markdown.append(f"### {_md(figure.get('title', figure['id']))}\n")
        files = figure.get("files", {})
        preview = files.get("png") or files.get("svg")
        if preview:
            href = _local_link(preview)
            html.append(f'<img src="{escape(href, quote=True)}" alt="{_html(figure.get("title", figure["id"]))}" loading="lazy">')
            markdown.append(f"![{_md(figure.get('title', figure['id']))}]({href})\n")
        html.append(f'<figcaption>{_html(figure.get("caption", ""))}</figcaption>')
        markdown.append(_md(figure.get("caption", "")) + "\n")
        if figure.get("note"):
            html.append(f'<p class="muted">{_html(figure["note"])}</p>')
            markdown.append(_md(figure["note"]) + "\n")
        links = [(fmt.upper(), value) for fmt, value in files.items()]
        if figure.get("data"):
            links.append(("CSV data", figure["data"]))
        html.append('<p class="downloads">' + " · ".join(f'<a href="{_local_link(path)}">{_html(label)}</a>' for label, path in links) + "</p></figure>")
        markdown.append(" · ".join(f"[{_md(label)}]({_local_link(path)})" for label, path in links) + "\n")
    html.append("</section>")
    if report.get("skipped"):
        html.append("<section><h2>Unavailable figures</h2><ul>")
        markdown.append("## Unavailable figures\n")
        for item in report["skipped"]:
            message = f"{item.get('run', '')}: {item.get('figure', '')} — {item.get('reason', '')}"
            html.append(f"<li>{_html(message)}</li>")
            markdown.append(f"- {_md(message)}")
        html.append("</ul></section>")
        markdown.append("")
    html.append("<section><h2>Reproducibility</h2>")
    markdown.append("## Reproducibility\n")
    ht, md = _tables(report.get("settings", {}))
    html.append(ht)
    markdown.append(md)
    html.append('<p><a href="report.json">Machine-readable report (JSON)</a></p>')
    markdown.append("[Machine-readable report (JSON)](report.json)\n")
    if report.get("sources"):
        html.append('<details><summary>Source file hashes</summary><table><thead><tr><th>Path</th><th>SHA-256</th><th>Bytes</th></tr></thead><tbody>')
        markdown.append("### Source file hashes\n\n| Path | SHA-256 | Bytes |\n| --- | --- | --- |")
        for source in report["sources"]:
            values = [source.get(k, "") for k in ("path", "sha256", "size")]
            html.append("<tr>" + "".join(f"<td>{_html(value)}</td>" for value in values) + "</tr>")
            markdown.append("| " + " | ".join(_md(value) for value in values) + " |")
        html.append("</tbody></table></details>")
        markdown.append("")
    html.append("</section>")
    if report.get("references"):
        html.append("<section><h2>Method and figure references</h2><ul>")
        markdown.append("## Method and figure references\n")
        for reference in report["references"]:
            name = reference.get("title", reference.get("url", "Reference"))
            href = _reference_link(reference.get("url", ""))
            html.append(f'<li><a href="{escape(href, quote=True)}" rel="noreferrer">{_html(name)}</a></li>' if href else f"<li>{_html(name)}</li>")
            markdown.append(f"- [{_md(name)}]({quote(href, safe=':/?=&%#-_.')})" if href else f"- {_md(name)}")
        html.append("</ul></section>")
    css = """
body{max-width:1100px;margin:2rem auto;padding:0 1.4rem;color:#20252b;background:#fff;font:16px/1.6 system-ui,sans-serif}
h1,h2,h3{line-height:1.25}h1{font-size:2rem}h2{margin-top:2.5rem;border-bottom:1px solid #dce2e8;padding-bottom:.4rem}
a{color:#005ea8}table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}td,th{text-align:left;vertical-align:top;padding:.45rem .65rem;border-bottom:1px solid #e0e5ea;overflow-wrap:anywhere}
th{font-weight:600}thead{background:#f4f7fa}.muted,figcaption{color:#515d68}.notice{border-left:4px solid #0072b2;padding:.8rem 1rem;background:#f1f7fb}
figure{margin:2rem 0 3rem}img{max-width:100%;height:auto;display:block;margin:.6rem 0}figcaption{margin:.7rem 0}.downloads{font-size:.9rem}li{margin:.5rem 0}details{margin:1rem 0}summary{cursor:pointer}
@media print{body{font-size:10pt;max-width:none}figure{break-inside:avoid}details{display:block}a{color:inherit}}
"""
    document = f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_html(title)}</title><style>{css}</style></head><body><main>' + "\n".join(html) + "</main></body></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
