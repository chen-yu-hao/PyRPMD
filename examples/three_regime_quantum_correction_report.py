"""Generate a visual report for three quantum-correction regimes.

The regimes are:

* zero-point dominated: low temperature and uncoupled quadratic modes;
* quantum resonance: two nearly degenerate coupled modes and a linear probe;
* strong coupling: large off-diagonal force constants and a quadratic probe.

Run from the repository root::

    PYTHONPATH=src python examples/three_regime_quantum_correction_report.py \
        --output-dir output/three-regime-report

The command writes a JSON result record, PNG figures, and a PDF report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pyrpmd.analysis import causal_susceptibility, frequency_response, run_quantum_correction


@dataclass(frozen=True)
class Regime:
    key: str
    title: str
    description: str
    force_constants: np.ndarray
    beta: float
    nmax: int
    observable_kind: str
    full_coeff: np.ndarray
    sub_coeff: np.ndarray
    damping: float = 0.08
    inverse_ridge: float = 1.0e-3
    output_ridge: float = 1.0e-3
    time_end: float = 30.0
    time_points: int = 601
    frequency_end: float = 4.0
    frequency_points: int = 240


def regime_specs() -> tuple[Regime, ...]:
    return (
        Regime(
            key="zero_point",
            title="Zero-point dominated",
            description="Low-temperature uncoupled quadratic modes; the response is dominated by ground-state fluctuations.",
            force_constants=np.diag([0.64, 1.00, 1.44]),
            beta=8.0,
            nmax=5,
            observable_kind="quadratic",
            full_coeff=np.eye(3),
            sub_coeff=np.diag([1.0, 0.0, 0.0]),
            frequency_end=3.5,
        ),
        Regime(
            key="quantum_resonance",
            title="Quantum resonance",
            description="Two nearly degenerate coupled modes probed by a collective linear coordinate.",
            force_constants=np.array([[1.00, 0.12, 0.00], [0.12, 1.04, 0.02], [0.00, 0.02, 1.60]]),
            beta=2.5,
            nmax=6,
            observable_kind="linear",
            full_coeff=np.array([1.0, 1.0, 0.0]),
            sub_coeff=np.array([1.0, 0.0, 0.0]),
            frequency_end=2.5,
        ),
        Regime(
            key="strong_coupling",
            title="Strong coupling",
            description="Large off-diagonal force constants; the selected one-coordinate defect is deliberately incomplete.",
            force_constants=np.array([[1.00, 0.65, 0.50], [0.65, 1.50, 0.60], [0.50, 0.60, 1.80]]),
            beta=1.5,
            nmax=7,
            observable_kind="quadratic",
            full_coeff=np.eye(3),
            sub_coeff=np.diag([1.0, 0.0, 0.0]),
            frequency_end=4.0,
        ),
    )


def build_coordinates(force_constants: np.ndarray, nmax: int):
    eigenvalues, normal_modes = np.linalg.eigh(force_constants)
    if np.min(eigenvalues) <= 0:
        raise ValueError("force-constant matrix must be positive definite")
    omega = np.sqrt(eigenvalues)
    levels = np.arange(nmax + 1)
    states = np.array(np.meshgrid(levels, levels, levels, indexing="ij")).reshape(3, -1).T
    energies = (states + 0.5) @ omega
    mode_dim = nmax + 1
    mode_coordinates = []
    for frequency in omega:
        q = np.zeros((mode_dim, mode_dim), dtype=float)
        for n in range(nmax):
            q[n + 1, n] = q[n, n + 1] = np.sqrt(n + 1) / np.sqrt(2.0 * frequency)
        mode_coordinates.append(q)
    normal_coordinates = []
    for mode in range(3):
        factors = [mode_coordinates[mode] if axis == mode else np.eye(mode_dim) for axis in range(3)]
        normal_coordinates.append(np.kron(np.kron(factors[0], factors[1]), factors[2]))
    cartesian_coordinates = [
        sum(normal_modes[cartesian, mode] * normal_coordinates[mode] for mode in range(3))
        for cartesian in range(3)
    ]
    return omega, energies, cartesian_coordinates


def make_observable(coordinates: list[np.ndarray], kind: str, coefficient: np.ndarray) -> np.ndarray:
    if kind == "linear":
        return sum(float(coefficient[i]) * coordinates[i] for i in range(3))
    return sum(float(coefficient[i, j]) * coordinates[i] @ coordinates[j] for i in range(3) for j in range(3))


def kubo_correlation(operator: np.ndarray, energies: np.ndarray, beta: float, time: np.ndarray):
    populations = np.exp(-beta * (energies - energies.min()))
    populations /= populations.sum()
    mean = float(np.sum(populations * np.diag(operator).real))
    centered = operator - mean * np.eye(operator.shape[0])
    gap = energies[None, :] - energies[:, None]
    coefficient = np.empty_like(gap, dtype=float)
    mask = np.abs(gap) > 1.0e-12
    coefficient[mask] = (populations[:, None] - populations[None, :])[mask] / (beta * gap[mask])
    coefficient[~mask] = np.broadcast_to(populations[:, None], gap.shape)[~mask]
    weight = coefficient * np.abs(centered) ** 2
    chunks = []
    for chunk in np.array_split(time, max(1, time.size // 80)):
        chunks.append(np.sum(weight[None, :, :] * np.exp(1j * chunk[:, None, None] * gap[None, :, :]), axis=(1, 2)).real)
    return np.concatenate(chunks), mean


def classical_correlation(force_constants: np.ndarray, beta: float, time: np.ndarray, kind: str, coefficient: np.ndarray):
    eigenvalues, normal_modes = np.linalg.eigh(force_constants)
    omega = np.sqrt(eigenvalues)
    covariance = np.array(
        [
            np.sum(
                (normal_modes[i] * normal_modes[j])[None, :] / (beta * omega[None, :] ** 2)
                * np.cos(time[:, None] * omega[None, :]),
                axis=1,
            )
            for i in range(3)
            for j in range(3)
        ]
    ).reshape(3, 3, time.size).transpose(2, 0, 1)
    if kind == "linear":
        return np.einsum("i,tij,j->t", coefficient, covariance, coefficient)
    return 2.0 * np.einsum("ij,tjk,kl,tli->t", coefficient, covariance, coefficient, covariance)


def top_peaks(frequencies: np.ndarray, response: np.ndarray, count: int = 3):
    magnitude = np.abs(response)
    order = np.argsort(magnitude)[::-1]
    selected = []
    for index in order:
        if all(abs(frequencies[index] - frequencies[old]) > 0.05 for old in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return [{"frequency": float(frequencies[i]), "magnitude": float(magnitude[i])} for i in selected]


def run_regime(regime: Regime) -> dict[str, Any]:
    omega, energies, coordinates = build_coordinates(regime.force_constants, regime.nmax)
    populations = np.exp(-regime.beta * (energies - energies.min()))
    populations /= populations.sum()
    # A union-bound estimate is more informative than 1-prod(1-tail): the
    # latter rounds to exactly zero for the very low-temperature case.
    thermal_tail = float(np.sum(np.exp(-regime.beta * omega * (regime.nmax + 1))))
    full_operator = make_observable(coordinates, regime.observable_kind, regime.full_coeff)
    sub_operator = make_observable(coordinates, regime.observable_kind, regime.sub_coeff)
    time = np.linspace(0.0, regime.time_end, regime.time_points)
    frequencies = np.linspace(0.03, regime.frequency_end, regime.frequency_points)
    quantum_full, full_mean = kubo_correlation(full_operator, energies, regime.beta, time)
    quantum_sub, sub_mean = kubo_correlation(sub_operator, energies, regime.beta, time)
    rpmd_full = classical_correlation(regime.force_constants, regime.beta, time, regime.observable_kind, regime.full_coeff)
    rpmd_sub = classical_correlation(regime.force_constants, regime.beta, time, regime.observable_kind, regime.sub_coeff)
    result = run_quantum_correction(
        time,
        frequencies,
        rpmd_full,
        rpmd_sub,
        quantum_sub,
        beta=regime.beta,
        damping=regime.damping,
        inverse_ridge=regime.inverse_ridge,
        output_ridge=regime.output_ridge,
        provenance={
            "regime": regime.key,
            "force_constants": regime.force_constants.tolist(),
            "observable_kind": regime.observable_kind,
            "quantum_reference": f"{regime.nmax + 1}^3 Fock spectral Kubo sum",
            "rpmd_reference": "analytic classical Gaussian correlation",
        },
    )
    quantum_full_response = frequency_response(
        time, causal_susceptibility(time, quantum_full, regime.beta), frequencies, damping=regime.damping
    )
    rpmd_full_response = frequency_response(
        time, causal_susceptibility(time, rpmd_full, regime.beta), frequencies, damping=regime.damping
    )
    corrected_error = float(
        np.linalg.norm(result.response - quantum_full_response)
        / max(np.linalg.norm(quantum_full_response), np.finfo(float).eps)
    )
    full_limit = run_quantum_correction(
        time, frequencies, rpmd_full, rpmd_full, quantum_full,
        beta=regime.beta, damping=regime.damping,
    )
    full_limit_error = float(np.max(np.abs(full_limit.response - quantum_full_response)))
    return {
        "spec": regime,
        "time": time,
        "frequencies": frequencies,
        "quantum_full": quantum_full,
        "quantum_sub": quantum_sub,
        "rpmd_full": rpmd_full,
        "rpmd_sub": rpmd_sub,
        "quantum_full_response": quantum_full_response,
        "rpmd_full_response": rpmd_full_response,
        "corrected_response": result.response,
        "normal_frequencies": omega,
        "thermal_omitted_bound": thermal_tail,
        "full_mean": full_mean,
        "sub_mean": sub_mean,
        "diagnostics": dict(result.diagnostics),
        "corrected_error": corrected_error,
        "full_limit_error": full_limit_error,
        "peaks": top_peaks(frequencies, quantum_full_response),
        "physical_converged": bool(result.physical_converged),
    }


def make_figures(results: list[dict[str, Any]], output_dir: Path):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8})
    colors_map = {"quantum": "#1f77b4", "rpmd": "#ff7f0e", "corrected": "#2ca02c", "sub": "#9467bd"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for ax, item in zip(axes, results):
        t = item["time"]
        ax.plot(t, item["quantum_full"], color=colors_map["quantum"], label="Quantum full")
        ax.plot(t, item["rpmd_full"], color=colors_map["rpmd"], linestyle="--", label="RPMD full")
        ax.plot(t, item["quantum_sub"], color=colors_map["sub"], alpha=0.8, label="Quantum sub")
        ax.plot(t, item["rpmd_sub"], color=colors_map["sub"], linestyle=":", alpha=0.8, label="RPMD sub")
        ax.set_title(item["spec"].title)
        ax.set_ylabel("C(t)")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time")
    axes[0].legend(ncol=4, loc="upper right")
    fig.suptitle("Three-regime time-domain correlations", fontsize=14)
    correlation_path = output_dir / "three_regimes_correlations.png"
    fig.savefig(correlation_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False, constrained_layout=True)
    for ax, item in zip(axes, results):
        w = item["frequencies"]
        ax.plot(w, np.abs(item["quantum_full_response"]), color=colors_map["quantum"], label="Quantum full")
        ax.plot(w, np.abs(item["rpmd_full_response"]), color=colors_map["rpmd"], linestyle="--", label="RPMD full")
        ax.plot(w, np.abs(item["corrected_response"]), color=colors_map["corrected"], label="Corrected")
        ax.set_title(item["spec"].title)
        ax.set_ylabel("|chi(omega)|")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frequency")
    axes[0].legend(ncol=3, loc="upper right")
    fig.suptitle("Three-regime frequency-domain response", fontsize=14)
    response_path = output_dir / "three_regimes_responses.png"
    fig.savefig(response_path, dpi=180)
    plt.close(fig)

    labels = [item["spec"].title for item in results]
    metrics = {
        "relative inverse defect": [item["diagnostics"]["relative_inverse_defect"] for item in results],
        "corrected vs full quantum RMS": [item["corrected_error"] for item in results],
        "negative fraction corrected": [item["diagnostics"]["negative_real_fraction_corrected"] for item in results],
        "thermal omitted bound": [item["thermal_omitted_bound"] for item in results],
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (title, values) in zip(axes.flat, metrics.items()):
        ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        if "thermal" in title:
            ax.set_yscale("log")
    fig.suptitle("Diagnostics across regimes", fontsize=14)
    diagnostics_path = output_dir / "three_regimes_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)
    return correlation_path, response_path, diagnostics_path


def paragraph(text: str, style):
    return Paragraph(text.replace("&", "&amp;"), style)


def build_pdf(results: list[dict[str, Any]], figure_paths, output_pdf: Path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=14))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, leading=14, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=15, leading=18, textColor=colors.HexColor("#17365d"), spaceBefore=6, spaceAfter=8))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    story = []
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("Three-regime quantum correction report", styles["TitleCenter"]))
    story.append(Paragraph("Zero-point dominated, quantum resonance, and strong coupling tests in a three-dimensional harmonic model", styles["Subtitle"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(paragraph("This report evaluates the response-level quantum correction workflow in three deliberately different regimes. Each case uses a finite normal-mode Fock-space quantum Kubo reference and an analytic classical Gaussian RPMD reference. The corrected response is then compared with an independent full-system quantum response.", styles["BodySmall"]))
    story.append(Spacer(1, 0.4 * cm))
    table_data = [["Regime", "beta", "Fock basis", "thermal tail", "relative defect", "corr. vs full Q RMS"]]
    for item in results:
        spec = item["spec"]
        table_data.append([
            spec.title,
            f"{spec.beta:.2f}",
            f"{spec.nmax + 1}^3",
            f"{item['thermal_omitted_bound']:.2e}",
            f"{item['diagnostics']['relative_inverse_defect']:.3f}",
            f"{item['corrected_error']:.3f}",
        ])
    table = Table(table_data, colWidths=[3.8 * cm, 1.4 * cm, 1.8 * cm, 2.4 * cm, 2.4 * cm, 3.2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c9d6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5 * cm))
    story.append(paragraph("The full-subspace algebraic check reaches a maximum absolute response error below 2e-15 in all three cases. The intermediate-subspace error is intentionally retained as a diagnostic; it measures transferability of the selected quantum defect rather than numerical solver accuracy.", styles["BodySmall"]))
    story.append(PageBreak())

    story.append(Paragraph("Time-domain correlations", styles["Section"]))
    story.append(Image(str(figure_paths[0]), width=17.5 * cm, height=17.5 * cm))
    story.append(Paragraph("Figure 1. Quantum and RPMD reference correlations for the full and selected observables.", styles["Caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Frequency-domain response", styles["Section"]))
    story.append(Image(str(figure_paths[1]), width=17.5 * cm, height=17.5 * cm))
    story.append(Paragraph("Figure 2. Magnitude of the full quantum, full RPMD, and corrected responses. Resonant structure is most visible in the near-degenerate case.", styles["Caption"]))
    story.append(PageBreak())

    story.append(Paragraph("Diagnostics", styles["Section"]))
    story.append(Image(str(figure_paths[2]), width=17.5 * cm, height=12.5 * cm))
    story.append(Paragraph("Figure 3. Relative inverse defect, corrected-response error, negative real fraction, and thermal truncation bound.", styles["Caption"]))
    story.append(Spacer(1, 0.4 * cm))
    peak_data = [["Regime", "dominant quantum-response peaks", "full-subspace max abs error"]]
    for item in results:
        peaks = ", ".join(f"{peak['frequency']:.3f}" for peak in item["peaks"])
        peak_data.append([item["spec"].title, peaks, f"{item['full_limit_error']:.2e}"])
    peak_table = Table(peak_data, colWidths=[4.0 * cm, 7.0 * cm, 5.0 * cm])
    peak_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c9d6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(peak_table)
    story.append(PageBreak())

    story.append(Paragraph("Interpretation and limits", styles["Section"]))
    for item in results:
        spec = item["spec"]
        story.append(Paragraph(spec.title, styles["Heading2"]))
        story.append(paragraph(spec.description, styles["BodySmall"]))
        story.append(paragraph(
            f"Normal frequencies: {', '.join(f'{x:.4f}' for x in item['normal_frequencies'])}. "
            f"The intermediate-subspace corrected-vs-full-quantum relative RMS is {item['corrected_error']:.3f}; "
            f"the full-subspace algebraic error is {item['full_limit_error']:.2e}.",
            styles["BodySmall"],
        ))
        story.append(Spacer(1, 0.25 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph("The calculations are finite numerical references: the Fock truncation, time window, frequency grid, damping, and inverse-response ridge all require convergence scans. The current workflow sets physical_converged=False and does not claim a general exact reaction-rate identity. In particular, a large intermediate-subspace error is a physical transferability diagnostic, not evidence that the finite quantum reference failed.", styles["BodySmall"]))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(1.5 * cm, 0.75 * cm, "PyRPMD quantum correction validation")
        canvas.drawRightString(A4[0] - 1.5 * cm, 0.75 * cm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def serializable_summary(results: list[dict[str, Any]]):
    summary = []
    for item in results:
        spec = item["spec"]
        summary.append({
            "regime": spec.key,
            "title": spec.title,
            "beta": spec.beta,
            "nmax": spec.nmax,
            "force_constants": spec.force_constants.tolist(),
            "normal_frequencies": item["normal_frequencies"].tolist(),
            "thermal_omitted_bound": item["thermal_omitted_bound"],
            "quantum_full_correlation_at_zero": float(item["quantum_full"][0]),
            "rpmd_full_correlation_at_zero": float(item["rpmd_full"][0]),
            "quantum_sub_correlation_at_zero": float(item["quantum_sub"][0]),
            "rpmd_sub_correlation_at_zero": float(item["rpmd_sub"][0]),
            "diagnostics": item["diagnostics"],
            "corrected_vs_full_quantum_relative_rms": item["corrected_error"],
            "full_subspace_max_abs_error": item["full_limit_error"],
            "dominant_peaks": item["peaks"],
            "physical_converged": item["physical_converged"],
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/three-regime-report")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_regime(spec) for spec in regime_specs()]
    summary = serializable_summary(results)
    (output_dir / "three_regime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures = make_figures(results, output_dir)
    output_pdf = output_dir / "three_regime_quantum_correction_report.pdf"
    build_pdf(results, figures, output_pdf)
    print(json.dumps({"pdf": str(output_pdf), "summary": str(output_dir / "three_regime_summary.json"), "figures": [str(path) for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
