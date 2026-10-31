"""Attempt-level reporting must survive missing successful diagnostics."""
from __future__ import annotations

import json
from html import unescape
from pathlib import Path
import re

import pytest

reportlab = pytest.importorskip("reportlab")
from benchmarks.h3 import render_preparation_report as report


def audit_cases(*, prepared=0, failed=2):
    return [{"variant": variant, "v": v, "nominal_total_label_ev": energy,
             "attempts": prepared + failed, "prepared": prepared, "failed": failed}
            for variant in report.AUDIT_VARIANTS for v, energy in report.POINTS]


def mean_row(case, *, mean=.25, sem=.01):
    return {**case, "window": "hold", "column": "primitive_internal_per_bead_hartree",
            "n_trajectory_means": case["prepared"], "mean": mean, "sem": sem}


def test_all_failed_audits_retain_attempts_and_have_no_fabricated_mean():
    data = {"complete": True, "audit_cases": audit_cases(), "audit_statistics": []}
    cases = report.audit_case_counts(data)
    assert sum(row["attempts"] for row in cases) == 24
    rows = report.audit_table_rows(data, cases)
    assert len(rows) == 12
    assert all(row[2] == "2 / 0 / 2" and row[3] == "无成功制备样本" for row in rows)


def test_single_success_keeps_undefined_sem_and_all_other_failed_cases():
    cases = audit_cases()
    cases[0].update(prepared=1, failed=1)
    data = {"complete": True, "audit_cases": cases,
            "audit_statistics": [mean_row(cases[0], sem=None)]}
    rows = report.audit_table_rows(data, report.audit_case_counts(data))
    assert len(rows) == 12
    assert rows[0][2] == "2 / 1 / 1"
    assert rows[0][3] == "0.25000 ± 未定义"


@pytest.mark.parametrize("mutation", ["missing", "wrong_count", "nan_mean", "zero_sem"])
def test_successful_audit_cannot_silently_lose_required_means(mutation):
    cases = audit_cases()
    cases[0].update(prepared=1, failed=1)
    mean = mean_row(cases[0], sem=None)
    if mutation == "wrong_count":
        mean["n_trajectory_means"] = 0
    elif mutation == "nan_mean":
        mean["mean"] = float("nan")
    elif mutation == "zero_sem":
        mean["sem"] = 0.
    data = {"complete": True, "audit_cases": cases,
            "audit_statistics": [] if mutation == "missing" else [mean]}
    with pytest.raises(ValueError):
        report.audit_table_rows(data, cases)


def test_legacy_report_requires_complete_unambiguous_audit_counts():
    cases = audit_cases(prepared=2, failed=0)
    rows = [mean_row(case) for case in cases]
    data = {"complete": True, "audit_statistics": rows + [dict(rows[0], window="final_50fs")]}
    assert len(report.audit_case_counts(data)) == 12
    data["audit_statistics"] = rows[:-1]
    with pytest.raises(ValueError, match="reanalyze"):
        report.audit_case_counts(data)
    data["audit_statistics"] = rows + [dict(rows[0], attempts=3, failed=1)]
    with pytest.raises(ValueError, match="inconsistent"):
        report.audit_case_counts(data)


def test_complete_report_requires_each_audit_case_once():
    cases = audit_cases()
    data = {"complete": True, "audit_cases": cases + [cases[0]], "audit_statistics": []}
    with pytest.raises(ValueError, match="Duplicate"):
        report.audit_case_counts(data)
    data["audit_cases"] = cases[:-1]
    with pytest.raises(ValueError, match="reanalyze"):
        report.audit_case_counts(data)


def test_missing_numerical_metrics_are_unavailable_not_zero():
    assert report.finite_maximum([], "error") is None
    assert report.finite_maximum([{"error": None}], "error") is None
    assert report.finite_maximum([{}, {"error": .1}], "error") == .1
    with pytest.raises(ValueError, match="Non-finite"):
        report.finite_maximum([{"error": float("inf")}], "error")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_report_json_is_rejected_before_output(tmp_path, token):
    (tmp_path / "preparation_analysis.json").write_text('{"complete": true, "bad": ' + token + '}')
    output = tmp_path / "report.pdf"
    with pytest.raises(ValueError, match="Non-finite"):
        report.render(tmp_path, output, tmp_path / "missing-font.ttf")
    assert not output.exists()


def test_render_all_failed_audits_and_distinct_collision_outcomes(tmp_path, capsys, monkeypatch):
    pdf = pytest.importorskip("pypdf")
    image = pytest.importorskip("PIL.Image")
    font = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    if not font.is_file():
        pytest.skip("ReportLab's bundled Vera font is unavailable")
    (tmp_path / "figures").mkdir()
    for name in ("preparation_scattering.png", "preparation_energy_hold.png"):
        image.new("RGB", (640, 300), "white").save(tmp_path / "figures" / name)
    records = [{"variant": variant, "v": v, "nominal_total_label_ev": energy,
                "samples": 3, "product": 0, "nonreactive": 0,
                "failed": 2, "unresolved": 1, "unknown": 3,
                "collision_energy_ev": .1, "sigma_confirmed_angstrom2": 0.,
                "sigma_ci95_including_unknown_lower_angstrom2": 0.,
                "sigma_ci95_including_unknown_upper_angstrom2": 1.}
               for variant in report.VARIANTS for v, energy in report.POINTS]
    data = {"complete": True, "cross_sections": records, "paired_comparisons": [],
            "internal_energy_references": [], "audit_cases": audit_cases(),
            "audit_statistics": [], "preparation_numerics": []}
    (tmp_path / "preparation_analysis.json").write_text(json.dumps(data, allow_nan=False))
    output = tmp_path / "report.pdf"
    paragraphs = []
    original_paragraph = report.Paragraph

    def record_paragraph(text, *args, **kwargs):
        paragraphs.append(unescape(text))
        return original_paragraph(text, *args, **kwargs)

    monkeypatch.setattr(report, "Paragraph", record_paragraph)
    report.render(tmp_path, output, font)
    summary = json.loads(capsys.readouterr().out)
    assert summary["collision_samples"] == 36
    assert summary["audit_samples"] == 24
    assert summary["audit_prepared"] == 0
    assert summary["audit_failed"] == 24
    # The packaged Latin font makes the rendering test portable. Assert the
    # actual paragraph content before glyph substitution; visual CJK inspection
    # is performed with a user-selected CJK font outside the optional test suite.
    assert len(pdf.PdfReader(output).pages) >= 5
    text = re.sub(r"\s+", "", "".join(paragraphs))
    assert "新增碰撞中24条计算失败、12条末态未决" in text
    assert "孤立初态测试中0个成功、24个失败" in text
    assert text.count("无成功制备样本") == 12
    assert "没有可用的成功制备数值诊断" in text
