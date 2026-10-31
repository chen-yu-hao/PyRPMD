"""Render the calculated comparison report as a readable Chinese PDF.

The numerical inputs are the same JSON/CSV files used by the HTML report.
This renderer never executes trajectories or modifies their ledgers.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               PageBreak, Table, TableStyle, Preformatted)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.analyze_state_selected_reproduction import (
    load_cases, load_csv, compare_literature, report_sections, paired_checks)


def render(root, output, font):
    pdfmetrics.registerFont(TTFont("ReportCJK", str(font), subfontIndex=0))
    pdfmetrics.registerFontFamily("ReportCJK", normal="ReportCJK", bold="ReportCJK", italic="ReportCJK")
    body = ParagraphStyle("Body", fontName="ReportCJK", fontSize=9.6,
                          leading=15, spaceAfter=8, wordWrap="CJK", textColor=colors.HexColor("#202936"))
    heading = ParagraphStyle("Heading", parent=body, fontSize=15, leading=21,
                             textColor=colors.HexColor("#176BA0"), spaceBefore=10,
                             spaceAfter=12, keepWithNext=True)
    title = ParagraphStyle("Title", parent=body, fontSize=23, leading=33, spaceAfter=14)
    subtitle = ParagraphStyle("Subtitle", parent=body, fontSize=10, leading=16,
                              textColor=colors.HexColor("#657083"), spaceAfter=20)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.9, leading=11.5,
                          spaceAfter=0, alignment=TA_LEFT)
    code = ParagraphStyle("Code", fontName="Courier", fontSize=7.2, leading=10,
                          textColor=colors.HexColor("#303b47"), spaceAfter=10)
    cases = load_cases(root)
    refs = load_csv(root/"literature_reference/marjollet_2021_requested_energy_interpolation.csv")
    comparisons = compare_literature(cases, refs)
    h2 = json.loads((root/"h2_reference.json").read_text())
    audit_path = root/"initial_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    sections = report_sections(cases, comparisons, h2, audit, paired_checks(root, cases))
    width = A4[0]-36*mm
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18*mm,
                            rightMargin=18*mm, topMargin=17*mm, bottomMargin=18*mm,
                            title="PyRPMD: H + H2 BKMP2 state-selected reproduction",
                            author="PyRPMD reproduction calculation", pageCompression=1)
    flow = [Paragraph("H + H₂ 态选择散射<br/>文献复现与数值审计", title),
            Paragraph("真实 BKMP2 势能面 · NE-RPMD / 谐振 QCT · 2026-10-31", subtitle)]
    page_starts = {"截面与误差", "H₂ 的独立量子参考", "初态切换审计", "数值检查与限度", "外侧环带与分层截面", "可复现文件与命令"}
    for name, content in sections:
        # The bundled CJK font lacks U+207B and U+2212: preserve mathematical
        # signs explicitly instead of allowing silent missing-glyph output.
        content = content.replace("⁻¹", "^-1").replace("−", "-")
        if name in page_starts:
            flow.append(PageBreak())
        flow.append(Paragraph(escape(name), heading))
        figure = {"截面与误差": "scattering_comparison.png",
                  "H₂ 的独立量子参考": "h2_reference.png"}.get(name)
        if figure:
            image = Image(str(root/"figures"/figure))
            image.drawHeight *= width / image.drawWidth
            image.drawWidth = width
            flow.extend([image, Spacer(1, 10)])
        fragment = ET.fromstring("<section>"+content+"</section>")
        for element in fragment:
            if element.tag == "p":
                serialized = ET.tostring(element, encoding="unicode")
                inner = serialized[serialized.index(">")+1:serialized.rindex("</")]
                flow.append(Paragraph(inner, body))
            elif element.tag == "table":
                data = [[Paragraph(escape("".join(c.itertext())), cell)
                         for c in row] for row in element.findall(".//tr")]
                ncol = len(data[0])
                if ncol == 7:
                    fractions = [.04, .105, .115, .12, .105, .245, .17]
                elif ncol == 8:
                    fractions = [.13, .045, .08, .055, .075, .075, .11, .43]
                elif ncol == 6:
                    fractions = [.04, .095, .09, .09, .32, .365]
                else:
                    fractions = [1/ncol]*ncol
                table = Table(data, colWidths=[width*f for f in fractions], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F7")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#D5DDE6")),
                ]))
                flow.extend([table, Spacer(1, 11)])
            elif element.tag == "pre":
                flow.append(Preformatted("".join(element.itertext()), code, maxLineLength=93))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5DDE6"))
        canvas.line(18*mm, 13*mm, A4[0]-18*mm, 13*mm)
        canvas.setFont("ReportCJK", 7.7)
        canvas.setFillColor(colors.HexColor("#657083"))
        canvas.drawString(18*mm, 8.7*mm, "PyRPMD · BKMP2 文献复现 · 原始轨迹与配置随数据包保存")
        canvas.drawRightString(A4[0]-18*mm, 8.7*mm, str(document.page))
        canvas.restoreState()
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(output), "comparison_points": len(comparisons)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"runs/state_selected_reproduction")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"))
    args = parser.parse_args()
    output = args.output or args.input/"pyrpmd_bkmp2_reproduction_report.pdf"
    render(args.input, output, args.font)


if __name__ == "__main__":
    main()
