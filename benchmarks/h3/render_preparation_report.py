"""Render the completed preparation study as a Chinese, data-driven PDF.

Only analysis outputs are read. This renderer never propagates trajectories or
changes outcome labels. The optional PDF dependency is ReportLab plus a CJK font.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[2]
POINTS = ((0, .55), (1, .90), (1, 1.10))
VARIANTS = {
    "ne_paper_subset": "论文筛选 / 直接切换",
    "ne_uncut": "无筛选 / 直接切换",
    "ne_switch200": "无筛选 / 200 fs 切换",
    "qct_harmonic": "谐振 QCT",
    "qct_action_collision": "作用量 QCT / 固定碰撞能",
    "qct_action_total": "作用量 QCT / 固定实际总能",
}
AUDIT_VARIANTS = {
    "direct-paper": "论文筛选 / 直接",
    "direct-none": "无筛选 / 直接",
    "switch50-none": "无筛选 / 50 fs",
    "switch200-none": "无筛选 / 200 fs",
}
COMPARISONS = {
    "Remove centroid cutoff": "删除质心键长筛选",
    "Add 200 fs switch to uncut NE": "无筛选 NE 加入 200 fs 切换",
    "Use action QCT at fixed Ecol": "同碰撞能：作用量 QCT - 谐振 QCT",
    "Retune action QCT to fixed actual Etot": "作用量 QCT：固定总能 - 固定碰撞能",
}


def number(value, digits=4):
    return "未定义" if value is None else f"{value:.{digits}f}"


def render(source, output, font, code_commit=None):
    report = json.loads((source/"preparation_analysis.json").read_text())
    if not report["complete"]:
        raise ValueError("Refusing to render an incomplete study as the final report")
    pdfmetrics.registerFont(TTFont("PreparationCJK", str(font), subfontIndex=0))
    pdfmetrics.registerFontFamily("PreparationCJK", normal="PreparationCJK", bold="PreparationCJK")
    body = ParagraphStyle("Body", fontName="PreparationCJK", fontSize=9.4,
                          leading=14.8, spaceAfter=8, wordWrap="CJK",
                          textColor=colors.HexColor("#253347"))
    title = ParagraphStyle("Title", parent=body, fontSize=23, leading=33, spaceAfter=18)
    heading = ParagraphStyle("Heading", parent=body, fontSize=16, leading=23,
                             spaceAfter=13, textColor=colors.HexColor("#126a95"), keepWithNext=True)
    subhead = ParagraphStyle("Subhead", parent=body, fontSize=11.2, leading=17,
                             spaceBefore=7, spaceAfter=8, keepWithNext=True)
    small = ParagraphStyle("Small", parent=body, fontSize=8.2, leading=12.4)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.0, leading=11.6, spaceAfter=0)
    code = ParagraphStyle("Code", fontName="Courier", fontSize=6.9, leading=9.5, spaceAfter=10)
    width = A4[0]-36*mm
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18*mm,
                            rightMargin=18*mm, topMargin=16*mm, bottomMargin=18*mm,
                            title="PyRPMD H + H2: state preparation sensitivity",
                            author="PyRPMD calculation", pageCompression=1)
    flow = []

    def paragraph(text, style=body):
        text = text.replace("⁻¹", "^-1").replace("−", "-")
        flow.append(Paragraph(escape(text), style))

    def table(headers, rows, fractions):
        data = [[Paragraph(escape(str(x)).replace("−", "-"), cell) for x in row]
                for row in [headers, *rows]]
        item = Table(data, colWidths=[width*f for f in fractions], repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2f7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#d8e2e9")),
        ]))
        flow.extend([item, Spacer(1, 10)])

    def figure(name):
        item = Image(str(source/"figures"/name))
        item.drawHeight *= width/item.drawWidth
        item.drawWidth = width
        flow.extend([item, Spacer(1, 8)])

    records = report["cross_sections"]
    lookup = {(r["variant"], r["v"], r["nominal_total_label_ev"]): r for r in records}
    pairs = report["paired_comparisons"]
    new = [r for r in records if r["variant"] in
           ("ne_uncut", "ne_switch200", "qct_action_collision", "qct_action_total")]
    means = [r for r in report["audit_statistics"] if r["window"] == "hold"
             and r["column"] == "primitive_internal_per_bead_hartree"]
    unknowns = sum(r["unknown"] for r in new)
    failures = sum(r["failed"] for r in new)

    paragraph("H + H₂ 态选择散射\n初态制备敏感性研究", title)
    paragraph("PyRPMD · BKMP2 势能面 · NE-RPMD / QCT · 2026-10-31", subhead)
    paragraph("结论与适用范围", heading)
    paragraph("本轮计算把初态筛选、谐振近似、平滑切换和能量约定分开比较。"
              "所有动力学数据来自实际计算；没有按论文曲线调参数，也没有将未决轨迹删除。"
              "这是一项初态敏感性研究，不能据此宣称已完成论文的定量复现。")
    paragraph(f"新增 {sum(r['samples'] for r in new):,} 条碰撞轨迹及 "
              f"{sum(r['attempts'] for r in means):,} 个独立孤立 H₂ 初态测试。"
              f"新增碰撞中 {failures} 条计算失败、{unknowns} 条末态未决。"
              "NE 对照每点使用相同的 1,024 个轨迹 ID；QCT 对照每点使用相同的 4,096 个 ID。")
    paragraph("已经能回答的问题", subhead)
    paragraph("1. 质心键长筛选确实写在 2021 年论文中。删除它改变了激发态初态系综，并显著增加低能反应；不能把删除筛选直接称为纠错。")
    paragraph("2. 全 BKMP2 势上的半经典作用量 QCT 可避免用谐振轨道直接初始化非谐势的误差。固定碰撞能与固定实际总能是两项不同实验，结果和实际碰撞能均分开列出。")
    paragraph("3. 200 fs 平滑切换是实验性扩展。切换后平均能量接近某个量子能级，并不证明获得了该振动态，也不证明多珠子激发态得到了绝热保持。")
    table(["对照", "目的", "固定内容"], [
        ["有 / 无论文筛选", "评估初态条件化", "ID、碰撞能、冲量参数随机流"],
        ["直接 / 200 fs 切换", "评估切换制备效应", "无筛选初态；A 原子及 BC 质心"],
        ["谐振 / 作用量 QCT", "评估非谐初态误差", "碰撞能、配对相位、发射随机流"],
        ["两种作用量 QCT 能量约定", "区分初态与可用碰撞能", "同一作用量轨道与抽样相位"],
    ], [.27, .28, .45])
    paragraph("注意：删除筛选时，原先被拒绝的轨迹 ID 会改变整组初态抽样，因此该对照不是逐坐标相同。"
              "在无筛选直接切换与平滑切换之间，代码保留 A 原子和 BC 质心的坐标、动量。", small)

    flow.append(PageBreak())
    paragraph("1 / 碰撞截面：初态效应有多大", heading)
    figure("preparation_scattering.png")
    paragraph("图中 NE 的论文筛选基线是原始 4,096 条数据中精确配对的 1,024 条子集。"
              "虚线为论文 NE 曲线的数字化插值，色带是数字化容差；它不是论文的统计置信区间。QCT 图仅比较本项目计算。", small)
    ne_rows = []
    for v, energy in POINTS:
        for variant in ("ne_paper_subset", "ne_uncut", "ne_switch200"):
            r = lookup[variant, v, energy]
            ne_rows.append([f"{v} / {energy:.2f}", VARIANTS[variant],
                            f"{r['product']} / {r['unknown']}",
                            number(r["sigma_confirmed_angstrom2"]),
                            f"[{number(r['sigma_ci95_including_unknown_lower_angstrom2'])}, "
                            f"{number(r['sigma_ci95_including_unknown_upper_angstrom2'])}]"])
    table(["v / 总能标记 eV", "NE 初态", "反应 / 未决", "截面 Å²", "95% 区间 Å²"],
          ne_rows, [.14, .30, .13, .13, .30])
    paragraph("截面 = πbmax² × 确认反应数 / 全部尝试数。未决或失败样本保留在分母；"
              "区间从最小可能反应数与最大可能反应数扩展 Clopper-Pearson 端点。"
              "零次观测反应只给出截面的统计上界，不能证明真实截面严格为零。", small)

    flow.append(PageBreak())
    paragraph("2 / 配对差异与 QCT 能量约定", heading)
    pair_rows = []
    for r in pairs:
        delta = (number(r["delta_sigma_angstrom2"]) if r["delta_sigma_angstrom2"] is not None
                 else f"[{number(r['delta_sigma_identification_lower_angstrom2'])}, "
                      f"{number(r['delta_sigma_identification_upper_angstrom2'])}]")
        pair_rows.append([f"{r['v']} / {r['nominal_total_label_ev']:.2f}",
                          COMPARISONS.get(r["comparison"], r["comparison"]), delta,
                          f"[{number(r['delta_sigma_conservative_ci95_lower_angstrom2'])}, "
                          f"{number(r['delta_sigma_conservative_ci95_upper_angstrom2'])}]"])
    table(["v / 标记 eV", "对照：新值 - 参考值", "Δ截面 Å²", "保守 95% 区间 Å²"],
          pair_rows, [.13, .40, .20, .27])
    paragraph("每个对照通过新增反应与丢失反应的两个 97.5% Clopper-Pearson 区间，构造至少 95% 覆盖率的差值区间。"
              "未决配对扩展为确定/可能事件数，不丢弃样本。该覆盖率针对单个对照，未对全部 12 个对照校正。", small)
    table(["v", "谐振标记 eV", "作用量能量 eV", "独立量子能级 eV", "作用量 - 量子 eV"],
          [[r["v"], number(r["harmonic_internal_label_ev"], 6),
            number(r["action_internal_energy_ev"], 6), number(r["exact_quantum_internal_energy_ev"], 6),
            number(r["action_minus_exact_ev"], 6)] for r in report["internal_energy_references"]],
          [.05, .24, .24, .24, .23])
    paragraph("作用量量子化采用 J(E) = (1/π)∫√[2μ(E-V(r))]dr = v + 1/2，并按轨道时间均匀抽样。"
              "这是最低阶 EBK/WKB 半经典参考，尚不是精确量子本征态。对于 v=1，"
              "固定实际总能比固定原碰撞能多出约 0.03442 eV 的碰撞能；这部分变化必须与初态改善分开解释。")

    flow.append(PageBreak())
    paragraph("3 / QCT 截面及实际碰撞能", heading)
    qct_rows = []
    for v, energy in POINTS:
        for variant in ("qct_harmonic", "qct_action_collision", "qct_action_total"):
            r = lookup[variant, v, energy]
            qct_rows.append([f"{v} / {energy:.2f}", VARIANTS[variant],
                             number(r["collision_energy_ev"], 5), r["product"],
                             number(r["sigma_confirmed_angstrom2"]),
                             f"[{number(r['sigma_ci95_including_unknown_lower_angstrom2'])}, "
                             f"{number(r['sigma_ci95_including_unknown_upper_angstrom2'])}]"])
    table(["v / 标记 eV", "QCT 初态与能量约定", "Ecol eV", "反应数", "截面 Å²", "95% 区间 Å²"],
          qct_rows, [.11, .28, .14, .10, .11, .26])
    paragraph("每行 N=4,096。谐振 QCT 的内部能量只是初始谐振标记，切到全势后不再是固定的实际内能；"
              "因此其总能标记不能与作用量轨道的实际总能混为一谈。固定碰撞能对照更适合隔离初态方法变化。")
    paragraph("末态未决样本的含义", subhead)
    paragraph("未决是现有 2+1 分子通道分类器在最大传播时间内未给出产品或非反应标签。"
              "三条原子间距离都很大、且都在分离的末态提示三碎片过程，不能仅因到达 1 ps 就重标为非反应。"
              "完整端点距离、径向分离速度和初态诊断保存在 unresolved_endpoint_diagnostics.json。")
    paragraph("这些诊断不等同于新增经过验证的三体解离分类器。本次保留原状态，给出截面范围；"
              "如果继续开发通道分类，应另行定义几何与能量判据并验证，避免把不物理初态产生的散射事件混入目标反应。")
    paragraph("本轮没有扩大 bmax 或重新收敛全部参数。v=0 使用 bmax=3.5 bohr，v=1 使用 4.0 bohr；"
              "先前外侧环带结果仍需与本轮初态变化分开看待。")

    flow.append(PageBreak())
    paragraph("4 / 孤立 H₂：切换后能量测试", heading)
    figure("preparation_energy_hold.png")
    audit_rows = []
    for r in means:
        audit_rows.append([f"{r['v']} / {r['nominal_total_label_ev']:.2f}",
                           AUDIT_VARIANTS.get(r["variant"], r["variant"]),
                           r["n_trajectory_means"], f"{number(r['mean'], 5)} ± {number(r['sem'], 5)}"])
    table(["v / 标记 eV", "制备方案", "独立轨迹数", "100 fs 均值 ± SEM / eV"],
          audit_rows, [.17, .32, .16, .35])
    paragraph("定量均值使用原生积分器在每个 0.02 fs 步长上的梯形积分，再对独立轨迹平均；"
              "SEM 也在轨迹之间计算。图中的 5 fs 稀疏帧只展示变化，不用于计算均值。", small)
    paragraph("碰撞实验在 200 fs 切换结束后立即发射，没有额外 hold；本页的独立初态测试才追加 100 fs 无热浴传播。"
              "因此此处均值不能作为碰撞发射瞬间的能量。", small)
    paragraph("这里的 primitive 内能估计量为每珠子的 K + Vfull - Vmin - S。"
              "它既不是守恒的环聚合物 H/P，也不是每条轨迹的物理内能，更不是振动态布居。"
              "有限样本的初末单帧波动较大，均值接近量子能级不能证明态纯度。", small)

    flow.append(PageBreak())
    paragraph("5 / 算法、验证与复现入口", heading)
    paragraph("平滑切换实现", subhead)
    paragraph("只传播孤立 BC 相对坐标，势能使用 Vλ=(1-λ)Vharm+λVfull；"
              "λ(s)=10s³-15s⁴+6s⁵。A 原子与 BC 质心不参与制备传播，"
              "碰撞能不重新调整。固定 β 与珠子数，采用 C++ 原生 BAB 分裂及精确自由环传播。")
    paragraph("显式累计 W/P=∫λdot·平均(Vfull-Vharm)dt，分别报告 Δ(H/P)-W/P 的离散残差"
              "和切换结束后无热浴传播的 H/P 漂移；驱动切换阶段的能量变化不能被误称为守恒误差。")
    paragraph("验证与一个独立的核心修复", subhead)
    paragraph("作用量采样通过谐振子、Morse 势及 BKMP2 检查；平滑切换与独立 ODE、"
              "解析谐振正则模和步长减半结果比较。测试覆盖配对相位、A/BC 质心保持、"
              "失败样本计数以及未决事件区间。原生每步平均还与 0.2 fs 密集轨迹积分交叉核对。")
    numerics = report["preparation_numerics"]
    balance = max(r["max_absolute_work_balance_residual_ev"] for r in numerics)
    drift = max(r["max_absolute_hold_energy_drift_ev"] for r in numerics
                if r["max_absolute_hold_energy_drift_ev"] is not None)
    largest_centroid = max(r["final_centroid_bond_max_bohr"] for r in numerics)
    paragraph(f"本批次全体制备的最大做功平衡残差为 {balance*1000:.4f} meV，"
              f"最大 hold 漂移为 {drift*1000:.4f} meV。尽管误差较小，"
              f"无筛选初态的最大最终质心键长仍达到 {largest_centroid:.4f} bohr，"
              "因此数值完成和能量守恒不能代替束缚态稳定性验证。")
    paragraph("核心 dRPMD 检查点的时间一致性检查改为随累计步数增长的浮点舍入误差界。"
              "这使长期重复加法得到的合法时间能恢复，同时保留整数步数及明显错误时间的拒绝。"
              "该修改不改变动力学积分方程。具体测试记录与源文件 SHA256 随结果保存。")
    paragraph("复现命令（从仓库根目录；先按 SOURCE.md 构建 BKMP2）", subhead)
    flow.append(Preformatted(
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \\\n"
        "  python benchmarks/h3/run_preparation_study.py --group all --workers 8\n"
        "PYTHONPATH=src:. python benchmarks/h3/analyze_preparation_study.py\n"
        "python benchmarks/h3/render_preparation_report.py", code))
    paragraph("原始轨迹、配置、源文件与本机二进制 SHA256、能量诊断、配对结果及分析脚本共同构成复现记录。"
              "数据包不包含第三方 BKMP2 源码/二进制或出版社 PDF；PES 通过固定来源的构建流程取得。", small)
    if code_commit:
        paragraph("计算源代码归档提交："+code_commit, small)
    paragraph("代码审阅：https://github.com/chen-yu-hao/PyRPMD/pull/37", small)
    paragraph("主要原始文献", subhead)
    for ref in (
        "Marjollet & Welsch (2020), J. Chem. Phys. 152, 194113. DOI: 10.1063/5.0004179. 原始态选择 NE-RPMD 方法及适用范围。",
        "Marjollet & Welsch (2021), Int. J. Quantum Chem. 121, e26447. DOI: 10.1002/qua.26447. H + H₂ 初态与截面对照。",
        "Marjollet, Inhester & Welsch (2022), J. Chem. Phys. 156, 044101. DOI: 10.1063/5.0076216. 态选择初态与谐振正则模定义。",
        "Ceriotti et al. (2010), J. Chem. Phys. 133, 124104. DOI: 10.1063/1.3489925. 自由环正则模传播与积分背景。",
    ):
        paragraph(ref, small)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d8e2e9"))
        canvas.line(18*mm, 13*mm, A4[0]-18*mm, 13*mm)
        canvas.setFont("PreparationCJK", 7.5)
        canvas.setFillColor(colors.HexColor("#657083"))
        canvas.drawString(18*mm, 8.5*mm, "PyRPMD · 初态制备敏感性 · 定量结果来自已完成的计算")
        canvas.drawRightString(A4[0]-18*mm, 8.5*mm, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(output), "collision_samples": sum(r["samples"] for r in new),
                      "audit_samples": sum(r["attempts"] for r in means)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"runs/state_preparation_study/analysis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"))
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    render(args.input, args.output or args.input/"pyrpmd_state_preparation_report.pdf",
           args.font, args.code_commit)


if __name__ == "__main__":
    main()
