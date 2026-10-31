"""Assemble the actual BKMP2 calculation ledgers and digitized literature data.

No published curve is fitted to the new calculations. Figure digitization
tolerances are kept separate from binomial trajectory confidence intervals.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distribution

AREA_FACTOR = 0.2800285205390781
ROOT = Path(__file__).resolve().parents[2]


def load_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_cases(directory):
    cases = []
    for path in sorted(Path(directory).glob("**/summary.json")):
        report = json.loads(path.read_text())
        if "config" not in report or "counts" not in report:
            continue
        config = report["config"]
        ci = np.array(report["sigma_confirmed_ci95_bohr2"]) * AREA_FACTOR
        case = {"scope": path.relative_to(directory).parts[0],
                "method": config["method"], "v": config["v"],
                "total_energy_ev": config["total_energy_ev"],
                "collision_energy_ev": report["collision_energy_ev"],
                "beads": config["beads"], "beta_au": report["beta_au"],
                "dt_fs": config["dt_fs"], "bmin_bohr": config["bmin"],
                "bmax_bohr": config["bmax"], "samples": report["launched_attempts"],
                "reactive": report["counts"]["product"],
                "nonreactive": report["counts"]["nonreactive"],
                "unresolved": report["counts"]["unresolved"],
                "failed": report["counts"]["failed"],
                "sigma_angstrom2": report["sigma_angstrom2"],
                "sigma_ci95_lower_angstrom2": float(ci[0]),
                "sigma_ci95_upper_angstrom2": float(ci[1]),
                "preparation_rejections": report["preparation_rejections"],
                "preparation_draws": report["preparation_draws"],
                "extended": report["extended_trajectories"],
                "max_observed_energy_drift_ev": report["max_observed_energy_drift_ev"],
                "sigma_unknown_lower_angstrom2": report["sigma_unknown_bounds_bohr2"][0]*AREA_FACTOR,
                "sigma_unknown_upper_angstrom2": report["sigma_unknown_bounds_bohr2"][1]*AREA_FACTOR,
                "sigma_ci95_including_unknown_lower_angstrom2": report["sigma_ci95_including_unknown_bohr2"][0]*AREA_FACTOR,
                "sigma_ci95_including_unknown_upper_angstrom2": report["sigma_ci95_including_unknown_bohr2"][1]*AREA_FACTOR,
                "time_median_fs": report["flight_time_quantiles_fs"][0],
                "time_p90_fs": report["flight_time_quantiles_fs"][1],
                "time_p99_fs": report["flight_time_quantiles_fs"][2],
                "time_max_fs": report["flight_time_quantiles_fs"][3],
                "native_steps": report["native_steps"], "python_steps": report["python_steps"],
                "elapsed_seconds": report["elapsed_seconds"],
                "summary_path": str(path.relative_to(directory)),
                "sensitivity_kind": report.get("sensitivity", {}).get("kind", "")}
        if case["bmin_bohr"] > 0:
            area = np.pi * (case["bmax_bohr"]**2-case["bmin_bohr"]**2)*AREA_FACTOR
            if "annulus_upper_bound" in report:
                case["annulus_one_sided_95_upper_angstrom2"] = report["annulus_upper_bound"]["sigma_upper_angstrom2"]
            elif case["reactive"] + case["unresolved"] + case["failed"] == 0:
                case["annulus_one_sided_95_upper_angstrom2"] = float(area * (-np.expm1(np.log(.05) / case["samples"])))
        cases.append((case, report))
    return cases


def compare_literature(cases, reference):
    rows = []
    for case, _ in cases:
        if case["scope"] != "main" or case["method"] != "ne_rpmd":
            continue
        found = [r for r in reference if int(r["v"]) == case["v"] and
                 abs(float(r["requested_total_energy_ev"])-case["total_energy_ev"]) < 1e-8]
        if len(found) != 1:
            continue
        ref = found[0]
        value = float(ref["linear_interpolated_cross_section_angstrom2"])
        lower = float(ref["digitization_only_lower_angstrom2"])
        upper = float(ref["digitization_only_upper_angstrom2"])
        rows.append({**{k: case[k] for k in ("v", "total_energy_ev", "collision_energy_ev", "beads", "samples", "reactive", "sigma_angstrom2", "sigma_ci95_lower_angstrom2", "sigma_ci95_upper_angstrom2")},
                     "paper_interpolated_angstrom2": value,
                     "paper_digitization_only_lower_angstrom2": lower,
                     "paper_digitization_only_upper_angstrom2": upper,
                     "difference_angstrom2": case["sigma_angstrom2"]-value,
                     "relative_difference_percent": (case["sigma_angstrom2"]/value-1)*100 if value > .1 else None,
                     "interval_overlap_descriptive_only": not (case["sigma_ci95_upper_angstrom2"] < lower or case["sigma_ci95_lower_angstrom2"] > upper),
                     "paper_interpolation_and_MC_error_quantified": False})
    return sorted(rows, key=lambda x: (x["v"], x["total_energy_ev"]))


def paired_checks(directory, cases):
    """Compare exactly the reused IDs; never compare a subset to the full mean."""
    def ledger(case):
        path = Path(directory)/case["summary_path"]
        return {r["id"]:r for r in map(json.loads, path.with_name("trajectories.jsonl").read_text().splitlines())}
    output = []
    for case, _ in cases:
        if case["scope"] != "sensitivity" or case["sensitivity_kind"] not in ("timestep", "radial"):
            continue
        baseline = [c for c, _ in cases if c["scope"] == "main" and c["method"] == "ne_rpmd" and
                    all(c[k] == case[k] for k in ("v", "total_energy_ev", "beads", "bmin_bohr", "bmax_bohr"))]
        if len(baseline) != 1:
            continue
        first, second = ledger(baseline[0]), ledger(case)
        if not set(second).issubset(first):
            continue
        pairs = [(first[i], second[i]) for i in sorted(second)]
        initial_checks = all(a["b_bohr"] == b["b_bohr"] and a["initial_bond_bohr"] == b["initial_bond_bohr"] for a, b in pairs)
        if not initial_checks:
            raise AssertionError("Paired case did not reuse the same initial random draws")
        ratios = [b["max_observed_energy_drift_ev"]/a["max_observed_energy_drift_ev"] for a,b in pairs if a["max_observed_energy_drift_ev"] > 0]
        output.append({"kind": case["sensitivity_kind"], "v": case["v"],
                       "total_energy_ev": case["total_energy_ev"], "paired_samples": len(pairs),
                       "initial_draws_verified": initial_checks,
                       "base_reactive": sum(a["status"] == "product" for a, _ in pairs),
                       "variant_reactive": sum(b["status"] == "product" for _, b in pairs),
                       "reactivity_label_changes": sum((a["status"] == "product") != (b["status"] == "product") for a,b in pairs),
                       "status_or_channel_changes": sum((a["status"],a["channel"]) != (b["status"],b["channel"]) for a,b in pairs),
                       "median_energy_drift_ratio": float(np.median(ratios)),
                       "mean_termination_time_change_fs": float(np.mean([b["time_fs"]-a["time_fs"] for a,b in pairs]))})
    return output


def stratified_estimates(cases):
    """Add disjoint disk/annulus area estimates, with conservative 95% bounds."""
    rows = []
    for annulus, annulus_report in cases:
        if annulus["scope"] != "sensitivity" or annulus["sensitivity_kind"] != "annulus":
            continue
        bases = [(c,r) for c,r in cases if c["scope"] == "main" and c["method"] == "ne_rpmd" and
                 c["v"] == annulus["v"] and c["total_energy_ev"] == annulus["total_energy_ev"] and
                 c["beads"] == annulus["beads"] and c["bmax_bohr"] == annulus["bmin_bohr"]]
        if len(bases) != 1:
            continue
        base, base_report = bases[0]
        intervals = [(r["config"]["id_offset"], r["config"]["id_offset"]+r["config"]["samples"]) for r in (base_report,annulus_report)]
        if max(intervals[0][0],intervals[1][0]) < min(intervals[0][1],intervals[1][1]):
            raise AssertionError("Stratified reports unexpectedly share initial RNG IDs")
        lower = upper = 0.0
        for c,r in ((base,base_report),(annulus,annulus_report)):
            n, k = c["samples"], c["reactive"]
            possible = k+c["failed"]+c["unresolved"]
            area = r["area_bohr2"]*AREA_FACTOR
            lower += area*(0.0 if k == 0 else beta_distribution.ppf(.0125,k,n-k+1))
            upper += area*(1.0 if possible == n else beta_distribution.ppf(.9875,possible+1,n-possible))
        rows.append({"v":base["v"],"total_energy_ev":base["total_energy_ev"],"beads":base["beads"],
                     "disk_bmax_bohr":base["bmax_bohr"],"combined_bmax_bohr":annulus["bmax_bohr"],
                     "disk_samples":base["samples"],"annulus_samples":annulus["samples"],
                     "disk_reactive":base["reactive"],"annulus_reactive":annulus["reactive"],
                     "disk_sigma_angstrom2":base["sigma_angstrom2"],
                     "annulus_sigma_angstrom2":annulus["sigma_angstrom2"],
                     "combined_sigma_angstrom2":base["sigma_angstrom2"]+annulus["sigma_angstrom2"],
                     "conservative_ci95_lower_angstrom2":float(lower),
                     "conservative_ci95_upper_angstrom2":float(upper),
                     "ci_method":"Sum of per-stratum 97.5% Clopper-Pearson intervals; Bonferroni coverage at least95%; unknowns included",
                     "beyond_combined_bmax_tested":False})
    return sorted(rows,key=lambda r:(r["v"],r["total_energy_ev"]))


def scattering_plot(cases, digitized, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 11, "legend.fontsize": 9,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    for v, ax in enumerate(axes):
        ref = [r for r in digitized if int(r["v"]) == v]
        x = np.array([float(r["total_energy_ev"]) for r in ref])
        y = np.array([float(r["cross_section_angstrom2"]) for r in ref])
        limits = (.515, 1.155) if v == 0 else (.85, 1.16)
        keep = (x >= limits[0]-.03) & (x <= limits[1]+.03)
        x, y = x[keep], y[keep]
        ax.plot(x, y, color="#575d68", marker="o", markersize=4, markerfacecolor="white",
                linewidth=1.2, label="Paper RPMD (digitized)")
        ax.fill_between(x, np.maximum(0, y-.01), y+.01, color="#575d68", alpha=.1,
                        linewidth=0, label="Vertical digitization tolerance")
        for method, color, marker, label in [
            ("ne_rpmd", "#176BA0", "o", "This calculation: NE-RPMD, P=32"),
            ("qct", "#CD6B29", "^", "This calculation: harmonic QCT")]:
            selected = sorted([c for c, _ in cases if c["scope"] == "main" and
                               c["method"] == method and c["v"] == v], key=lambda c:c["total_energy_ev"])
            if not selected:
                continue
            xx = np.array([c["total_energy_ev"] for c in selected])
            yy = np.array([c["sigma_angstrom2"] for c in selected])
            errors = np.array([[c["sigma_ci95_lower_angstrom2"], c["sigma_ci95_upper_angstrom2"]] for c in selected]).T
            ax.errorbar(xx, yy, yerr=[yy-errors[0], errors[1]-yy], fmt=marker, color=color,
                        markersize=6, capsize=3.2, linewidth=1.4, label=label, zorder=5)
        ax.set(xlim=limits, ylim=(0, None),
               xlabel="Nominal total energy (eV)", ylabel=r"Integral cross section ($\mathrm{\AA^2}$)",
               title=rf"H + H$_2$ ($v={v}$, $j=0$)")
        ax.grid(alpha=.18)
        ax.legend(loc="upper left", frameon=False)
    fig.suptitle("BKMP2 scattering: comparison with the published curves", fontsize=14, fontweight="bold", y=.985)
    fig.get_layout_engine().set(rect=(0, .10, 1, .86))
    fig.text(.5, .018, "Bars: exact 95% binomial CIs. Grey band: vertical digitization tolerance only.\nPaper: Marjollet & Welsch, IJQC 121 e26447 (2021); original simulation errors are not available.",
             ha="center", va="bottom", fontsize=8.2, color="#444a53")
    fig.savefig(output / "scattering_comparison.png", dpi=300)
    fig.savefig(output / "scattering_comparison.svg")
    plt.close(fig)


def html_table(headers, rows):
    return "<table><thead><tr>"+"".join("<th>"+html.escape(str(h))+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"


def report_sections(cases, comparisons, h2, audit, paired=()):
    main = [c for c, _ in cases if c["scope"] == "main"]
    ne = [c for c in main if c["method"] == "ne_rpmd"]
    n = sum(c["samples"] for c in main)
    unknown = sum(c["unresolved"]+c["failed"] for c in main)
    overlap = sum(c["interval_overlap_descriptive_only"] for c in comparisons)
    table = [[c["v"], f'{c["total_energy_ev"]:.2f}', f'{c["collision_energy_ev"]:.6f}',
              f'{c["reactive"]}/{c["samples"]}', f'{c["sigma_angstrom2"]:.4f}',
              f'[{c["sigma_ci95_lower_angstrom2"]:.4f}, {c["sigma_ci95_upper_angstrom2"]:.4f}]',
              f'{c["paper_interpolated_angstrom2"]:.4f}'] for c in comparisons]
    results = html_table(["v", "名义总能量 / eV", "碰撞能 / eV", "反应 / 样本", "截面 / Å²", "本次 95% CI / Å²", "论文插值 / Å²"], table)
    levels = html_table(["v (j=0)", "BKMP2 本征值 / eV", "本征值 / cm⁻¹", "4407 cm⁻¹ 谐近似 / eV"],
                        [[l["v"], f'{l["energy_above_well_minimum_eV"]:.8f}', f'{l["energy_above_well_minimum_cm_minus1"]:.5f}',
                          f'{.546398362495114*(l["v"]+.5):.8f}'] for l in h2["levels"]])
    sections = [
        ("完成了什么", f"<p>本次在真实 BKMP2 势能面上完成 H + H₂(v=0,1; j=0) 的态选择散射计算，并与 2021 年原文 Fig. 5B / 6B 数字化数据对照。正式 NE-RPMD 与谐振 QCT 对照合计 {n:,} 条轨迹；其中未决或数值失败 {unknown} 条。所有积分步骤均采用 C++ 原生传播。</p>"
         f"<p>本次共比较 {len(comparisons)} 个 NE-RPMD 能量点，其中 {overlap} 个点的本次统计区间与论文读图区间重叠。这是描述性对照，不能当作完整复现成功率：论文原始统计误差与插值模型误差未公开。近阈值点的偏差应继续作为方法和实现问题研究。</p>"),
        ("论文与复现范围", "<p>主要对象：A. Marjollet and R. Welsch, <i>State-selective cross sections from ring polymer molecular dynamics</i>, Int. J. Quantum Chem. 121, e26447 (2021), <a href='https://doi.org/10.1002/qua.26447'>doi:10.1002/qua.26447</a>。前篇：J. Chem. Phys. 152, 194113 (2020), <a href='https://doi.org/10.1063/5.0004179'>doi:10.1063/5.0004179</a>。</p>"
         "<p>选择 2021 三原子初态约定：H₂ 仅采样振动环模式；入射 H 的自由内部环仅沿入射方向；centroid 额外激发 vω；NE-RPMD 拒绝初始 centroid 键长 &lt;1.078 bohr 的样本，并独立记录拒绝率。正式 NE-RPMD 使用固定 32 beads，谐振 QCT 对照使用 1 bead；均为 0.02 fs、无热浴，H 质量 1.00782503223 u。论文每点 40,000 条，本次每点 4,096 条。</p>"
         "<p>本次主计算采用 exact free-ring normal-mode BAB；论文称 modified velocity-Verlet，未给出足够细节来唯一恢复拆分。主计算将 10+Rg(A)+max[Rg(B),Rg(C)] bohr 用作轴向距离，再加横向碰撞参数 b；径向解释的敏感性单独记录。此基准独立于 PyRPMD 通用初态默认值。</p>"),
        ("截面与误差", results+"<p>总能量标签 E_tot=E_col+(v+1/2)ω，ω=4407 cm⁻¹=0.54639836 eV。它是初态的名义能量标签，不要求每条轨迹的全势能相空间样本能量相同。截面采用均匀 b² 采样：σ=πb_max² N_R/N；两个 H 交换通道已经相加，不额外乘二。</p>"
         "<p>主表和图是主圆盘截面：v=0 的 b_max=3.5 bohr，v=1 为 4.0 bohr；外侧环带贡献单独列出。置信区间采用双侧 Clopper–Pearson 二项分布区间。未决和失败保留在分母并报告可能截面上下界；不能当作已确认非反应。论文数值来自原图圆点的像素标定与相邻点线性插值，读图容差为 σ±0.01 Å²，横轴 ±0.003 eV (v=0) / ±0.002 eV (v=1)，这些不是统计置信区间。</p>"),
        ("H₂ 的独立量子参考", levels+"<p>表中能量相对于 BKMP2 孤立 H₂ 势阱底。数值求解 j=0 的约化径向 Schrödinger 方程，使用同一 BKMP2 渐近势、μ=918.5763237198094 原子单位。径向概率按 ∫|u(r)|²dr=1 归一化。四个网格步长 0.004、0.002、0.001、0.0005 bohr 及独立区间检查均通过；结果用 h² Richardson 外推。此数值一致性不代表势能面对实验的误差。</p>"
         "<p>平衡键长为 1.4014719755 bohr；势曲率对应谐频 4403.54998 cm⁻¹。该 BKMP2 势上的数值 v=0→1 间隔为 4161.08493 cm⁻¹。谐近似初态不等同于完整 BKMP2 的振动态本征函数，v=1 的能量差尤其值得保留在方法误差分析中。</p>"),
        ("数值检查与限度", "<p>能量漂移监控对象为守恒环聚合物 Hamiltonian H_P/P；物理 primitive estimator 为 K+V−spring，二者不可混用。P=32、有限 β 正则谐振子的平均 primitive 能量与 ω/2 的比值，在 β=200、300、400 时分别约为 1.0348、1.0005、0.9929；这不是纯基态的本征能。有限珠数偏差与额外热能同时存在，不能从珠数本身保证散射精度。</p>"
         "<p>敏感性结果单独保留其样本量、随机流和原始计数。减半步长和距离解释比较使用相同轨迹 ID，可检查逐轨迹结果变化；珠数比较使用不同的 P-specific 随机流。环带检查使用独立 IDs；零反应只能给有限环带的上界，不能证明任意更大 b 都无反应。不得把小批次收敛检查升级成出版精度的证明。</p>"
         "<p>初始时间预算 250 fs，仍未决的轨迹继续至最多 1000 fs；按唯一短键、atom–diatom 距离 ≥10 bohr、向外运动并持续 1 fs 判定。对本文范围，实际终止时间分布和所有扩展/未决数都保存在每点 summary.json 与逐轨迹 ledger 中。</p>"),
        ("可复现文件与命令", "<p>数据包包含：逐轨迹 JSONL、每点配置/汇总、PES 校验与编译来源、论文数字化数据和标定记录、H₂ 网格数据及环境说明。全部新增基准脚本在 <a href='https://github.com/chen-yu-hao/PyRPMD/pull/37'>PyRPMD PR #37</a>；数据包记录准确代码版本和 SHA256。第三方 BKMP2 源码和二进制不随数据包重新分发；构建脚本通过固定上游提交和 SHA256 校验获取。</p>"
         "<pre>PYTHONPATH=src:. python benchmarks/h3/build_pes.py\nPYTHONPATH=src:. python benchmarks/h3/pes.py --check\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \\\n  python benchmarks/h3/state_selected_reproduction.py \\\n  --v 0 --etot 0.55 0.70 0.90 1.10 --samples 4096 \\\n  --beads 32 --workers 8 --id-offset 10000 --output runs/reproduction\nPYTHONPATH=src:. python benchmarks/h3/h2_reference.py</pre>"
         "<p>β 使用 2021 Eq. 26–27 的印刷约定：β=max[(2/π)4.5/(E_col+vω), ln((ω−0.01E_tot)/(0.01E_tot))/ω]，上限 1500，能量以 Hartree 代入。Eq.27 的减号按原文执行；它不应被同时描述为严格满足标准谐振子热能的 1% 上界。</p>"),
    ]
    if audit:
        audit_rows = []
        for a in audit["cases"]:
            shift = a["energies_eV"]["switch_potential_delta"]
            expected = a["comparison"]
            audit_rows.append([a["config"]["v"], f'{a["config"]["total_energy_ev"]:.2f}', a["accepted"],
                               f'{100*a["rejection_fraction"]:.3f}%',
                               f'{shift["mean"]:+.6f} ± {shift["sem"]:.6f}',
                               f'{expected["variance_reduced_expected_primitive_bkmp2_eV"]:.6f} ± {expected["variance_reduced_expected_primitive_bkmp2_sem_eV"]:.6f}'])
        audit_table = html_table(["v", "总能量 / eV", "初态数", "拒绝率", "势能切换差 / eV (SEM)", "BC primitive 初始期望 / eV (SEM)"], audit_rows)
        sections.insert(4, ("初态切换审计", audit_table+
            "<p>初始期望采用已知的有限珠数、条件化谐振期望加上同一初态的配对势能差，以降低 primitive 估计量的噪声。直接样本均值、标准差和解析条件化公式均保留在 initial_audit.json。这里的误差为 SEM，与截面表的 95% CI 不同。</p>"
            "<p>在 v=0、总能量 0.55 eV，只有 4/8196 次初态被 centroid 筛选拒绝，但接受的环中 26.34% 至少有一个 bead 短于 1.078 bohr。短键组和其余组对总体势能切换均值的贡献分别约为 +0.01875 和 −0.00843 eV，净差为 +0.01032 eV。在 v=1、总能量 1.10 eV 的审计点，centroid 筛选本身显著改变谐振期望，再加上全势切换后平均初态低于名义谐能量约 0.0625 eV。</p>"
            "<p>这些是明确的制备效应，不能通过增加轨迹数消除。但它们本身不能证明已找到截面偏差的唯一原因，也不能将 primitive 单条样本当成真实微正则能量。此审计仅在 t=0 进行，没有直接复现论文 Fig.2 的时间曲线。</p>"))
    sensitivity = [c for c, _ in cases if c["scope"] == "sensitivity"]
    if sensitivity:
        check_rows = [[c["sensitivity_kind"], c["v"], f'{c["total_energy_ev"]:.2f}', c["beads"],
                       c["samples"], c["reactive"], f'{c["sigma_angstrom2"]:.4f}',
                       f'[{c["sigma_ci95_lower_angstrom2"]:.4f}, {c["sigma_ci95_upper_angstrom2"]:.4f}]'] for c in sensitivity]
        check_table = html_table(["检查", "v", "E / eV", "P", "N", "反应", "σ / Å²", "95% CI / Å²"], check_rows)
        index = next(i for i, s in enumerate(sections) if s[0] == "数值检查与限度")
        sections[index] = (sections[index][0], check_table+sections[index][1])
    if ne:
        index = next(i for i, s in enumerate(sections) if s[0] == "数值检查与限度")
        duration = (f"<p>正式 32-bead NE-RPMD 的各点终止时间中位数为 "
                    f"{min(c['time_median_fs'] for c in ne):.1f}–{max(c['time_median_fs'] for c in ne):.1f} fs，"
                    f"全部轨迹最晚 {max(c['time_max_fs'] for c in ne):.1f} fs 结束；"
                    f"正式样本中扩展时间预算的轨迹为 {sum(c['extended'] for c in main)} 条。"
                    "这些时间只适用于本次入射距离、能量和终止判据，不构成普遍的轨迹时长建议。</p>")
        sections[index] = (sections[index][0], sections[index][1]+duration)
    if paired:
        index = next(i for i, s in enumerate(sections) if s[0] == "数值检查与限度")
        descriptions = []
        for p in paired:
            kind = "减半步长" if p["kind"] == "timestep" else "径向距离解释"
            descriptions.append(
                f"{kind}：v={p['v']}、E={p['total_energy_ev']:.2f} eV，"
                f"{p['paired_samples']} 对相同初态中有 {p['reactivity_label_changes']} 对改变反应性，"
                f"反应数 {p['base_reactive']}→{p['variant_reactive']}"
                +(f"，能量漂移比中位数 {p['median_energy_drift_ratio']:.4f}" if p['kind'] == 'timestep' else "")
                +"。")
        sections[index] = (sections[index][0], sections[index][1]+"<p>"+" ".join(descriptions)+"</p>")
    stratified = stratified_estimates(cases)
    if stratified:
        combined_rows = [[r["v"],f'{r["total_energy_ev"]:.2f}',f'{r["disk_bmax_bohr"]:.1f}–{r["combined_bmax_bohr"]:.1f}',
                          f'{r["annulus_reactive"]}/{r["annulus_samples"]}',f'{r["annulus_sigma_angstrom2"]:.5f}',
                          f'{r["combined_sigma_angstrom2"]:.4f}',
                          f'[{r["conservative_ci95_lower_angstrom2"]:.4f}, {r["conservative_ci95_upper_angstrom2"]:.4f}]'] for r in stratified]
        combined_table=html_table(["v","E / eV","外侧环带 / bohr","环带反应 / N","环带 σ / Å²","相加后 σ / Å²","保守 95% 区间 / Å²"],combined_rows)
        index=next(i for i,s in enumerate(sections) if s[0]=="数值检查与限度")
        sections.insert(index+1,("外侧环带与分层截面",combined_table+
            "<p>圆盘与外侧环带采用独立轨迹 IDs，按各自面积分别估计后相加，未把不同权重的轨迹等权混池。保守区间把每层 97.5% Clopper–Pearson 区间相加，Bonferroni 保证至少 95% 的联合覆盖；表中结果仅覆盖相加后给定的最大 b，更远区域未测试。</p>"
            "<p>v=1、总能量 0.90 eV 在 4–5 bohr 环带观察到 1/1024 条反应，因此不能称 b_max=4 bohr 已收敛。这一正贡献使低能点相对于论文的偏高继续增加，不能解释为漏采大 b 导致的偏高。</p>"))
    return sections


def make_html(sections, figures, output):
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>PyRPMD — BKMP2 文献复现</title><style>"
             "body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#202936;line-height:1.65}h1{font-size:30px}h2{border-bottom:2px solid #176ba0;padding-bottom:6px;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{border-bottom:1px solid #d6dce4;text-align:left;padding:8px}th{background:#edf3f8}img{max-width:100%;height:auto}pre{background:#f4f6f8;padding:16px;overflow:auto;white-space:pre-wrap}a{color:#176ba0}.subtitle{color:#627084}</style>"
             "<body><h1>H + H₂：态选择 NE-RPMD / QCT-RPMD 文献复现</h1><p class='subtitle'>真实 BKMP2 · 独立量子参考 · 原始轨迹与统计误差 · 2026-10-31</p>"]
    for title, content in sections:
        parts.append("<h2>"+title+"</h2>"+content)
        image_name = "scattering_comparison.png" if title == "截面与误差" else "h2_reference.png" if title == "H₂ 的独立量子参考" else None
        if image_name and (figures / image_name).exists():
            parts.append("<img alt='"+title+"' src='data:image/png;base64,"+base64.b64encode((figures/image_name).read_bytes()).decode()+"'>")
    parts.append("</body></html>")
    output.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"runs/state_selected_reproduction")
    args = parser.parse_args()
    root = args.input
    reference = root/"literature_reference"
    cases = load_cases(root)
    digitized = load_csv(reference/"marjollet_2021_rpmd_fig5b_fig6b_digitized.csv")
    interpolated = load_csv(reference/"marjollet_2021_requested_energy_interpolation.csv")
    comparisons = compare_literature(cases, interpolated)
    paired = paired_checks(root, cases)
    stratified = stratified_estimates(cases)
    table = [dict(c) for c, _ in cases]
    fields = list(dict.fromkeys(k for c in table for k in c))
    write_csv(root/"all_case_results.csv", [{k:c.get(k) for k in fields} for c in table])
    write_csv(root/"paper_comparison.csv", comparisons)
    write_csv(root/"paired_sensitivity_results.csv", paired)
    write_csv(root/"stratified_cross_sections.csv", stratified)
    audit_path = root/"initial_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    h2 = json.loads((root/"h2_reference.json").read_text())
    analysis = {"cases": table, "paper_comparison": comparisons, "paired_sensitivities": paired,
                "stratified_cross_sections":stratified,
                "scope": "Finite-sample protocol reproduction; neither an exact quantum-scattering calculation nor a claim of full published-curve replication",
                "publication_reference_error": "Digitization tolerance available; original MC error and interpolation model error unquantified",
                "relative_error_reporting": "Percent difference is reported only when the interpolated reference exceeds 0.1 angstrom^2; otherwise null to avoid near-zero ratios",
                "formal_sample_total": sum(c["samples"] for c in table if c["scope"] == "main")}
    (root/"analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    scattering_plot(cases, digitized, figures)
    sections = report_sections(cases, comparisons, h2, audit, paired)
    make_html(sections, figures, root/"pyrpmd_bkmp2_reproduction_report.html")
    print(json.dumps({"completed_cases": len(cases), "comparison_points": len(comparisons),
                      "formal_sample_total": analysis["formal_sample_total"]}))


if __name__ == "__main__":
    main()
