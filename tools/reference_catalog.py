#!/usr/bin/env python3
"""Render the method/paper catalog from ``docs/method-references.json``.

The JSON file is the single source of truth.  This script regenerates

* ``docs/manual/references.md`` and ``docs/zh/manual/references.md``
* ``docs/manual/reproduction.md`` and ``docs/zh/manual/reproduction.md``
* ``docs/references.bib``

and validates the catalog: unique identifiers, resolvable cross references,
DOI syntax, required bilingual fields and repository paths that exist.

Usage::

    python tools/reference_catalog.py --check   # exit 1 if a generated file is stale
    python tools/reference_catalog.py --write   # regenerate after editing the JSON

Only the standard library is used, and the repository working tree (not git
history) is consulted, so the check also works in shallow CI clones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "method-references.json"

GROUPS = {
    "en": {
        "core": "Equilibrium, dynamics, rates and spectroscopy",
        "sampling": "Free energy, enhanced sampling and reaction paths",
        "nonadiabatic": "Nonadiabatic methods and matrix models",
        "materials": "Materials, electrochemistry, potentials and interfaces",
    },
    "zh": {
        "core": "平衡统计、动力学、反应速率与光谱",
        "sampling": "自由能、增强采样与反应路径",
        "nonadiabatic": "非绝热方法与矩阵模型",
        "materials": "材料、电化学、势能与接口",
    },
}

METHOD_FIELDS = (
    "id", "name", "name_zh", "code", "references", "scope", "scope_zh",
    "original_system", "original_system_zh", "reproduce_observables",
    "reproduce_observables_zh", "existing_entry", "gaps", "gaps_zh", "priority", "group",
)
REFERENCE_FIELDS = ("id", "authors", "title", "year", "venue", "doi", "url", "role")
PRIORITY_RE = re.compile(r"^P[0-3](/P[0-3])*$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
COMMAND_RE = re.compile(r"^(PYTHONPATH=\S+ )?(pyrpmd|python|mpiexec)( |$)")
# "Journal Name 123, 4567–4570" -> structured BibTeX @article fields.
ARTICLE_RE = re.compile(
    r"^(?P<journal>[A-Za-z][^,]*?)(?: (?P<volume>\d+))?, "
    r"(?P<pages>[A-Za-z]?\d+[A-Za-z]?\d*(?:[–-][A-Za-z]?\d+)?)$"
)

# ---------------------------------------------------------------------------
# Hand-written introductions.  {short}, {commit}, {date}, {methods} and
# {references} are filled from the catalog metadata.
# ---------------------------------------------------------------------------

REFERENCES_INTRO_EN = """\
# Original papers and implemented features

This inventory groups implemented **scientific methods, models and interfaces**, rather than inventing
a paper for every helper function. It audits commit
[`{short}`](https://github.com/{repo}/commit/{commit}) on **{date}**.
The **{methods} entries and {references} bibliographic records** connect implementation paths to primary sources
and explicit scope boundaries. The [reproduction checklist](reproduction.md) adds the original systems,
observables, existing entry points, missing ingredients and priorities.
Entries for features merged after the previous audit (`cb60679`, 2026-11-02), and entries whose scope those
features changed, were written or revised against this commit; the remaining scope notes carry the earlier
review forward with updated code links.

## How to cite a calculation

Cite the **original algorithms, the actual PES/electronic-structure model, and the upstream software** used,
as well as the exact PyRPMD version/commit. An H + H2/BKMP2 rate calculation needs the Bennett–Chandler
foundation, RPMD rate theory and bimolecular normalization, the BKMP2 surface and the software actually used.
A software paper or a review does not replace the originating algorithm paper.

Cite PyRPMD itself as software, using the metadata in
[CITATION.cff](https://github.com/{repo}/blob/main/CITATION.cff) (GitHub's **Cite this repository** button
produces APA and BibTeX from it). Add the version reported by `pyrpmd --version`, which each run also
stores in `manifest.json`, and, for an unreleased checkout, the commit hash from `git rev-parse HEAD`.
PyRPMD has no journal article yet, so do not substitute RPMDrate, Caracal or a method paper for the
software citation.

Sources are deliberately distinguished as original methods, later corrections, models/data, software,
formula references or background. **A citation is not a claim of full implementation or successful
reproduction.** Package-specific compositions cite their components; an adapter does not implement all
of the upstream method; generic controls and engineering utilities have no invented scientific origin.
Respect the source version, erratum and implementation boundary recorded below.

### Workflow checklist

Collect the citation keys for a run from the entries below. Common starting sets:

| Calculation | Catalog entries to cite |
| --- | --- |
| Bimolecular RPMD rate (`rate`) | [rpm-rate](#rpm-rate), [integrators](#integrators), [samp-umbrella](#samp-umbrella), [samp-ui](#samp-ui) or [samp-mbar](#samp-mbar), plus the PES |
| Direct collisions (`drpmd`) | [direct-rpmd](#direct-rpmd), [integrators](#integrators), plus [state-rpmd](#state-rpmd) or [qct](#qct) for selected initial states |
| Free-energy surfaces (`fes`) | The chosen method: [samp-umbrella](#samp-umbrella), [samp-standard-meta](#samp-standard-meta), [samp-wt-meta](#samp-wt-meta), [samp-opes](#samp-opes), [samp-ves](#samp-ves), [samp-abf](#samp-abf) or [samp-steered](#samp-steered) |
| Equilibrium PIMD (`pimd`) | [pimd](#pimd), the thermostat ([pile](#pile), [gle](#gle), [piglet](#piglet), [nhc](#nhc)) and, for NPT, the barostat ([bzp](#bzp), [mtk](#mtk), [scr](#scr)) |
| Replica exchange and tempering | [samp-reus](#samp-reus) or [samp-tempering](#samp-tempering) |
| Path sampling (`path_sampling`) | [samp-tis](#samp-tis), [samp-retis](#samp-retis), [samp-infretis](#samp-infretis), [samp-pptis](#samp-pptis) or [samp-ffs](#samp-ffs) |
| Nonadiabatic populations (`nonadiabatic`) | The selected method entry, such as [na-nrpmd-focused](#na-nrpmd-focused), [na-carp-mash](#na-carp-mash), [na-ba-rp-mash](#na-ba-rp-mash), [na-ms-mash](#na-ms-mash), [na-iesh](#na-iesh) or [na-rpsh-ca](#na-rpsh-ca), plus the model |
| Electrochemical PCET (`pcet`) | [pcet](#pcet), plus the electrode and environment model |
| External force engines and ML potentials | The adapter entry ([ase](#ase), [openmm](#openmm), [lammps](#lammps), [ipi-socket](#ipi-socket), [ml-adapters](#ml-adapters)) and the model paper and checkpoint |

Machine-readable records and BibTeX are available as repository files
[method-references.json](../method-references.json) and
[references.bib](../references.bib). Both pages and the BibTeX file are generated from the JSON by
`python tools/reference_catalog.py --write`.
"""

REFERENCES_INTRO_ZH = """\
# 原始论文与实现功能对照

本页按**已实现的方法/模型/接口**归并功能，而不是为每个辅助函数硬配论文。核查对象是
[`{short}`](https://github.com/{repo}/commit/{commit})，核查日期为 **{date}**。
下列 **{methods} 个条目、{references} 条文献记录**将源码、原始方法及实现范围联系起来；
[复现清单](reproduction.md)另列原文体系、应算物理量、现有入口、缺失条件和优先级。
上次核查（`cb60679`，2026-11-02）之后合入的功能，以及范围因这些功能而改变的条目，均按本提交新写或修订；
其余条目的范围说明沿用上次核查结论，仅更新源码链接。

## 如何引用

论文/报告应同时引用使用的**原始算法、具体势能面或电子结构模型、依赖软件**，并记录 PyRPMD 的版本与提交。
例如 H + H₂/BKMP2 的速率研究需同时引用 Bennett–Chandler、RPMD 速率、双分子归一化、BKMP2 势及实际使用的软件。
仅引用 RPMDrate、Caracal 或一篇综述不能代替原始算法出处。

PyRPMD 本身按软件引用，元数据见
[CITATION.cff](https://github.com/{repo}/blob/main/CITATION.cff)（GitHub 的 **Cite this repository**
按钮可据此生成 APA 与 BibTeX）。同时写明 `pyrpmd --version` 给出的版本（每次运行的 `manifest.json`
也会记录）；未发布的检出还需注明 `git rev-parse HEAD` 给出的提交哈希。
PyRPMD 目前没有期刊论文，不要用 RPMDrate、Caracal 或方法论文代替软件引用。

本目录有意保留不同出处的身份：原始方法、后续修正、模型/数据、上游软件、公式或背景文献。
**引用不等于完整实现或已完成复现**。组合流程分别引用各组成方法；外部接口只表明能调用该软件；
通用解析控制模型、输入/检查点/并行/报告等工程功能没有唯一科学原始论文时，直接说明这一点。
原始论文的具体版本、勘误及实现差别以各条目的范围说明为准。

### 按工作流选择条目

从下列条目收集一次计算需要引用的文献。常见起点：

| 计算 | 应引用的条目 |
| --- | --- |
| 双分子 RPMD 速率（`rate`） | [rpm-rate](#rpm-rate)、[integrators](#integrators)、[samp-umbrella](#samp-umbrella)、[samp-ui](#samp-ui) 或 [samp-mbar](#samp-mbar)，以及势能面 |
| 直接碰撞（`drpmd`） | [direct-rpmd](#direct-rpmd)、[integrators](#integrators)；选态初始条件另加 [state-rpmd](#state-rpmd) 或 [qct](#qct) |
| 自由能面（`fes`） | 所用方法：[samp-umbrella](#samp-umbrella)、[samp-standard-meta](#samp-standard-meta)、[samp-wt-meta](#samp-wt-meta)、[samp-opes](#samp-opes)、[samp-ves](#samp-ves)、[samp-abf](#samp-abf) 或 [samp-steered](#samp-steered) |
| 平衡 PIMD（`pimd`） | [pimd](#pimd)、所用热浴（[pile](#pile)、[gle](#gle)、[piglet](#piglet)、[nhc](#nhc)）；NPT 另加压浴（[bzp](#bzp)、[mtk](#mtk)、[scr](#scr)） |
| 副本交换与回火 | [samp-reus](#samp-reus) 或 [samp-tempering](#samp-tempering) |
| 路径采样（`path_sampling`） | [samp-tis](#samp-tis)、[samp-retis](#samp-retis)、[samp-infretis](#samp-infretis)、[samp-pptis](#samp-pptis) 或 [samp-ffs](#samp-ffs) |
| 非绝热布居（`nonadiabatic`） | 所选方法条目，如 [na-nrpmd-focused](#na-nrpmd-focused)、[na-carp-mash](#na-carp-mash)、[na-ba-rp-mash](#na-ba-rp-mash)、[na-ms-mash](#na-ms-mash)、[na-iesh](#na-iesh) 或 [na-rpsh-ca](#na-rpsh-ca)，以及所用模型 |
| 电化学 PCET（`pcet`） | [pcet](#pcet)，以及电极与环境模型 |
| 外部力引擎与机器学习势 | 接口条目（[ase](#ase)、[openmm](#openmm)、[lammps](#lammps)、[ipi-socket](#ipi-socket)、[ml-adapters](#ml-adapters)）以及模型论文和所用 checkpoint |

完整结构化目录和 BibTeX 位于仓库的
[method-references.json](../../method-references.json) 与
[references.bib](../../references.bib)。两种语言的本页及 BibTeX 均由
`python tools/reference_catalog.py --write` 从 JSON 生成。
"""

REPRODUCTION_INTRO_EN = """\
# Paper systems and reproduction checklist

**First finish the H + H2/BKMP2 convergence record and, in parallel, the Richardson–Thoss 2013 two-state
model; then study q-TIP4P/F water equilibrium, diffusion and spectra.** These campaigns test rates,
nonadiabatic dynamics and condensed-phase nuclear quantum effects. All {methods} inventory entries
have a task below; their primary papers and code paths are in the [source catalog](references.md).

## Task types and priorities

An original-paper system is the model actually used in that paper. An existing entry point may instead
be a substitute regression or a short workflow demonstration. A **paper reproduction** must match the PES,
parameters, initial distribution, dynamics, estimator and processing; a **substitute regression** tests
implementation; a **new application** needs a separately justified model/reference. Label these separately.
This is a documentation/source audit and calculation plan, not a new production-simulation result.
Existing evidence is recorded in the frozen
[literature-validation inventory](https://github.com/{repo}/blob/{commit}/docs/literature-validation.md) and
[campaign registry](https://github.com/{repo}/blob/{commit}/benchmarks/literature/campaign.json).

| Priority | Meaning | Deliverable |
| --- | --- | --- |
| P0 | Analytic/numerical controls and interface checks | Reference values, errors and dt/grid/bead checks |
| P1 | Recommended early paper-model or central-method campaign | Target figure/table, independent-seed intervals and convergence record |
| P2 | Needs data, substantial sampling or method completion | Resolve the stated gaps and verify intermediate quantities first |
| P3 | Complex composition or substantially incomplete source workflow | Complete model/sampler/estimator support before registering production |

These priorities are recommendations, not paper claims or runtime estimates. Mixed labels such as P0/P2
mean a control experiment followed by the more demanding physical-system campaign.

## Eight campaigns to start with

| Order | System and methods | Compute | Gate |
| --- | --- | --- | --- |
| 1 | Harmonic/quartic controls; PIMD, integrators, baths, estimators | Position variance, energy, heat capacity, momentum distribution; P/dt/autocorrelation | Agreement with analytic or independent references |
| 2 | H + H2 / BKMP2; RPMD rate | PMF, k_QTST, kappa(t), k(T), H/D KIE | Complete parent-time, independent-umbrella precision and dynamic P256 checks |
| 3 | Richardson–Thoss 2013 coupled harmonic model | Fig.1(c) position/electronic correlations, average sign and ESS | lowest_adiabatic splitting; bead, dt and sample convergence |
| 4 | Tully I–III; FSSH/MASH/RPSH-CA/BA/DM | Four scattering channels, population, coherence, purity and active-state fractions | Identical physical incoming distribution and correct asymptotic channels |
| 5 | q-TIP4P/F water; PIMD, RPMD, PA-CMD/TRPMD | RDF, energy, D from MSD/VACF, dipole correlation and IR | Separate classical/quantum and equilibration/real-time dynamics; source Fig.5 is PA-CMD |
| 6 | OGRe double-well/Ackley and OPES Müller models | FES, barriers, RMSE, coverage, ESS and total force evaluations | Include pilot cost; separate OPES variants and compare fixed history with PLUMED |
| 7 | Eckart and bound symmetric double wells; instantons | k(T), path, action, fluctuation determinant and splitting | Bead/Hessian convergence; molecular rotation/translation treatment first |
| 8 | 2008 PCET model then 2019 Au/TEAH+ | k(U,T), forward/reverse rates, H/D KIE and vibronic contributions | Validate local model first; add paper distance averaging/double-layer terms |

Sources: [RPMD rates](references.md#rpm-rate), [thermal NRPMD](references.md#na-thermal-nrpmd),
[Tully](references.md#na-tully-models), [water](references.md#water09), [sampling](references.md#samp-adaptive),
[instantons](references.md#rpi), [PCET](references.md#pcet).

## Outstanding conditions in existing evidence

H + H2 has fixed-P=128 agreement within the paper precision in the four-pair extension. The supplementary
parent-half equivalence check still fails; dynamic P=256 production is incomplete; independent umbrella
precision remains a separate condition. A matching final rate alone does not finish the convergence study.
The eight-seed classical OpenMM q-TIP4P/F reference is complete, but preparation-energy checks, the
short-lag MSD screen and timestep-equivalence intervals still leave issues. Quantum and PyRPMD production
transport are unrun. H + CH4, OH + H2, F + H2 and Ni-surface campaigns have distinct readiness states;
consult the frozen inventory rather than labeling them all reproduced.

## Required record for every calculation

1. Paper DOI/version, target figure/table and errata; reference numbers, digitization uncertainty or independent reference method.
2. PES/model bytes and SHA256, software commit/build/device, units, masses, electronic/nuclear states, boundary and symmetry conventions.
3. Temperature, beads, dt, initialization/equilibration, independent seeds, windows/CVs, parent/child durations and output interval.
4. The same estimator and normalization as the paper. An arbitrary FES additive constant may be removed; a fitted scale cannot conceal disagreement.
5. Independent-chain or correlated-block uncertainty, ESS, unresolved trajectories and average sign. Shared walkers/exchange slots are correlated.
6. Predetermined refinement ranges and equivalence margins. Separate physical-model, statistical and discretization errors; no universal 5% pass rule applies.

Archive failed/unresolved runs and original inputs. Choose gates for the actual observable before looking
at agreement with the paper, rather than tuning a threshold to obtain a pass.

## Existing starting commands

Install the package and run from the checkout root. These budgets are starting points, not convergence
guarantees. External PES/PLUMED/electronic-structure dependencies follow their guides. Entry points were
checked in this audit; new production simulations were not run.

```bash
python examples/nonadiabatic/published_nrpmd_benchmark.py --plan
python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre
python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre
python benchmarks/tier2_literature/sampling.py reference --output runs/tier2-sampling
python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling
python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example
```

For the paper-specific NRPMD production run retain `--potential-split lowest_adiabatic` and the supplied
paper digitization. Continue the documented unfinished H + H2/water conditions instead of replacing
their registered campaigns with a short default input.

## Complete task inventory

Only verified original-system information is recorded. Where the accessible source did not recover the
specific UI biological model, a complete constrained-dynamics application protocol or the six TREQ
reaction identities, the gap is stated explicitly; no guessed alanine/water/NH3 substitute is labeled original.

"""

REPRODUCTION_INTRO_ZH = """\
# 论文体系与复现计算清单

**建议先完成 H + H₂/BKMP2 的收敛链，并行做 NRPMD 2013 的两态谐振模型；再开展 q-TIP4P/F 水的平衡、扩散和光谱。**
这三条主线能分别验证速率、电子非绝热动力学和凝聚相核量子效应。下方列出全部 {methods} 个条目的任务。
来源及源码映射见[原始论文目录](references.md)。

## 任务性质与优先级

“原论文体系”指所引原文实际计算的模型；“现有入口”可能只是替代回归或短流程。
**原论文复现**必须匹配势文件、参数、初态、动力学、估计量和后处理。
**替代模型回归**用于检查实现；**新体系应用**需要独立建立模型和参考。三者在记录中分别标识。
这里是计算计划和源码审计，未执行新的物理生产模拟。已有结果以固定提交下的
[literature-validation](https://github.com/{repo}/blob/{commit}/docs/literature-validation.md)
和 [campaign.json](https://github.com/{repo}/blob/{commit}/benchmarks/literature/campaign.json) 为准。

| 优先级 | 含义 | 建议交付物 |
| --- | --- | --- |
| P0 | 解析/数值控制与接口校验 | 参考值、误差、时间步/网格/珠数检查 |
| P1 | 优先开展的论文模型或核心方法计算 | 原文目标图/表、独立种子区间、参数收敛记录 |
| P2 | 需要模型数据、较大采样或方法补全 | 先交付缺口清单和可验证的中间量，再计算目标曲线 |
| P3 | 复杂组合或目前明显不完整的原文流程 | 模型/采样/估计器补全后单独注册研究计划 |

优先级是本次建议，不是原论文给出的排序，也不是耗时保证。P0/P2 等表示先控制、后物理体系的两阶段任务。

## 推荐首先完成的八组计算

| 顺序 | 体系与方法 | 应计算的核心结果 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| 1 | 谐振子、四次非谐振子；PIMD、积分器、温控、估计器 | 位置方差、能量、热容、动量分布；P、dt、相关时间 | 解析/独立参考一致；热浴采样时间不当实时间 |
| 2 | H + H₂ / BKMP2；RPMD rate | PMF、k_QTST、κ(t)、k(T)、H/D KIE | 父链时间、独立伞窗精度、P256 动态收敛补齐 |
| 3 | Richardson–Thoss 2013 两态谐振模型 | Fig.1(c) 位置相关及电子相关；平均符号、有效样本 | 使用 lowest_adiabatic 分裂；beads/dt/样本收敛 |
| 4 | Tully I–III；FSSH/MASH/RPSH-CA/BA/DM | 反射/透射四通道、布居、相干、纯度、活动态比例 | 相同物理入射分布，正确宽度及终态定义 |
| 5 | q-TIP4P/F 水；PIMD、RPMD、PA-CMD/TRPMD | RDF、能量、MSD/VACF 的 D；偶极相关与 IR | 分开古典/量子与平衡/实时间生产；RPMD 用 NVE，PA-CMD/TRPMD 按内模温控定义；原 Fig.5 用 PA-CMD |
| 6 | OGRe 双阱/Ackley 与 OPES Müller 模型 | FES、势垒、RMSE、覆盖率、ESS、总力调用数 | 包含 pilot 成本；OPES 变体分别核验，固定历史对照 PLUMED |
| 7 | Eckart、束缚对称双阱；RPI、劈裂 | k(T)、路径、作用量、涨落行列式、Δ | 收敛珠数和 Hessian；分子体系先补平移/转动处理 |
| 8 | 2008 PCET 参数模型 → 2019 Au/TEAH⁺ | k(U,T)、正反速率、H/D KIE、振动态贡献 | 先核验局域模型；原文距离平均/双电层需另补 |

这些主线的出处分别见 [RPMD](references.md#rpm-rate)、[NRPMD](references.md#na-thermal-nrpmd)、
[Tully](references.md#na-tully-models)、[水模型](references.md#water09)、[采样](references.md#samp-adaptive)、
[瞬子](references.md#rpi)、[PCET](references.md#pcet)。

## 已有证据中仍待完成的环节

H + H₂ 的固定 P=128 四对扩展记录已在论文精度内一致，但补充父链半时长等效性未通过，
P=256 动态 campaign 尚未完成，独立伞窗精度也仍是单独条件。不要只凭最终速率接近论文就结束研究。
q-TIP4P/F 古典 OpenMM 参考已有完整八种子记录，但准备阶段能量检查、短时 MSD 窗口及步长等效性仍有问题，
量子与 PyRPMD 生产输运尚未运行。H + CH₄、OH + H₂、F + H₂、Ni 表面的状态各不相同，见上述原始记录，不能统一标“已复现”。

## 每一项都要保存什么

1. 原文 DOI/版本、目标图表和是否使用勘误；原文数字、数字化误差或独立参考方法。
2. PES/模型文件与 SHA256、软件提交、编译/设备信息、单位、质量、电子/核状态、边界与对称性因子。
3. 温度、珠数、dt、初态/平衡方法、独立随机种子、窗口/CV、父链与子轨迹长度、输出频率。
4. 与论文相同的估计量与归一化；PMF 只可移除任意加性常数，不能用拟合比例掩盖差异。
5. 独立链或相关块误差、有效样本、未决轨迹、平均符号；共享 walkers 和交换槽位不能当独立数据。
6. 预先确定的收敛范围与等效性容限；分别报告物理模型差异、统计误差和离散误差。没有统一的“误差小于5%即可”规则。

固定原始输入和全部失败/未决记录；通过条件由具体物理量决定，不能为贴合论文曲线而事后调整。

## 可直接定位的起步命令

以下是已有入口，默认预算是起点，不是收敛保证；先完成安装并在仓库根目录执行。
外部势、PLUMED、电子结构模型等依赖仍按各指南安装。命令本次仅核对入口，没有运行生产模拟。

```bash
# Paper-specific NRPMD setup; inspect its plan before production.
python examples/nonadiabatic/published_nrpmd_benchmark.py --plan

# Adaptive umbrella on a published analytic test model.
python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre
python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre

# Published OPES model and independent reference workflow.
python benchmarks/tier2_literature/sampling.py reference --output runs/tier2-sampling
python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling

# Fixed-full-ring committor: a small demonstration budget.
python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example
```

NRPMD 正式计算应按脚本说明保留 `--potential-split lowest_adiabatic` 并使用原文数字化数据。
H + H₂ 和水已有 campaign，不要绕过原记录另以默认短输入声称完成；从其未完成条件继续。

## 完整逐项清单

下列原体系描述只使用已核实信息。UI 原文具体生物模型、部分 constrained-dynamics 应用协议与 TREQ 六个反应的完整清单等
尚未从可得全文/补充信息恢复的内容均在“缺口”中保留，不以猜测的 alanine、水或 NH₃ 代替。

"""


class CatalogError(ValueError):
    """The catalog JSON is inconsistent."""


def load(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _path_of(item: str) -> str:
    return item.split("#", 1)[0]


def _entries(method: dict) -> list[str]:
    value = method["existing_entry"]
    return list(value) if isinstance(value, list) else value.split("; ")


def validate(cat: dict, root: Path = ROOT) -> list[str]:
    """Return a list of problems; an empty list means the catalog is consistent."""
    problems: list[str] = []
    refs = cat["references"]
    methods = cat["methods"]
    ref_ids = [r["id"] for r in refs]
    method_ids = [m["id"] for m in methods]
    for kind, ids in (("reference", ref_ids), ("method", method_ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            problems.append(f"duplicate {kind} ids: {', '.join(dupes)}")
    clash = sorted(set(ref_ids) & set(method_ids))
    if clash:
        problems.append(f"ids used for both a method and a reference: {', '.join(clash)}")
    if not re.fullmatch(r"[0-9a-f]{40}", cat.get("audited_commit", "")):
        problems.append("audited_commit must be a full 40-character commit hash")

    known = set(ref_ids)
    cited: set[str] = set()
    for r in refs:
        missing = [f for f in REFERENCE_FIELDS if f not in r]
        if missing:
            problems.append(f"reference {r.get('id')}: missing {missing}")
            continue
        doi = r["doi"]
        if doi is not None and not DOI_RE.match(doi):
            problems.append(f"reference {r['id']}: malformed DOI {doi!r}")
        if doi is None and not r["url"]:
            problems.append(f"reference {r['id']}: needs a DOI or a URL")
        if not isinstance(r["year"], int):
            problems.append(f"reference {r['id']}: year must be an integer")

    seen_groups: list[str] = []
    for m in methods:
        missing = [f for f in METHOD_FIELDS if f not in m]
        if missing:
            problems.append(f"method {m.get('id')}: missing {missing}")
            continue
        if m["group"] not in GROUPS["en"]:
            problems.append(f"method {m['id']}: unknown group {m['group']!r}")
        if not seen_groups or seen_groups[-1] != m["group"]:
            if m["group"] in seen_groups:
                problems.append(f"method {m['id']}: group {m['group']!r} is not contiguous")
            seen_groups.append(m["group"])
        if not PRIORITY_RE.match(m["priority"]):
            problems.append(f"method {m['id']}: unexpected priority {m['priority']!r}")
        for r in m["references"]:
            cited.add(r)
            if r not in known:
                problems.append(f"method {m['id']}: unknown reference {r!r}")
        for path in m["code"]:
            if not (root / path).exists():
                problems.append(f"method {m['id']}: code path {path} does not exist")
        for item in _entries(m):
            path = _path_of(item)
            if "/" in path and not COMMAND_RE.match(item) and " " not in path:
                if not (root / path).exists():
                    problems.append(f"method {m['id']}: entry point {path} does not exist")
    return problems


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _text(method: dict, key: str, lang: str) -> str:
    value = method[key + "_zh"] if lang == "zh" else method[key]
    return "; ".join(value) if isinstance(value, list) else value


def _repo_link(cat: dict, path: str, root: Path, label: str | None = None) -> str:
    kind = "tree" if (root / _path_of(path)).is_dir() else "blob"
    return f"[{label or path}](https://github.com/{cat['repository']}/{kind}/{cat['audited_commit']}/{path})"


def _render_entries(cat: dict, method: dict, root: Path) -> str:
    out = []
    for item in _entries(method):
        if COMMAND_RE.match(item):
            out.append(f"`{item}`")
        elif (root / _path_of(item)).exists() and " " not in item:
            out.append(_repo_link(cat, item, root))
        else:
            out.append(item)
    return "; ".join(out)


def _intro_values(cat: dict) -> dict:
    return {
        "repo": cat["repository"],
        "commit": cat["audited_commit"],
        "short": cat["audited_commit"][:7],
        "date": cat["audit_date"],
        "methods": len(cat["methods"]),
        "references": len(cat["references"]),
    }


def _bibliography_line(r: dict) -> str:
    line = f"{r['authors']}. **{r['title']}**. *{r['venue']}* ({r['year']})."
    if r["doi"]:
        line += f" DOI: [{r['doi']}](https://doi.org/{r['doi']})."
    if r["url"] and r["url"] != f"https://doi.org/{r['doi']}":
        line += f" [Source / author version]({r['url']})."
    return line


def render_references(cat: dict, lang: str, root: Path = ROOT) -> str:
    zh = lang == "zh"
    lines = [(REFERENCES_INTRO_ZH if zh else REFERENCES_INTRO_EN).format(**_intro_values(cat))]
    group = None
    for m in cat["methods"]:
        if m["group"] != group:
            group = m["group"]
            lines += [f"## {GROUPS[lang][group]}", ""]
        sources = ", ".join(f"[{r}](#{r})" for r in m["references"]) or "—"
        lines += [f"### {m['id']}", "", f"**{_text(m, 'name', lang)}**", ""]
        lines += [("出处：" if zh else "Sources: ") + sources, ""]
        lines += [_text(m, "scope", lang), ""]
        code = ", ".join(_repo_link(cat, c, root) for c in m["code"])
        lines += [("源码：" if zh else "Code: ") + code, ""]
        lines += [("[该项复现任务]" if zh else "[Reproduction task]") + f"(reproduction.md#task-{m['id']}).", ""]
    if zh:
        lines += ["## 完整参考文献", "", "下列“用途”描述保留英文以精确区分出处身份；论文题名使用出版方原题名。", ""]
    else:
        lines += ["## Bibliography", "",
                  "The role notes identify the relationship to the implementation, including background-only sources.", ""]
    for r in cat["references"]:
        role = r["role"] if r["role"].endswith(".") else r["role"] + "."
        lines += [f"### {r['id']}", "", _bibliography_line(r), "",
                  ("用途 / Role: " if zh else "Role: ") + role, ""]
    return "\n".join(lines)


def render_reproduction(cat: dict, lang: str, root: Path = ROOT) -> str:
    zh = lang == "zh"
    lines = [(REPRODUCTION_INTRO_ZH if zh else REPRODUCTION_INTRO_EN).format(**_intro_values(cat))]
    group = None
    for m in cat["methods"]:
        if m["group"] != group:
            group = m["group"]
            lines += [f"### {GROUPS[lang][group]}" + ("复现任务" if zh else " reproduction tasks"), ""]
        papers = ", ".join(f"[{r}](references.md#{r})" for r in m["references"]) or "—"
        lines += [f"#### task-{m['id']}", "", f"**{_text(m, 'name', lang)} — {m['priority']}**", ""]
        lines += [("论文：" if zh else "Papers: ") + papers + ".", ""]
        lines += [("**原文体系。** " if zh else "**Original system.** ") + _text(m, "original_system", lang), ""]
        lines += [("**应该算什么。** " if zh else "**Compute.** ") + _text(m, "reproduce_observables", lang), ""]
        lines += [("**现有入口。** " if zh else "**Existing entry.** ") + _render_entries(cat, m, root), ""]
        lines += [("**缺口与边界。** " if zh else "**Gaps and boundaries.** ") + _text(m, "gaps", lang), ""]
        lines += [("[实现与范围]" if zh else "[Implementation and scope]") + f"(references.md#{m['id']}).", ""]
    return "\n".join(lines)


def _bib_escape(text: str) -> str:
    return re.sub(r"(?<!\\)([_&%#])", r"\\\1", text)


def _bib_authors(authors: str) -> str:
    authors = authors.replace(" et al.", " and others")
    if ";" not in authors:
        authors = re.sub(r",? and ", "; ", authors)
    parts = [p.strip() for p in re.split(r";|, (?=[A-Z][.-])", authors) if p.strip()]
    return " and ".join(parts)


def render_bibtex(cat: dict) -> str:
    lines = [
        f"% PyRPMD primary-source catalog; audited commit {cat['audited_commit']}",
        "% UTF-8. Generated from method-references.json by tools/reference_catalog.py;",
        "% roles and implementation limits are documented in manual/references.md.",
        "",
    ]
    for r in cat["references"]:
        match = ARTICLE_RE.match(r["venue"])
        kind = "article" if match else "misc"
        lines.append(f"@{kind}{{{r['id']},")
        lines.append(f"  author = {{{_bib_escape(_bib_authors(r['authors']))}}},")
        lines.append(f"  title = {{{{{_bib_escape(r['title'])}}}}},")
        if match:
            lines.append(f"  journal = {{{_bib_escape(match['journal'])}}},")
            if match["volume"]:
                lines.append(f"  volume = {{{match['volume']}}},")
            lines.append(f"  pages = {{{match['pages'].replace('–', '--')}}},")
        lines.append(f"  year = {{{r['year']}}},")
        if not match:
            lines.append(f"  howpublished = {{{_bib_escape(r['venue'])}}},")
        if r["doi"]:
            lines.append(f"  doi = {{{r['doi']}}},")
        if r["url"]:
            lines.append(f"  url = {{{r['url']}}},")
        lines.append(f"  note = {{{_bib_escape(r['role'])}}},")
        lines += ["}", ""]
    return "\n".join(lines)


def outputs(cat: dict, root: Path = ROOT) -> dict[Path, str]:
    return {
        root / "docs/manual/references.md": render_references(cat, "en", root),
        root / "docs/zh/manual/references.md": render_references(cat, "zh", root),
        root / "docs/manual/reproduction.md": render_reproduction(cat, "en", root),
        root / "docs/zh/manual/reproduction.md": render_reproduction(cat, "zh", root),
        root / "docs/references.bib": render_bibtex(cat),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    mode.add_argument("--write", action="store_true", help="regenerate the files")
    args = parser.parse_args(argv)

    cat = load()
    problems = validate(cat)
    if problems:
        print("Catalog problems:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    stale = []
    for path, text in outputs(cat).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.write:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
        else:
            stale.append(path.relative_to(ROOT))
    if stale:
        print("Stale generated files (run `python tools/reference_catalog.py --write`):",
              *stale, sep="\n  ", file=sys.stderr)
        return 1
    print(f"catalog OK: {len(cat['methods'])} methods, {len(cat['references'])} references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
