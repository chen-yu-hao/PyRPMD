# 软件论文算例规划（低计算量文献复现）

本文为 PyRPMD 软件论文挑选**每个模块一个**代表性算例。选择原则：

1. **复现已发表论文中的图/表**，且有独立参考（解析解、DVR/HEOM 精确量子、或原文数值）。
2. **计算量小**：单个算例在一台工作站（≤ 32 核）上数分钟到数小时完成；最重的 H + H₂ 已有完成的记录，可直接复用。
3. **每个算例对应一个论文图或表的一个面板**，并附收敛（P、Δt、样本量）与不确定度。
4. 区分三类证据：**数值验证**（与解析/精确解比较）、**方法复现**（与原文同一方法同一模型）、**物理比较**（RPMD vs 精确量子）。论文中分别标注。

现有完整清单见 [reproduction checklist](manual/reproduction.md)（111 项）和 [literature validation](literature-validation.md)；本文只是其中适合软件论文的精简子集。

## 总览

| # | 模块（代码） | 论文 | 模型 | 观测量 | 独立参考 | 估计耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | PIMD 平衡统计 `path_integral.py`, `quantum_statistics.py` | Takahashi–Imada 1984；Kapil, VandeVondele, Ceriotti, JCP 144, 054111 (2016) | 1D 谐振子 + 四次方势 | ⟨E⟩、⟨x²⟩ 随 P 的收敛（primitive P⁻²，TI/SC P⁻⁴） | 解析 / DVR | 分钟 |
| A2 | 恒温器与积分器 `propagation.py`, `dynamics.py` | Ceriotti et al., JCP 133, 124104 (2010) PILE；Korol, Bou-Rabee, Miller, JCP 151, 124103 (2019) Cayley；Korol et al., JCP 152, 104102 (2020) BCOCB | 1D 谐振子/四次方势，P = 32–512 | 自相关时间 τ 随 γ；最大稳定 Δt 随 P（标准 vs Cayley） | 解析 PILE 最优 γ = 2ω_k | 分钟 |
| A3 | PIGLET/GLE `_gle_fitting.py` | Ceriotti & Manolopoulos, PRL 109, 100604 (2012) | 谐振子 + Morse（OH 参数） | 量子动能 vs P（PILE vs PIGLET） | DVR | 分钟 |
| A4 | 玻色交换 PIMD | Hirshberg, Rizzi, Parrinello, PNAS 116, 21445 (2019) | N = 2–16 个无相互作用玻色子，2D 谐振阱 | ⟨E⟩/N 随 βħω | 解析正则配分函数递推 | 分钟–1 h |
| B1 | RPMD 实时关联 `engine.py`, `dynamics.py` | Craig & Manolopoulos, JCP 121, 3368 (2004) | 1D：V = x⁴/4 与 V = x²/2 + x³/10 + x⁴/100，β = 1, 8 | Kubo 位置自相关 c̃ₓₓ(t) | DVR 精确量子 + 经典 MD | 分钟 |
| B2 | 振动光谱 TRPMD/PA-CMD `workflows/spectroscopy.py` | Witt et al., JCP 130, 194510 (2009)；Rossi, Ceriotti, Manolopoulos, JCP 140, 234116 (2014) | 1D OH Morse 振子（及 2D 弯曲模型可选） | 偶极/速度谱：RPMD 伪共振、TRPMD 展宽、PA-CMD 红移 | DVR 精确谱线位置 | 分钟–1 h |
| C1 | RPMD 速率（1D）`rate`, `vtst*.py` | Craig & Manolopoulos, JCP 122, 084106 (2005) | 对称 Eckart 势垒（原文参数） | k_RPMD(T)、κ(t)、QTST，T 跨越隧穿交叉温度 | 精确量子速率（通量-通量） | 分钟–1 h |
| C2 | RPMD 速率（3D 气相）Bennett–Chandler 全流程 | Collepardo-Guevara, Suleimanov, Manolopoulos, JCP 130, 174713 (2009)；Suleimanov, Allen, Green, CPC 184, 833 (2013) RPMDrate | H + H₂ / BKMP2，300 K | PMF、κ(t)、k_RPMD；可选 600/1000 K | 原文数值；已完成 P128 记录（+0.92%） | 复用（新温度点约 1 天） |
| C3 | 瞬子 `instanton.py`, `low_temperature.py`, `tunnelling.py` | Richardson & Althorpe, JCP 131, 214106 (2009)；JCP 134, 054109 (2011) | Eckart 势垒；1D 对称双阱 | k_inst(T) vs 精确；隧穿劈裂 Δ vs N | 精确量子 / DVR 劈裂 | 分钟 |
| D1 | 自由能：US + WHAM/MBAR `workflows/rate_umbrella_analysis.py`, `mbar` | Torrie & Valleau 1977；Shirts & Chodera, JCP 129, 124105 (2008) | 1D 双阱（量子 P = 16 和经典 P = 1） | PMF 误差 vs 窗口数/样本量 | 网格数值积分（经典）/ 环聚合物质心 PMF 的 PIMD 长跑 | 分钟 |
| D2 | MetaD / WT-MetaD / OPES `opes.py`, `workflows/metadynamics.py` | Barducci, Bussi, Parrinello, PRL 100, 020603 (2008)；Invernizzi & Parrinello, JPCL 11, 2731 (2020) | Müller–Brown 2D 势（Müller & Brown 1979） | FES RMSE vs 力评估次数；ΔF(A→B) | 网格 Boltzmann 积分 | 分钟–1 h |
| D3 | 路径采样 TIS/RETIS/FFS/committor `retis.py`, `ffs.py`, `pptis.py` | van Erp, Moroni, Bolhuis, JCP 118, 7762 (2003)；van Erp, PRL 98, 268301 (2007) | 2D 双阱或 Müller–Brown，Langevin（P = 1 与小 P） | 速率常数 k_AB；交叉概率曲线 | 长时间直接 MD 计数 + Kramers 估计 | 1 h |
| D4 | 过渡态/路径 `workflows/saddle.py`, `neb.py`, `finite_temperature_string.py` | Henkelman, Uberuaga, Jónsson, JCP 113, 9901 (2000) CI-NEB；Baker 1986 P-RFO | Müller–Brown | 三个极小 + 两个鞍点坐标/能量；MEP | 原文/解析鞍点（Müller & Brown 1979 表） | 秒 |
| E1 | 非绝热散射 FSSH / MASH / RPSH `surface_hopping.py`, `mash.py` | Tully, JCP 93, 1061 (1990)；Mannouch & Richardson, JCP 158, 104111 (2023) | Tully I–III | 透射/反射到各态的概率 vs 入射动量 k | 精确波包（split-operator，可内置 1D 网格脚本） | 分钟–1 h |
| E2 | 自旋-玻色子群布居 MASH / NRPMD `mash.py`, `nonadiabatic.py` | Mannouch & Richardson 2023；Richardson & Thoss, JCP 139, 031102 (2013) | 自旋-玻色子（Debye 或 Ohmic 浴，原文参数，离散 ~100 模） | ⟨σ_z(t)⟩ | HEOM（`analysis/heom.py`）/ 原文曲线 | 分钟–1 h |
| E3 | 非绝热速率 KC-RPMD / 金规则瞬子 `kc_rpmd.py`, `golden_rule.py` | Menzeleev, Bell, Miller, JCP 140, 064103 (2014)；Richardson, Bauer, Thoss, JCP 143, 134115 (2015) | 系统-浴电子转移模型 / 自旋-玻色子 | k vs 电子耦合 Δ（非绝热→绝热）；k vs 驱动力 ε（Marcus 反转区） | Fermi 金规则精确量子 / 原文图 | 1 h |
| F1 | PCET 速率 `pcet.py`, `workflows/pcet.py` | Hammes-Schiffer & Soudackov, JPCB 112, 14108 (2008) | 原文谐振子 PCET 模型 | k(T)、H/D KIE、振动态贡献 | pyPCET（已有 0.062% 一致） | 分钟 |
| F2 | 电化学 / 恒电势 `electrochemistry/` | Lam, Soudackov, Hammes-Schiffer, JPCL 10, 5312 (2019)（Au/TEAH⁺ 模型部分） | 原文局部模型 | k(U) Tafel 曲线、传递系数 α | 原文图 / 解析 Marcus–Hush–Chidsey | 分钟–1 h |
| G1 | 凝聚相 NQE `workflows/pimd.py`, `barostat.py` | Habershon, Markland, Manolopoulos, JCP 131, 024501 (2009) | q-TIP4P/F，32–64 分子，298 K | O–H/O–O RDF、量子动能、⟨E⟩、密度（NPT 可选） | 原文表/图；OpenMM 经典参考已有 | 小时 |
| G2 | 材料扩散/表面 `workflows/materials.py`, `electronic_friction.py` | Suleimanov, JPCC 116, 11141 (2012) H/Ni(100) | 已固定的 EAM4 静态接口；或 1D 周期余弦势 | 跳跃速率 / D(T) 的 Arrhenius 图 | 1D 周期势的 TST/精确量子；EAM4 原文值（若协议核对完成） | 1D：分钟；Ni：天（可选） |
| H1 | 势接口与性能 `potentials/`, `parallel/`, `torch_backend.py` | —（工程，不是论文复现） | H + H₂ BKMP2、LJ 晶体 | NumPy vs C++/OpenMP vs Torch 每步耗时；MPI/进程池扩展性；固定误差下的成本 | 同一结果的逐位/统计一致性 | 小时 |

> 星级建议：**正文必选** A1、B1、C1、C2、C3、D2、E1、E2、F1（覆盖 PIMD、RPMD 动力学、速率、瞬子、增强采样、非绝热、PCET）；**SI 或正文可选** 其他。全部必选项可在一台工作站上 1–2 天内完成（C2 复用已有记录）。

## 各模块详细计划

下列每项给出：目的 → 设置 → 输出图表 → 收敛与通过标准 → 现有入口/缺口。数值参数以原文为准，复现前把原文图/表数字化或抄录入 `benchmarks/paper_suite/<case>/reference.*`，并记录 DOI 与图号。

### A1. PIMD 平衡统计与高阶分解

- **目的**：验证 primitive、Takahashi–Imada（TI）、Suzuki–Chin（SC）估计量的收敛阶。
- **设置**：ħ = m = ω = 1，β = 10（深量子区）；P = 4, 8, 16, 32, 64；1D 谐振子与 V = x⁴/4。PILE-L 恒温，每个 P 8 个独立种子。
- **图**：log|⟨E⟩_P − E_exact| vs log P，三条线斜率 −2 / −4 / −4；插图为 virial 与 primitive 动能估计量的方差对比。
- **标准**：拟合斜率与理论阶在 2σ 内；P→∞ 外推值与解析/DVR 在统计误差内一致。
- **入口**：`examples/quantum_statistics/harmonic_validation.py`、`examples/path_integral/suzuki_chin.inp`；需补四次方势 DVR 参考（约 30 行 NumPy）。

### A2. 恒温器效率与积分器稳定性

- **目的**：复现 PILE 2010 的“最优摩擦使所有正则模自相关时间最短”，以及 Cayley/BCOCB 在大 P 下 Δt 稳定性不随 P 退化。
- **设置**：谐振子，P = 32；对 γ_k = λ·2ω_k 扫描 λ ∈ [0.1, 10]，测 ⟨V⟩ 与 ⟨K_virial⟩ 的积分自相关时间。另做 P = 32…512、Δt 扫描，记录 NVE 下能量漂移/发散阈值（标准 Verlet vs Cayley vs BCOCB）。
- **图**：(a) τ vs λ（对应 Ceriotti 2010 Fig.）；(b) 最大稳定 Δt vs P。
- **标准**：λ ≈ 1 处最小值；Cayley 的稳定 Δt 与 P 无关、标准自由环聚合物传播子在 P 大时随 ω_P 退化。
- **入口**：`benchmarks/torch_bcocb_validation.py`、`docs/integrators.md`。

### A3. PIGLET

- **目的**：以极少珠子数得到收敛量子动能（Ceriotti & Manolopoulos 2012 的核心结论）。
- **设置**：OH Morse 振子（或谐振子），300 K；P = 1, 2, 4, 6, 8, 16, 32；PILE vs PIGLET（用仓库自带 GLE 矩阵 `benchmarks/condensed_phase/piglet_4_750K.json` 或按目标温度重新拟合）。
- **图**：⟨K⟩ vs P，PIGLET 在 P ≈ 4–6 即达到 PILE 在 P ≈ 32 的值。
- **入口**：`examples/path_integral/gle_parameter_validation.py`。

### A4. 玻色交换（可选）

- **目的**：复现 Hirshberg et al. 2019 Fig. 2 类的 2D 谐振阱玻色子能量曲线。
- **设置**：N = 2, 4, 8, 16（视 O(N²) 实现耗时），P 按 βħω 选取；观测 ⟨E⟩/(Nħω) vs βħω（或 T/T_c）。
- **参考**：无相互作用玻色子正则配分函数递推公式，精确可算。
- **入口**：`examples/path_integral/bosonic.inp`、`benchmark_exchange.py`。

### B1. RPMD 实时关联函数（Craig–Manolopoulos 2004）

- **目的**：RPMD 的“招牌”算例：短时精确、长时与经典/精确量子的差异。
- **设置**：ħ = m = 1；两个势，β = 1 与 β = 8；P = 32（β = 8）/ 8（β = 1）；从 PIMD 平衡构型发射 NVE 轨迹，~10⁵ 条样本（1D 很便宜）。
- **图**：c̃ₓₓ(t) 四个面板：精确（DVR Kubo 变换）、RPMD、经典 MD。对应原文 Fig. 1–2。
- **标准**：t = 0 处与精确值吻合（静态量必须精确）；RPMD 曲线与原文图形一致；P 加倍不改变结果。
- **缺口**：需一个 1D DVR Kubo 关联参考脚本（约 60 行），建议放 `benchmarks/paper_suite/rpmd_correlation/`。

### B2. 振动光谱：RPMD 伪共振 / TRPMD / PA-CMD

- **目的**：展示 `spectroscopy` 工作流的三种方法差异，与 Witt 2009、Rossi 2014 的结论一致。
- **设置**：1D OH Morse（D_e、a、μ 用 Witt 2009 参数），T = 300 K 与低温 T = 100 K；P = 32；速度/偶极自相关。
- **图**：谱线：精确（DVR 0→1 跃迁）、经典、RPMD（出现内部模伪峰）、TRPMD（伪峰被抑制、峰展宽）、PA-CMD（红移）。
- **入口**：`benchmarks/trpmd/harmonic_spectrum.py`、`examples/spectroscopy/pacmd.inp`、`benchmarks/tier2_literature/spectroscopy.py`。

### C1. 1D 对称 Eckart 势垒速率（Craig–Manolopoulos 2005）

- **目的**：Bennett–Chandler 分解（QTST × κ）在可精确求解系统上的检验；展示深隧穿区 RPMD 的已知偏差。
- **设置**：原文 Eckart 参数（H 原子质量，势垒约 0.42 eV 量级，以原文为准）；T 覆盖交叉温度 T_c 上下（约 6 个温度点）；P 随 β 增加（16–128）。
- **图**：(a) k_RPMD/k_exact vs 1000/T，叠加经典 TST、Wigner、Eckart 校正；(b) 一个温度下的 κ(t) 平台。
- **参考**：精确量子速率 k(T) = (2πħQ_r)⁻¹∫P(E)e^{−βE}dE，Eckart 透射概率 P(E) 有解析式——零成本独立参考。
- **入口**：`examples/instanton/eckart.py`（势已存在），需接入 rate 工作流的 1D 输入。

### C2. H + H₂ / BKMP2 三维速率

- **目的**：软件主功能的旗舰算例；与 RPMDrate 及原文直接对比。
- **现状**：300 K、P = 128 已完成：k = 1.968 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹，比原文 +0.916%，95% 区间落在原文 ±5% 内；经典 1.5454 × 10⁻¹⁸ 与原文差 0.30%。
- **低成本补充**（建议）：再加 2–3 个较高温度（如 600、1000 K），可用较少珠子（P = 32–64），给出 Arrhenius 图与 D + H₂ 动力学同位素效应（KIE）。
- **图**：(a) PMF W(ξ)；(b) κ(t)；(c) k(T) 与原文/RPMDrate/精确量子（原文表）对比。
- **入口**：`examples/rate/bkmp2_pilot.inp`、`bkmp2_production.inp`、`benchmarks/h3/`；`pyrpmd study` 做温度/同位素扫描。
- **注意**：P256 动态收敛与 umbrella 精度门槛仍未通过，论文中照实写成“固定 P 下的复现”。

### C3. 环聚合物瞬子：速率与隧穿劈裂

- **目的**：低温区补足 RPMD 的方法；结果与 C1 同一 Eckart 势放在同一张 Arrhenius 图上。
- **设置**：(a) Eckart：T < T_c 的 5 个温度，N = 64–512 珠，检验 k_inst/k_exact（原文报告的低温误差在约 20% 内，以原文表为准）；(b) 1D 对称双阱：N 收敛的 Δ_inst vs DVR 精确劈裂。
- **图**：(a) 与 C1 合并；(b) Δ vs N 收敛曲线。
- **入口**：`examples/instanton/eckart.py`、`low_temperature.py`、`cross_checks.json`。

### D1. 伞形采样 + WHAM/MBAR

- **目的**：PMF 估计量正确性与误差条可信度。
- **设置**：1D 双阱，经典（P = 1）与量子质心 PMF（P = 16）；窗口数 8–32；MBAR、WHAM、UI 三种分析。
- **图**：PMF vs 精确（经典为解析 −kT ln ρ；量子为长时 PIMD 质心直方图）；误差 vs 样本量（应 ∝ N^{−1/2}），并检验 bootstrap 误差覆盖率（重复 50 次看 95% CI 覆盖率）。
- **入口**：`examples/rate/pmf_umbrella_mbar.inp`、`benchmarks/pmf_accuracy/`。

### D2. MetaD / WT-MetaD / OPES 在 Müller–Brown 上

- **目的**：增强采样模块的统一比较。
- **设置**：Müller–Brown（原始参数，kT 取使势垒 ≈ 10–15 kT），Langevin，两个 CV = (x, y)；每方法 5 个种子，固定力评估预算。
- **图**：(a) 参考 FES 与各方法 FES；(b) 区域内 RMSE vs 力评估次数；(c) ΔF(A→B) 时间收敛（OPES 应收敛更快、更平稳——对应 Invernizzi & Parrinello 2020 的论点，但用同一模型自行比较，不声称复现其原图）。
- **入口**：`benchmarks/fes_methods/`、`benchmarks/tier2_literature/sampling.py`（已有 OPES Müller 模型）、`examples/opes/`。可选：与 PLUMED 固定历史的逐点一致性（`plumed.py`）。

### D3. 路径采样速率

- **目的**：TIS/RETIS/FFS 给出的 k_AB 与暴力 MD 一致。
- **设置**：2D 双阱，势垒 ≈ 8 kT（暴力 MD 可在 ~10⁸ 步内统计上百次跃迁作参考）；界面 6–10 个。
- **图**：交叉概率 P(λ_i|λ_A) 累积曲线；k_AB 四种方法 + 暴力 MD 的条形图（含误差）。committor 直方图在 TS 区域呈 0.5 峰。
- **入口**：`retis.py`、`ffs.py`、`examples/sampling/committor.py`。

### D4. 过渡态搜索（SI）

- **设置**：Müller–Brown 的三个极小与两个鞍点；P-RFO 和 CI-NEB；报告与文献坐标（精确到 10⁻⁴）以及力调用次数。
- **入口**：`benchmarks/transition_state_search/`。

### E1. Tully 三个模型

- **目的**：非绝热模块的经典标准算例。
- **设置**：Tully 1990 原参数，质量 2000 a.u.；k = 2–30 a.u. 扫描 ~15 点，每点 2000 条轨迹；FSSH、MASH（及 RPSH，P = 1 时应退化为 FSSH，P > 1 作为附加曲线）。
- **图**：3 × (透射下/透射上/反射下) 概率 vs k，与精确波包对比。对应 Tully 1990 Fig. 3–5、Mannouch & Richardson 2023 的散射图。
- **参考**：1D 两态 split-operator 波包（约 80 行 NumPy，秒级）。
- **入口**：`benchmarks/paper_validation/independent_tully_dac.py`、`tully_sac_pilot.py`（已有 Liu 2024 参考 CSV）。

### E2. 自旋-玻色子布居动力学

- **目的**：凝聚相非绝热：MASH、NRPMD、Ehrenfest 对比 HEOM。
- **设置**：对称与非对称偏置两组（ε/Δ = 0、1），Debye 浴，λ、ω_c、β 采用 MASH 2023 原文参数；~100 个离散模；10⁴ 轨迹。
- **图**：⟨σ_z(t)⟩，方法 vs HEOM。
- **入口**：`examples/nonadiabatic/heom_spin_boson.py`、`src/pyrpmd/analysis/heom.py`、`DiscreteSpinBoson`；Richardson–Thoss 2013 模型见 `examples/nonadiabatic/published_nrpmd_benchmark.py`（已有原文数字化）。

### E3. 非绝热速率：KC-RPMD 与金规则瞬子

- **目的**：电子耦合从非绝热到绝热区的速率转变。
- **设置**：Menzeleev–Bell–Miller 2014 的系统-浴 ET 模型；Δ 跨 3 个数量级（~8 个点）；另以 Richardson 2015 自旋-玻色子模型做 ε 扫描（Marcus 正常区→反转区）。
- **图**：(a) log k vs log Δ（KC-RPMD、普通 RPMD、金规则）；(b) log k vs ε（金规则瞬子 vs 精确金规则 vs Marcus）。
- **入口**：`kc_rpmd.py`、`examples/instanton/golden_rule.py`、`golden_rule_reference.py`、`validate_golden_rule.py`。

### F1. PCET 速率

- **目的**：复用已有 pyPCET 0.062% 一致结果，补 H/D KIE 与温度依赖。
- **图**：k(T) Arrhenius、KIE(T)，各振动通道贡献堆叠图。
- **入口**：`examples/pcet/`、`workflows/pcet.py`。

### F2. 电化学恒电势（SI 或正文可选）

- **设置**：Au/TEAH⁺ 原文局部模型（不含双电层/距离平均，论文里写清楚）；U 扫描 ~10 点。
- **图**：log k vs U（Tafel），提取 α；与 Marcus–Hush–Chidsey 解析式比较。
- **入口**：`examples/electrochemistry/`、`benchmarks/electrochemical_string/`。

### G1. q-TIP4P/F 液态水（中等成本）

- **目的**：凝聚相 + 外部力引擎（OpenMM）接口的展示。
- **设置**：32 或 64 个水分子，298 K，P = 32（或 PIGLET P = 6，与 A3 呼应），每个 50–100 ps；经典对照已有完整 8 种子 OpenMM 参考。
- **图**：g_OO、g_OH 经典 vs PIMD（与原文图对比）；H 的量子动能、⟨V⟩。
- **标准**：仅比较平衡结构/能量；扩散系数和 IR 需要长时动力学，**本论文不声称**（现有记录中 Δt 等价门槛未通过）。
- **入口**：`benchmarks/water/`。

### G2. 表面/材料扩散（可选）

- 建议软件论文用 **1D/2D 周期余弦势上的 H 跳跃速率**：RPMD k(T) 与精确量子（周期势能带 + 通量）/TST 比较，几分钟可完成；Ni(100) EAM4 留作后续应用论文。
- **入口**：`PeriodicCosinePotential`、`workflows/site_rate.py`、`examples/surface/`。

### H1. 性能与可扩展性（软件论文必备，非复现）

- 每步墙钟时间：NumPy 参考 vs C++/OpenMP vs Torch(CPU/GPU)，P = 16–512，H + H₂ 与 LJ-256 两个体系。
- 强扩展：1–32 线程 OpenMP、进程池、MPI G × B 组；轨迹层面并行效率。
- **以固定误差为目标的成本**（如达到 κ 半宽 3% 所需 CPU·h），而非仅仅报告加速比。
- 检查点重启的逐位可复现性：中断-恢复与不中断结果一致（直接展示软件特色）。
- **入口**：`benchmarks/efficiency/`、`benchmarks/loop_native/`、`benchmarks/batched_dynamics.py`、`docs/t19-performance.md`。

## 建议的论文图表布局

| 图 | 内容 |
| --- | --- |
| Fig. 1 | 软件架构 + 单输入文件 → TS → PMF → κ → k 的流程图 |
| Fig. 2 | PIMD：A1 收敛阶 (a)，A2 恒温器/积分器 (b)，A3 PIGLET (c) |
| Fig. 3 | RPMD 动力学：B1 关联函数 (a–b)，B2 光谱 (c) |
| Fig. 4 | 速率：C2 H + H₂ 的 PMF/κ(t)/k(T) (a–c)，C1+C3 Eckart Arrhenius 图（RPMD、瞬子、精确）(d) |
| Fig. 5 | 增强采样：D2 Müller–Brown FES 与收敛 (a–b)，D3 路径采样速率 (c) |
| Fig. 6 | 非绝热：E1 Tully (a–c)，E2 自旋-玻色子 (d)，E3 KC-RPMD/金规则 (e) |
| Fig. 7 | PCET/电化学：F1、F2 |
| Fig. 8 | 性能与扩展性：H1 |
| Table 1 | 所有算例汇总：参考值、PyRPMD 值、95% CI、相对偏差、P/Δt/样本量、CPU·h |
| SI | A4、D1、D4、G1、G2，以及所有收敛扫描和失败/未收敛记录 |

## 执行与归档规范

1. 新建 `benchmarks/paper_suite/<case_id>/`，每个算例包含：`run.py` 或 `.inp`、`reference.{json,csv}`（含 DOI、图/表号、数字化不确定度）、`analyze.py`、`README.md`。
2. 统一的顶层脚本 `benchmarks/paper_suite/run_all.py --tier {quick,paper}`：`quick` 用于 CI（每个算例 < 1 min 的冒烟版本），`paper` 生成论文数据。
3. 每个结果写入 JSON：commit、种子、P、Δt、样本数、估计值、95% CI、参考值、墙钟时间、硬件。论文 Table 1 由这些 JSON 自动生成。
4. 判据**在运行前写入** `README.md`（例如“与 DVR 在 2σ 内一致”），不在看到结果后调整。
5. 未通过的检验照实保留并在 SI 报告（与仓库现有 [publication readiness](publication-readiness.md) 一致）。

## 优先级与预计工作量

| 阶段 | 算例 | 新代码 | 计算 |
| --- | --- | --- | --- |
| 第 1 周 | A1、B1、C1、C3、E1 | DVR/波包/Eckart 解析参考脚本（各 < 100 行） | 每个 < 1 h |
| 第 2 周 | A2、A3、D1、D2、D3、E2、F1 | Müller–Brown 势（若未内置）、暴力 MD 参考 | 每个 1–3 h |
| 第 3 周 | C2 补温度点/KIE、E3、H1 | 性能脚本整理 | 1–2 天 |
| 可选 | A4、B2、D4、F2、G1、G2 | — | 小时到天 |
