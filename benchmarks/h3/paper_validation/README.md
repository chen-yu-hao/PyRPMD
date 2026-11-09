# H + H₂ / BKMP2 论文验证计划：PMF 方法对比与多温度速率

本目录是软件论文中 H + H₂ 算例的**预先登记计划和输入**，不是结果。所有输入由
[`generate.py`](generate.py) 从一张参数表生成，改参数请改生成器后重新生成。
每个输入均已通过 `pyrpmd validate`（不含 `--check-pes`，运行前需先构建 BKMP2 库）。

```
paper_validation/
├── generate.py              # 唯一参数来源
├── inputs/scan/T*.inp       # Track B：6 个温度的生产速率（伞形 + MBAR，同样本给出 UI）
├── inputs/methods/*.inp     # Track A：PMF 方法对比，300/1000 K × P=1/量子 P × 4 种方法
└── studies/T*.json          # Track C：P、Δt、R∞ 收敛，以及 300/1000/1500 K 的 H/D KIE
```

## 1. 要回答的问题

| 编号 | 问题 | 证据类型 |
| --- | --- | --- |
| Q1 | 不同 PMF 方法（固定伞形 + UI / MBAR、自适应伞形、WT-MetaD、OPES）在相同算力下给出的 ΔG(ξ*) 和 k_QTST 是否一致？各自成本多少？ | 软件内部一致性 + 独立参考 |
| Q2 | P = 1 时，PMF 是否与确定性积分得到的经典 TST 一致？ | 数值验证（近精确参考） |
| Q3 | 300 K 的 k_RPMD、k_QTST、κ 是否复现 RPMDrate 2013 表 2？ | 实现复现 |
| Q4 | 200–1500 K 的 k_RPMD(T) 是否与已发表 RPMD 值一致？与精确量子（QM）差多少？ | 实现复现 + 物理比较 |
| Q5 | 结果对 P、Δt、R∞、ξ*、直方图分箱、伞形刚度是否收敛或不变？ | 收敛 |
| Q6 | H + H₂ 与 D + D₂ 的同位素效应 | 物理比较（相对 QM） |

## 2. 参考数据

数据来自 [`benchmarks/published_rates/references.json`](../../published_rates/references.json)（单位：cm³ molecule⁻¹ s⁻¹）。

| 来源 | T / K | 量 | 值 | 用途 |
| --- | --- | --- | --- | --- |
| RPMDrate, CPC 184, 833 (2013) 表 2 | 300 | k_RPMD（P = 128，R∞ = 30 bohr，Δt = 0.1 fs） | 1.95 × 10⁻¹⁶ | 实现复现，原文统计精度约 5% |
| 同上 | 300 | k_QTST / κ | 2.51 × 10⁻¹⁶ / 0.776 | 分别检验 PMF 和 recrossing |
| 同上 | 300 | 经典速率 | 1.55 × 10⁻¹⁸ | P = 1 检验 |
| JCP 123, 054108 (2005) 表 VII | 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000, 1200, 1500 | H + H₂ 精确量子 | 2.01e-18 … 1.35e-11 | 物理比较。核自旋/宇称约定尚未匹配，**不是通过/失败目标** |
| 同上 | 300, 1000, 1500 | D + D₂ 精确量子 | 4.65e-17, 1.03e-12, 5.96e-12 | KIE 物理比较 |
| Collepardo-Guevara, Suleimanov, Manolopoulos, JCP 130, 174713 (2009)；PCCP 15, 2013（H + H₂ 同位素体 RPMD） | 多温度 | k_RPMD(T) | **登记表中尚无数值** | **第 0 步**：抄录表格数值、P、R∞，写入 `references.json`，然后才看本研究结果 |

仓库内可计算的独立参考：

- **经典 TST 确定性积分**：[`classical_tst.py`](../classical_tst.py) 中的 `quadrature(potential, temperature=T)`，任意温度均可算。它给出 P = 1 的 k_QTST，几乎没有统计误差。CLI 固定为 300 K，其他温度需要从 Python 调用。
- **量子分割面配分函数**：[`surface_partition.py`](../surface_partition.py) 在 TS 面和反应物面上直接采样，不依赖伞形采样。它曾在 300 K、P = 128 给出与论文一致的 QTST。使用其他温度前，需要确认脚本支持温度参数，不支持就添加。

## 3. 全局设置（所有 Track 共用）

| 项 | 值 | 理由 |
| --- | --- | --- |
| PES | BKMP2（[SOURCE.md](../SOURCE.md)，SHA256 固定），三个 ¹H 质量 1.00782503223 u | 与 RPMDrate 相同 |
| 反应坐标 | RPMDrate ξ；两个等价交换通道都在 `equivalent_paths` 中，`symmetry_factor = 1`（**不要再乘 2**） | 与表 2 的通道约定一致 |
| R∞ | 基准 10 bohr，study 中检查 20 和 30 bohr | RPMDrate 用 30 bohr |
| 积分器 | Cayley-BCB，Δt = 0.1 fs，study 中检查 0.05 fs | |
| 伞形窗口 | ξ ∈ [−0.05, 1.10]，间距 0.025，共 47 个窗口；**βK = 400 在每个温度保持不变**，即 K(T) = 1.900086938 × T/1500 hartree | 窗口高斯宽度 1/√(βK) = 0.05 = 2 × 间距，各温度重叠相同 |
| PMF 采样 | 每窗口 4 个副本 × (0.3 ps 平衡 + 5 ps 生产)，Andersen 恒温 | 总计 996 ps 环聚合物时间 |
| Recrossing | 4 条约束母链 × 96 个构型 × 4 对 ± 动量子轨迹；bootstrap 300 | |

### 各温度的珠子数与 recrossing 时长

| T / K | P（生产） | 收敛对照 P | K / hartree | 子轨迹时长 | 预计步数 | 珠子势能调用 |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 192 | 96 | 0.253345 | 200 fs | 16.1 M | 3.1 × 10⁹ |
| 300 | 128 | 64 | 0.380017 | 200 fs | 16.1 M | 2.1 × 10⁹ |
| 400 | 64 | 32 | 0.506690 | 200 fs | 16.1 M | 1.0 × 10⁹ |
| 600 | 48 | 24 | 0.760035 | 150 fs | 14.6 M | 0.7 × 10⁹ |
| 1000 | 32 | 16 | 1.266725 | 100 fs | 13.1 M | 0.4 × 10⁹ |
| 1500 | 16 | 8 | 1.900087 | 100 fs | 13.1 M | 0.2 × 10⁹ |

珠子数的取法：300 K 与 RPMDrate 相同（128），其余温度大致按 βħω_max 等比例缩放；P/2 对照用来证明收敛。仓库已有 300 K、P = 256 的数据可作补充。
势能调用数用于预算。先跑 `bkmp2_pilot.inp`，测出本机每次珠子调用的耗时，再换算成 CPU·h。

## 4. Track A：PMF 方法对比（Q1, Q2）

**设计**：2 个温度（300 K 量子主导、1000 K 接近经典）× 2 个珠子数（P = 1 和该温度的生产 P）× 4 种方法，共 16 个输入：`inputs/methods/T{T}_P{P}_{method}.inp`。

| 代号 | `rate.pmf.method` / estimator | 设置要点 |
| --- | --- | --- |
| `us` | `umbrella` / `mbar`，`compare_ui = true` | 47 个固定窗口。**同一批样本**同时输出 MBAR 和 UI，因此一个输入覆盖两种估计量 |
| `aus` | `umbrella` + `[rate.pmf.adaptive_umbrella]` / `mbar` | 初始 24 个窗口（间距 0.05），最多 8 轮、96 个窗口，目标重叠 0.03 |
| `wtmetad` | `metadynamics`（well-tempered）/ `reweighted_histogram` | σ = 0.04，高度 0.1 kT，pace 100，γ ≈ 12 kcal/mol ÷ kT（300 K 时 20，1000 K 时 6）；前 1.2 M 步学习偏置，冻结后再生产 1.19 M 步 |
| `opes` | `opes` / `reweighted_histogram` | σ = 0.04，barrier 12 kcal/mol，γ 同上，学习/冻结步数同上 |
| `ext`（不单独生成） | `external` | 用 `us` 存下的 PMF 只重跑 recrossing，检查“复用 PMF”这条路径与一次跑完是否逐位一致；也用于下面的 ξ* 不变性检验 |

**控制变量**：

- 四种方法的**偏置采样总量相同**：都是 4 个副本 × 约 2.49 M 步（0.1 fs），等于固定伞形的 47 × 4 × 5.3 ps。
- **分割面固定为 ξ* = 1.0**，**随机种子相同**。如果 recrossing 阶段的随机流与 PMF 阶段独立派生，四种方法的约束母链和子轨迹就完全相同，κ 逐位相同，差异只来自 PMF。第一批运行后要核对 κ 是否确实一致；如果不一致，则直接比较 k_QTST，κ 另行统计。
- 只改 PMF 相关设置；其他设置都与 Track B 相同。

**每次运行要报告的量**：W(ξ*) − W(反应物参考) 及 95% CI；k_QTST；所用势能调用数；墙钟时间；每窗口/每 bin 的有效样本数；重叠矩阵最小值；PMF 在反应物参考与 ξ* 处的支持度；被拒的 bootstrap draw 数。

**预先登记的判据（看结果之前固定）**：

1. **P = 1，与经典 TST 积分比较**：每种方法的 k_QTST 相对积分值的 z 分数 |z| < 2，且相对偏差 < 5%。这是唯一有近精确参考的检验，作为**主判据**。
2. **量子 P，方法间等价**：以 `us`/MBAR 为基准，其他方法的 k_QTST 比值的 95% CI 落在 [0.90, 1.10] 内。该区间在 300 K 对应 |ΔΔG| < 0.057 kcal/mol，在 1000 K 对应 < 0.19 kcal/mol。
3. **UI 与 MBAR**：两者用同一批样本，比较配对差值的 CI。仓库历史上在 1500 K、P = 16 出现过 3 kcal/mol 的 UI/MBAR 差异（见 [docs/mbar-rate-pmf.md](../../../docs/mbar-rate-pmf.md)），本项目必须报告这一项。
4. **300 K 量子 k_QTST 与 RPMDrate 的 2.51 × 10⁻¹⁶ 比较**，以原文的 ±5% 精度为参照；与 `surface_partition.py` 的独立 QTST 比较。

**MetaD/OPES 的已知风险（预先写明，事后不调参）**：[enhanced-sampling.md](../../../docs/enhanced-sampling.md) 指出，ξ 不受约束时，H + H₂ 的 MetaD 可能逃出反应走廊或使 H₂ 解离；冻结偏置的速率路线又**不允许静态墙**。如果 PMF 在反应物参考或 ξ* 处支持不足，PyRPMD 会按设计**不给出速率**。这本身是要报告的结果，用来说明软件的安全门槛，而不是在看到结果后去改 σ、γ 或时长。如果需要第二轮参数，只能作为新登记的 case 运行，并保留第一轮结果。

## 5. Track B：多温度生产速率（Q3, Q4）

`inputs/scan/T{0200,0300,0400,0600,1000,1500}.inp`：伞形 + MBAR（`compare_ui = true`），ξ* 由 PMF 最大值自动选取，每个温度的设置见第 3 节表格。
用 Track C 的 study 文件运行，每个温度有 3 个独立重复。

**输出**：k_RPMD(T)、k_QTST(T)、κ(T)、ξ*(T)、W(ξ)，均带 95% CI。

**图（论文正文）**：

- (a) 6 个温度的 W(ξ) 叠在一张图上；
- (b) 6 个温度的 κ(t)；
- (c) Arrhenius 图：本工作 RPMD、文献 RPMD、精确 QM、经典 TST 积分、本工作 QTST；
- (d) 本工作 RPMD 相对文献 RPMD 和 QM 的比值随 T 的变化。

**判据**：

- 与文献 RPMD：每个温度的比值 CI 与原文精度区间（原文未给精度时按 ±10%）有交集，且点估计在区间内。
- 与 QM：只报告比值，不设通过线。约定未匹配之前，QM 只能作为参照曲线。
- 预期趋势：RPMD 在低温（≤ 300 K）偏低于 QM，高温时趋于一致。如果趋势相反，先查约定和归一化，**不调参数**。

## 6. Track C：收敛与不变性（Q5, Q6）

`studies/T*.json` 使用 `design = "one_at_a_time"`：

| 轴 | 取值 | 说明 |
| --- | --- | --- |
| beads | P/2，P | |
| timestep | 0.1，0.05 fs | 0.05 fs 是 study 的参考档，成本约为 0.1 fs 的 2 倍 |
| separation (R∞) | 10，20，30 bohr | 30 bohr 与 RPMDrate 一致 |
| replicates | 3 | 种子由 study 派生，彼此独立 |
| isotopologues（300/1000/1500 K） | H，D（三个原子都换成 2.01410177812 u） | 给出 KIE = k_H / k_D，与 QM 表 VII 比较 |

`relative_tolerance = 0.05`：相邻档位的速率差的 CI 落在 ±5% 内即判为收敛。

study 之外的手动检验（只在 300 K 和 1000 K 做）：

1. **ξ* 不变性（Bennett–Chandler）**：用 `rate.pmf.method = "external"` 复用 Track A 的 `us` PMF，分别在 ξ* = 0.95、1.00、1.05 重跑 recrossing。k_QTST 和 κ 应各自变化，但乘积 k 应在 CI 内不变。这一项很直观地证明 recrossing 实现正确，适合放进论文。
2. **直方图分箱**：85 / 170 / 340 bins，只用于检查曲线敏感性。
3. **伞形刚度与间距**：βK = 800、间距 0.020（对应 [`convergence.py`](../convergence.py) 中 k800 的设置），与 βK = 400 比较。
4. **recrossing 统计量**：母链数加倍，或每个构型改为 8 对子轨迹，看 κ 的半宽是否按 1/√N 下降。

## 7. 计算预算

| Track | 运行数 | 珠子势能调用（合计） |
| --- | --- | --- |
| A（16 个输入） | 16 | 约 2 × (2.1 + 0.4) × 10⁹ ≈ 5 × 10⁹（P = 1 的部分可忽略） |
| B（3 个重复） | 18 | ≈ 2.2 × 10¹⁰ |
| C（study 的其余档位，含 D） | ≈ 117 | 约为 B 的 4–6 倍，主要来自 0.05 fs 和 D 同位素 |

按每次珠子调用 1–3 µs 粗算（以 pilot 实测为准），A + B 约几十 CPU·h，C 约一两百 CPU·h，单节点（32–64 核）几天内可以完成。**第一步先跑 1500 K 和 1000 K**，它们最便宜，能尽早暴露流程问题。

## 8. 运行顺序

```bash
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3-paper/pes-check.json
cd benchmarks/h3/paper_validation && python generate.py && cd -
```

```bash
for f in benchmarks/h3/paper_validation/inputs/*/*.inp; do pyrpmd validate "$f" --check-pes || break; done
```

Track A（每个输入使用独立的输出目录）：

```bash
for f in benchmarks/h3/paper_validation/inputs/methods/*.inp; do n=$(basename "$f" .inp); pyrpmd run "$f" --output "runs/h3-paper/methods/$n" && pyrpmd analyze "runs/h3-paper/methods/$n"; done
```

Track B + C（以 1500 K 为例，其余温度相同）：

```bash
pyrpmd study benchmarks/h3/paper_validation/studies/T1500.json --plan
```

```bash
pyrpmd study benchmarks/h3/paper_validation/studies/T1500.json --output runs/h3-paper/study/T1500
```

中断后加 `--resume` 续跑。不要在运行中修改输入或 PES：科学指纹会拒绝把不同设置混进同一结果。

## 9. 论文中的呈现

| 位置 | 内容 |
| --- | --- |
| 正文图 | Track B 的四个面板 (a)–(d) |
| 正文图 | Track A：P = 1 时各方法的 k_QTST 与经典积分（条形图 + CI）；量子 P 时各方法的 W(ξ)；精度–成本散点图 |
| 正文表 | 6 个温度：P、k_QTST、κ、k_RPMD（95% CI）、文献 RPMD、QM、比值 |
| 正文或 SI | ξ* 不变性（三个 ξ* 下的 QTST、κ、k） |
| SI | 所有收敛轴、UI/MBAR 配对差、被拒或失败的 case、KIE 表 |

## 10. 已知未决事项（论文中照实写）

- 现有 300 K、P = 128 的结果（k = 1.968 × 10⁻¹⁶，比原文 +0.92%）中，母链时间对半的等价检验仍未通过；P = 256 的动态比较和独立伞形精度门槛也尚未完成。见 [literature-validation.md](../../../docs/literature-validation.md)。本计划的 Track B/C 会补全多温度和收敛证据，但不会自动消除这些已登记的失败项。
- QM 参考的核自旋/电子约定尚未匹配（见 `references.json` 的 notes），所以 QM 只作物理比较。
- 文献 RPMD 多温度数值必须在运行前抄录并登记（第 0 步）；不能在看到结果之后再挑选参考文献。
