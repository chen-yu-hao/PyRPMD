---
orphan: true
---

# L4 平衡 PIMD、量子统计与凝聚相估计量

| 字段 | 内容 |
| --- | --- |
| CPU | 20–23（4 核） |
| 预算 | 184 核时（计划 150） |
| 工作流 | `pimd`（`sampling.engine` = python / native / torch）、`[path_integral]`（Suzuki–Chin、玻色子、费米子、PIGS）、热浴（Andersen、Langevin、PILE-L/G、SVR、NHC、GLE、PIGLET）、压浴（BZP、MTK、SCR）、凝聚相 SC/PIGLET/RPC 估计量、TRPMD/PA-CMD 谐振光谱 |
| 外部模型 | 无；torch 引擎需要 L0 安装 CPU 版 PyTorch，否则该列标为阻塞 |
| 建议分支 | `agent/c32-l4-pimd` |
| 记录 | `docs/validation/c32-l4-pimd-<日期>/README.md` |
| 复核 | L4 复核 L5；L5 复核 L4 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L4 ...` |

## 背景

- 三套 PIMD 引擎（Python 旧路径、`native_pimd`、Torch）共享分派与 checkpoint（T16b 部分合并），
  但"三引擎确定性 NVE 等价"仍是待办；它们在统计意义上是否给出同一个平衡分布也没有系统比较过。
- NP2（#119）加入原生 PILE-L/TRPMD 采样块（无偏置 E0）；长时程统计未验证。
- [Tier 2 文献记录](../../validation/tier2-literature-2026-11-01/README.md)：16 个理想二维玻色子 β=6，
  三种子结果 15.88987612 ± 0.49239761，参考 P=36 精确作用量 15.94975672、连续极限 16.00500687——
  区间太宽，不能检验论文 0.5% 的最大差异。

## 子测试

### L4-a 解析谐振/四次势：估计量 × 热浴 × 引擎矩阵（约 70 核时）

- 模型：一维/三维谐振子（βħω = 1、4、8）与一个四次势（参考用独立的数值对角化，写进驱动并单独测试）。
- 维度：P ∈ {1, 4, 16, 64}；步长 dt₀ 与 dt₀/2；热浴 PILE-L、PILE-G、SVR、NHC、Langevin、Andersen、
  GLE/PIGLET（用仓库自带参数）；引擎 python、native、torch。不需要全因子：预注册一个覆盖每个因素至少两个水平的
  正交子集（约 60 个配置），每个配置 8 个独立种子。
- 观测量：⟨V⟩、初级/维里/质心维里动能、Cv、⟨x²⟩ 与质心方差、积分自相关时间。
- 参考：精确有限 P 值（谐振势解析，四次势数值）和连续极限分别报告。
- 判据：每个配置的 95% 区间应包含精确有限 P 值；预期约 5% 的格子不包含，报告不包含的数量与二项分布的比较，
  并把显著超额的因素（某个热浴、某个引擎、某个步长）单独列出。同一配置三引擎之间的差的区间单独列表。

### L4-b 三引擎 NVE 与原生采样块的逐位/舍入级检查（约 5 核时）

同一初态（q、p 相同、无热浴），python / native / torch 各传播 1000 步：逐帧比较 q、p 和能量，
报告最大差异与首次超过 1e-12 的步数（这是 T16b 待办的测量，不是修复）。对 PILE-L 采样块，
同种子 `loop_backend` native 与 python 的前 200 步按 R1 比较。

### L4-c 玻色子、费米子与 PIGS 加量（约 40 核时）

`benchmarks/tier2_literature/path_integral.py --mode sample --workers 4 --steps ...`：
把 16 玻色子 β=6 的种子数提高到 16、每种子步数由 pilot 定，目标 95% 半宽 ≤ 0.08（约 0.5%）。
判据：区间包含 P=36 精确作用量值；同时报告与连续极限的差（离散化误差，不是采样误差）。
PIGS（α=0.25、投影时间 8，P=33 与 P=129）和 `examples/path_integral/fermionic.inp` 的平均符号与有效样本数一并报告。

### L4-d 凝聚相估计量完整运行（约 25 核时）

`benchmarks/condensed_phase/validate.py --case harmonic` 与 `--case lj`，用默认（非 `--quick`）设置，
`--jobs` 不超过 4（`--slots 4`）。本机实测 `--case harmonic --quick` 单核约 10 分钟，按此估计完整运行的规模。
报告 SC 能量、Cv/Cp、压力估计量与参考的区间比较，以及 PIGLET 压力诊断。

### L4-e 压浴（约 8 核时）

`examples/materials/npt_free.inp`（理想气体，体积分布有解析参考）分别用 BZP、MTK、SCR，
以及原生 `examples/ensembles/pimd_native_npt.inp`：比较 ⟨V⟩ 与体积分布（KS 检验，预注册显著性水平）；
在一个 LJ 模型上比较三个压浴的 ⟨V⟩ 是否在区间内一致。

### L4-f TRPMD / PA-CMD 谐振光谱（约 5 核时）

`benchmarks/trpmd/harmonic_spectrum.py --replicas ... --steps ... --beads ...` 与
`benchmarks/tier2_literature/spectroscopy.py`：峰位与精确值的差、质心不变性，种子数加到 pilot 允许的 4 倍。

## 不做

- 不跑 q-TIP4P/F 量子水（成本见 README）；不修三引擎差异，只测量并记录。

## 交付物

- `benchmarks/campaign_c32/L4/`：驱动（含四次势参考的独立测试）、预注册、分析脚本；验证记录与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L4 agent，负责"平衡 PIMD、量子统计与凝聚相估计量"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L4-pimd-equilibrium.md；docs/quantum-statistics.md；docs/condensed-phase-nqe.md；
docs/native-ensembles.md。

要求：
- 你的 CPU 是 20–23（4 核），预算 184 核时；所有计算用 $C32_LAUNCH run --lane L4 启动。
- L4-a 是因素矩阵：预注册一个正交子集（约 60 配置 × 8 种子），参考值（精确有限 P 与连续极限）先算好并测试。
- 报告"区间不包含参考"的格子数并与二项期望比较；不删除不好看的格子。
- 三引擎差异只测量不修复；torch 不可用时标为阻塞。
- T+24 h、T+40 h 更新 $C32_DATA/L4/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l4-pimd 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L5。
```
