---
orphan: true
---

# L3 自由能与增强采样

| 字段 | 内容 |
| --- | --- |
| CPU | 16–19（4 核） |
| 预算 | 184 核时（计划 150） |
| 工作流 | `fes`：umbrella（固定、adaptive）、REUS（`[replica_exchange]`）、metadynamics（standard、well-tempered、shared walkers）、OPES、ABF、eABF、VES、steered；`[tempering]` PT/HREMD；`analyze --recompute`、`project` |
| 外部模型 | L0 构建的 OGRe 模型（`$C32_DATA/shared/ogre`）；pymbar 可选 |
| 建议分支 | `agent/c32-l3-fes` |
| 记录 | `docs/validation/c32-l3-fes-<日期>/README.md` |
| 复核 | L3 复核 L7；L7 复核 L3 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L3 ...` |

## 背景

- [FES 方法记录](../../validation/fes-methods-2026-10-30/README.md) 用 3 个种子比较了五种方法，
  OPES/ABF/eABF/VES 没有在同一模型和同一预算下与它们比较。
- [Tier 2 文献记录](../../validation/tier2-literature-2026-11-01/README.md)：OPES Müller 共 456 万 MD 步，
  论文为每条 2 亿步；坏 CV 的 METAD 案例零次跨盆。
- NP4（#122）把 MetaD 与墙偏置的描述符移进 C++（E1）；长时程 FES 的统计等价（R1）没有在工作流层面验证过。

所有子测试都在解析模型上，参考值精确（P=1 对裸势；谐振势对任意 P 的质心 FES）。

## 子测试

### L3-a 同预算方法比较：OGRe 双阱与 Ackley，P=1、300 K（约 70 核时）

- 方法：`fixed`、`adaptive`、`unbiased`、`standard`、`well_tempered`（`benchmarks/fes_methods/ogre_benchmark.py run`），
  再加 OPES、ABF、eABF、VES——后四者在 `benchmarks/campaign_c32/L3/` 中写同一模型、同一 CV、
  同一温度和步长的 `fes` 输入（参考 `examples/opes/`、`examples/abf_double_well.inp`、`eabf_double_well.inp`、
  `ves_double_well.inp`）。
- **预算按力评估次数对齐**，包含 adaptive 的 pilot 轮和 VES/OPES 的学习阶段；每个方法 8 个独立种子。
- 指标（在预注册中固定区域与阈值）：区域内 F 的 RMSE（去掉一个最优加性常数，不允许缩放）、
  两盆自由能差和势垒的误差、覆盖率、跨盆次数、有效样本数、总力评估次数、墙钟。
- 判据：每个方法每个指标给 8 种子的 95% 区间；只有区间不相交时才说一个方法"更好"。
  任何种子的覆盖率不足或零次跨盆都单独列出，不平均掉。

### L3-b OPES Müller 加量（约 40 核时）

`benchmarks/tier2_literature/sampling.py opes --steps ... --seeds ... --variants ...`：先 pilot 测每百万步耗时，
然后把主变体的每条轨迹步数提高到 pilot 允许的上限（目标 ≥ 2000 万步/条，8 个种子）。
检查 [Tier 2 记录](../../validation/tier2-literature-2026-11-01/README.md) 中"零跨盆"案例在加长后
是否出现跨盆，以及 FES 误差随步数的变化（报告 1/4、1/2、全长三个前缀，不挑前缀）。
半步长、半带宽对照保留为独立阶段。

### L3-c 原生与 Python 历史偏置的长时程统计等价（NP4，约 20 核时）

同一 WT-MetaD 输入（`examples/metadynamics/well_tempered_2d.inp` 放大 20 倍步数；另加一个 1D 双阱），
`loop_backend="native"` 与 `"python"` 各 8 个种子（两组种子不同）。比较最终 FES、偏置高度累积和
重加权分布：预注册"两组种子均值之差的 95% 区间落在 ±0.1 kBT 内（区域内逐点，Holm 校正）"。
同一种子在两个后端上的前 200 步逐帧差异按 R1 的容差另报。

### L3-d 谐振势上质心 FES 的精确性（约 10 核时）

`benchmarks/fes_methods/benchmark.py build|run|dynamics|quantum`：P=1、8、32，五种方法 × 4 种子，
质心 FES 对任意 P 都是 `(x²+y²)/2 + C`。判据：每个种子区域内最大偏差 < 预注册阈值（建议 0.3 kBT），
珠子二阶矩与精确有限 P 值的 95% 区间相容。

### L3-e 副本交换与 tempering（约 10 核时）

`examples/fes/replica_exchange_2d.inp`（REUS）与 `examples/tempering/double_well.inp`、`hremd_umbrella.inp`
放大采样：交换接受率、往返时间、MBAR 重组分布与精确分布的偏差；与不交换的同预算运行比较误差。

### L3-f 续算与重新分析（约 5 核时）

对 L3-a 中 MetaD、OPES、VES 各取一个种子，在生产中途中断后 `resume`（改 `--block-steps`），
与不中断的同种子运行逐位比较（G5、R5）；再对 umbrella 结果做 `pyrpmd analyze --recompute --bins ... --bootstrap ...`
和 `pyrpmd project`，确认重新分析不传播新轨迹，pymbar 可用时与 PyMBAR 全样本结果比较。

## 不做

- 不比较量子质心 FES 与裸势（非谐势 P>1 时二者不同）；不用 MetaD 的偏置时间推速率。
- PLUMED 只在 L0 报告已安装时做（`sampling.py plumed`），否则标为阻塞。

## 交付物

- `benchmarks/campaign_c32/L3/`：输入、预注册、分析与作图脚本；验证记录（含方法 × 指标的区间表）与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L3 agent，负责"自由能与增强采样"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L3-fes-enhanced-sampling.md；docs/enhanced-sampling.md；benchmarks/fes_methods/README.md。

要求：
- 你的 CPU 是 16–19（4 核），预算 184 核时；所有计算用 $C32_LAUNCH run --lane L3 启动。
- 方法比较按"总力评估次数"对齐预算（含 pilot 与学习阶段），每个方法 8 个独立种子；
  指标、区域和阈值在生产前提交。只有区间不相交时才排序方法。
- 原生/Python 后端的长时程比较用不同的种子组，按 R1 分两层报告。
- 失败、零跨盆、覆盖不足的种子全部保留并单独列出。
- T+24 h、T+40 h 更新 $C32_DATA/L3/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l3-fes 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L7。
```
