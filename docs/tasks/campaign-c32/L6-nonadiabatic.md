---
orphan: true
---

# L6 非绝热动力学与热非绝热速率

| 字段 | 内容 |
| --- | --- |
| CPU | 27–29（3 核） |
| 预算 | 138 核时（计划 110） |
| 工作流 | `nonadiabatic`（NRPMD、SM-NRPMD、CA-RP-MASH、BA-RP-MASH、RPSH-CA/BA/DM、FSSH、MASH、IESH）、`nonadiabatic-rate`（热 NRPMD 速率）、MV-/KC-RPMD、golden-rule、HEOM 参考（`pyrpmd.analysis.heom.SpinBosonHEOM`） |
| 外部模型 | 无（全部为解析模型）；NaH₂ 分子势为可选 |
| 建议分支 | `agent/c32-l6-nonadiabatic` |
| 记录 | `docs/validation/c32-l6-nonadiabatic-<日期>/README.md` |
| 复核 | L6 复核 L0；L0 复核 L6 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L6 ...` |

## 背景

- [Richardson–Thoss 2013 记录](../../validation/published-nrpmd-2013-2026-11-01/README.md)：P=1、4 的核关联与
  论文曲线 RMSE 为 0.04254、0.02580；**P=8 因 7/256 次 bootstrap 抽样不满足符号归一化门槛，整个置信区间被扣留**；
  电子关联的时间零点差 0.05690 与其标准误 0.05486 相当。默认预算是 4 条链 × 512 样本。
- [Tier 2 文献记录](../../validation/tier2-literature-2026-11-01/README.md)：Tully I 每个方法/动量 512 条轨迹，
  128 条的步长子集"无法分辨几个百分点的收敛"。
- [DAC 记录](../../../benchmarks/paper_validation/DAC_VALIDATION.md)：单珠 focused MMST 与 W 映射在第二交叉区与
  低动量返回区明显偏离发表的量子参考（负结果）。
- 本机实测：`published_nrpmd_benchmark.py --beads 8 --chains 1 --samples 32 --burn-in 256` 单核约 6 s（含启动）。

## 子测试

### L6-a Richardson–Thoss 2013 Fig. 1(c) 加量（约 35 核时）

```bash
$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L6 --name rt2013-main \
  --kind production --slots 3 --prereg benchmarks/campaign_c32/L6/prereg-rt2013.json -- \
  python examples/nonadiabatic/published_nrpmd_benchmark.py --output "$C32_DATA/L6/rt2013" \
  --beads 1 4 8 --chains 64 --samples 4096 --workers 3 --seed 20261106
```

链数和样本数由 pilot 定（上面是目标规模：16 倍链、8 倍样本）。判据：P=1、4、8 的 bootstrap 区间是否可以报告
（全部抽样满足符号门槛）；可以报告时，与论文顶点的 RMSE 和区间覆盖率；P=8 的平均符号与有效样本数随样本量的变化。
另用 `--potential-split trace` 跑一个 P=4 对照，并用 `--dt` 减半的 refinement 子集估计步长误差。
P=8 仍被扣留时如实报告——这本身是关于符号问题的结果。

### L6-b Tully I 散射加量（约 35 核时）

`benchmarks/tier2_literature/nonadiabatic.py --output ... --workers 3`：每个方法/动量 4096 条主轨迹
（原 512），步长子集 1024 条（原 128）。判据：每个出射通道的 Wilson 区间与独立单色量子参考的关系；
步长对半的配对差区间是否落在预注册边界（建议 ±0.02 概率）内。未判定轨迹按原规则单独计数。

### L6-c 自旋–玻色子：环聚合物方法对 HEOM（约 20 核时）

参考：`SpinBosonHEOM`（示例 `examples/nonadiabatic/heom_spin_boson.py`）。先做 HEOM 层级深度与浴展开项数的
收敛（加深一层的变化小于预注册阈值），再在两组参数（对称、偏置）上比较 NRPMD、SM-NRPMD、CA-RP-MASH、
BA-RP-MASH、RPSH 的布居 P(t)（P = 1、4、8，每个 ≥ 2048 条轨迹）。报告每个方法相对 HEOM 的最大偏差及其区间；
这些是近似方法，偏差本身是结果，不设"通过"阈值，只设"统计可分辨"的报告规则。

### L6-d 热非绝热速率与 golden-rule 极限（约 10 核时）

`pyrpmd nonadiabatic-rate examples/nonadiabatic/thermal_rate.inp --output ...`，以及
`examples/nonadiabatic/thermal_rates.py`、`method_models.py` 中的 MV-RPMD、KC-RPMD、isomorphic 速率：
在弱耦合（非绝热）区扫 3 个耦合强度，比较与 Fermi golden-rule（含 golden-rule instanton，`pyrpmd.golden_rule`）
速率的比值是否随耦合减小趋于 1（预注册单调性与极限区间）。

### L6-e 布居动力学与 DAC 对照加量（约 10 核时）

- `benchmarks/nonadiabatic/population_dynamics.py`（非 `--smoke`）：轨迹数 2048，`steps*dt` 固定下做步长对照。
- `benchmarks/paper_validation/tully_sac_pilot.py --n ...` 与 DAC 驱动：把轨迹数加到原记录的 4 倍，检查负结果
  是否在更小的统计误差下依然成立（不是"修正"负结果）。

## 不做

- 不修改映射、跃迁或退相干算法；不为接近参考而改变初始分布（例如不把单色动量换成高斯展宽）。

## 交付物

- `benchmarks/campaign_c32/L6/`：驱动、HEOM 收敛脚本、预注册；验证记录与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L6 agent，负责"非绝热动力学与热非绝热速率"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L6-nonadiabatic.md；docs/nonadiabatic-methods.md；docs/thermal-nrpmd.md；
docs/validation/published-nrpmd-2013-2026-11-01/README.md。

要求：
- 你的 CPU 是 27–29（3 核），预算 138 核时；所有计算用 $C32_LAUNCH run --lane L6 启动。
- 符号、未判定轨迹和被扣留的区间全部保留；不裁剪映射样本，不丢弃失败的 bootstrap 抽样。
- HEOM 参考先做层级收敛，再用于比较。近似方法与参考的偏差是结果，不是"失败"。
- 负结果（DAC）只加量检验是否稳定，不改方法使之"通过"。
- T+24 h、T+40 h 更新 $C32_DATA/L6/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l6-nonadiabatic 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L0。
```
