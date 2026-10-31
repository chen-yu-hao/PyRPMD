# 自适应伞抽样可靠性审查与修复（2026-10-31）

**结论：PyRPMD 的两阶段自适应伞抽样设计可以用于可靠计算，但默认参数只是启动预算，不能保证最终 FES、PMF 或速率的精度。** 本次修复可复现的诊断和误差报告缺陷；没有把一次运行或通过测试解释为真实体系已经物理收敛。

审查基线为 `main` 的 `c4fa743f552c1d71db35fab791a2e63409bc1c3e`。本目录的
[counterexamples.json](counterexamples.json) 保存同一组合成样本在基线与修改后代码上的实际结果，包含随机种子、输入条件和关键实现文件 SHA256。

## 审查范围与已有正确设计

对输入解析、单位与 Hamiltonian 约定、CV Jacobian、偏置力、热化和采样、adaptive 决策、MBAR/UI 重建、相关性与 bootstrap、速率误差、保存/恢复、并行执行和用户指南进行了交叉检查。全量自动化回归覆盖其它工作流的集成影响。它不是对每一种外部 PES、硬件后端和应用体系的物理认证。

重点实现位于 `src/pyrpmd/{input,bias,dynamics}.py`、`workflows/{adaptive_umbrella,fes,rate}.py` 和 `analysis/fes.py`。审查确认以下机制已经存在，本次保留：

- pilot 只选择窗口。每轮使用固定偏置；最终窗口冻结后使用不同随机流重新平衡并生产，pilot 样本不进入最终估计。
- 自适应目前增加中心和必要的 pilot 长度，**不自动优化刚度矩阵**。
- 物理偏置为 `W = 1/2 delta.T @ K @ delta`；求和形式的环聚合物 Hamiltonian 加入 `P*W`，MBAR 去偏使用 `beta*W`。未在所检查的实现及解析回归中发现额外的 bead 因子错误。
- MBAR 使用对数域计算并检查最终方程残差；UI 对整条曲线重采样，保留不同位置及端点间的统计关联。
- ESS 分别考虑各条流，包含 CV 振幅平方或周期谐波，不把独立流边界当成连续动力学；无可用相关块时不报告误差。
- 预算耗尽返回 `adaptation_incomplete`，不会冒充选窗成功；检查点与计划有完整性和物理配置匹配检查。

## 已修复的缺陷

### 1. 多个高度相似窗口会稀释旧重叠分数

旧指标 `sqrt(O_ij * O_ji)` 是 MBAR 状态转移重叠的对称化值。对于 K 个等样本量、近乎相同的状态，它约为 `1/K`。因此固定阈值可能把充分重叠的大窗口集合判为断连，并不断增加无用窗口。

可复现实验使用 64 个中心（范围 `[-1e-4, 1e-4]`）、每窗 400 个高斯样本、`beta=Kappa=1`、随机种子 2026：

| 量 | 基线 | 修改后 |
| --- | ---: | ---: |
| 最弱生成树连接分数 | 0.015625 | 0.999999999995 |
| 最低每窗 ESS | 145.33 | 145.33 |
| 建议新增窗口 | 63 | 0 |
| 是否通过默认 0.03 重叠门槛 | 否 | 是 |

新 schema-3 计划使用 `normalized_mbar_cosine`：

`S_ij = sqrt(O_ij * O_ji / (O_ii * O_jj))`。

它是 MBAR 权重列的余弦相似度；相同权重列为 1，完全断开的经验支持为 0。这个优化消除了上述无用加窗，不构成对全局混合或物理精度的保证。新旧指标的同一个数值阈值有不同含义，用户应重新检查所设目标。

schema-1/2 计划保留 `mbar_symmetric_transition` 和原调度规则；包升级不会静默改变正在进行的 pilot。新的指标、旧计划重分析、断连反例、交错且不等长 replica 聚合均有回归覆盖。离线报告明确显示指标版本。

### 2. pilot 的成功不能代替生产检查

基线独立复核生产 ESS，但生产 overlap 和可选 coverage 没有按 pilot 目标完整复核，原 `coverage` 仍来自 pilot。

现在分别报告 `pilot_target_met` 和 `production_target_met`，并独立检查生产 overlap、窗口 ESS、可选覆盖网格。旧 `target_met` 作为 pilot 兼容字段保留，附带 scope；旧 `coverage` 明确标为 pilot，并增加 `pilot_coverage` / `production_coverage`。

生产检查复用已有 MBAR 权重、重叠矩阵和相关性结果，无需额外求解。诊断在 root 上、剥离数组之前处理，summary-only MPI 路径不额外广播轨迹或权重。速率结果在 `summary.pmf.adaptive_umbrella` 和 `rate_pmf.json` 保留这些诊断，不必只查子目录。

**生产目标失败仍保留点估计并警告，不自动延长轨迹，也不自动阻止所有后续速率步骤。** 用户必须读取生产诊断；非 null 的结果不是可靠性证书。`converged` 仍为 `false`。

### 3. 失败 bootstrap draw 不能被静默排除后当作主置信区间

基线的气相速率会丢弃缺少支持或拟合失败的 draw，再将幸存子集的误差放在主字段。相同问题也存在于通用 FES 的单个 bin。

反例的三个 histogram bin 分别有 120、1、120 个样本；使用 shuffle seed 7、bootstrap seed 52，共 40 次重采样：

- MBAR 势垒只有 26/40 次有完整支持，14 次因中间区间缺乏支持失败；基线仍报告主 SE `0.0670789016 hartree` 和主 95% 区间。修改后主 SE/区间为 `null`，原数值只保留在明确的 `supported_draw_*` 条件诊断中。
- 通用 FES 中间 bin 只有 26/40 次支持；主 SE 现在为 NaN，条件 SE `0.2435694829 hartree` 仍可查。其它支持完整的 bin 保留主误差，FES 点估计不变。
- 如果固定参考 bin 在某次 draw 中消失，则该次不能提供任何相对参考的主曲线误差。不会改用另一个参考来掩盖失败。

UI、MBAR rate、气相/表面最终速率采用一致的主/条件误差区分。通用 FES 和 rate PMF 曲线逐 bin 检查支持；外围 bin 缺少支持不会无故取消完整势垒区间的误差。相关块完全不可用时，条件字段同样不提供误差。

### 4. bootstrap 不得使原先 masked 的点“恢复”为有误差条

一个只有 1 个样本、`min_count=2` 的原始 bin 可能在有放回重采样中达到 2 个样本。基线公共 rate PMF API 因而曾为 `free_energy=NaN` 的位置返回有限 SE `0.0826486021`。现在原始 mask 同样作用于主/条件误差和有效 draw 计数。重采样不增加原数据的真实支持。

## 默认参数为什么不能保证精确结果

完整且中英同步的实际默认值见[英文输入表](../../manual/input.md#defaults)和[中文输入表](../../zh/manual/input.md)。最关键的设置是：

| 设置 | 当前默认 | 限制 |
| --- | --- | --- |
| bead 数 / 时间步长 | 32 / 0.1 fs | 需要 P→2P、dt→dt/2 的目标量收敛检查 |
| 平衡 / adaptive 生产 | 100 fs / 1000 fs，每流 | 不能覆盖未知体系的慢时间尺度 |
| 每窗独立 replica | 1 | 不能提供独立重复之间的一致性证据 |
| 初始 pilot / 上限 | 1000 / 16000 步 | 默认步长下 0.1 / 1.6 ps，只是预算 |
| pilot ESS 门槛 | 20 | 筛选条件，不是势垒或速率误差指标 |
| 空间 coverage | 默认未启用 | 连通图不等于所需区域已充分采样 |
| FES/PMF bootstrap | 20 次 | 只能给粗略重采样诊断，尤其不宜凭此声称尾部区间稳定 |
| 默认气相 rate 伞窗 | 23 窗，ξ=-0.05…1.05 | Δξ=0.05；需要窗口和刚度收敛 |
| 默认气相刚度 | `50/beta` | `beta*kappa=50`；自适应不会自动改变刚度 |
| 默认气相 PMF | Gaussian UI，401 个积分点 | 增加积分点或 bootstrap 不会消除非高斯模型误差 |

`preset="production"` 要求部分顶层设置存在；嵌套项仍可以继承默认值。它没有切换到“高精度参数集”。本次没有用任意放大所有预算来假装保证精度，而是修复错误判据和误差语义，并明确必须验证的量。

仓库已有的 [H₃ 有限预算研究](../enhanced-sampling-2026-10-31/README.md)也不能为普适精度背书：短生产自适应势垒为 `12.82 ± 3.52 kcal/mol`，相同每窗长预算为 `18.95 ± 1.98`，匹配固定窗总步数后的自适应为 `19.68 ± 0.69`。这些是历史记录，不是本次重新测出的数值；固定窗本身也不是真值，且该研究未建立 P、dt、独立种子和完整速率的物理收敛。

## To-do list

### 本次完成

- [x] 固定审查基线，核对 pilot/production 隔离、偏置与 bead 约定及保存/恢复。
- [x] 用可复现反例修复重叠分数的窗口数稀释；明确版本并保持旧计划恢复。
- [x] 独立复核生产 overlap、ESS、coverage；复用拟合并保持 MPI summary-only 行为。
- [x] 统一 UI/MBAR/FES/速率的主误差与幸存 draw 条件误差语义。
- [x] 防止 bootstrap 复活原始 masked bin 的误差；保留失败原因与实际次数。
- [x] 更新报告 reader、最终 rate 诊断和中英默认值/使用指南。
- [x] 加入真实数值反例、恢复/并行集成回归和 4-rank MPI CI 项。
- [x] 保留机器可读的基线/修改后结果，记录验证范围和未完成的物理验证。

### 后续需要目标体系与误差预算

- [ ] 对实际目标体系，至少做不同种子和初态的独立重复，延长平衡及生产，并比较前后分段与分块误差稳定性。
- [ ] 对 P、dt、窗口刚度/密度、bin 宽度、反应物渐近距离及 recrossing 平台逐项收敛。
- [ ] 如果继续采用默认 UI，检查偏斜/多峰和端点外推，并以同一测度、同一生产样本做 MBAR/UI 对照。
- [ ] 另行设计可选的严格 UI 实测支持门槛及按目标误差延长生产的工作流；本 PR 保留现行 UI 模型点估计兼容行为。

一个可操作的误差预算例子：当 `k_QTST ∝ exp(-beta*DeltaF)`，仅允许势垒误差带来最多 1.10 倍的速率偏差，需 `|deltaF| ≤ kBT*ln(1.10)`；300 K 下约为 **0.0568 kcal/mol**。这还没有包括 PES、RPMD 动力学近似和传输系数误差。验收流程见[中英科学指南](../../manual/science.md#assessing-adaptive-umbrella-results)。

## 验证

最终测试与构建结果记录在 [validation.json](validation.json)：

| 验证 | 结果 |
| --- | --- |
| 完整 pytest | 3,829 通过、95 跳过；14 项因沙箱禁止本机 socket 而失败 |
| 允许本机通信后重跑相关三个文件 | 36 通过、2 跳过，包含上述全部 14 项；无代码绕过 |
| 合并上述结果 | 3,843 个不同测试通过，95 个可选集成测试跳过，无未解决失败；不是声称单次全套命令零失败 |
| 最终修改的定向回归 | 197 通过，15 个 subtest 通过 |
| 实际 4-rank MPI（G2/B2） | 新运行、生产诊断、所有 rank 一致性和精确重分析通过 |
| 固定校验和的 BKMP2 / 原生采样相关回归 | 155 通过，5 个 subtest 通过 |
| 中英文 Sphinx（警告视作错误）/ wheel 构建 | 全部通过 |

这些验证集合有重叠，不应相加作为额外的独立证据。可选集成的跳过不算验证成功；完整套件的 14 条警告来自既有 Matplotlib 弃用提示。

针对性复现入口：

```bash
python -m pytest -q tests/test_adaptive_umbrella.py tests/test_fes_bootstrap_support.py \
  tests/test_mbar_rate_analysis.py tests/test_rate_mbar_backend.py \
  tests/test_surface_uncertainty.py tests/test_report_sampling_validation.py
python -m pytest -q
mpiexec -n 4 python tests/mpi_adaptive_smoke.py /fresh/shared/output
python -m sphinx -b html -W --keep-going docs /tmp/pyrpmd-docs/en
python -m sphinx -b html -W --keep-going docs/zh /tmp/pyrpmd-docs/zh
```

本次另构建了仓库固定 SHA256 的 BKMP2 并运行相应回归。解析/合成模型、真实 PES 接口和多进程检查解决不同风险，均不能替代目标反应的物理收敛研究。

## 一手方法资料

- Kästner & Thiel, statistical error in umbrella integration: [JCP 124, 234106 (2006)](https://doi.org/10.1063/1.2206775)。高斯近似及跨坐标误差相关性。
- Shirts & Chodera, MBAR: [JCP 129, 124105 (2008)](https://doi.org/10.1063/1.2978177)。
- [PyMBAR timeseries](https://pymbar.readthedocs.io/en/latest/timeseries.html)：平衡检测和相关时间是诊断，有限轨迹不能证明遍历。
- [GROMACS WHAM](https://manual.gromacs.org/current/onlinehelp/gmx-wham.html)：有限采样可能低估相关时间和 bootstrap 误差。
- [OpenMM RPMD](https://docs.openmm.org/latest/userguide/library/09_rpmd_plugin.html)：bead 收敛与 RPMD 动力学近似的边界。
