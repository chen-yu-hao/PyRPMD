# 可选 MBAR 速率 PMF 与 H₃ 独立重复研究

## 为什么增加这个后端

[先前真实 H₃ / BKMP2 测试](validation/h3-bkmp2-2026-10-29/README.md)
在同一分割面 ξ*=1.00925 上得到 Gaussian UI 势垒 15.3032 kcal/mol、
histogram MBAR 势垒 18.3085 kcal/mol。此前的 40 次配对重采样中，
差值为 3.0053 kcal/mol，条件 95% 区间为 [1.9222, 3.4011] kcal/mol。
这说明需要检查窗口高斯近似和有限采样，不能只因软件测试通过就选一个
数值当作收敛结果。旧的短/长数据使用相同随机种子、存在前缀关系，
不能用作两个独立重复。

新后端让速率工作流直接使用 MBAR，并保留默认 UI 的兼容行为。
输入和失败语义见 [输入规范](input.md#速率-pmf-后端与分割面)，
测度和速率公式见 [科学约定](science.md)。

## 数值实现与核验

求解器在固定自由能 gauge 后使用阻尼 Newton；完全相同的约化势能行
可以精确合并，连续时间序列仍分别重采样。原求解器保留为 fallback
和 `method="legacy"` 对照。最终验收使用原始固定点方程的残差。

`benchmarks/h3/solver_check.py` 对相同保存样本同时调用新求解器、
旧求解器和官方 [PyMBAR](https://pymbar.readthedocs.io/en/stable/mbar.html)。
独立实现得到一个显式无采样目标态 `u_target=0, N_target=0`，
通过公开的 `W_nk` 列比较目标权重，同时比较 gauge 对齐的窗口自由能。
stride 只控制核验成本，不宣称已去除时间相关性；不用 PyMBAR 的
渐近误差替代本项目的 block bootstrap。

数值测试还包括已知非高斯双阱的解析势垒、独立的端点协方差参考、
缺失中间 bin、断连、仅一次有效 bootstrap、重复状态和旧算法等价性。
这些测试分别检查统计估计、失败处理和非线性方程，不证明真实体系遍历性。

## 支撑、权重与窗口门槛

直方图 PMF 只在 bin **中心**有定义。`analysis.ranges=[[-0.1,1.1]]`、`bins=240`
时最后一个中心是 1.0975，而不是 1.1。输入预检会拒绝落在
`[first_center, last_center]` 之外的参考点 ξ=0、`rate.dividing_surface` 或
`rate.pmf_search_range` 端点，同时报告 edge/center 范围和需要的边界，
绝不把 1.1 静默截断为 1.0975。端点恰在中心或内部 edge 上可以接受。
省略 `analysis.ranges`（按数据决定 bin）时，在分析阶段给出同样的分类。

`rate_pmf.json` 保存解析后的分箱与支撑：`support.histogram`
（edges、centers、首末中心、宽度）、`support.search_endpoints`
（`on_center`、`between_centers`、`outside_histogram_centers` 或 `masked`，
含相邻 bin 和被屏蔽区间）、`support.reference_endpoint` 与
`support.dividing_surface_endpoint`。

MBAR 速率直方图由归一化 log 权重以 log-sum-exp 累积 bin 质量，
极小但非零的权重不再因下溢变成屏蔽/NaN。`weight_diagnostics` 报告全局
Kish ESS、每条流的 ESS 与权重占比、端点 bin ESS、最大归一化权重和
log 权重跨度。log 域运算只能恢复**诊断**曲线，最终速率仍受门槛约束：

| 门槛 | 默认 | `[rate]` 选项 | 失败状态 |
| --- | --- | --- | --- |
| 全局 Kish 权重 ESS（坍缩报警） | ≥ 2 | `pmf_min_weight_ess` | `weight_ess_collapse` |
| 参考点/分割面 bin 的 ESS | ≥ 2 | `pmf_min_endpoint_ess` | `endpoint_weight_ess_collapse` |
| 流均值距中心不超过 N 个偏置宽度 `1/sqrt(beta K)` | N = 20 | `pmf_max_window_displacement` | `umbrella_stream_anomaly` |
| 末态质心位于 RPMDrate ξ 的正常分支 `s0-s1>0` | 始终 | — | `umbrella_stream_anomaly` |
| 自适应伞形：冻结生产的重叠、每窗 ESS 与覆盖 | 自适应目标 | `[adaptive_umbrella]` | `adaptive_production_targets_not_met` |

即使数据健康，无偏伞形权重的全局 ESS 也很小：边缘窗口的权重按
`exp(beta K (xi-c)^2/2)` 增长，因此该下限只用于拒绝坍缩到一两帧的情形；
端点 bin 的 ESS 才衡量进入势垒的支撑。仅在有意做诊断时把门槛设为 `"off"`。任一门槛失败时：`ready=false`、
`status="unsupported_pmf"`、`curve_role="diagnostic_only"`、`k_qtst=null`，
不启动 recrossing；潜在势垒只写入 `diagnostic_free_energy_barrier_hartree`。
`rate_pmf.npz` 含 `diagnostic_only=true` 与 `log_density`，`gates` 列出全部检查。

偏离中心或位于错误分支的流还会写出 `umbrella_stream_anomalies.json`
（中心、刚度、均值、宽度、以 σ 计的偏移、均值处偏置能 kcal/mol、
由末态质心重算的 ξ、s0、s1 与分支，以及检查清单）。越过奇异面
`s0=s1` 后，RPMDrate ξ 跳到 −∞，再沿产物通道回升到约 1/3；因此中心为
ξ=1.1 的窗口可能报告 ξ≈0.4–0.7、偏置能巨大而回复力很弱。
继续该流不是收敛手段。

自适应 pilot 只负责选择窗口。即使所有 pilot 目标都满足，速率仍要求
冻结生产满足 `target_overlap`、`min_effective_samples` 及声明的覆盖。

### 失败的偏置 history

伞形窗口、WT-MetaD 与 OPES history 在传播或 PES 失败时（BKMP2 中任一
原子对距离小于 0.2 bohr）写出 `failures/<history>.json`，记录错误、最后一个
已提交状态的 step/time/时钟、步长、珠子与质心最小原子对距离、质心 CV、
最后记录的 CV、偏置状态（hills 或 kernels）、native/Python 传播计数和
checkpoint。此时 `rate_pmf.json` 以 `status="sampling_incomplete"` 列出全部失败
history，运行抛出错误；不会挑选已完成的 history 拼接 PMF。

## 预先固定的四组计算

以下参数在读取本轮结果之前确定，完整机器可读计划由
`python benchmarks/h3/convergence.py plan` 生成。

| Case | 随机种子 | βK | 窗口中心范围 / 间距 | 窗口 × replica | 保存样本 |
| --- | ---: | ---: | --- | ---: | ---: |
| k400-a | 821071 | 400 | −0.05…1.10 / 0.025 | 47 × 2 | 188,000 |
| k400-b | 821072 | 400 | −0.05…1.10 / 0.025 | 47 × 2 | 188,000 |
| k800-a | 831071 | 800 | −0.06…1.10 / 0.020 | 59 × 2 | 236,000 |
| k800-b | 831072 | 800 | −0.06…1.10 / 0.020 | 59 × 2 | 236,000 |

共同设置：1500 K、16 珠子、0.1 fs、R∞=10 bohr；每条流 300 fs
平衡加 1000 fs 生产、每 0.5 fs 采样。主直方图范围 [−0.30, 1.40]、
170 bins、40 次 block bootstrap；85 和 340 bins 只作曲线敏感性，
不额外重复 40 次 bootstrap。UI 对照与 MBAR 使用相同 draw。

每组独立计算 2 条受约束母链，每条 32 个构型，每构型 2 对正反动量
子轨迹，共 256 条、每条 80 fs；κ 使用 100 次重采样。
分割面均固定为 ξ*=1.00925，避免把 PMF 算法差异与分割面移动混在一起。
两种设置同时改变刚度与间距；差异不能归因于刚度单一参数。

## 执行、保存和验收

```bash
python -m pip install -e '.[test,analysis]' matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/convergence.py run --case k400-a --output runs/k400-a
python benchmarks/h3/solver_check.py --run runs/k400-a/run \
  --stride 10 --reference-pymbar --output runs/k400-a/solver-check.json
# 用相同命令运行另三个 case，再汇总共同父目录。
python benchmarks/h3/convergence_report.py runs --output runs/PMF-CONVERGENCE.md
```

每个 case 保存实际输入、源文件/PES/样本哈希、全部 checkpoint、HDF5
原始样本与能量、PMF/κ 曲线、逐 draw 失败原因、端点支撑及窗口分布矩。
汇总器核对四组计划、独立种子、生产门槛与官方 PyMBAR 核验；不完整
或失败的研究仍生成诊断报告，但返回失败状态。
`--smoke` 缩短轨迹来检查程序连接，明确排除在正式研究汇总外。

验收保留有效样本数、真实子轨迹、父链约束、重叠、能量误差、
样本/checkpoint 一致性等门槛；完整记录全部 bootstrap draw。
当前输出只在全部请求的 PMF draw 均成功且相关性分块支持充分时报告主势垒误差与置信区间；
若存在拒绝 draw，主字段为 `null`，曲线主误差为 NaN，接受子集的统计只放在 `supported_draw_*` 中。
最终速率使用同一规则；条件统计不代表无条件置信区间。应一起检查请求、尝试、有效和拒绝次数。
本页与历史记录中的条件区间保留其原始限定，不应改称当前实现的无条件主误差。
报告区分同流 bootstrap 误差、独立 seed 差异和 bin 宽敏感性。
即使所有工程门槛通过，`converged` 仍为 `false`：只有两个独立种子，
也未改变温度、珠子数、步长、R∞，不足以给出正式收敛速率。

GitHub 工作流 **H3 PMF independent-seed study** 同时执行四组计算，
每组保存原始 artifact，最终汇总另存一个完整 artifact。
实际运行数字在完成后单独归档，历史测试记录不覆盖。

## 本次实际运行

[2026-10-29 四组独立研究记录](validation/h3-pmf-refinement-2026-10-29/README.md)
保存成功的 GitHub run `40921070882`：四组与汇总共 6 个 job 全部成功，
所有 160 次 PMF 配对重采样有效，保存样本总数 848,000。
UI/MBAR 配对差值为 −0.0022、+0.3646、+0.1162、+0.0406 kcal/mol，
四个条件 95% 区间均包含零。结果支持方法一致性改善；独立 seed 与窗口
配置之间仍有波动，最终报告保留 `converged=false`。
