# 统一自由能面与增强采样

三种采样方法统一使用 `task="fes"`，由 `[fes] method` 选择。未指定
`method` 时仍使用伞形采样。积分、CV 观测、偏置更新时序、checkpoint 和
replica 调度共用基础设施；偏置更新算法与自由能重构仍分别实现。

| 方法 | CV 维度 | 自由能重构 | `uncertainty_kind` |
| --- | --- | --- | --- |
| `umbrella` | 1–3 | 含 bin 体积修正的 MBAR | `block_bootstrap` |
| `metadynamics` | 1–3 | standard / well-tempered 偏置估计 | `replica_spread` |
| `opes` | 1–2 | KDE 或冻结偏置后的重加权直方图 | `replica_spread` |

## 输入：公共部分与方法参数

公共 CV、采样和分析设置可以分别放在 `[[fes.cvs]]`（或按名称组织的
`[fes.cvs]`）、`[fes.sampling]`、`[fes.analysis]`。偏置参数放在所选方法的
`[fes.umbrella]`、`[fes.metadynamics]` 或 `[fes.opes]` 中。

```toml
task = "fes"

[fes]
method = "metadynamics"

[fes.sampling]
equilibration_steps = 100
production_steps = 600
sample_interval = 5
replicas = 2

[fes.metadynamics]
cvs = ["x", "y"]
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0

[fes.analysis]
method = "bias_estimate"
bins = [24, 24]
bootstrap = 0
```

这个片段还需要普通的 system、potential、温度、珠数和 x/y CV 定义；完整例子见
[`multiple_walkers.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/metadynamics/multiple_walkers.inp)。
`system`、`potential`、`dynamics`、`parallel`、`output` 仍为顶层设置。

原有顶层 `cvs`、`sampling`、`analysis` 和方法表仍受支持，但同一部分不能同时写在
顶层与 `[fes]` 下。未选方法的表会被拒绝，包括空表。跨方法参数不会被忽略，例如
MetaD 表中不能写伞形 `stiffness`，伞形任务不能附带 `[fes.opes]`。
`adaptive_umbrella` 和 `replica_exchange` 只适用于伞形方法。

旧的 `task="metadynamics"`、`task="opes"` 是保留的别名，会通过 `FutureWarning`
提示新写法。旧算例中仍保留这些写法以检验兼容性。

## 统一输出：`FESResult`

三种方法都写出以下文件：

| 文件 | 内容 |
| --- | --- |
| `fes.npz` | 带完整性校验、内嵌方法元数据的权威结果 |
| `fes.csv` | 统一的网格、自由能、误差和 mask 表 |
| `fes.json` | schema、方法、单位和方法元数据 |
| `summary.json` | 运行诊断和文件引用 |
| `results/summary.json` | CLI 工作流的汇总，增加软件和运行信息 |

伞形的 `fes.h5` 以及各方法的轨迹、hill/kernel 历史仍作为附加诊断保留。
公共结果包括 `edges_0` 等网格边界、`grid` 格心坐标、`free_energy`、`uncertainty`、
`mask`、`uncertainty_mask`、CV 单位、能量单位、方法和方法元数据。能量单位为 hartree。
没有采样支持的格子被 mask；无法估计的误差为 NaN，并由 `uncertainty_mask` 标识。

```python
from pyrpmd.analysis import FESResult

surface = FESResult.read("runs/my-fes")
print(surface.method, surface.uncertainty_kind)
grid = surface.grid
free_energy = surface.free_energy
uncertainty = surface.uncertainty
supported = ~surface.mask
```

旧格式只在这个读取接口中做兼容转换。`analyze`、报告、对比及投影工具读取同一核心
对象；`analyze --recompute` 仍会按所选方法从 checkpoint 重新重构，可修改分析网格，
不推进动力学。3D 报告提供穿过最低有支持格点的三个正交切片，切片不是边缘自由能。
跨方法比较保留各自的方法与误差语义；仅物理体系、CV、测度和静态墙一致时叠图，
不会合并不同算法的样本或误差。

### 误差含义

伞形沿用时间相关的 block bootstrap，副本交换梯子内按同一时间索引共同重采样。
如果请求的重采样在某格缺少支持，或共同参考点没有支持，该格的主要 `uncertainty`
及兼容字段 `standard_error` 为 NaN。`supported_draw_standard_error` 只描述有支持的
接受子集，`bootstrap_valid_per_bin` 保留支持次数，不能把这些条件统计当作完整误差。
MetaD/OPES 对各独立 replica 的自由能对齐零点后，报告样本标准差，也就是
`replica_spread`。它是描述性离散程度，不是均值标准误，也不是已校准的置信区间。
只有一个独立历史时无法估计 spread。公共字段始终为 `uncertainty`，统计含义由
`uncertainty_kind` 说明；历史的 `standard_error` / `replica_spread` 数组作为兼容字段保留。

## 三维伞形 FES

选择总计三维的 CV，在 `[fes.umbrella]` 提供三个网格轴，以及标量、三个对角元素
或 3×3 对称正定的刚度矩阵。`analysis.bins` 和 `ranges` 也需要三个轴。
MBAR、mask 和 bootstrap 在完整三维直方图上计算，保留 bin 体积。
`rate_flux` 的专用反应坐标限制仍为原有的一维/二维情形。

```bash
pyrpmd run examples/fes/harmonic3d.inp --output runs/umbrella3d
pyrpmd analyze runs/umbrella3d --recompute --bins 6 6 6 --bootstrap 10
```

[`harmonic3d.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic3d.inp) 使用解析谐振子势。它演示三维输入与
输出，不代表已经达到采样收敛。自适应伞形的覆盖网格也支持一至三维，保留最多一百万
格的检查；pilot 达标后才冻结窗口并开始独立生产段。

### 自适应计划与生产诊断

新 schema-3 计划使用归一化重叠分数
`sqrt(O_ij * O_ji / (O_ii * O_jj))`，记录为 `normalized_mbar_cosine`。
旧 schema-1/2 计划保留 `mbar_symmetric_transition` 分数及已提交决策，续算时不重新选窗。
两种分数下的相同阈值含义不同，不能直接比较。

读取 `adaptive_umbrella.pilot_target_met` 判断选窗阶段是否达标，读取
`production_target_met` 判断独立生产段是否通过已启用的重叠、ESS 与覆盖检查。
`pilot_coverage` 和 `production_coverage` 分别记录两个阶段；生产诊断会同时保存在
统一的 FESResult 元数据与摘要中。pilot 达标或存在有限自由能，并不表示生产结果已收敛。
详细默认预算及验收步骤见[输入缺省值](manual/input.md#缺省值)和
[评估自适应伞抽样结果](manual/science.md#评估自适应伞抽样结果)。

## MetaD 共享偏置 multiple-walker

默认 `sampling.replicas` 仍创建相互独立的偏置历史。启用共享模式时：

```toml
[fes.metadynamics]
cvs = ["x", "y"]
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0
multiple_walkers = true
block_steps = 100
```

`sampling.replicas`（或 `fes.sampling.replicas`）成为 walker 数。
`block_steps` 必须为正整数，默认等于 `pace`；它决定偏置同步协议，属于 checkpoint
的科学身份。它不同于 `dynamics.block_steps` 这种仅影响执行性能的分块设置。

每个 block 从上一个同步边界的全局 hills 开始。block 内，walker 能看到这份全局
历史和自己新沉积的 hills。所有 walker 到达同一绝对积分步后，协调器按
`(step, walker)` 排序合并新 hills，再分发给下一个 block。WT hill 高度保留沉积时
使用的局部偏置值，不在合并时重新计算。最后一个 block 可以不足整长；同步、采样与
沉积的周期不要求互为整数倍。同一步内的力始终使用同一偏置历史，观测发生在该步
成功后的沉积之前。

`checkpoints/metadynamics_walkers.npz` 原子保存一份全局 hill 历史，以及所有 walker
的 q/p、RNG、恒温器状态、样本和时钟。根目录的 `HILLS.csv` 是可重建的视图，包含
`walker` 列。共享模式在每个同步 block（及最后不足整长的 block）提交 checkpoint；
`output.checkpoint_steps` 不会产生不同步的独立历史。

任一 walker 失败后，续算从最后一个全局已提交 block 恢复整个集合。串行和多进程
调度保留相同的随机流身份。不能把已经独立运行的 HILLS 文件简单拼接成一次共享偏置
运行，也不能在同一输出目录把独立模式直接切换为共享模式。

共享 walker 相互相关，只产生一份独立 bias history。结果以全部 walker 的访问范围
重构这份全局偏置，记录 `multiple_walkers=true`、`walkers=N`、
`independent_histories=1`；`uncertainty` 为 NaN。需要独立集合才能评价重复计算的 spread。

```bash
pyrpmd run examples/metadynamics/multiple_walkers.inp --output runs/shared-meta
pyrpmd resume runs/shared-meta
pyrpmd analyze runs/shared-meta --recompute --bins 30 30
```

## 旧 checkpoint 与 fingerprint

新 fingerprint 显式包含方法，统一旧别名与新入口的写法。读取旧 checkpoint 时仍计算
原版本的精确 fingerprint；不会放宽温度、珠数、CV、偏置、种子或时间步等科学参数的
检查。旧伞形、MetaD、OPES 的坐标、动量、RNG 和偏置历史可继续使用。旧运行的
manifest 必须先通过校验，才会迁移保存的入口写法。更改方法或共享协议会拒绝复用
旧物理状态。

## `rate` 选择 PMF 来源

不指定 `rate.pmf_source`，或者设置为 `"umbrella"`，仍按现有流程采样伞形 PMF。
复用已有结果时可写：

```toml
[rate.pmf_source]
directory = "runs/umbrella-rate-flux"
```

默认只接受伞形 `FESResult`，并要求 `measure="rate_flux"`、一维无量纲的质心
RPMDrate 坐标及 hartree 单位。来源必须带有 manifest、resolved input 和 system。
温度、珠数、势能、体系及反应坐标身份必须一致；来源文件会进入新运行的校验清单。
使用外部 PMF 只跳过偏置采样，受约束母链及无偏回穿子轨迹照常执行。

MetaD/OPES 来源须同时满足 `allow_non_umbrella=true`、结果元数据
`reweighted=true` 及相同的 `rate_flux`/物理身份检查。原始 `-V` 或 WT 缩放偏置会被
拒绝，并明确提示 PMF 必须重加权。此选项没有新增时变 MetaD 重加权器；当前 MetaD
输出的是 `cv_density` 偏置估计，不能直接给出气相速率 PMF。OPES 冻结偏置后的
密度重构也不会自动获得 rate-flux 测度。

逐格误差不足以确定参考点与势垒点的协方差。因此外部来源不伪造 PMF 势垒误差或总
速率误差；这些误差会被标为不可用。

## 回归验证

[`unification_regression.py`](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/fes_methods/unification_regression.py)
对旧 checkout 与新代码运行现有二维伞形、自适应、副本交换、standard/WT/grid MetaD
和 OPES 算例，逐项比较 FES、轨迹、RNG 与偏置历史。它还实际用旧代码制造三种方法的
中断 checkpoint，再由新代码续算并与完整运行对比。单独测试覆盖三维解析高斯格积分、
公共结果、跨方法参数拒绝、PMF 来源及共享 walker 的失败恢复。这些验证说明实现
兼容性，不代替珠数、时间步或长时间采样的物理收敛研究。

算法公式、墙与网格偏置的细节见[英文增强采样规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)。
