# 气相反应、自由能与增强采样

本章介绍 `rate`、`drpmd`、`fes`、`network` 和 `metadynamics`。从仓库根目录运行命令，每个计算使用新的输出目录。
周期材料与固定表面见[材料与表面](periodic.md)，局域电化学见 [PCET](pcet.md)，单位与归一化见[科学概念](science.md)。

## 气相 RPMD 速率：`rate`

对两个远离的反应物片段和一个过渡态参考结构，`rate` 执行 Bennett–Chandler 流程：准备反应坐标、采样伞 PMF、
生成受约束母构型，并发射成对的无偏回穿子轨迹。速率为 $k_{\mathrm{RPMD}}=k_{\mathrm{QTST}}\kappa$。

### 运行示例

```bash
pyrpmd validate examples/rate/pilot.inp --check-pes
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/manual-rate
pyrpmd analyze runs/manual-rate
pyrpmd resume runs/manual-rate
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/rate/pilot.inp)使用解析势 `toy_exchange.py`，
从 `ts.xyz` 开始，用 4 个珠子和较短的采样，约 15 秒完成。`prepare --pilot` 执行数值准备。

只给出 TS 时，准备程序检查力、找到数值 Hessian 的不稳定模，并沿两个方向下降以识别两个入口片段。通道有歧义时，
显式给出 `reaction.fragments`、`forming_bonds` 和 `breaking_bonds`。

### 设置自己的体系

| 输入部分 | 决定的内容 | 检查 |
| --- | --- | --- |
| `system.transition_state` | TS 参考几何与原子顺序 | 与 PES、同位素和成断键一致 |
| `reaction.fragments`、成断键 | 入口片段与反应坐标 | 0 起始索引、原子守恒、通道定义 |
| `reaction.rinf` | 远区归一化距离 | 改变时片段是否解耦、长程势的影响 |
| `umbrella` | 窗口中心、范围、刚度 | 从反应物参考点到分割面的连通采样支持 |
| `sampling` | 每窗口的平衡、生产、保存间隔与副本 | 分布形状、有效样本数、独立重复 |
| `rate.pmf_method` | `ui` 或 `mbar` | UI 的高斯窗口假设；MBAR 的 bin 与支持 |
| `recrossing` | 母链数、采样间隔、child pairs、传播时间 | 独立母构型、稳定平台、能量误差 |

速率 PMF 使用 `rate_flux` 测度，含质量加权的通量度量；通用 `fes` 使用 `cv_density`。
`rate.pmf_method="mbar"` 在同一测度下切换 PMF 估计器。用 `recrossing.dividing_surface` 固定分割面；否则工作流把它
放在 `rate.pmf_search_range`（缺省为伞中心范围）内的 PMF 最大值处。

母链在分割面上受约束采样。每对子轨迹抽取无约束的 Maxwell 动量，以 `+p` 和 `-p` 发射，不加恒温器或伞势。

### 输出

| 文件 / 摘要字段 | 含义 |
| --- | --- |
| `preparation.json` | TS 准备：Hessian 本征值、下降端点、识别的反应与诊断 |
| `rate_pmf.csv`、`rate_pmf.npz` | 速率 PMF、支持区域及相关数组 |
| `rate_pmf.json` | PMF 势垒及误差、$k_{\mathrm{QTST}}$、分割面 `xi_star` 与诊断 |
| `recrossing.csv`、`recrossing.npz` | 完整 $\kappa(t)$ 曲线、误差与 bootstrap 信息 |
| `umbrella/` | 各窗口采样、检查点与中间分析 |
| `results/summary.json` | 运行摘要、警告与诊断 |
| `kappa`、`plateau_detected` | 末端传输系数及是否找到平台 |
| `rate_estimate_cm3_s` | 有限时间速率估计 |
| `rate_cm3_s` | 平台与能量诊断通过时的速率，否则为 `null` |
| `rate_standard_error_cm3_s`、`rate_percentile_95_ci_cm3_s` | 由 PMF 与回穿得到的联合采样误差和百分位区间 |
| `conditional_rate_standard_error_cm3_s` | 固定 QTST 因子时的回穿误差 |
| `rate_uncertainty` | 误差是否可用、有效 bootstrap 次数与限定说明 |

速率单位为 `cm^3 molecule^-1 s^-1`。若 PMF 重叠不连通或分割面缺少采样支持，先增加窗口或采样，再读取速率。

数学定义见[速率约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md)和
[UI / MBAR 说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md)。
[H₃/BKMP2 基准](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md)在已发表的反应势能面上运行完整工作流。

## 直接碰撞：`drpmd`

`drpmd` 对两个反应物采样母链，从中发射独立碰撞并对产物分类。发射分布决定结果是热速率还是固定能量截面。
在耦合矩阵势上从指定入射 diabatic 电子态出发的碰撞，使用实验性的 [NRPMD 扩展](nonadiabatic.md)。

```bash
pyrpmd validate examples/drpmd/free_scattering.inp --check-pes
pyrpmd run examples/drpmd/free_scattering.inp --output runs/manual-scattering
pyrpmd analyze runs/manual-scattering
pyrpmd resume runs/manual-scattering
```

[自由散射例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/free_scattering.inp)使用 `V=0`，
所有碰撞都是非反应的；它检验发射、传播、完成条件与续跑。

`collision.trajectories` 是所有母链共享的碰撞总数；例子把 5 次碰撞按 3+2 分给两条母链。`sampling.equilibration`
是每条母链的首次热化，`sampling.interval` 是同一母链两次发射之间推进的时间。显式设置 `sampling.parents`，
可在改变工作进程数时保持任务身份。

| `collision.sampling` | 发射分布 | 结果 |
| --- | --- | --- |
| `thermal_flux` | 相对能量 $E\sim\mathrm{Gamma}(2,k_BT)$ | 热通量平均速率 |
| `maxwell` | 相对速度的 Maxwell 分布 | 按每条轨迹速度加权的热速率 |
| `fixed_energy` | 显式 `collision.energy` | 该碰撞能下的截面 |

`system.separation` 是初始质心径向距离 R0，包含碰撞参数的贡献；`collision.bmax` 须小于 R0。碰撞参数在 $b^2$
上均匀采样。R0、bmax 和最大传播时间分别做收敛。

内置连通图分类器使用键长阈值、分离条件与持续时间。同分子式异构体和同种元素之间的交换通道需要通过
`outcomes.factory` 提供自定义分类器。没有显式规则时，PyRPMD 按共价半径推断阈值，把结果标为候选，
并且不报告化学速率和截面。

在 `results/summary.json` 中先看 `counts`、`all_trajectories_resolved` 和 `classification`，再看 `total_reaction`
和 `channels`。通道热速率在 `rate_cm3_molecule_s` 中，固定能量截面在 `cross_section_bohr2` 中。未决和失败的轨迹
单独计数，给出观察下界与可能上界。`parent_cluster_bootstrap95` 计入同一母链轨迹之间的相关性。

`results/channels.csv` 逐通道列出计数、点估计与上下界。运行目录还保存 `states/initial-*.npz`、`trajectories/`
和检查点，可以检查每条轨迹的初态与分类。见 [dRPMD 示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/README.md)。

不写 `[collision]` 时，pilot 以 `max_time="100 fs"` 和 `bmax=R0/4` 运行 4 条 `thermal_flux` 轨迹，不报告热速率；
缺省值见[输入](input.md#缺省值)。

### 振动初态选择

`initial_ensemble.method = "ne_rpmd"` 从谐振环聚合物分布出发，在质心正常模上加入随机相位的振动激发。
`method = "qct"` 准备单珠子的准经典对照。两者都独立准备初态、采用固定碰撞能，并跳过热母链采样。
这是实验性的入射振动态近似，首版只支持片段质心角动量为零；产物不自动获得严格的转振量子态标签。

```bash
pyrpmd validate examples/state_selected/ne_rpmd.inp
pyrpmd state-check examples/state_selected/ne_rpmd.inp --time '10 fs' --samples 8 --output runs/state-check.json
pyrpmd run examples/state_selected/ne_rpmd.inp --output runs/selected
pyrpmd extend runs/selected --max-time '4 fs'
```

NE-RPMD 必须显式提供 `initial_ensemble.beta`，并省略顶层 `temperature`：这里的 beta 是初态分布和环弹簧参数，
不代表选定振动态气体的真实温度。输入各片段 Hessian 或质量加权正常模、实际同位素质量和 `vibrational_quanta`。
QCT 要求一个珠子，NE-RPMD 至少两个。仓库示例使用不发生交换反应的 Morse 教学势，只验证程序流程。

`state-check` 不赋碰撞速度，在完整 PES 上传播分离反应物，并在稀疏观测帧进行正常模投影。
`extend` 只提高轨迹从发射开始的总时间上限，保留未结束轨迹的完整 q/p 和分类持续性状态；
已结束和数值失败轨迹不会重新发射。所有已发射轨迹始终留在截面分母中，未知结局上下界与满足条件的统计置信区间分别报告。
原生 PES 的实际碰撞仍由现有 C++ 连续 block 传播。

公式、文献与适用边界见[态选择方法说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/state-selected.md)，
可运行配置及原生 PES 用法见[示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/state_selected/README.md)。

## 一维与二维自由能：`fes`

`fes` 在一个或两个 CV 上做伞抽样，用 MBAR 去偏置，得到这些坐标上的概率密度自由能。

```bash
pyrpmd validate examples/fes/harmonic2d.inp --check-pes
pyrpmd run examples/fes/harmonic2d.inp --output runs/manual-fes
pyrpmd analyze runs/manual-fes --recompute --bins 20 20 --bootstrap 50
```

[二维谐振子输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic2d.inp)定义质心 x/y 两个 CV、
3×3 伞网格和每窗口两个副本。`umbrella.stiffness` 可以是标量、对角数组或完整对称正定矩阵。窗口应覆盖要分析的整个区域。

结果保存在 `fes.csv`、`fes.npz` 和 `fes.h5` 中，含 `free_energy`、`probability_mass`、`density`、bin 边界、计数与 `mask`。
检查重叠是否连通，以及各时间序列的有效样本数和加权 ESS。无采样支持的格子被 mask（NaN），
`overlap_connected=false` 表示部分窗口组之间没有共同的归一化。

`analyze --recompute` 从保存的窗口检查点重建分析，可更换 bin、计数门槛或 bootstrap 设置，不调用 PES。

### 自适应伞抽样

自适应工作流依据 pilot 的重叠和有效样本数插入窗口并延长 pilot，然后冻结窗口，用新的独立随机流运行生产段：

```bash
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/manual-adaptive
pyrpmd resume runs/manual-adaptive
pyrpmd analyze runs/manual-adaptive --recompute --bins 30 --bootstrap 20
```

`adaptive-plan.json` 记录每一轮的决定、窗口、随机种子与校验值。生产段只使用自己的样本。若 `max_rounds` 或
`max_windows` 用尽时仍未达到目标，任务不启动生产段而停止，并在计划中记录 `failure_reason`。

自适应伞抽样适用于 `fes`、表面 `fes` 和 `rate` PMF，在声明的区域内细化窗口；`surface_rate` 和 `site_rate`
使用固定窗口。参数见[增强采样规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)。

## 多入口、多过渡态：`network`

`task="network"` 把多个气相双分子速率问题作为一个可续跑的计算运行。每条路径各自完成 PMF 与回穿，并保留自己的
入口归一化、PES、TS 几何和误差。

```bash
pyrpmd prepare examples/network/two_entrances.py --pilot
pyrpmd run examples/network/two_entrances.py --output runs/manual-network
pyrpmd analyze runs/manual-network
pyrpmd resume runs/manual-network
```

[例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/network/two_entrances.py)有两个入口，每个入口两个解析
TS/PES。所有路径共享有序元素、同位素质量、电荷与多重度。每个入口可设置自己的片段和 `rinf`；每个 TS 可设置自己的
结构、成断键、PES 和对称因子。不同化学计量或同位素组成放在不同任务中。

PES 按全局 → 入口 → TS 继承，更具体的 `potential` 表整体替换上一级。采样和速率参数表可在每一级覆盖。
`parallel` 和分块设置属于共享配置。

结果按 `matrix[entrance][channel]["transition_states"][ts_id]` 索引。每条路径在 `paths/<entrance-id>/<ts-id>/`
中运行，`network-progress.json` 记录已完成的路径；续跑时复用它们并继续其余路径。

不同分割面的通量可能重叠，因此 PyRPMD 分别报告每个 TS（`aggregation="separate"`），`total_rate_cm3_s` 和
`summed_rate_cm3_s` 保持 `null`。见[反应网络定义](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/reaction-networks.md)。

## Metadynamics

`task="metadynamics"` 在一维/二维 CV 上支持 standard 和 well-tempered 模式，多个副本各自积累偏置历史。
CV 可以声明周期。该工作流用于不含 PBC 和固定原子的分子体系。

```bash
pyrpmd validate examples/metadynamics/well_tempered_2d.inp
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/manual-meta
pyrpmd analyze runs/manual-meta --recompute --bins 30 30
pyrpmd resume runs/manual-meta
```

| 参数 | 含义 |
| --- | --- |
| `metadynamics.cvs` | 一个或两个已声明 CV 的名称 |
| `mode` | `standard` 或 `well_tempered` |
| `sigma` | 各 CV 的 Gaussian 宽度，带单位 |
| `height` | hill 高度 |
| `pace` | 两次沉积之间完成的生产步数 |
| `bias_factor` | WT 模式的 $\gamma>1$；standard 模式省略 |
| `analysis.method` | `bias_estimate` |
| `analysis.bootstrap` | 0 |

hills 从生产段开始：第 `pace` 步完成后沉积第一个 hill，同一步内的所有力评估看到同一份历史。续跑时以
`checkpoints/metadynamics_XXXX.npz` 中的偏置历史为准，`replicas/XXXX/HILLS.csv` 由它重建。

standard 模式的自由能估计为 $F\approx -V+C$，WT 模式为 $F\approx-\gamma V/(\gamma-1)+C$。`fes.npz` / `fes.csv`
保存有限时间估计与支持 mask。各副本在共同支持区域上对齐后比较，`replica_spread` 给出副本之间的离散程度。

历史偏置在 Python 时间步循环中运行并使用编译子步骤，因此拒绝 `loop_backend="native"`；计算量随 hill 数增长。
Metadynamics 只输出自由能估计。方法与续跑时序见[增强采样规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)。
