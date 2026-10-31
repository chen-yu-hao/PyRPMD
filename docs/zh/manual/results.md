# 输出、分析与结果判读

## 摘要与运行身份

```python
from pyrpmd import analyze_run

summary = analyze_run("my-first-run/run")
print(summary["task"])
print(summary.get("converged"))
print(summary.get("warnings", []))
print(summary.get("files", {}))
```

每个工作流都把最终摘要写到 `results/summary.json`。高层入口加入软件版本、输出路径、preset 和输入诊断；任务本身加入
物理结果、统计量和文件索引。报告结果时一并给出单位、未决比例、拟合区间和归一化。

| 路径 | 内容 |
| --- | --- |
| `manifest.json` | 软件与依赖、任务身份、引用文件的哈希、原生能力 |
| `input/resolved.json` | 解析了单位和缺省值的配置 |
| `input/system.json` | 分子或周期体系；PCET 使用 `input/model.json` |
| `input/original.*` | 原输入文件的副本 |
| `checkpoints/` | 任务状态、随机流与阶段检查点 |
| `samples/`、工作流数据目录 | 保存的轨迹、窗口和其他任务数据 |
| `summary.json`（运行目录顶层） | 工作流写出的摘要；未完成的 network 写出 `network-progress.json` |
| `results/summary.json` | 工作流摘要加上软件版本、输出路径、preset 与输入提示；`analyze` 优先读取它 |

检查摘要中的 `status`、样本量以及失败或未决的数量；中断的运行用 `pyrpmd resume` 继续。

## FES：密度、概率质量与自由能

FES 写出 `fes.npz`、`fes.csv` 和 `fes.h5`；摘要中的 `files` 给出路径。CSV 每个 CV 一个坐标列，并含以下列：

| 列 | 含义 |
| --- | --- |
| `cv0_bohr` 等 | 网格中心；列名带单位 |
| `free_energy_hartree` | 由概率密度得到的相对自由能 |
| `standard_error_hartree` | 计算时给出的 bootstrap 标准误 |
| `probability_mass` | 该格的概率质量 |
| `density` | 单位 CV 体积上的概率密度 |
| `masked` | 没有采样支持或被诊断排除的格子 |
| `raw_count` | 该格的样本数 |

概率质量与密度相差格子体积。在相同的 CV、分箱和测度下比较 FES 曲线；气相速率使用单独的 `rate_flux` PMF。

```python
import numpy as np
from pyrpmd import analyze_run

summary = analyze_run("my-first-run/run")
table = np.genfromtxt(summary["files"]["table"], delimiter=",", names=True)
supported = table["masked"] == 0
print("supported bins:", int(supported.sum()))
print(table["free_energy_hartree"][supported])
```

窗口重叠图不连通时，增加桥接窗口或采样，把各段连接成一个自由能面。

重新分箱复用保存的样本，不调用 PES：

```bash
pyrpmd analyze my-first-run/run --recompute --bins 8 --bootstrap 20
```

bootstrap 给出保存样本的统计误差。没有有效误差时该字段为 `null`。

## 反应速率：PMF 与回穿

气相 `rate` 遵循 Bennett–Chandler 流程：静态 PMF/QTST 因子与动态传输系数共同给出速率。检查 PMF 测度、分割面、
反应物归一化、受约束母链和回穿平台。`site_rate` 与 `surface_rate` 使用显式反应物盆地和一个有向边界，给出单位为
`s^-1` 的一阶速率。

摘要按实际情况报告有限时间估计、有符号的 κ、平台检测、失败的 bootstrap 抽样和缺失的联合区间。在盆地归一化中，
每次 PMF bootstrap 抽样同时计算密度与盆地积分以保留协方差。联合误差把 PMF/QTST 与 κ 的 bootstrap 作为独立样本组合，
母链 burn-in 与独立性假设记录在 `rate_uncertainty` 中。各工作流的字段名见[工作流](workflows.md)和[材料与表面](periodic.md)。

对 dRPMD，把 `counts` 中的 product、nonreactive、unresolved 和 failed 计数与 `results/channels.csv` 及分类规则一起阅读。
未决和失败的轨迹与非反应轨迹分开计数，化学速率和截面需要显式的分类规则。`network` 分别报告每个入口、通道和 TS。

启用 NRPMD 时，另读 `summary["nonadiabatic"]` 和 `results/electronic_channels.csv`。电子布居是有符号的 MMST 估计量，
可以落在 [0,1] 之外。有轨迹失败或未决时，最终的态分辨速率或截面为空，同时报告已观察到的贡献。见[非绝热 NRPMD](nonadiabatic.md)。

## PIMD 与扩散

PIMD 的 primitive 和 centroid-virial 估计量给出物理热力学能量；环 Hamiltonian 是传播诊断。PIMD 缺省使用固定晶胞，
显式设置受支持的恒压器时使用 NPT；请阅读报告的系综，NPT 运行还需检查保存的体积与压力。

扩散保存连续、展开的质心位置和速度。MSD 拟合需要长时扩散区，`analysis_status` 标出弹道运动和不稳定的拟合。
VACF 积分和副本间误差提供另外两项检查。

```bash
pyrpmd analyze runs/diffusion --recompute \
  --fit-range '10 fs' '100 fs' --axes 0 1 --atoms 0 --max-lag 100
```

这是对数据充足的扩散运行进行重分析的模板；拟合范围须在保存的延迟时间之内，所选轴和原子须存在。重分析可以重新选择
保存的原子、轴、拟合区间和 bootstrap；时间分辨率由保存间隔决定。（`--bins` 用于 FES 和 metadynamics。）
字段与状态说明见[材料章节](periodic.md)。

## PCET 与 metadynamics

PCET 保存质子谱、波函数、通道贡献和扫描结果。速率下溢为 0 时读取 log 速率。谱网格、态数和尾部检查衡量局域模型的数值收敛。

Metadynamics 保存 HILLS、偏置和有限时间偏置 FES。`analyze --recompute` 重新评估保存的偏置；bootstrap 须保持为 0。

## 可复现的报告

记录代码 commit、已安装依赖、势能来源与哈希、输入与单位、运行身份、珠子数与时间步、平衡与生产长度、独立副本、
窗口或拟合方案、每项诊断和每个缺失量。比较两个结果时，说明哪些科学参数变了、哪些只是重分析设置。报告物理预测之前，
先完成[科学约定](science.md)中的收敛系列。
