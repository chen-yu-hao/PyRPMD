# 输出、分析与结果判读

## 从摘要和运行身份开始

```python
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
print(summary["task"])
print(summary.get("converged"))
print(summary.get("warnings", []))
print(summary.get("files", {}))
```

多数工作流最终摘要在 `results/summary.json`。高层入口统一提供软件版本、输出路径、
preset 与输入诊断；任务自身保留物理量、统计限定和文件索引。
不要只保存一个 scalar 而丢弃单位、未决比例、拟合窗口或归一化约定。

| 路径 | 用途 |
| --- | --- |
| `manifest.json` | 软件/依赖、任务身份、引用文件哈希与原生能力 |
| `input/resolved.json` | 单位与默认值解析后的配置 |
| `input/system.json` | 分子/周期体系；PCET 使用 `input/model.json` |
| `input/original.*` | 来自文件输入时保存的原输入副本 |
| `checkpoints/` | 任务状态、随机流和阶段检查点 |
| `samples/`、工作流数据目录 | 保存的轨迹、窗口或其他任务数据 |
| `results/summary.json` | 首先阅读的任务结果与诊断 |

文件存在不等于整个任务完成。检查摘要中的状态、样本量、失败轨迹和未决结果；
中断后使用 `resume`，不要手工把半成品补成“成功”。

## FES：密度、概率质量与自由能

FES 保存 `fes.npz`、`fes.csv` 和可用的 `fes.h5`；精确路径由摘要的 `files` 给出。
CSV 包含以下列，CV 列数随维度改变：

| 列 | 解释 |
| --- | --- |
| `cv0_bohr` 等 | 网格中心，列名写明坐标单位 |
| `free_energy_hartree` | 基于概率密度的相对自由能 |
| `standard_error_hartree` | 可获得时的 bootstrap 标准误 |
| `probability_mass` | 该网格的概率质量 |
| `density` | 单位 CV 体积上的概率密度 |
| `masked` | 缺少采样支持或诊断不允许估计的区域 |
| `raw_count` | 原始落点计数 |

概率质量与密度相差网格体积。不同分箱、不同 CV 度量和不同 Jacobian 下的曲线不能
未经转换直接比较。`cv_density` 自由能也不能直接代替气相速率所要求的 `rate_flux` 测度。

```python
import numpy as np
from pyrpmd import analyze_run

summary = analyze_run("runs/manual-first")
table = np.genfromtxt(summary["files"]["table"], delimiter=",", names=True)
supported = table["masked"] == 0
print("supported bins:", int(supported.sum()))
print(table["free_energy_hartree"][supported])
```

未采样区域应保留 mask。窗口 overlap 图不连通时，全局相对自由能不可辨识；
应增加桥接窗口或采样，而不是平移各段直到图形看起来连续。

重新分箱不会调用 PES：

```bash
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 20
```

bootstrap 处理保存样本的统计不确定度，不能修复错误势能、未收敛珠子数或缺失反应通道。
没有有效误差条时，`null`/缺失值不能解释为零误差。

## 反应速率：一起读 PMF 与回穿

气相 `rate` 报告 Bennett–Chandler 流程：静态部分与动态传输系数共同形成速率。
必须检查 PMF 测度、分割面、反应物归一化、受约束母链和回穿平台。
`site_rate` / `surface_rate` 使用显式反应物盆地及一个有向边界，输出一阶通道率 `s^-1`，
不能乘上气相双分子因子，也不自动变成每面积、每位点或覆盖度归一化的实验速率。

有限时间估计、负的有符号回穿估计、无平台、失败 bootstrap draw 和缺失联合区间
都需要保留。盆地归一化中，密度与盆地积分在同一个 PMF bootstrap draw 内计算，保留二者协方差。
静态 PMF/QTST 与动态 κ 的联合误差按独立采样假设组合，并保留母链 burn-in 与独立性限定。
这些检查的用途与各工作流实际字段见[工作流](workflows.md)和[材料与表面](periodic.md)。

dRPMD 需要同时阅读 completed、failed、unresolved 事件及分类规则；缺失轨迹不能
全部按非反应处理。没有经验证分类器或碰撞参数时，几何事件计数不等于正式化学截面。
`network` 当前保留各入口/通道/TS 的独立结果，不隐式相加为总速率。

启用实验性 NRPMD 时，另外阅读 `summary["nonadiabatic"]` 和
`results/electronic_channels.csv`。电子 population 是有符号 MMST 估计量，允许为负或大于一；
不能裁剪到 [0,1] 或用 argmax 给轨迹分配一个电子面。存在失败/未决轨迹时，最终电子态分辨
速率/截面会保留为空；已观察到的贡献仍可用于诊断。详见[非绝热 NRPMD](nonadiabatic.md)。

## PIMD 与扩散

PIMD 的 primitive 和 centroid-virial 估计量用于物理热力学能量；环 Hamiltonian
用于积分/传播诊断，两者并非同一个能量。固定晶胞 PIMD 不给出 NPT 平衡晶格常数。

diffusion 保存连续展开的质心位置与速度。MSD 拟合需要真实的长时扩散区；
短时弹道运动也会产生正斜率，但不能因此接受扩散系数。VACF 积分是另一诊断，
有限时间平台和独立副本误差都应检查。

```bash
pyrpmd analyze runs/diffusion --recompute \
  --fit-range '10 fs' '100 fs' --axes 0 1 --atoms 0 --max-lag 100
```

这是已有足够长 diffusion 数据的重分析模板。拟合范围必须落在保存的 lag 时间中，
轴和原子选择必须存在；短教学示例不一定支持这个范围。
可以重新选择已保存原子、轴、拟合区间与 bootstrap，但不能恢复从未保存的时间分辨率。
扩散分析不使用 `--bins`。完整字段和接受/拒绝解释见[材料章节](periodic.md)。

## PCET 与 metadynamics

PCET 保存量子质子谱、波函数、通道贡献和条件扫描结果。普通速率下溢时，应同时读
有限的 log-rate；谱网格、保留态数与尾部检查只评价局域固定模型的数值充分性。
模型外的电极结构、pH、电流密度归一化不能从这份输出中推得。

metadynamics 的 HILLS、偏置与有限时间 bias FES 描述有偏采样历史。
`analyze --recompute` 可以重新评估保存的偏置，当前非零 bootstrap 请求会拒绝。
有偏轨迹时间不能直接当作物理反应时间。

## 保存一次可复现的报告

至少记录代码 commit、安装依赖、势能来源/哈希、输入及单位、运行身份、珠子数与步长、
平衡/生产长度、独立副本、窗口或拟合方案、所有诊断与缺失量。
若比较两个结果，应明确哪些科学参数变了，哪些只是重新分析设置。
依据[科学约定](science.md)完成收敛系列后，再讨论体系的物理预测。
