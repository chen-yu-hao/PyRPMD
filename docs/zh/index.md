# PyRPMD 使用手册

[English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/index.md) · **中文**

PyRPMD 将势能面连接到环聚合物分子动力学、反应速率、自由能采样、量子统计和光谱计算。
初次使用请阅读[安装](manual/installation.md)、[快速入门](manual/quickstart.md)和
[输入与命令行](manual/input.md)。

本手册对应源码检出版本。包版本仍为 0.2.0；[下一版本发布说明](releases/unreleased.md)列出
[0.2.0 发布记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/releases/0.2.0.md)之后合入的功能。
[能力与限制](manual/capabilities.md)说明支持的组合与验证证据。短示例用于检查接口；正式计算还需完成
[科学收敛检查](manual/science.md)。下面标注“英文”的链接指向详细技术指南，中文核心章节与英文一一对应。

## 按目标选择工作流

| 计算目标 | 入口 | 指南 |
| --- | --- | --- |
| 热气相 RPMD 速率与反应网络 | `rate`、`network` | [气相与自由能](manual/workflows.md) · [显式速率输入](rate-input-v2.md) · [PMF 选择](rate-pmf-selection.md) |
| 有预算、预先声明统计与数值目标的速率 | `pyrpmd precision plan/run/status` | [精度驱动速率控制（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate_precision.md) |
| 瞬子速率、隧穿劈裂与正常区 golden-rule 速率 | `pyrpmd instanton`、Python API | [瞬子工作流（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md) · [低温归一化（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/low-temperature.md) |
| 热分布或指定振动态碰撞、NE-RPMD 和 QCT | `drpmd`、`state-check`、`extend` | [直接碰撞](manual/workflows.md) |
| 材料位点跳跃与固定基底上的逃逸 | `site_rate`、`surface_rate` | [材料与表面](manual/periodic.md) · [广义坐标速率（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/general-coordinate-rates.md) |
| 伞形 FES、metadynamics、OPES 与副本交换 | `fes` | [增强采样](enhanced-sampling.md) · [副本交换](replica-exchange.md) · [OPES（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/opes.md) |
| ABF、eABF 与 steered 自由能差 | `fes` 的 `abf`、`eabf`、`steered` 方法 | [增强采样（英文详细指南）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md) |
| 有限温度 string 与学习型 CV | Python API / 导出的 CV | [String 采样（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/finite-temperature-string.md) · [mlcolvar（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mlcolvar.md) |
| 预先声明文件重启策略的 PLUMED 偏置 | `plumed` | [PLUMED 桥（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md) |
| TIS、RETIS minus 系综通量、REPPTIS 与直接 FFS | Python API | [路径采样（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md) |
| 平衡 PIMD、原生恒温与恒压 | `pimd`、`sampling.engine="native"`、`pyrpmd-native` | [原生工作流（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md) · [原生系综（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-ensembles.md) |
| GLE/PIGLET 参数导入、生成和估计量核验 | Python API / 恒温器参数 | [凝聚相量子统计（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md) |
| 设备驻留 PIMD 与 RPMD 输运 | `pimd` / `diffusion`、`sampling.engine="torch"` | [张量工作流（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/torch-materials-workflows.md) |
| Suzuki–Chin、玻色/费米交换、周期 winding 与交换 NPT、PIGS | `pimd` + `[path_integral]` | [路径积分作用量（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-path-integral.md) · [量子估计量（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-statistics.md) |
| 自扩散、MSD 与 VACF | `diffusion` | [材料](manual/periodic.md) · [统一传播（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/propagation-stack.md) |
| TRPMD、PA-CMD、二次型 f-CMD 与实验性分子 f-QCMD 光谱 | `spectroscopy` | [属性光谱（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-spectroscopy.md) · [f-QCMD（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/qcmd.md) · [TRPMD（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/trpmd-spectroscopy.md) |
| 映射动力学、表面跳跃、热速率与电子摩擦 | `nonadiabatic`、`nonadiabatic-rate`、Python API | [布居工作流（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-workflow.md) · [方法比较（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-comparison.md) |
| 经典 IESH 与多态 MASH | `nonadiabatic`、P=1 | [IESH 与多态 MASH（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-iesh-multistate.md) |
| PySCF SA-CASSCF 电子数据与 spin–boson HEOM 对照 | Python API | [电子结构与 HEOM（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-nonadiabatic.md) |
| PCET、恒电势模型与受限非绝热路径 | `pcet`、`fes`、Python API | [PCET](manual/pcet.md) · [电化学](manual/electrochemistry.md) · [自适应路径](manual/adaptive-nrpmd.md) |
| 批量 ML 势、OpenMM 和 LAMMPS | 势能 provider | [批量 ML（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-ml-potentials.md) · [Provider 接口](manual/potentials.md) · [实测性能与限制（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/t19-performance.md) |
| 快速解析势、批量轨迹与 RPC/MTS | 原生绑定 / Python API | [内置原生势（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/builtin-native-potentials.md) · [批量（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-dynamics.md) · [RPC/MTS（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rpc-mts.md) |
| PES 拟合与可恢复的采样—标注—训练循环 | `python -m pyrpmd.pesgen`、Python API | [势能生成](manual/pes-generation.md) · [JaxPIP](manual/jaxpip-generation.md) · [主动学习（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-ml-lifecycle.md) |
| FES 重分析、速率后处理、轨迹导出与报告 | `analyze`、`project`、`rate-postprocess`、`report` | [结果](manual/results.md) · [速率后处理（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-postprocessing.md) · [轨迹导出（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/trajectory-export.md) · [报告](manual/analysis-reports.md) |

方法出处见[原始论文目录](manual/references.md)。[复现清单](manual/reproduction.md)保留其固定源码版本上的
研究计划；当前支持的组合以各功能指南为准。

## 基本循环

大多数输入工作流使用同一组命令，每个新计算使用独立的输出目录。PLUMED 自动续跑要求在首次运行前
设置 `restart="files"`；精度控制 campaign 和热非绝热速率各有独立命令与输入格式。
需要先创建可运行输入时，使用 `pyrpmd init my-first-run` 并按[快速入门](manual/quickstart.md)操作。
下面的 `job.inp` 是你自己的输入文件占位符。

```bash
pyrpmd validate job.inp --check-pes     # 检查输入与单位，构造势能
pyrpmd run job.inp --output runs/job    # 运行；结果位于 runs/job/results/summary.json
pyrpmd analyze runs/job                 # 读取摘要；三种 FES 方法、扩散及非绝热布居支持 --recompute
pyrpmd resume runs/job                  # 从检查点续跑，科学设置保持不变
pyrpmd report runs/job                  # 可选：由保存数据生成图、index.html 与 report.json
```

命令索引与常用选项见[输入与命令行](manual/input.md#命令行参考)；`pyrpmd COMMAND --help` 列出当前安装版本的全部选项。
对应的 Python 入口是 `prepare`、`run`、
`analyze_run` 和 `resume_run`，见[快速入门](manual/quickstart.md)。

## 章节

| 主题 | 章节 |
| --- | --- |
| 开始使用 | [安装](manual/installation.md) · [快速入门](manual/quickstart.md) · [输入与命令行](manual/input.md) · [科学约定](manual/science.md) · [能力与限制](manual/capabilities.md) |
| 计算任务 | [气相与自由能](manual/workflows.md) · [材料与表面](manual/periodic.md) · [PCET](manual/pcet.md) · [电极模型](manual/electrochemistry.md) · [非绝热 NRPMD](manual/nonadiabatic.md) · [示例](manual/examples.md) |
| 模型与运行 | [势能与插件](manual/potentials.md) · [势能生成](manual/pes-generation.md) · [JaxPIP 拟合](manual/jaxpip-generation.md) · [执行、并行与续跑](manual/execution.md) · [输出与分析](manual/results.md) · [自动分析与出图](manual/analysis-reports.md) |
| 动力学扩展（英文指南） | [统一传播栈](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/propagation-stack.md) · [昂贵势能](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/performance-dynamics.md) · [批量轨迹](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-dynamics.md) · [RPC/MTS](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rpc-mts.md) · [混合采样](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mixed-sampling.md) · [TRPMD 与谱](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/trpmd-spectroscopy.md) |
| 参考 | [原始论文](manual/references.md) · [复现清单](manual/reproduction.md) · [Python API](manual/api.md) · [常见问题](manual/troubleshooting.md) · [开发者指南](manual/development.md) · [下一版本](releases/unreleased.md) |

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:

安装 <manual/installation>
快速入门 <manual/quickstart>
输入与命令行 <manual/input>
科学约定 <manual/science>
能力与限制 <manual/capabilities>
```

```{toctree}
:maxdepth: 2
:caption: 用户指南
:hidden:

气相与自由能 <manual/workflows>
速率 PMF 方法选择 <rate-pmf-selection>
显式速率输入与分区伞窗口 <rate-input-v2>
统一增强采样 <enhanced-sampling>
材料与表面 <manual/periodic>
PCET <manual/pcet>
电极模型 <manual/electrochemistry>
非绝热 NRPMD <manual/nonadiabatic>
自适应恒电势 NRPMD <manual/adaptive-nrpmd>
势能与插件 <manual/potentials>
势能生成 <manual/pes-generation>
JaxPIP 拟合与原生执行 <manual/jaxpip-generation>
执行与并行 <manual/execution>
输出与分析 <manual/results>
自动分析与出图 <manual/analysis-reports>
示例索引 <manual/examples>
副本交换伞形采样 <replica-exchange>
```

```{toctree}
:maxdepth: 2
:caption: 参考与开发
:hidden:

原始论文与实现 <manual/references>
论文体系与复现 <manual/reproduction>
Python API <manual/api>
常见问题 <manual/troubleshooting>
开发者指南 <manual/development>
下一版本发布说明 <releases/unreleased>
```

## 关于本手册

手册有[英文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/index.md)与中文两个版本，核心章节一一对应；
HTML 版本中，导航栏的语言按钮会打开同一页面的另一语言版本。目前仅有英文版的扩展指南和验证报告，其语言按钮会打开中文手册目录。
[输入规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)、
[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)等详细规范和
[验证记录](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)与手册一同保存在仓库中。
