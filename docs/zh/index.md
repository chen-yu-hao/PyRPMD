# PyRPMD 使用手册

[English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/index.md) · **中文**

PyRPMD 用一个 Python 或 TOML 输入运行环聚合物分子动力学。它把势能面、环聚合物积分器与可续跑的
工作流连接起来，覆盖气相反应速率、直接碰撞、自由能面、增强采样、周期材料、带固定层的表面以及
局域电化学 PCET；实验性 NRPMD 扩展处理从指定入射电子态出发的散射。本手册对应 0.2.0 版本。

初次使用，请阅读[安装](manual/installation.md)、[快速入门](manual/quickstart.md)和
[输入与命令行](manual/input.md)。已有体系和势能面时，直接进入对应任务：

| 计算目标 | `task` | 章节 |
| --- | --- | --- |
| 气相双分子 RPMD 速率 | `rate` | [气相与自由能](manual/workflows.md) |
| 直接碰撞与产物通道 | `drpmd` | [气相](manual/workflows.md)；[NRPMD](manual/nonadiabatic.md) |
| 一维/二维自由能面 | `fes` | [快速入门](manual/quickstart.md)、[气相与自由能](manual/workflows.md) |
| 多入口、多过渡态的速率矩阵 | `network` | [气相与自由能](manual/workflows.md) |
| Standard 或 well-tempered metadynamics | `metadynamics` | [气相与自由能](manual/workflows.md) |
| 平衡核量子统计与能量估计量 | `pimd` | [材料与表面](manual/periodic.md) |
| RPMD 自扩散、MSD、VACF | `diffusion` | [材料与表面](manual/periodic.md) |
| 材料中的有向位点跃迁 | `site_rate` | [材料与表面](manual/periodic.md) |
| 固定基底上的有向逃逸 | `surface_rate` | [材料与表面](manual/periodic.md) |
| 弱振电耦合极限下的电化学 PCET | `pcet` | [PCET](manual/pcet.md) |

示例使用解析模型势，几秒内即可完成。进行正式研究前，请按[科学约定与收敛](manual/science.md)完成收敛流程。

## 基本循环

所有任务使用同样四条命令，每个新计算使用独立的输出目录。

```bash
pyrpmd validate job.inp --check-pes     # 检查输入与单位，构造势能
pyrpmd run job.inp --output runs/job    # 运行；结果位于 runs/job/results/summary.json
pyrpmd analyze runs/job                 # 读取摘要；FES、metadynamics、扩散支持 --recompute
pyrpmd resume runs/job                  # 从检查点续跑，科学设置保持不变
```

全部选项见[输入与命令行](manual/input.md#命令行参考)。对应的 Python 入口是 `prepare`、`run`、
`analyze_run` 和 `resume_run`，见[快速入门](manual/quickstart.md)。

## 章节

| 主题 | 章节 |
| --- | --- |
| 开始使用 | [安装](manual/installation.md) · [快速入门](manual/quickstart.md) · [输入与命令行](manual/input.md) · [科学约定](manual/science.md) |
| 计算任务 | [气相与自由能](manual/workflows.md) · [材料与表面](manual/periodic.md) · [PCET](manual/pcet.md) · [非绝热 NRPMD](manual/nonadiabatic.md) · [示例](manual/examples.md) |
| 模型与运行 | [势能与插件](manual/potentials.md) · [执行、并行与续跑](manual/execution.md) · [输出与分析](manual/results.md) |
| 参考 | [Python API](manual/api.md) · [常见问题](manual/troubleshooting.md) · [开发者指南](manual/development.md) |

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:

安装 <manual/installation>
快速入门 <manual/quickstart>
输入与命令行 <manual/input>
科学约定 <manual/science>
```

```{toctree}
:maxdepth: 2
:caption: 用户指南
:hidden:

气相与自由能 <manual/workflows>
材料与表面 <manual/periodic>
PCET <manual/pcet>
非绝热 NRPMD <manual/nonadiabatic>
势能与插件 <manual/potentials>
执行与并行 <manual/execution>
输出与分析 <manual/results>
自动分析与出图 <manual/analysis-reports>
示例索引 <manual/examples>
```

```{toctree}
:maxdepth: 2
:caption: 参考与开发
:hidden:

Python API <manual/api>
常见问题 <manual/troubleshooting>
开发者指南 <manual/development>
```

## 关于本手册

手册有[英文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/index.md)与中文两个版本，章节一一对应；
HTML 版本中，导航栏的语言按钮会打开同一页面的另一语言版本。[输入规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)、
[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)等详细规范和
[验证记录](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)与手册一同保存在仓库中。
