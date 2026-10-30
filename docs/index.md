# PyRPMD 使用手册

**从一个可运行的输入开始，完成环聚合物采样、动力学与结果分析。**

PyRPMD 是可扩展的 Python 科研计算包，使用统一的 Python/TOML 输入，连接势能面、
环聚合物积分器和可恢复的计算工作流。本手册对应仓库中的 **0.2.0 整合版本**，
覆盖气相反应、自由能、周期材料、固定基底表面和局域 PCET 模型，并介绍指定入射电子态
散射的实验性 NRPMD 扩展。

第一次使用，依次阅读[安装](manual/installation.md)、[快速入门](manual/quickstart.md)
和[输入与计算对象](manual/input.md)。已有体系和势能面时，可直接选择下表中的任务。

| 想计算什么 | `task` | 阅读入口 |
| --- | --- | --- |
| 气相双分子 RPMD 速率 | `rate` | [气相与自由能工作流](manual/workflows.md) |
| 独立直接碰撞与产物事件 | `drpmd` | [气相工作流](manual/workflows.md)；[实验性 NRPMD](manual/nonadiabatic.md) |
| 一维/二维平衡自由能面 | `fes` | [快速入门](manual/quickstart.md)、[工作流](manual/workflows.md) |
| 多入口、多 TS 的独立速率矩阵 | `network` | [气相与自由能工作流](manual/workflows.md) |
| Standard / well-tempered metadynamics | `metadynamics` | [气相与自由能工作流](manual/workflows.md) |
| 平衡核量子统计与能量估计量 | `pimd` | [材料与表面](manual/periodic.md) |
| RPMD 自扩散、MSD、VACF | `diffusion` | [材料与表面](manual/periodic.md) |
| 材料中一个有向位点跃迁率 | `site_rate` | [材料与表面](manual/periodic.md) |
| 固定基底上的一个有向逃逸率 | `surface_rate` | [材料与表面](manual/periodic.md) |
| 弱振电耦合极限的局域电化学 PCET | `pcet` | [PCET](manual/pcet.md) |
| 恒电位有效势、电子数平衡采样与 HER 电流 | `fes` / Python API | [电极模型](manual/electrochemistry.md) |

解析势示例用于学习输入、检验算法和复现程序行为。成功执行、数值检查通过与物理模型适用、
统计及离散化收敛是不同的问题。每一章都解释结果的单位、归一化和使用边界；
正式计算前请阅读[科学约定与收敛](manual/science.md)。

## 学习与使用

以下目录也可直接在 GitHub 上阅读：

| 路线 | 章节 |
| --- | --- |
| 开始使用 | [安装](manual/installation.md) · [快速入门](manual/quickstart.md) · [输入与命令行](manual/input.md) · [科学约定](manual/science.md) |
| 选择计算任务 | [气相与自由能](manual/workflows.md) · [材料与表面](manual/periodic.md) · [PCET](manual/pcet.md) · [非绝热 NRPMD](manual/nonadiabatic.md) · [完整示例索引](manual/examples.md) |
| 连接和运行模型 | [势能与插件](manual/potentials.md) · [执行、并行与续跑](manual/execution.md) · [输出与分析](manual/results.md) |
| 查阅与开发 | [Python API](manual/api.md) · [常见问题](manual/troubleshooting.md) · [开发者指南](manual/development.md) |

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:

manual/installation
manual/quickstart
manual/input
manual/science
```

```{toctree}
:maxdepth: 2
:caption: 用户指南
:hidden:

manual/workflows
manual/periodic
manual/pcet
manual/electrochemistry
manual/nonadiabatic
manual/potentials
manual/execution
manual/results
manual/examples
```

```{toctree}
:maxdepth: 2
:caption: 参考与开发
:hidden:

manual/api
manual/troubleshooting
manual/development
```

## 文档组织与来源

本手册参考 [PySCF User Guide](https://pyscf.org/user/index.html) 和
[Quickstart](https://pyscf.org/quickstart.html) 的层次：安装与最小示例、按计算任务组织的教程、
公共 API、开发者指南；使用 Sphinx、MyST 与 PyData Sphinx Theme 构建可搜索的 HTML。
文字、示例和接口说明依据 PyRPMD 的源码编写。PySCF 是文档组织的参考，
不是本项目已实现的势能适配器名称。

既有[输入规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)、
[公共接口约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)和
[历史验证记录](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)
保留在仓库中。历史报告的提交哈希、样本量和硬件范围属于报告所记载的运行，
不自动成为任何后来版本或用户体系的验证结论。

