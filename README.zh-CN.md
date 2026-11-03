# PyRPMD

**环聚合物分子动力学与可复现的分子模拟工作流。** PyRPMD 将 C++17/OpenMP 数值内核、NumPy 参考实现、严格的势能接口与带检查点的 Python/TOML 工作流结合起来，覆盖反应速率、量子平衡采样、光谱、材料和非绝热模型动力学。数值验证不等于物理收敛。

[English](README.md) · [中文手册](docs/zh/index.md) · [安装](docs/zh/manual/installation.md) · [快速入门](docs/zh/manual/quickstart.md) · [示例](docs/zh/manual/examples.md) · [API](docs/zh/manual/api.md)

## 安装

默认原生构建需要 Python 和 C++17 编译器。

```bash
python -m pip install -e '.[test]'
```

文档环境可安装 `python -m pip install -e '.[docs]'`。GPU、电子结构、MPI 与外部引擎适配器需要额外可选依赖，参见[安装](docs/zh/manual/installation.md)和[势能接口](docs/zh/manual/potentials.md)。不支持的组合会明确报错，不会静默替换设备、精度或引擎。

## 第一次运行

```bash
python -m pyrpmd init my-first-run
python -m pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
python -m pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
python -m pyrpmd analyze my-first-run/run
python -m pyrpmd resume my-first-run/run
```

模板使用解析势能。连接真实势面前请阅读[快速入门](docs/zh/manual/quickstart.md)；科学设置改变必须生成新的运行身份。续算和收敛边界见[执行指南](docs/zh/manual/execution.md)与[科学约定](docs/zh/manual/science.md)。

## 能力表

| 计算目标 | 入口 | 指南 |
| --- | --- | --- |
| 热 RPMD 速率、回穿、PMF、反应网络 | `rate`、`network`、`pyrpmd study` | [气相工作流](docs/zh/manual/workflows.md) · [精度控制（英文）](docs/rate_precision.md) |
| 直接碰撞与振动态选择 | `drpmd` | [气相工作流](docs/zh/manual/workflows.md) |
| 增强采样、自由能面、OPES、ABF 与副本交换 | `fes` | [增强采样](docs/zh/enhanced-sampling.md) · [副本交换](docs/zh/replica-exchange.md) |
| 平衡 PIMD、量子统计及全同粒子交换 | `pimd`、Python API | [材料](docs/zh/manual/periodic.md) · [路径积分作用量（英文）](docs/tier2-path-integral.md) |
| 原生与 Torch 批量 PIMD/扩散 | `sampling.engine` | [原生引擎（英文）](docs/native-pimd-workflow.md) · [Torch（英文）](docs/torch-materials-workflows.md) |
| 扩散、位点、表面、电化学 PCET | `diffusion`、`site_rate`、`surface_rate`、`pcet` | [材料](docs/zh/manual/periodic.md) · [PCET](docs/zh/manual/pcet.md) |
| Instanton、隧穿速率与能级劈裂 | `pyrpmd instanton`、Python API | [Instanton（英文）](docs/instanton.md) |
| 光谱与质心方法 | `spectroscopy`、Python API | [光谱（英文）](docs/tier2-spectroscopy.md) · [f-QCMD（英文）](docs/qcmd.md) |
| 非绝热模型、IESH、MASH、热 NRPMD | `nonadiabatic`、Python API | [非绝热指南](docs/zh/manual/nonadiabatic.md) · [IESH/MASH 边界（英文）](docs/nonadiabatic-iesh-multistate.md) |
| RETIS、REPPTIS、FFS、committor | Python API | [路径采样（英文）](docs/transition-interface-sampling.md) |
| ML/ASE/OpenMM/LAMMPS/PyTorch 势能适配 | Python API 与模型 provider | [势能](docs/zh/manual/potentials.md) · [批量 ML（英文）](docs/batched-ml-potentials.md) |

能力表区分输入文件工作流与仅提供 Python API 的功能；具体支持范围和独立参考验证请以各指南为准。

## 文档与可复现性

[中文手册](docs/zh/index.md)与[英文手册](docs/index.md)的核心章节一一对应。科研引用与复现请查阅[原始论文目录](docs/zh/manual/references.md)、[复现清单](docs/zh/manual/reproduction.md)、[验证与限制（英文）](docs/literature-validation.md)和[发布说明（英文）](docs/releases/0.2.0.md)。测试、CI 与贡献指南见[开发文档](docs/zh/manual/development.md)。

> 软件测试通过、统计误差有限或检查点完成，都不意味着材料性质、反应速率或光谱已达到物理收敛。
