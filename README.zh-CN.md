# PyRPMD

**[使用手册](docs/zh/index.md)** · [快速入门](docs/zh/manual/quickstart.md) ·
[示例](docs/zh/manual/examples.md) · [Python API](docs/zh/manual/api.md) ·
[English](README.md)

PyRPMD 提供路径积分采样与环聚合物分子动力学，面向反应速率、散射、自由能、材料、输运和振动光谱。
一个 Python 或 TOML 输入即可把势能面接入工作流，并提供严格的单位检查、诊断记录与可续跑计算。
C++17/OpenMP、NumPy 和显式选择的 Torch 路径分别支持相应的计算任务。

第一次使用请运行下面附带的谐振子示例。开展研究计算时，先从能力表选择方法，
再按对应指南检查模型、采样与收敛条件。

<a id="installation"></a>

## 安装

原生构建需要 **Python 3.11+** 和 **C++17 编译器**；Linux 默认启用 OpenMP。
安装器会安装 NumPy、SciPy、h5py 和 threadpoolctl 等依赖。

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pyrpmd info
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。
[安装指南](docs/zh/manual/installation.md)介绍编译器选择、显式 NumPy 构建、MPI 与可选引擎。
按计算需要安装扩展，例如分析工具 `'.[analysis]'`、力引擎 `'.[openmm]'` / `'.[lammps]'`，
或配合适当 PyTorch 发行版的 `'.[gpu]'`。机器学习框架的专用扩展与模型要求见
[批量势能指南](docs/batched-ml-potentials.md)。

<a id="first-run"></a>

## 第一次运行

下面的一维伞形采样使用内置谐振势，安装后可从任意工作目录运行：

```bash
python -m pyrpmd init my-first-run
python -m pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
python -m pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
python -m pyrpmd analyze my-first-run/run
python -m pyrpmd resume my-first-run/run
```

结果在 `my-first-run/run/results/summary.json`：示例有 3 个伞窗口、6 条采样流和 120 个保存样本。
它用于检查工作流，`converged` 保持 false。`resume` 继续中断的计算；已经完成的计算保持完成状态。
改变科学设置时应使用新的输出目录。[快速入门](docs/zh/manual/quickstart.md)解释输入、保存文件与重新分析；
[命令行参考](docs/zh/manual/input.md)列出各命令及其不同的输入格式。

<a id="capabilities"></a>

## 能力

表中区分输入工作流与 Python API。两种功能分别可用，并不代表所有组合都受支持；
具体限制见对应指南。[能力与验证指南](docs/zh/manual/capabilities.md)集中说明方法边界、检查点规则与结果依据。
扩展方法的详细指南目前主要为英文，核心手册提供对应中文章节。

| 计算目标 | 入口与范围 | 指南 |
| --- | --- | --- |
| 反应速率与散射 | `rate`、`drpmd`、`network`、`site_rate`、`surface_rate`；PMF 与回穿、直接碰撞、态选择 NE-RPMD/QCT | [反应工作流](docs/zh/manual/workflows.md) · [态选择散射](docs/state-selected.md) · [材料与表面](docs/zh/manual/periodic.md) |
| 速率精度与参数研究 | `precision plan/run/status` 提供有预算、可续算的热速率生产；`study` 扫描温度、同位素与数值设置 | [精度控制](docs/rate_precision.md) · [速率研究](docs/rate-studies.md) |
| 自由能与增强采样 | `fes`：伞形采样、metadynamics、OPES、ABF/eABF 与 steered MD；REUS、学习型 CV 和有限温度 string 使用各自的控制项/API | [增强采样](docs/zh/enhanced-sampling.md) · [CV 与采样](docs/tier2-sampling.md) · [String](docs/finite-temperature-string.md) |
| 外部偏置与稀有事件路径 | PLUMED 质心偏置，可显式启用文件续跑；Python TIS/RETIS 含 minus 系综、REPPTIS 和直接 FFS | [PLUMED](docs/plumed.md) · [路径采样](docs/transition-interface-sampling.md) |
| 平衡 PIMD 与输运 | `pimd`、`diffusion`；原生恒温器及支持的 NPT 恒压器、能量估计量、MSD/VACF 与流式轨迹 | [材料](docs/zh/manual/periodic.md) · [原生 PIMD](docs/native-pimd-workflow.md) |
| 设备常驻轨迹批次 | `sampling.engine="torch"` 用于固定晶胞 PIMD 与扩散，float64 CPU/CUDA 与整批检查点；有明确的模型/设备要求 | [Torch 工作流](docs/torch-materials-workflows.md) · [性能证据](docs/t19-performance.md) |
| 量子统计与交换 | Suzuki–Chin、PIGS、玻色交换、费米符号重加权、周期 winding 求和与全珠子各向同性交换 NPT；GLE/PIGLET 矩阵导入和生成 | [路径积分作用量](docs/tier2-path-integral.md) · [量子估计量](docs/quantum-statistics.md) · [凝聚相核量子效应](docs/condensed-phase-nqe.md) |
| 振动光谱 | `spectroscopy`：PA-CMD、二次 f-CMD、实验性弯曲三原子 f-QCMD 与相对 IR/Raman/SFG；TRPMD 关联函数 API | [光谱](docs/tier2-spectroscopy.md) · [f-QCMD](docs/qcmd.md) · [TRPMD](docs/trpmd-spectroscopy.md) |
| 低温速率与劈裂 | `instanton` 与 Python API：分子路径、Hessian provider、收敛扫描、涨落预因子和正常区 golden-rule instanton | [Instanton](docs/instanton.md) · [低温方法](docs/low-temperature.md) |
| 非绝热动力学 | `nonadiabatic`、选定散射及独立热速率工作流；NRPMD、自旋映射与表面跳跃；IESH 和多态 MASH 当前使用 P=1 | [非绝热工作流](docs/nonadiabatic-workflow.md) · [IESH 与多态 MASH](docs/nonadiabatic-iesh-multistate.md) |
| 电子结构与量子参考 | Python PySCF SA-CASSCF 桥接、spin–boson HEOM，以及限定模型下的 PCET 与恒电位计算 | [电子数据与 HEOM](docs/tier2-nonadiabatic.md) · [PCET](docs/zh/manual/pcet.md) · [电极模型](docs/zh/manual/electrochemistry.md) |
| 势能与批量评估 | 原生内置解析势；Python、C ABI、RPMDrate/f2py、ASE、Torch、i-PI、OpenMM、LAMMPS；批量 MACE/NequIP/fairchem/DeePMD/metatomic | [势能接口](docs/zh/manual/potentials.md) · [原生内置势](docs/builtin-native-potentials.md) · [适配器](docs/potential-adapters.md) · [批量 ML](docs/batched-ml-potentials.md) |
| 模型构建与主动学习 | PES 生成、committee、delta 修正与可续算的采样—标注—训练闭环；用户回调或 CPU MACE 参考训练器 | [PES 生成](docs/zh/manual/pes-generation.md) · [ML 生命周期](docs/tier2-ml-lifecycle.md) |
| 执行、输出与分析 | 支持的进程/MPI 并行、原子检查点、轨迹导出、离线分析与报告；RPC/MTS 和键约束 API | [执行](docs/zh/manual/execution.md) · [输出](docs/zh/manual/results.md) · [传播](docs/propagation-stack.md) |

实时 QCMD 与 h-CMD 尚未实现，f-QCMD 也尚未完成所要求的收敛光谱验证。
现有性能报告包含 CPU 测量和 CUDA 基准驱动，**没有 GPU 提速实测结果**。
数值测试与短模型对照不代表物理收敛。

<a id="documentation"></a>

## 文档

| 入口 | 后续内容 |
| --- | --- |
| [English manual](docs/index.md) · [中文手册](docs/zh/index.md) | 对应核心章节、可搜索 HTML 与专题指南 |
| [示例](docs/zh/manual/examples.md) · [输入与命令行](docs/zh/manual/input.md) | 可运行输入、科学设置和专用命令 |
| [科学约定](docs/zh/manual/science.md) · [能力与验证](docs/zh/manual/capabilities.md) | 原子单位、收敛、限制、已发表模型对照与审查记录 |
| [Python API](docs/zh/manual/api.md) · [开发指南](docs/zh/manual/development.md) | 势能/CV 扩展、测试、原生构建与并行执行 |
| [方法参考文献](docs/zh/manual/references.md) · [复现清单](docs/zh/manual/reproduction.md) | 原始论文、实现范围与可复现研究计划 |
| [下一版发布说明](docs/zh/releases/unreleased.md) · [开发路线](docs/roadmap.md) | 已合并新增功能、集成状态与剩余工作 |

在本地构建中英文手册：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

Documentation 工作流构建手册并运行快速入门示例。研究中应引用实际采用的方法、PES、参数、上游软件和
PyRPMD commit；可用的引用格式包括 [BibTeX](docs/references.bib) 与[结构化目录](docs/method-references.json)。
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)记录来源归属、固定的上游 commit 与第三方许可。
PyRPMD 自身采用 [MIT](LICENSE) 许可；附带第三方组件保留各自的许可。
