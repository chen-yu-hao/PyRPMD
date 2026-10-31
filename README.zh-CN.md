# PyRPMD

**[使用手册](docs/zh/index.md)** · [安装](docs/zh/manual/installation.md) ·
[快速入门](docs/zh/manual/quickstart.md) · [示例](docs/zh/manual/examples.md) ·
[Python API](docs/zh/manual/api.md) · [English](README.md)

**一个输入文件完成环聚合物分子动力学——从过渡态到经回穿校正的速率，支持带科学指纹的检查点续跑。**

给 PyRPMD 一个过渡态和一个势能，它会自动搭建完整的 Bennett–Chandler 计算：数值 Hessian 与双向下降识别反应通道，伞抽样
给出 PMF，受约束母链发射成对回穿轨迹得到透射系数。同一套 Python 或 TOML 输入驱动 11 类任务，从气相速率、直接碰撞到自由能面、
metadynamics、PIMD、扩散、表面跃迁和电化学 PCET；积分器内核为 C++17/OpenMP，并配有 NumPy 参考实现。

## 为什么选择 PyRPMD

- **一条命令，从 TS 到速率。** 不需要手工布置窗口或另写回穿脚本：一个输入把计算从结构带到速率，并给出不确定度。
- **结果经得起推敲。** 每次运行分别报告估计值、失败或未决轨迹以及诊断信息。重叠、回穿平台或能量守恒未通过检查时，PyRPMD
  不给出该数值，并说明原因。
- **放心续跑。** 检查点保存完整 q/p、RNG 状态、偏置历史和科学指纹，原子写入并逐数组 SHA256 校验。续跑时可调整分块大小和
  循环后端；改变科学设置会被拒绝。
- **接入你自己的势能。** Python、RPMDrate/f2py、版本化 C ABI `.so`、由 manifest 链接的普通 `.o`、ASE、Torch 或 i-PI socket。
- **从笔记本到集群。** 轨迹/窗口进程池、常驻珠子 PES 进程池、MPI `G × B` 分组、OpenMP，以及多张 GPU 上每卡多条轨迹。
- **对照已发表参考验证。** BKMP2 势能面上的 H + H₂、与 pyPCET 相差不超过 0.062% 的 PCET 速率，以及与解析积分和 PyMBAR
  对照的 PMF（见[验证](#验证)）。

昂贵势能计算可使用 [worker 势能复用与批量传播](docs/performance-dynamics.md)、
[显式 RPC/MTS Python API](docs/rpc-mts.md)和
[TRPMD、相关函数及相对 IR 谱](docs/trpmd-spectroscopy.md)。
[文献速率比较工具](benchmarks/published_rates/README.md)检查模型身份、统计约定和收敛证据。
这些扩展指南目前提供英文版。


## 安装

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git PyRPMD
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pyrpmd --version
pyrpmd info
```

PyRPMD 需要 Python 3.11+、NumPy、SciPy、h5py、threadpoolctl 和 C++17 编译器；Linux 上编译 OpenMP。用 `CC`/`CXX`
指定编译器。没有编译器时安装 NumPy 路径：

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

| 可选功能 | 安装 |
| --- | --- |
| MPI | 站点 MPI 与匹配的 mpi4py，再安装 `.[mpi]` |
| ASE | `.[ase]` |
| Torch / GPU | 与节点 CUDA 匹配的 PyTorch，再安装 `.[gpu]` |

`native='required'` 要求编译内核；没有 OpenMP 时请求多个 dynamics 线程会报错。MPI 支持静态分组和显式启用的动态轨迹队列，后者使用一个协调 rank；测试覆盖
单机四个 rank 的静态分组、五个 rank 的动态调度、Torch CPU 执行和多 GPU 分配。见[执行](docs/zh/manual/execution.md)。

## 第一次运行

在内置谐振势上做一次可续跑的一维自由能短测试。安装完成后可在任意工作目录开始；`init` 提供输入文件，
无需依赖源码目录中的示例或下载外部模型：

```bash
pyrpmd init my-first-run
pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
pyrpmd analyze my-first-run/run
pyrpmd resume my-first-run/run
```

所有任务都使用同样的 `validate` → `run` → `analyze` → `resume` 流程。`--check-pes` 还会评估势能，对只给 TS 的速率输入还会
执行数值准备。每个新计算使用新的输出目录；`resume`（或对原输入使用 `run --resume`）继续已有运行。
入门例子产生 3 个窗口、6 条采样流和 120 个保存样本，摘要位于 `my-first-run/run/results/summary.json`。
这个小样本用于检查工作流，不代表物理结果已收敛。`init` 拒绝写入非空目录。若 shell 找不到 `pyrpmd`，
可以用 `python -m pyrpmd` 加相同参数。[快速入门](docs/zh/manual/quickstart.md)逐行讲解输入、输出与离线重分析。

## 可以计算什么

所有示例都使用解析模型势，单核即可运行。时间为 `OMP_NUM_THREADS=1` 下的实际耗时。

| 任务 | `task` | 示例 | 时间 |
| --- | --- | --- | --- |
| 从 TS 出发的双分子 RPMD 速率 | `rate` | `pyrpmd prepare examples/rate/pilot.inp --pilot`，再 `pyrpmd run examples/rate/pilot.inp` | ~15 s |
| 直接碰撞（dRPMD） | `drpmd` | `examples/drpmd/free_scattering.inp` | ~2 s |
| 多入口多 TS 速率矩阵 | `network` | `examples/network/two_entrances.py` | ~25 s |
| 一维/二维自由能面 | `fes` | `examples/fes/harmonic2d.inp` | ~6 s |
| 自适应伞窗口 | `fes` | `examples/fes/adaptive_harmonic.inp` | ~4 s |
| Standard 或 well-tempered metadynamics | `metadynamics` | `examples/metadynamics/well_tempered_2d.inp` | ~2 s |
| OPES 概率增强采样 | `opes` | `examples/opes/harmonic_2d.inp` | — |
| 固定晶胞 PIMD 与能量估计量 | `pimd` | `examples/materials/pimd.inp` | ~2 s |
| RPMD 自扩散、MSD、VACF | `diffusion` | `examples/materials/diffusion.inp` | ~3 s |
| 材料中的有向位点跃迁 | `site_rate` | `examples/materials/site_rate.inp` | ~25 s |
| 固定基底上的有向逃逸 | `surface_rate` | `examples/surface/diffusion.inp` | ~1.5 min |
| 电化学 PCET，H/D、电势与温度扫描 | `pcet` | `examples/pcet/harmonic_scan.inp` | ~2 s |
| 态选择非绝热散射（NRPMD，实验性） | `drpmd` | `examples/nonadiabatic/exchange.inp` | ~15 s |

用 `pyrpmd run INPUT --output runs/NAME` 运行其中任意一个。另外两种常用方式：

```bash
# 用不同分块大小续跑；科学设置保持运行身份。
pyrpmd resume runs/meta --loop-backend auto --block-steps 128
pyrpmd checkpoint runs/meta/checkpoints/metadynamics_0000.npz --require-integrity

# 2 个轨迹组 × 每组 2 个珠子 PES rank。
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi
```

[示例索引](docs/zh/manual/examples.md)给出学习路径，以及从示例走向研究计算的步骤。

## 功能

| 方面 | PyRPMD 提供 |
| --- | --- |
| 输入 | Python `build()` 返回 `JobSpec` 或字典，或 TOML，严格检查字段、单位、原子索引和路径；命令行可覆盖 PES、温度、珠子数和 worker 数 |
| RPMD 速率 | TS + PES → 数值 Hessian 与双向下降 → 通道识别 → PMF（Gaussian UI 或直方图 MBAR）→ 受约束母链 → 成对回穿子轨迹；同一抽样上的 UI/MBAR 配对比较 |
| 等价通道 | 由原子置换得到等价 TS，按 RPMDrate 的 `min_bond(max_TS(s1))` 组合；区分 H + H₂ 的两个交换出口 |
| 反应网络 | 同一原子与同位素组成下的不同入口、TS 和 PES；每条路径填入速率矩阵，按入口、通道和 TS 分别报告 |
| dRPMD | 两个反应物 XYZ 和初始质心距离；常驻母链按设定间隔发射碰撞 |
| 核初态选择 | 固定碰撞能下近似振动态选择 NE-RPMD / 单珠子 QCT，独立初态采样、碰撞前检查与未决轨迹延长续跑（[方法](docs/state-selected.md)、[示例](examples/state_selected/README.md)） |
| 自由能 | 带完整 CV Jacobian 的一维/二维伞抽样、MBAR、重叠图、相关块 bootstrap、未采样区域 mask 与离线重分析；可依据 MBAR 重叠、有效样本数及可选覆盖目标自适应插入窗口 |
| Metadynamics | Standard 与 well-tempered，1–3 维 CV、周期/固定层体系、NHC/PILE-L/Andersen、软墙、可选 1D/2D 偏置网格、独立历史与有限时间 FES |
| 副本交换伞抽样 | 同温固定窗口 REUS、PILE-L 或 Andersen、同步集合续跑与跨窗口联合 bootstrap（[指南](docs/replica-exchange.md)） |
| OPES | 固定带宽加权概率估计、可选冻结偏置重加权与可续跑的独立历史（[指南](docs/opes.md)） |
| PLUMED 与路径抽样 | 可选的质心偏置 PLUMED 桥接；无偏双向 shooting 的跃迁界面集合与独立通量归一化（[PLUMED](docs/plumed.md)、[TIS](docs/transition-interface-sampling.md)） |
| 速率后处理 | 离线二维分界面通量积分与带符号的通道贡献，显式归一化与严格平台检查（[指南](docs/rate-postprocessing.md)） |
| 材料与表面 | 固定晶胞或各向同性 BZP 恒压 PIMD、固定层、primitive 与 centroid-virial 估计量、流式轨迹；普通或质心电子摩擦扩散；盆地归一化的位点/表面速率与源 FES 投影（[详情](docs/surface-catalysis.md)） |
| 电化学 PCET | 局域复合物、单质子、宽带金属与经典 Gaussian 环境下的 golden-rule 正反向速率（s⁻¹）；H/D KIE 与振动态贡献（[方法](docs/pcet.md)） |
| 积分器 | `cayley-bcb`、`normal-mode-bab`、`velocity-verlet`，均支持原生分段传播、Andersen 母链和质心 RATTLE |
| 续跑 | v2 原子检查点逐数组 SHA256 校验（仍可读取 v1）；续跑时可调整分块和后端 |

[原生集合传播器](docs/native-ensembles.md)提供显式启用的 C++ 恒温/恒压轨迹、常驻状态与编译 PES 回调。
[势能适配器](docs/potential-adapters.md)涵盖 C ABI v2、周期 Torch、具名机器学习计算器、tblite、PySCF 与多客户端 i-PI。
[离线 PES 生成流程](docs/manual/pes-generation.md)将 PySCF 参考数据连接到固定源码版本的 Caracal QMDFF/EVB/TREQ，
并要求先完成试算与验证。

PIMD 与固定晶胞扩散支持冻结基底原子；各向同性 BZP 恒压 PIMD 要求三维完全周期、全部原子可动及物理势能 virial。
材料轨迹默认按带校验的分块流式保存。质子导体需要适合其组成、电荷与机理的周期势。输入、单位、周期 CV 与统计见
[材料说明](docs/materials.md)。本版本的新内容见 [0.2.0 发行说明](docs/releases/0.2.0.md)。

## 最小输入

一个过渡态和一个势能就是完整的速率输入：

```toml
task = "rate"

[system]
transition_state = "ts.xyz"

[potential]
provider = "cabi"
library = "libpes.so"
```

`[potential]` 也可以在命令行用 `--pes ./libpes.so` 给出。`.so` 实现 PyRPMD PES v1 ABI；f2py 模块使用
`provider='rpmdrate'`，普通 `.o` 另加 `--pes-manifest`，写明编译器、wrapper 和链接依赖。

两个反应物和一个间距就是完整的 dRPMD 输入：

```toml
task = "drpmd"

[system]
reactants = ["A.xyz", "B.xyz"]
separation = "12 angstrom"

[potential]
provider = "cabi"
library = "libpes.so"
```

两者都使用 `pilot` 预设：300 K、32 个珠子、0.1 fs 步长。在此基础上，每添加一项设置就多得到一项结果：

- 给出经过截断检验的 `bmax`，得到 dRPMD 热速率（不给时碰撞从 `R0/4` 开始，不报告热速率）；
- 给出分类规则，得到化学速率和截面（不给时 PyRPMD 按共价半径推断连通性，把结果标为候选）；
- 异构化和交换通道使用自定义分类器。

## Python 扩展

Python 输入可以计算参数、加载模型并构造任意静态可微 CV。把 factory 放在可导入模块中；每个进程或 MPI rank 构造自己的势能上下文：

```python
# job.py
from pyrpmd import JobSpec

def build():
    return JobSpec.from_mapping({
        "task": "drpmd",
        "temperature": "500 K",
        "beads": 32,
        "system": {
            "reactants": ["A.xyz", "B.xyz"],
            "separation": "12 angstrom",
        },
        "potential": {
            "provider": "python",
            "factory": "my_pes.py:build_pes",
            "params": {"model": "model.dat"},
        },
        "collision": {"bmax": "4 angstrom", "trajectories": 1000},
        "parallel": {"backend": "process", "mode": "serial", "trajectory_workers": 8},
    })
```

PES 接收原子单位的 `float64[B,N,3]` 坐标，返回 `[B]` 能量和 `[B,N,3]` 力（负梯度）。自定义 CV 返回 `value[D]` 和完整的
`jacobian[D,P,N,3]`。可运行的例子见 [Python PES](examples/drpmd/python_pes.py) 和[自定义二维 CV](examples/fes/custom_input.py)。
Python 输入作为普通的受信任代码执行。

## 验证

**BKMP2 上的 H + H₂。** [`benchmarks/h3`](benchmarks/h3/README.md) 在已发表的 BKMP2 势能面上、1500 K 与 16 个珠子下运行
dRPMD、二维质心 FES、伞抽样 PMF 和回穿。构建时按 commit 和 SHA256 固定势能源码，并在任何传播之前验证作者给出的能量/梯度参考值、
H₂ 平衡结构和一阶鞍点。四组使用 MBAR 速率 PMF 的独立随机种子重复覆盖 848,000 个样本和 1,024 条新回穿轨迹：四组均通过，同一
分割面上 UI/MBAR 势垒差的每个配对区间都包含零，速率的联合相对标准误为 20–23%
（[记录](docs/validation/h3-pmf-refinement-2026-10-29/README.md)、[方法](docs/mbar-rate-pmf.md)、
[估计器审查](docs/h3-pmf-estimator-audit.md)）。

**PCET。** 在谐振模型上，PyRPMD 与固定 commit 的 pyPCET 在 12 个正反向速率上相差不超过 0.062%
（[记录](docs/validation/pcet-model-2026-10-30/README.md)）。

**PMF 数值。** 一维/二维 PMF 在不同珠子数、积分器、周期 CV 和相关误差下与解析积分和 PyMBAR 一致
（[记录](docs/validation/pmf-accuracy-2026-10-30/README.md)）。

**材料。** [Ni–H EAM pilot](benchmarks/materials/README.md) 在固定来源的 EAM 模型上，以 1 和 4 个珠子运行 Ni(100) slab 上的
H 和 D（[记录](docs/validation/materials-2026-10-30.md)）。

GitHub Actions 的 **Real H3 BKMP2 validation** 与 **H3 PMF independent-seed study** 工作流重新运行这些研究，并保存输入、
样本、检查点、PMF、κ(t)、诊断与图表。

<details>
<summary>复现 BKMP2 基准</summary>

```bash
# 需要 GNU Fortran；图表需要 matplotlib。
python -m pip install matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3/pes-check.json
python benchmarks/h3/numerics.py --output runs/h3/numerics.json
python benchmarks/h3/drpmd.py --output runs/h3/drpmd --workers 2
python benchmarks/h3/rpmd.py --stage fes2d --output runs/h3/fes2d --workers 2
python benchmarks/h3/rpmd.py --stage rate --output runs/h3/rate --workers 2
```

独立随机种子研究（case 为 `k400-a`、`k400-b`、`k800-a`、`k800-b`；`--smoke` 只运行简短检查）：

```bash
python -m pip install -e '.[analysis]' matplotlib
python benchmarks/h3/convergence.py plan
python benchmarks/h3/convergence.py run --case k400-a --output runs/h3-pmf/k400-a
python benchmarks/h3/solver_check.py --run runs/h3-pmf/k400-a/run \
  --stride 10 --reference-pymbar --output runs/h3-pmf/k400-a/solver-check.json
```

</details>

[H + HO₂ 二维扫描](benchmarks/hho2/README.md)可直接用于全维反应性 H + HO₂ 势能面，提供多起始几何、共同的有限距离墙、
六键长通道分类和连通分析。

开展正式研究前，请阅读[使结果收敛](docs/zh/manual/science.md#使结果收敛)，其中说明如何把计算从 pilot 推进到正式生产。

[自适应伞抽样可靠性审查](docs/validation/umbrella-reliability-2026-10-31/README.md)记录了可复现的重叠与误差报告缺陷、对应修复、实际默认预算，以及仍须针对具体体系完成的收敛检查。pilot 达标不代表精度已经得到保证。

## 原生传播

`Dynamics.propagate()` 以整块为单位原子提交；`Dynamics.propagate_block()` 逐步提交，提供观测 mask、完整 q/p 快照和原生质心
RATTLE，数值失败时保留成功的前缀。两者都支持三种积分器。内置连通图分类器与 H3 交换分类器在 C++ 块内运行并提前结束碰撞；
融合的 C++ PILE-L 内核以及正常模 BAB 在 2 的幂次 `P>=32` 时使用的原生 FFT 加速带恒温器和大珠子数的运行。缺省为
`loop_backend="auto"`、`block_steps=256`。详见[原生传播](docs/native-propagation.md)、
[分段 API 与故障处理](docs/block-propagation.md)、[积分器](docs/integrators.md)、
[优化记录](docs/validation/audit-optimization-2026-10-30/README.md)。

态选择碰撞复用同一 C++ 生产传播路径。可选原生观测记录每珠子势能、质心动能、内部 bead 动能、弹簧能和每原子环尺寸，
不增加 PES 调用。`pyrpmd state-check` 在生产块外将稀疏保存的状态投影到分子参考模式；
谐振初态近似、辅助 beta 与固定能量统计的适用条件见[态选择说明](docs/state-selected.md)。

[BKMP2 态选择对照](benchmarks/h3/STATE_SELECTED_REPRODUCTION.md)及其
[初态制备研究](benchmarks/h3/STATE_PREPARATION_STUDY.md)区分论文的质心筛选、全势作用量 QCT
与实验性平滑切换，并给出配对轨迹统计和孤立双原子能量诊断。

## 文档

使用手册涵盖安装、快速入门、全部 11 类任务的指南、API 与开发说明，有[英文](docs/index.md)和[中文](docs/zh/index.md)两个版本，
章节一一对应。构建带搜索的 HTML 手册（中文版输出到 `zh/`）：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

GitHub **Documentation** 工作流构建两个版本、运行入门示例，并上传 `pyrpmd-manual-html` 产物。

[用户体验审查与 to-do list](docs/ux-review-2026-10-31.md)记录当前上手流程的审查结果及后续工作。

运行目录包含 `manifest.json`、`input/`、`checkpoints/`、工作流数据和 `results/summary.json`。FES 另有
`fes.h5`/`fes.npz`/`fes.csv`；dRPMD 保存每条轨迹的初态、碰撞轨迹和分类记录。不同 MPI rank 不共同写一个 HDF5 文件。

参考规范：

- [输入规范](docs/input.md)
- [连接势能](docs/potentials.md)与 [C ABI](native/ABI.md)
- [并行与多节点运行](docs/parallel.md)
- [科学约定](docs/science.md)
- [开发与验证](docs/development.md)、[验证记录](docs/validation.md)
- [多入口与多 TS](docs/reaction-networks.md)
- [实验性 NRPMD 与态分辨碰撞](docs/nonadiabatic.md)
- [电子摩擦](docs/electronic-friction.md)、[热 NRPMD](docs/thermal-nrpmd.md)、[KC/MV/isomorphic 模型](docs/nonadiabatic-methods.md)、[SOC 与 RPSH](docs/nonadiabatic-soc-rpsh.md)
- [BZP 恒压 PIMD](docs/materials-npt.md)、[网格 metadynamics](docs/grid-metadynamics.md)、[流式轨迹](docs/streaming-trajectories.md)
- [恒电位电极模型与 HER 后处理](docs/electrochemistry.md)
- [自适应伞抽样与 metadynamics](docs/enhanced-sampling.md)
- [电化学 PCET 模型](docs/pcet.md)、[可运行 PCET 示例](examples/pcet/README.md)
- [固定晶胞、PIMD、材料扩散与位点速率](docs/materials.md)
- [表面催化与周期 slab](docs/surface-catalysis.md)
- [公共接口](docs/interfaces.md)、[检查点与续跑](docs/checkpoints.md)

| 路径 | 职责 |
| --- | --- |
| `src/pyrpmd/input.py`、`units.py`、`system.py`、`geometry.py` | 输入、单位、原子/同位素体系与固定晶胞几何 |
| `src/pyrpmd/dynamics.py`、`native/` | 三种环积分器、动量采样、恒温器、C++/OpenMP |
| `src/pyrpmd/potentials/` | 势能接口及其适配器 |
| `src/pyrpmd/cv.py`、`bias.py` | CV、完整 Jacobian、伞势、速率通量度量、历史 Gaussian 偏置 |
| `src/pyrpmd/workflows/`、`reactions.py` | rate、dRPMD、PIMD、扩散、位点/表面速率、PCET、FES、自适应窗口、metadynamics、多 TS 组织与准备 |
| `src/pyrpmd/analysis/` | MBAR、umbrella integration、metadynamics 偏置 FES、MSD/VACF/扩散张量、统计诊断 |
| `src/pyrpmd/checkpoint.py`、`runner.py` | 原子状态容器、完整性校验、运行身份与公共入口 |
| `src/pyrpmd/parallel/` | 轨迹池、珠子池、MPI、GPU 放置 |
| `examples/`、`tests/`、`docs/` | 可运行示例、数值/集成测试、规范 |

## 测试与来源

```bash
python -m pytest -q
mpiexec -n 4 python tests/mpi_integration.py
```

Cayley 积分器、反应坐标与速率算法改编自 [RPMDrate](https://github.com/GreenGroup/RPMDrate) 及作者的
[Cayley/thermal-dRPMD 分支](https://github.com/chen-yu-hao/RPMDrate/tree/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78)；速率工作流
还参照了 [Caracal](https://github.com/Trebonius91/Caracal/tree/7f95ce9818b7f6d81d7cb3b634f360373c090820) 的伞抽样/回穿组织方式。
适配器、输入规范和调度是新增的层。来源 commit、作者、MIT 声明和论文 DOI 记录在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
不包含 Caracal 的 QMDFF、xTB 和机器学习模型。

许可证：[MIT](LICENSE)。
