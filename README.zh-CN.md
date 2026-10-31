# PyRPMD

**[使用手册](docs/zh/index.md)** · [安装](docs/zh/manual/installation.md) ·
[快速入门](docs/zh/manual/quickstart.md) · [示例](docs/zh/manual/examples.md) ·
[Python API](docs/zh/manual/api.md) · [English](README.md)

**Python 环聚合物分子动力学。** PyRPMD 0.2.0 计算双分子 RPMD 速率、直接碰撞 dRPMD、多入口多过渡态的速率矩阵、
带自适应伞抽样和 metadynamics 的一维/二维自由能面、固定晶胞 PIMD、RPMD 扩散、位点与表面跃迁速率，以及局域电化学
PCET 速率。输入为普通 Python 或 TOML `.inp` 文件。C++17/OpenMP 内核实现 Cayley、精确正常模 BAB 和 velocity-Verlet
积分器，并配有 NumPy 参考实现。

每次运行分别报告有限样本估计、失败或未决轨迹以及诊断信息。手册的[收敛章节](docs/zh/manual/science.md#使结果收敛)
说明如何把结果从 pilot 推进到正式计算。

## 功能

| 方面 | PyRPMD 提供 |
| --- | --- |
| Python 输入 | `.py` 中的 `build()` 返回 `JobSpec` 或字典；自定义 PES、CV 和产物分类器来自可导入的 factory |
| TOML 输入 | 严格检查字段、单位、原子索引和相对路径；命令行可覆盖 PES、温度、珠子数和 worker 数 |
| 从 TS 出发的 RPMD 速率 | TS 与 PES 启动数值 Hessian 和双向下降；识别出的双分子通道依次进行 PMF、受约束母链与成对回穿子轨迹 |
| dRPMD | 两个反应物 XYZ、初始质心距离和 PES；常驻母链按设定间隔发射碰撞，总轨迹数按余数分配 |
| 核初态选择 | 固定碰撞能下近似振动态选择 NE-RPMD / 单珠子 QCT，独立初态采样、碰撞前稳定性检查、未结束轨迹延长续跑 |
| 势能 | Python、RPMDrate/f2py、版本化 C ABI `.so`、manifest 驱动的 `.o` 链接、ASE、Torch、i-PI socket |
| 并行 | 轨迹/窗口进程、常驻珠子 PES 进程池、MPI `G × B` 分组、OpenMP、多 CUDA 设备及每卡多轨迹 |
| 自由能面 | 一维/二维伞抽样、完整 CV Jacobian、MBAR、重叠图、相关块 bootstrap、未采样区域 mask、离线重分析 |
| 速率 PMF | Gaussian UI 或直方图 MBAR；固定分割面、同一抽样上的 UI/MBAR 配对比较、逐次 bootstrap 诊断 |
| 等价交换通道 | 由原子置换得到等价 TS，按 RPMDrate 的 `min_bond(max_TS(s1))` 组合；区分 H + H₂ 的两个交换出口 |
| 多入口与多 TS | 同一原子与同位素组成下的不同入口划分、TS 结构和 PES；每条路径运行 PMF 与回穿并填入速率矩阵 |
| 积分器 | `cayley-bcb`、`normal-mode-bab`、`velocity-verlet`，均支持两套原生 block API、Andersen 母链和质心 RATTLE |
| 自适应伞抽样 | 一维/二维 pilot 依据 MBAR 重叠和有效样本数插入窗口、延长采样，然后冻结并独立生产；适用于 FES 和速率 PMF |
| Metadynamics | Standard 与 well-tempered，一维/二维及周期 CV、完整 Jacobian、独立副本、HILLS 与有限时间偏置 FES |
| 电化学 PCET | 局域复合物、单质子、宽带金属与经典 Gaussian 环境下的 golden-rule 正反向速率（s⁻¹）；电势/温度扫描、H/D KIE、振动态贡献、数值诊断 |
| 固定晶胞与 PBC | 正交与倾斜晶胞、部分周期 slab、连续展开坐标；ASE、Python 和 i-PI 势能的周期契约 |
| PIMD | 与积分器匹配的 PILE-L 或 Andersen 采样；primitive 与 centroid-virial 能量估计量、回转半径 |
| 扩散 | 热化后的无恒温 RPMD；质心 MSD、扩散张量、VACF、副本误差、带扩散区检查的离线重分析 |
| 位点与表面速率 | 线性质心坐标、盆地归一化 QTST、受约束母链与成对回穿；一级速率（s⁻¹）；表面支持倾斜晶胞、部分 PBC 和固定层（[详情](docs/surface-catalysis.md)） |
| 续跑 | 完整 q/p、RNG、偏置历史、阶段与科学指纹；v2 原子检查点逐数组 SHA256 校验，可读取 v1 数据；续跑时可调整分块和后端 |

多 TS 结果按入口、通道和 TS 分别报告（`aggregation="separate"`），因为不同分割面的通量可能重叠。Metadynamics 给出
有限时间自由能估计。PCET 求解一维质子本征问题并做电极能量积分，约定见 [PCET 说明](docs/pcet.md)。新接口与兼容性见
[0.2.0 发行说明](docs/releases/0.2.0.md)。

MPI 在单机四个 rank 上测试，Torch CPU 执行和多 GPU 分配各有测试。MPI 采用静态分组，`pes_ranks=1`。

## 安装

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git PyRPMD
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pyrpmd info
```

PyRPMD 需要 Python 3.11+、NumPy、SciPy、h5py、threadpoolctl 和 C++17 编译器；Linux 上编译 OpenMP。用 `CC`/`CXX`
指定编译器。没有编译器时安装 NumPy 路径：

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

MPI 需安装站点 MPI、与之匹配的 mpi4py 以及 `.[mpi]`。ASE 使用 `.[ase]`。Torch 需先安装与节点 CUDA 匹配的 PyTorch，
再安装 `.[gpu]`。`native='required'` 要求编译内核；没有 OpenMP 时请求多个 dynamics 线程会报错。

## 快速运行

```bash
# 两条常驻母链按 3 + 2 分配 5 次碰撞。
pyrpmd validate examples/drpmd/free_scattering.inp
pyrpmd run examples/drpmd/free_scattering.inp --output runs/scattering
pyrpmd resume runs/scattering

# 两个 CV 上的自由能密度面。
pyrpmd run examples/fes/harmonic2d.inp --output runs/fes2d
pyrpmd analyze runs/fes2d --recompute --bins 20 20 --bootstrap 50

# 解析反应势：自动 TS 准备后运行完整 Bennett–Chandler 流程。
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/rate

# 两个入口、四条 TS/PES 路径。
pyrpmd run examples/network/two_entrances.py --output runs/network

# 自适应窗口与 well-tempered metadynamics。
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/adaptive
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/meta
pyrpmd analyze runs/meta --recompute --bins 30 30

# PCET 模型：H/D 的温度与电势扫描，局域一级速率。
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic

# 用不同分块大小续跑；科学设置保持运行身份。
pyrpmd resume runs/meta --loop-backend auto --block-steps 128
pyrpmd checkpoint runs/meta/checkpoints/metadynamics_0000.npz --require-integrity

# 2 个轨迹组 × 每组 2 个珠子 PES rank。
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi

# 固定周期表面上的一级跃迁速率，单位 s^-1。
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
```

`validate` 可检查任何输入；`validate --check-pes` 还会构造并评估势能，对只给 TS 的速率输入还会执行数值准备。对 PCET，
`--check-pes` 求解质子本征态；`run` 再给出速率与离散化检查。每次运行使用新目录，继续运行使用 `resume` 或对原输入使用
`run --resume`。

## 材料中的氢与质子

```bash
# 解析模型上的固定晶胞 PIMD。
pyrpmd run examples/materials/pimd.inp --output runs/materials-pimd

# 自由粒子：分析识别出弹道运动，不报告 D。
pyrpmd run examples/materials/diffusion.inp --output runs/materials-free
pyrpmd analyze runs/materials-free --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40

# 有限盆地中单一有向出口的速率。
pyrpmd run examples/materials/site_rate.inp --output runs/materials-site
```

[Ni–H EAM pilot](benchmarks/materials/README.md) 在固定来源的 EAM 模型上，以 1 和 4 个珠子运行 Ni(100) slab 上的 H 和 D。
材料任务使用固定晶胞且全部原子可动；质子导体需要适合其组成、电荷与机理的周期势。输入、单位、周期 CV 与统计见
[材料说明](docs/materials.md)，测试结果见[材料验证记录](docs/validation/materials-2026-10-30.md)。

## H + H₂ / BKMP2 验证

[2026-10-30 PMF 准确性验证](docs/validation/pmf-accuracy-2026-10-30/README.md)在不同珠子数、积分器、周期 CV 和相关误差下，
用解析积分和 PyMBAR 检查一维/二维 PMF，并运行 BKMP2 工作流。[H + HO₂ 二维扫描](benchmarks/hho2/README.md)提供多起始几何、
共同的有限距离墙、六键长通道分类和有支持网格上的连通分析，可直接用于全维反应性 H + HO₂ 势能面。

[`benchmarks/h3`](benchmarks/h3/README.md) 在已发表的 BKMP2 势能面上运行 dRPMD、二维质心 FES、伞抽样 PMF 和回穿。构建时
从固定 commit 下载势能源码并核对 SHA256，在传播前验证作者给出的能量/梯度参考值、H₂ 平衡结构和一阶鞍点。基准在 1500 K、
16 个珠子下运行。

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

GitHub Actions 的 **Real H3 BKMP2 validation** 工作流运行这一体系，保存输入、样本、检查点、PMF、κ(t)、诊断与图表。
[2026-10-29 记录](docs/validation/h3-bkmp2-2026-10-29/README.md)保存一次实际运行的数值、图表与提交信息，
[UI / MBAR 估计器审查](docs/h3-pmf-estimator-audit.md)解释两种估计器在同一分割面上的 PMF 差异。

独立随机种子研究使用 MBAR 作为速率 PMF，采用更强、更密的伞窗口，并对每个 case 重新计算受约束母链与回穿：

```bash
python -m pip install -e '.[analysis]' matplotlib
python benchmarks/h3/convergence.py plan
python benchmarks/h3/convergence.py run --case k400-a --output runs/h3-pmf/k400-a
python benchmarks/h3/solver_check.py --run runs/h3-pmf/k400-a/run \
  --stride 10 --reference-pymbar --output runs/h3-pmf/k400-a/solver-check.json
```

四个 case 为 `k400-a`、`k400-b`、`k800-a` 和 `k800-b`。**H3 PMF independent-seed study** 工作流并行运行它们，保存全部样本、
逐次诊断、独立的 PyMBAR 检查与跨种子汇总；`--smoke` 只运行简短检查。方法与计划见 [MBAR 速率 PMF](docs/mbar-rate-pmf.md)。

[四组独立重复的结果](docs/validation/h3-pmf-refinement-2026-10-29/README.md)覆盖 848,000 个样本和 1,024 条新回穿轨迹。
四组均通过，同一分割面上 UI/MBAR 势垒差的四个配对区间都包含零，速率的联合相对标准误为 20–23%。

## 最小输入

一个过渡态和一个势能：

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

最小 dRPMD 输入：

```toml
task = "drpmd"

[system]
reactants = ["A.xyz", "B.xyz"]
separation = "12 angstrom"

[potential]
provider = "cabi"
library = "libpes.so"
```

两者都使用 `pilot` 预设：300 K、32 个珠子、0.1 fs 步长。不给 `bmax` 时，dRPMD 从 `R0/4` 开始且不报告热速率；给出经过截断检验
的 `bmax` 即可得到热速率。不给分类规则时，PyRPMD 按共价半径推断连通性，把结果标为候选，并且不报告化学速率和截面；异构化和
交换通道使用自定义分类器。

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

## 文档

使用手册从安装、快速入门、任务指南一直到 API 与开发说明，参照 PySCF 手册的结构，覆盖气相反应、自由能、周期材料、固定表面
和 PCET 全部 10 类任务，并说明单位、输出与续跑。手册有[英文](docs/index.md)和[中文](docs/zh/index.md)两个版本，章节一一对应。

构建带导航与搜索的 HTML 手册（中文版输出到 `zh/`，导航栏可切换语言）：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

打开 `docs/_build/html/index.html`（英文）或 `docs/_build/html/zh/index.html`（中文）。GitHub **Documentation** 工作流构建两个
版本、运行入门示例，并上传 `pyrpmd-manual-html` 产物。

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

参考规范：

- [输入规范](docs/input.md)
- [连接势能](docs/potentials.md)与 [C ABI](native/ABI.md)
- [并行与多节点运行](docs/parallel.md)
- [科学约定](docs/science.md)
- [开发与验证](docs/development.md)、[验证记录](docs/validation.md)
- [原生传播](docs/native-propagation.md)、[分段 API 与故障处理](docs/block-propagation.md)
- [积分器](docs/integrators.md)、[多入口与多 TS](docs/reaction-networks.md)
- [实验性 NRPMD 与态分辨碰撞](docs/nonadiabatic.md)
- [自适应伞抽样与 metadynamics](docs/enhanced-sampling.md)
- [电化学 PCET 模型](docs/pcet.md)、[可运行 PCET 示例](examples/pcet/README.md)
- [固定晶胞、PIMD、材料扩散与位点速率](docs/materials.md)
- [表面催化与周期 slab](docs/surface-catalysis.md)
- [公共接口](docs/interfaces.md)、[检查点与续跑](docs/checkpoints.md)

运行目录包含 `manifest.json`、`input/`、`checkpoints/`、工作流数据和 `results/summary.json`。FES 另有
`fes.h5`/`fes.npz`/`fes.csv`；dRPMD 保存每条轨迹的初态、碰撞轨迹和分类记录。不同 MPI rank 不共同写一个 HDF5 文件。

## 原生 C++17 传播

态选择 dRPMD 复用现有 C++ 连续传播；在线记录势能、质心动能、内部 bead 动能、弹簧能和每原子环尺寸，
不增加 PES 求值。`state-check` 在稀疏观测帧检查分子正常模，`extend` 保留原始状态提高未结束轨迹的总时间上限。
输入与方法边界见[态选择说明](docs/state-selected.md)和[可运行示例](examples/state_selected/README.md)。

`Dynamics.propagate()` 连续传播并以整块为单位原子提交。`Dynamics.propagate_block()` 逐步提交，提供观测 mask、完整 q/p 快照和
原生质心 RATTLE，数值失败时保留成功的前缀。两种 API 都支持三种积分器。dRPMD 碰撞和受约束母链使用分段接口；原生 FES 及复用
其采样器的 PMF 工作流在一个分段内收集多个样本并保留每步能量检查，Andersen 热化与回穿子轨迹使用连续传播。

自定义 Python PES 与 CV 使用兼容路径，自定义分类器在其事件边界由 Python 调用。内置连通图通道和 H3 交换分类器在 C++ 块内按原
频率判定、累计持续时间并提前停止；临界几何回到 Python 判据。原生 dRPMD 在块内复用终点力，并用单个回滚快照代替逐步 q/p 历史。
缺省为 `loop_backend="auto"`、`block_steps=256`。

`auto` 模式下，PILE-L 对 2 的幂次和较小的珠子数使用融合的 C++ 自由环半步–OU–半步内核，随机数仍由 NumPy 生成，势能接口仍在
Python 中；固定原子保持精确固定。正常模 BAB 对 2 的幂次 `P>=32` 使用原生 FFT，缺省的 Cayley 算法不变。基准与回归覆盖见
[优化验证记录](docs/validation/audit-optimization-2026-10-30/README.md)；方法选择、数值比较与续跑规则见[积分器说明](docs/integrators.md)。

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
