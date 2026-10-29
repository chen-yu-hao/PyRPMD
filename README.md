# PyRPMD

**Python 可扩展的环聚合物分子动力学计算包。** PyRPMD 0.1.0 提供可运行的 RPMD 双分子速率、直接碰撞 dRPMD，以及一维/二维自由能面工作流。输入使用普通 Python 或 TOML 语法的 `.inp` 文件；数值传播使用 C++17/OpenMP Cayley 内核，并保留 NumPy 参考实现。

本项目是初始科研实现。随附示例用于验证算法和程序连接；真实化学体系仍需要合适的势能面、通道定义及珠子数、步长、采样、远区距离和回穿时间等收敛检查。程序会分别报告有限采样估计、失败/未决轨迹和诊断，不把一次运行成功标成物理收敛。

## 已实现

| 需求 | 0.1.0 行为 |
| --- | --- |
| 自由 Python 输入 | `.py` 中 `build()` 返回 `JobSpec` 或字典；自定义 PES、CV、产物分类器通过可导入 factory 构造 |
| 简单 `.inp` | TOML，严格检查字段、单位、原子索引、输入相对路径；命令行可覆盖 PES、温度、珠子数、轨迹进程数 |
| TS-only RPMD | TS + PES 启动数值 Hessian/双向下坡准备；唯一可识别双分子通道进入 PMF、受约束母轨迹和成对回穿子轨迹；歧义给出诊断 |
| dRPMD | 两个反应物 XYZ + 初始 COM 距离 + PES；独立母轨迹连续热化并按采样间隔发射，支持总轨迹数余数分配 |
| 势能适配 | Python、RPMDrate/f2py、版本化 C ABI `.so`、manifest 驱动 `.o` 链接、ASE、Torch、i-PI socket |
| 并行 | 轨迹/窗口进程并行、持续珠子势能进程池、MPI `G × B` 分组、OpenMP、CUDA 多设备和每卡多轨迹进程 |
| 2D FES | 二维 umbrella、完整 CV Jacobian、MBAR、窗口重叠图、相关块 bootstrap、未采样区域 mask、保存后重新分析 |
| 续跑 | 保存完整 q/p、随机数状态、父链进度、已完成轨迹与科学配置指纹；常规入口检查输入/PES 文件哈希 |

MPI 路径已在**同一台机器的 4 个真实 MPI 进程**中测试。跨物理节点和真实 CUDA 硬件尚未实测。Torch CPU 和多 GPU 分配逻辑分别有测试。当前 MPI 是静态分组，`pes_ranks=1`；不声称支持任意势能程序内部的 MPI communicator。

## 安装

从源码安装，而非假定已有同名 PyPI 发布：

```bash
git clone https://github.com/chen-yu-hao/Next-RPMD.git PyRPMD
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pyrpmd info
```

需要 Python 3.11+、NumPy、SciPy、h5py 和 C++17 编译器。Linux 默认编译 OpenMP。指定编译器可设置 `CC`/`CXX`。没有编译器时可安装 NumPy 路径：

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

MPI 环境需安装站点提供的 MPI 及与之匹配的 mpi4py；可安装 `.[mpi]`。ASE 使用 `.[ase]`。Torch 请按计算节点的 CUDA 环境安装对应 PyTorch，再使用 `.[gpu]`。`native='required'` 会要求编译内核；OpenMP 不可用时请求多 dynamics 线程会明确报错。

## 快速运行

```bash
# 两个持续母轨迹，5 条独立碰撞任务分配为 3 + 2。
pyrpmd validate examples/drpmd/free_scattering.inp
pyrpmd run examples/drpmd/free_scattering.inp --output runs/scattering
pyrpmd resume runs/scattering

# 两个 CV 的自由能密度面。
pyrpmd run examples/fes/harmonic2d.inp --output runs/fes2d
pyrpmd analyze runs/fes2d --recompute --bins 20 20 --bootstrap 50

# 教学解析势：自动 TS 准备后执行完整 Bennett–Chandler 管线。
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/rate

# 实际 2 组轨迹 × 每组 2 路珠子势能。
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi
```

所有示例都可以先运行 `validate`。`validate --check-pes` 还会初始化并评估势能；对 TS-only 速率输入，这也会进行数值准备。每次新运行使用新目录；续跑使用 `resume` 或原输入的 `run --resume`。

## 最小输入

仅过渡态结构与势能选择：

```toml
task = "rate"

[system]
transition_state = "ts.xyz"

[potential]
provider = "cabi"
library = "libpes.so"
```

也可不写 `[potential]`，在命令行指定 `--pes ./libpes.so`。这里 `.so` 必须实现 PyRPMD PES v1 ABI；已有 f2py 模块应选择 `provider='rpmdrate'`。普通目标文件 `.o` 还需要 `--pes-manifest`，其内容明确编译器、wrapper 和链接依赖。

最小 dRPMD：

```toml
task = "drpmd"

[system]
reactants = ["A.xyz", "B.xyz"]
separation = "12 angstrom"

[potential]
provider = "cabi"
library = "libpes.so"
```

这些输入使用 `pilot` 预设。缺省温度是 300 K、珠子数 32、步长 0.1 fs，不代表已经收敛。dRPMD 未指定 `bmax` 时以 `R0/4` 启动试算，并**暂不报告热速率点估计**；正式估计应显式提供经截断测试的 `bmax`。未给分类规则时会以共价半径推断连通图并给出暂定候选标签，同时抑制化学速率/截面；复杂异构化和交换通道需要自定义分类器。仅有结构文件不能唯一确定全部化学通道。

## Python 扩展

普通 Python 可以计算参数、读取模型和构造任意静态可微 CV。factory 放在独立可导入模块，进程/MPI 工作进程会各自构造势能上下文：

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

PES 接收 `float64[B,N,3]` 原子单位坐标，返回 `[B]` 能量和 `[B,N,3]` 力（负梯度）。自定义 CV 返回 `value[D]` 和完整 `jacobian[D,P,N,3]`。可直接运行的例子见 [Python PES](examples/drpmd/python_pes.py) 和 [自定义二维 CV](examples/fes/custom_input.py)。Python 输入执行普通受信任代码。

## 目录与文档

| 路径 | 职责 |
| --- | --- |
| `src/pyrpmd/input.py`, `units.py`, `system.py` | 输入、单位与原子/同位素系统 |
| `src/pyrpmd/dynamics.py`, `native/` | 完整环 Cayley、动量采样、热化、C++/OpenMP |
| `src/pyrpmd/potentials/` | 统一势能接口及各类适配器 |
| `src/pyrpmd/cv.py`, `bias.py` | CV、完整 Jacobian、umbrella 与速率 flux metric |
| `src/pyrpmd/workflows/` | rate、dRPMD、FES、TS 准备 |
| `src/pyrpmd/analysis/` | MBAR、umbrella integration、统计诊断 |
| `src/pyrpmd/parallel/` | 轨迹进程池、珠子池、MPI、GPU 放置 |
| `examples/`, `tests/`, `docs/` | 可运行示例、数值/集成测试、规范 |

- [输入规范](docs/input.md)
- [连接现有势能](docs/potentials.md)与 [C ABI](native/ABI.md)
- [并行与跨节点运行](docs/parallel.md)
- [科学约定与适用范围](docs/science.md)
- [开发与验证](docs/development.md)、[本次验证记录](docs/validation.md)

运行目录包含 `manifest.json`、`input/`、`checkpoints/`、工作流数据及 `results/summary.json`。FES 另有 `fes.h5`/`fes.npz`/`fes.csv`；dRPMD 保存每条初始状态、碰撞轨迹和分类记录。不同 MPI rank 不共同写一个 HDF5 文件。

## 测试与来源

```bash
python -m pytest -q
mpiexec -n 4 python tests/mpi_integration.py
```

Cayley、反应坐标及速率算法参考并移植了 [RPMDrate](https://github.com/GreenGroup/RPMDrate) 与用户 [Cayley/thermal-dRPMD 分支](https://github.com/chen-yu-hao/RPMDrate/tree/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78) 的相关实现；速率工作流也参考 [Caracal](https://github.com/Trebonius91/Caracal/tree/7f95ce9818b7f6d81d7cb3b634f360373c090820) 的 umbrella/recrossing 组织。新增适配器、输入规范和调度独立分层。来源 commit、作者、MIT 通知与论文 DOI 记录在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。未捆绑 Caracal 的 QMDFF/xTB/机器学习模型。

许可证：[MIT](LICENSE)。
