# 执行、并行与续跑

## 积分器与执行路径

积分器决定数值动力学；native/Python 与分块设置决定这一动力学怎样执行。两种路径使用相同的势能、质量、温度、偏置和随机流。

```toml
[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"
native = true
loop_backend = "auto"
block_steps = 256
```

| 设置 | 含义 |
| --- | --- |
| `native=true` | 可用时使用编译的积分子步骤，否则用 NumPy |
| `native=false` | 使用 NumPy 积分器 |
| `native="required"` | 要求所选积分器有编译实现 |
| `loop_backend="auto"` | PES、CV 与偏置允许时融合整段传播，否则使用 Python 循环 |
| `loop_backend="python"` | Python 外层循环，可用时使用编译子步骤 |
| `loop_backend="native"` | 要求融合的整段传播；不可用时报错 |
| `block_steps` | 执行分块大小，1–65536 |

默认材料引擎中的周期与固定原子体系、PILE-L 采样以及 metadynamics 使用 Python 外层循环和编译的确定性子步骤。
`pimd`、`diffusion` 和 `metadynamics` 的摘要在 `propagation` 下报告原生与 Python 步数及回退原因；API 中
`Dynamics.loop_stats` 给出同样的计数。

平衡 `pimd` 还可显式选择 `sampling.engine="native"`，使用原生恒温器、受支持的 NPT 恒压器与兼容的编译势能。
这条路径有独立的检查点身份与输入限制，见[原生 PIMD 工作流](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md)。
只设置 `loop_backend="native"` 不会选择这个采样引擎。

[NRPMD](nonadiabatic.md) 在 CPU 上用 NumPy 推进电子矩阵，可串行或使用本机轨迹进程；其核自由环子步骤使用编译内核。

## CPU 轨迹与珠子并行

先并行独立的窗口、母链或副本。单次 PES 调用很昂贵时，再把一条轨迹的珠子分给多个进程。

```toml
[parallel]
backend = "process"
mode = "serial"
trajectory_workers = 4
threads_per_rank = 1
```

`mode="serial"` 让每次 PES 调用在一个进程内完成，同时运行四个轨迹进程。`mode="bead"`、`bead_workers=2` 时，每条并发
轨迹拥有一个由两个珠子 PES 进程组成的常驻进程池。`collision.trajectories` 始终是整个任务的总数。

对计算量很小的解析势，进程通信占主要开销。按分配的 CPU 选择轨迹进程数 × 珠子进程数 × 势能/BLAS 线程数，通常设置：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## MPI 的 G × B 拓扑

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 2
bead_workers = 2
threads_per_rank = 1
pes_ranks = 1
coordinator_ranks = 0
scheduling = "static"
```

这需要四个 rank：两个轨迹组，每组两个珠子 PES rank。每组的首个 rank 推进状态，其余 rank 计算 PES 批次；world rank 0 汇总结果。

```bash
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi-first
mpiexec -n 4 pyrpmd resume runs/mpi-first
```

world size 等于 `trajectory_groups * bead_workers`。静态调度默认 `pes_ranks=1`、`coordinator_ranks=0`，各珠子分区在所分配的 rank 上求值。所有节点需要相同的代码、环境和模型路径，以及共享的输出目录。

## 动态 MPI 队列

对提前停止时间不同的 dRPMD 轨迹，启用动态队列：

```toml
[parallel]
backend = "mpi"
mode = "bead"
scheduling = "dynamic"
coordinator_ranks = 1
trajectory_groups = 2
bead_workers = 2
threads_per_rank = 1
```

rank 数为 `1 + trajectory_groups * bead_workers`，上例需要五个 rank。
world rank 0 负责调度，空闲轨迹组继续领取任务；GPU 模式下调度 rank 不占 GPU 槽。
静态调度仍为默认。动态选项适用于 MPI，`pes_ranks=1` 约束继续有效。

```bash
mpiexec -n 5 pyrpmd run examples/drpmd/mpi_dynamic.py --output runs/dynamic-mpi
```

dRPMD 首先沿持久热化母链生成可续跑的初始构型，再将每条碰撞轨迹独立加入队列。
早停后立即释放组，任务索引、随机流和检查点保持不变，输出仍按任务顺序排列。
比较布局或改变 worker 数时显式设置 `sampling.parents`。
任务异常停止新分配，等待已运行的组完成并释放势能上下文，再向全部 rank 报错；
MPI 进程崩溃和网络故障由 launcher 处理。

## GPU 放置

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

这会启动四个轨迹进程，每张卡两个。设备号是经 `CUDA_VISIBLE_DEVICES` 映射后的本地编号；多节点 MPI 按节点内 rank 分配。
设备不可用或模型实际在 CPU 上运行时，CUDA 请求会报错。GPU 模式下每条轨迹使用一个珠子 worker。

## 保存与续跑

```bash
pyrpmd resume runs/example
pyrpmd resume runs/example --workers 4 --loop-backend auto --block-steps 128
```

续跑读取保存的解析输入、体系或 PCET 模型、q/p、RNG 状态、阶段和历史。已完成的任务保持完成，校验失败的检查点会以
错误终止续跑。worker 布局、后端和分块大小可以在续跑之间改变。dRPMD 省略 `sampling.parents` 时母链数随执行槽数而定，
若计划改变 worker 数，请显式设置 `parents`。

温度、珠子数、同位素、势能、时间步、积分器、CV 和采样设置构成运行身份，改变其中任何一项都需要新的输出目录。
模型与插件文件的哈希属于运行身份；间接 Python 依赖、系统库和远程 i-PI 服务通过运行环境固定。

检查单个检查点时，从运行目录中选一个 `.npz` 文件：

```bash
pyrpmd checkpoint runs/example/checkpoints/ACTUAL_FILE.npz --require-integrity
```

把 `ACTUAL_FILE` 换成实际文件名；各工作流的文件名不同。NPZ 容器按数组做 SHA256 校验，没有校验值的旧容器报告
`unverified_legacy`。随机流在同一平台上可复现；不同硬件或归约顺序可能改变最后几位。

## 故障

`Dynamics.propagate()` 只在每一步都成功时提交整块；`propagate_block()` 保留成功的步骤前缀。工作流在热浴、观测、hill
沉积和检查点边界提交状态。PES 失败后，异常信息和保存的阶段信息给出运行停止的位置；修正原因后从最后的检查点续跑。

详细约定见[并行](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/parallel.md)、
[原生传播](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-propagation.md)、
[分段 API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/block-propagation.md)和
[检查点](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/checkpoints.md)。
