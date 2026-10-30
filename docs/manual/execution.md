# 执行、并行与续跑

## 区分积分方法与执行路径

积分器决定数值动力学；native/Python 与分块设置决定怎样执行同一个计算。
选择 C++ 不应改变势能、质量、温度、偏置或随机采样语义。

```toml
[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"
native = true
loop_backend = "auto"
block_steps = 256
```

| 设置 | 意义 |
| --- | --- |
| `native=true` | 优先编译积分子步骤，允许参考实现回退 |
| `native=false` | 使用 NumPy 积分路径 |
| `native="required"` | 必须具有所选积分器的编译能力 |
| `loop_backend="auto"` | 在满足 PES/CV/偏置能力时融合整段传播，否则兼容路径 |
| `loop_backend="python"` | Python 外层循环，积分子步骤仍可编译 |
| `loop_backend="native"` | 严格要求融合整段传播，不满足时报错 |
| `block_steps` | 执行分块大小；允许范围 1–65536 |

周期或固定原子体系目前走兼容的 Python 外层循环，可使用编译的确定性积分子步骤。
PILE-L 采样和动态 metadynamics 历史也有各自能力限制。
查看运行中的传播统计和回退原因，不能根据安装成功推断所有路径都已融合。

实验性 [NRPMD](nonadiabatic.md) 的电子矩阵传播使用 CPU/NumPy，支持串行或本地轨迹进程；
不支持 MPI、珠子 worker 拆分、GPU 或融合原生电子传播。核自由环子步骤可以使用现有编译内核。

## CPU 轨迹与珠子并行

通常先并行独立窗口、母链或副本。单个 PES 很昂贵时，再考虑将珠子批次分给多个进程。

```toml
[parallel]
backend = "process"
mode = "serial"
trajectory_workers = 4
threads_per_rank = 1
```

这里 `mode="serial"` 表示一次 PES 调用不拆珠子，仍有四个独立轨迹执行进程。
如果改为 `mode="bead"`、`bead_workers=2`，每条并发轨迹会拥有持续的珠子 PES 进程。
这些设置不把 `collision.trajectories` 解释成“每个 worker 的数量”：该字段始终是全任务总数。

很快的解析势可能受进程通信开销限制。合理分配
轨迹数 × 珠子进程数 × 外部势能/BLAS 线程数，避免超出分配 CPU。常见环境设置为：

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

对应四个 rank：两组独立轨迹，每组两个珠子势能 rank。
每组首 rank 推进状态，其他 rank 服务 PES 批次；world rank 0 汇总结果。

```bash
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi-first
mpiexec -n 4 pyrpmd resume runs/mpi-first
```

world size 必须等于 `trajectory_groups * bead_workers`。
当前实现为静态分组，`pes_ranks=1`、`coordinator_ranks=0`；
不支持任意外部 PES 内部再次嵌套一个 MPI communicator。
所有节点需要一致的代码、环境和模型路径，以及共同可见的输出目录。
单机四 rank 验证与跨物理节点的扩展性测量是不同证据。

## GPU 放置

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

上例建立四个独立轨迹进程，每张卡两个模型实例。设备号是 `CUDA_VISIBLE_DEVICES`
映射后的本地编号；MPI 多节点使用节点本地 rank 路由。
请求 CUDA 而模型实际运行在 CPU，或设备不可用，都会报错。
GPU 模式不同时做多进程珠子拆分。具体任务仍要满足该 provider 的支持范围。

## 保存和恢复

```bash
pyrpmd resume runs/example
pyrpmd resume runs/example --workers 4 --loop-backend auto --block-steps 128
```

恢复读取保存的 resolved input、系统/PCET 模型、q/p、RNG、阶段与历史状态。
已完成的任务不会再次增加轨迹，检查点校验失败不会被静默跳过。
仅更改 worker 布局、后端与块大小属于执行层调整；具体工作流仍有约束。
例如 dRPMD 若省略 `sampling.parents`，母链数可能取执行槽数；变更 worker 后若改变了
母链身份，则不能作为同一计算继续。为了可移植调度，显式设置 parents。

温度、珠子数、同位素、势能、时间步、积分器、CV 和科学采样设置属于运行身份。
改变这些设置应新建输出目录。模型和显式插件文件的哈希参与校验；间接 Python 依赖、
系统动态库及远程 i-PI 服务状态仍需由环境固定保证。

检查单个文件时，从运行目录中选择实际存在的 `.npz`：

```bash
pyrpmd checkpoint runs/example/checkpoints/ACTUAL_FILE.npz --require-integrity
```

`ACTUAL_FILE` 是占位符，各工作流的文件名不同。
NPZ 容器以逐数组 SHA256 检查完整性；旧容器的 `unverified_legacy` 与校验通过含义不同。
同一随机流也不保证不同硬件或不同归约顺序逐位相同。

## 故障时的状态边界

低层 `Dynamics.propagate()` 仅在整块成功时提交该块；`propagate_block()` 保留成功的步骤前缀。
工作流在热浴、观测、hill 沉积与检查点事件边界提交对应状态。
遇到 PES 失败时应查看异常与保留的阶段信息，不能直接把最后一帧当作完整生产结果。

详细契约见[并行](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/parallel.md)、
[原生传播](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-propagation.md)、
[分段 API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/block-propagation.md)和
[检查点](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/checkpoints.md)。

