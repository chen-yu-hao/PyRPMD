# 并行执行与可复现性

PyRPMD 分开配置任务传输方式 `parallel.backend` 和单条轨迹的势能计算方式
`parallel.mode`。`serial`、`bead`、`gpu` 三种势能模式都可以同时运行多条轨迹；
`serial` 模式只表示一次 PES 调用不拆分珠子。常规 RPMD 的独立窗口、parent
和 child 任务也使用同一执行器。

| 配置 | 实际执行方式 | 适用情况 |
| --- | --- | --- |
| `backend="serial"`, `trajectory_workers=1` | 当前进程运行独立任务 | 调试、轻量任务 |
| `trajectory_workers=n > 1`，非 MPI | `spawn` 启动 n 个轨迹进程，静态分配任务 | 单机并行轨迹、窗口或 replica |
| `mode="bead"`, `bead_workers=B > 1`，非 MPI | 每条并发轨迹持有 B 个持续 PES 子进程 | 较昂贵或不可重入的 PES |
| `backend="mpi"` | G 个轨迹组，每组 B 个 rank | 单机多 rank 或跨节点 MPI |
| `mode="gpu"` | 每个轨迹进程独立加载 GPU 模型，一次批处理全部珠子 | 单卡、多卡，每卡一条或多条并发轨迹 |

`threads_per_rank` 控制已编译的 OpenMP 动力学内核；要求真正可用的 OpenMP
构建。本地 task batch 使用 `threadpoolctl` 将已加载的 BLAS 线程池限制为 1，
同时在任务作用域内设置 OpenBLAS、MKL、BLIS、Accelerate 的线程环境变量，
使通常在任务中导入的库也采用单线程。任务完成或抛出异常后恢复原先已加载
线程池的设置和环境变量；只有一个实际 task、直接在调用进程执行时也如此。
持续 bead 子进程在初始化时限制线程池，并保持到子进程退出，无需每次 PES
调用重新扫描库。该限制只针对 BLAS，不修改 `OMP_NUM_THREADS` 或动力学
内核的 `threads_per_rank`。

这些 BLAS 设置是进程级状态，不能隔离同一进程中其他 Python 线程的并发
数值计算。MPI 路径和外部势能引擎的专有线程池仍需按部署与引擎配置，避免
轨迹数 × 珠子进程数 × 引擎线程数超过分配的 CPU 资源；MPI 启动前可设置
`OPENBLAS_NUM_THREADS=1` 和 `MKL_NUM_THREADS=1`。对于很快的解析势，
进程通信通常比 PES 本身更贵，优先增加独立轨迹。

## MPI 拓扑

当前实现要求 `MPI world size = G × B`，其中
`G = trajectory_groups`，`B = bead_workers`。组首 rank 推进积分与轨迹状态；
其余 rank 持续服务珠子 PES 请求。一次力调用使用本次提交坐标的结果，保持珠子
顺序，不复用过期的异步力。时间步只在所属组内同步，不插入 world barrier。

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 4
bead_workers = 2
threads_per_rank = 1
pes_ranks = 1
coordinator_ranks = 0
scheduling = "static"
```

以上配置需要 8 个 MPI rank。当前只支持 `pes_ranks=1`、静态任务分配和
`coordinator_ranks=0`；专用动态调度器和单次 PES 内部再嵌套 MPI 的 Q 层
没有实现，请勿把预留字段视作已有能力。不支持的值会明确报错。

跨节点运行需要所有节点使用兼容的 MPI/mpi4py、Python、PyRPMD 和势能库。
Python factory 必须能在每个执行进程中导入，数据文件须位于相同路径；输出目录
须是各节点共同可见的共享存储。仅组首写该组轨迹文件，最终统计由 world rank 0
完成，再广播阶段结果。Slurm 模板见 `examples/slurm/`；实际 `srun --mpi`
插件、模块环境和节点分配需与集群匹配。

## GPU 设备与每卡多轨迹

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

这会建立 4 个轨迹执行进程：GPU 0 两个、GPU 1 两个。设为
`trajectories_per_gpu=1` 即每卡一个。每个进程都有自己的模型、副本缓存和轨迹
状态，显存预算必须乘上每卡进程数；这不是多个轨迹共享一个模型实例。
当前 GPU 模式由 provider 批处理全部珠子，不同时启用 `bead_workers > 1`。
GPU provider 必须声明并使用所请求的 CUDA 设备；设备缺失不会静默回退 CPU。

`devices` 是经过 `CUDA_VISIBLE_DEVICES` 映射之后的本地设备编号。MPI GPU
执行通过 `MPI.COMM_TYPE_SHARED` 得到节点本地 rank，按本地 rank 选卡，
不使用全局 rank 直接推导 CUDA 编号。每个节点须运行
`len(devices) × trajectories_per_gpu` 个 rank，并让这些 rank 都看到该节点
配置的设备列表。例如两节点、每节点两卡、每卡两个轨迹进程，对应 world size
8、G=8、B=1。各节点的编号 `[0,1]` 指各自的两张卡。

## 持续 parent 链与续跑

dRPMD 使用 n 个独立热 parent。每个 parent 只做一次 equilibration，随后
每隔 `sampling.interval_steps` 从同一条持续链抽取完整 q/p，组装一条碰撞
轨迹。请求的 N 是全作业总数：前 `N % n` 个 parent 各获得 `N // n + 1`
条，其余各获得 `N // n` 条。生产轨迹与 parent 状态分离，生产阶段关闭热浴。

parent RNG 由全局 seed 和 parent ID 派生，碰撞抽样 RNG 由全局 trajectory
ID 派生。检查点原子保存 parent q/p、RNG、热化进度、待运行 child、child q/p、
分类保持时间及完成记录。中断后，已经完成的轨迹只计数一次；恢复终态检查点
也不会多推进一个时间步。

显式固定 `sampling.parents`，就可以在保持 parent 归属与物理设置的前提下
改变 CPU worker 数或 MPI G/B 布局。若省略 parents，它默认取轨迹执行槽数；
改变槽数也改变 parent 数，此时检查点会拒绝不兼容的续跑。模型、PES 库、
manifest、声明的 Python factory 源码等本地文件会参与 SHA256 校验。
任意 Python 传递依赖、系统动态库和远程 i-PI 引擎的内部状态没有被完整指纹化，
应通过固定运行环境保证一致。改变设备或并行归约方式可能改变浮点舍入，
相同随机流不等于不同硬件逐位相同。

最小 dRPMD pilot 输入可省略分类器。此时以已有共价半径表推断 centroid
键阈值，并将新分子组成记录为 candidate 通道；推断数值和假设写入 resolved
input、summary 与物理检查点身份。`classification_unvalidated=true` 时仅提供
几何事件计数，不输出正式化学速率、截面或化学反应界。同组成交换或内部键图
变化保持 unresolved；未知元素或初始片段不符合推断图时明确要求显式分类配置。
这些 pilot 规则不因使用更多进程或 MPI rank 而获得验证。

## 已验证的范围

单机测试覆盖真实 `spawn` 多轨迹、持续珠子进程、二者组合、5 条轨迹分配给
2 个 parent、非整除珠子批次、模型文件变更拒绝续跑、运行中及终态断点恢复。
初始碰撞构型的 PES 域错误会作为 failed 轨迹计入未知结果范围；其缺失能量
记录为 JSON null 和空能量轨迹，不冒充数值结果。

`tests/mpi_integration.py` 使用本机真实 4-rank MPI 验证 G=2、B=2 的势能
通信、有序结果、初始化异常传播、持续 parent 和续跑。命令行 MPI 示例也已
实际运行与恢复。`tests/mpi_workflow_smoke.py` 还以真实 G=2、B=2 验证了
FES，以及 rate 的 PMF、parent、child 多阶段广播与各自的续跑。
GPU 设备分配测试使用 mock provider 检查每卡 1 个或多个
进程和 MPI 节点本地路由；没有实际 CUDA 设备计算测试，也没有多物理节点的
性能或扩展性测量。Slurm 文件是可检查的提交模板，不代表已在目标集群提交运行。
