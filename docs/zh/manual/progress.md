# 进度监控与结构化日志

## 查询运行中的计算

在第二个终端运行状态查询；`status` 只读，不会初始化势能面、推进动力学、重新分析样本或修改运行目录。

```bash
pyrpmd status runs/job
pyrpmd status runs/job --watch
pyrpmd status runs/job --watch --interval 5
pyrpmd status runs/job --watch --json > progress-snapshots.jsonl
```

单次查询输出 JSON，包含 `status`、`stage`、`completed_steps`、`total_steps`、`percent`、分阶段计数器、耗时和
`eta_seconds`。`--watch` 默认每两秒刷新一次：在交互式终端中显示单行动态刷新，重定向或提供 `--json` 时输出
JSON Lines。刷新间隔必须是有限正数。计算进入终态后监控自动停止；Ctrl-C 只停止监控器（退出码 130），不会中断计算。
缺少快照时会给出可操作的错误说明：目录可能尚未开始运行，或早于该功能创建。`status` 不会根据已完成摘要编造实时状态，
也不会代用户调用 `resume`。

## 计数器含义

计数器统计的是**已提交到检查点的积分步数**（跨独立副本、窗口或轨迹求和），而非珠子力求值次数。在下一个检查点写入前，
它们会滞后于实时计算。协调进程每两秒刷新一次快照；提高 CLI 轮询频率不会使检查点提前出现。检查点频率由既有的
`output.checkpoint_steps` 或 `output.checkpoint_interval` 控制。

| 工作流 | 计入的工作量 |
| --- | --- |
| 独立固定窗口伞形 FES | 已保存的窗口/副本时钟，总预算取自声明的窗口计划 |
| 独立 metadynamics 与 OPES | 已保存的副本时钟及其有限采样预算 |
| 气相 `rate` | 伞形/PMF 采样、受约束母链，以及每对已提交回穿子轨迹的两条分支 |
| `pimd` 与 `diffusion` | 平衡与生产阶段（含 Python、原生 journal 及 Torch 批量存储路径） |
| `drpmd` | 仅碰撞传播阶段；不含母链热化准备 |
| 显式 `run --stage subspace` | 仅记录生命周期与阶段；无积分步百分比 |

环境设置、数值准备、后处理与科学收敛不计入积分步百分比。报告的阶段由已提交工作推断，在下一个检查点之前可能滞后于实时阶段。
无中间检查点的速率子轨迹仅在一对带符号子轨迹完整保存后推进计数。提前停止的 dRPMD 轨迹可在少于最大传播预算 100% 时正常完成；
同様に，当 PMF 缺少支持时，速率工作流会跳过子轨迹阶段。已完成的命令绝不会被伪造成执行了未实际运行的积分步。

自适应计划、副本交换、共享/分布式 walkers、并行回火（tempering）以及其他没有固定时钟/预算的工作流，将未知计数或总预算输出为
JSON `null`（终端显示为 `?`），不会伪造百分比或 ETA。读取失败或损坏的检查点时钟会记录为未知并附带警告，而非归零或视为已完成。
这些轻量读取器仅检查标量时钟，不读取完整坐标历史；它们仅作参考，不能替代 `resume` 执行的完整性校验。

`eta_seconds` 是**当前阶段**的剩余时间估计（`eta_scope="current_stage"`），仅基于本次调用新提交的工作量计算。
因此，恢复一段很长的历史计算不会产生接近零的虚假 ETA。阶段切换会重置该估计；在观察到可测量的新增工作和已知剩余预算之前，
ETA 保持未知。它不是对初始化、后续阶段或分析耗时的承诺。

## 文件与输出流

在输出目录初始化与预检成功后，Runner 会创建两个非科学产物文件：

- `progress.json`：schema 为 `pyrpmd.progress`（版本 1），通过唯一临时文件、flush/fsync 与原子重命名替换更新。
- `run.log.jsonl`：只追加的结构化生命周期/进度事件流。每条事件包含名称、级别、尝试 `run_id`、时间戳、阶段和计数器。

事件包括 `run.started`、节流的 `run.progress`，以及终态事件 `run.completed`、`run.failed`、`run.interrupted` 或 `run.partial`。
同样的事件写入 stderr。**Stdout 仍保留命令的最终结果 JSON**，既有结果管道无需改动：

```bash
pyrpmd run job.inp --output runs/job > result.json 2> terminal.log
pyrpmd status runs/job --watch
```

`terminal.log` 还可能包含 Python 警告和 `--debug` traceback；`runs/job/run.log.jsonl` 则只包含结构化运行事件。
`--debug` 保持原有的错误 traceback 行为，进度监控不需要开启它。日志写入异常最多触发一次尽力而为的 `progress.warning`，
不会覆盖科学计算异常或改变数值结果。

`resume` 会以新的 `run_id` 追加一轮新的尝试记录。`extend` 报告**已授权的有效总预算**，而非最初较短的输入预算。
速率扩展会将继续推进的伞形时钟与新一代母链和子轨迹组合，不会重复计入原始代次的结果。显式子空间后处理会更新运行根目录下的这两个参考文件，
但不会改写源 PMF、速率、输入或检查点。

## MPI、多进程与停止的作业

只有主进程（MPI world rank 0）持有进度报告器、写入这两个文件并输出运行事件。本机 spawn/fork 工作进程和非 root 的 MPI rank
只写入各自原有的任务检查点，不会重复打印进度消息，也不会争用 `progress.json`。心跳线程仅执行文件系统读取，从不发起 MPI 调用，
因此不会引入额外的 MPI 线程级别要求。MPI 运行仍需配置共享输出文件系统。

捕获到异常时记录 `failed`；`KeyboardInterrupt` / `SystemExit` 记录 `interrupted`。程序不会安装额外的信号处理器。
机器宕机、`SIGKILL` 或其他无法捕获的终止可能使最后一次快照停留在 `running`。`status` 会报告 `heartbeat_age_seconds`，
当超过 30 秒或三个心跳周期中的较大者时将其标记为陈旧（stale）并隐去 ETA。心跳过期**不能证明远程进程已经退出**：
请先检查作业调度系统和文件系统，切勿仅因心跳过期就并发启动 `resume`。

无论是 `completed`、100% 还是新鲜的心跳，都不代表物理收敛。请继续检查任务保存的诊断、警告以及
[如何使结果收敛](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/convergence.md)。
