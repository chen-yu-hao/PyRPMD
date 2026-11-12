# 进度监控与结构化日志

## 查询运行中的计算

在另一个终端运行只读的状态查询：

```bash
pyrpmd status runs/job
pyrpmd status runs/job --watch
pyrpmd status runs/job --watch --interval 5 --json
```

查询不会初始化势能面、推进动力学或修改运行目录。一次查询输出 JSON，包含阶段、已提交步数、总预算、百分比、
阶段计数器、耗时和当前阶段的 `eta_seconds`。`--watch` 默认每两秒刷新；终端中显示单行，重定向或使用 `--json`
时输出 JSON Lines。计算进入终态后查询自动停止；Ctrl-C 只停止查询并返回 130，不会停止计算。

## 计数器含义

计数器表示已写入检查点的积分步数，可能滞后于实时计算，且不包含设置、数值准备、后处理或科学收敛。速率子轨迹
只有在一对轨迹完整保存后才增加计数。没有固定预算或预算未知的工作流输出 JSON `null`，不会伪造百分比或 ETA。
恢复运行会以新的日志批次计算 ETA；之前的历史不会被当作本次调用的新工作。

完成、100% 或新鲜的心跳都不代表物理收敛，仍需检查任务自身的诊断、警告和收敛标准。

## 文件与输出流

运行初始化成功后，主进程（MPI rank 0）写入两个非科学文件：

- `progress.json`：`pyrpmd.progress` v1 的原子替换快照。
- `run.log.jsonl`：追加写入的结构化生命周期和进度事件。

事件包含 `run.started`、节流的 `run.progress` 以及 `run.completed`、`run.failed`、`run.interrupted` 或
`run.partial`。最终结果 JSON 仍写入 stdout；结构化事件和监控警告写入 stderr。心跳过期只表示快照可能陈旧，
不能据此判断远程作业已退出，也不要仅因心跳过期就并发执行 `resume`。
