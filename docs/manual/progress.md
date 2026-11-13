# Monitoring progress and structured logs

## Query a running calculation

Use a second terminal; status is read-only and does not initialize a potential,
propagate dynamics, reanalyze samples or modify the run.

```bash
pyrpmd status runs/job
pyrpmd status runs/job --watch
pyrpmd status runs/job --watch --interval 5
pyrpmd status runs/job --watch --json > progress-snapshots.jsonl
```

A one-shot query prints JSON, including `status`, `stage`, `completed_steps`,
`total_steps`, `percent`, per-stage counters, elapsed time and `eta_seconds`.
`--watch` refreshes every two seconds by default. It renders a single updating
line in an interactive terminal, or JSON Lines when redirected or when `--json`
is supplied. The interval must be finite and positive. Watch stops at a terminal
state; Ctrl-C stops only the watcher, with exit code 130, not the calculation.
A missing snapshot produces an actionable error: the directory may not have
started running or may predate this feature. `status` does not invent a live
state from a finished summary or call `resume` on the user's behalf.

## What the counters mean

Counters are **committed integration steps**, summed across independent
replicas, windows or trajectories, not bead force evaluations. They lag live
computation until its next checkpoint. The coordinator refreshes its snapshot
every two seconds; increasing the CLI polling frequency does not make a
checkpoint appear sooner. The existing `output.checkpoint_steps` or
`output.checkpoint_interval` controls checkpoint cadence.

| Workflow | Counted work |
| --- | --- |
| Independent fixed umbrella FES | Saved window/replica clocks, with the declared window plan's total |
| Independent metadynamics and OPES | Saved replica clocks and their finite sampling budgets |
| Gas-phase `rate` | Umbrella/PMF sampling, constrained parents and both members of each committed child pair |
| `pimd` and `diffusion` | Equilibration and production, including Python, native-journal and Torch-batch storage |
| `drpmd` | Collision propagation only; parent preparation is excluded |
| Explicit `run --stage subspace` | Lifecycle and stage only; no integration-step percentage |

Setup, numerical preparation, postprocessing and scientific convergence are
not measured by the integration-step percentage. The reported stage is inferred
from committed work; until the next checkpoint it can lag the live stage.
Rate children without intermediate checkpoints advance their counters only when
a complete signed pair is saved. Early-stopping dRPMD trajectories can complete
successfully using less than 100% of the maximum propagation budget. Likewise,
a rate workflow can deliberately skip children when its PMF is unsupported.
A completed command is never made to look as though it integrated unperformed
steps.

Adaptive plans, replica exchange, shared/distributed walkers, tempering and
other workflows without a supported fixed clock/budget expose unknown counters
or totals as JSON `null` (shown as `?` in the terminal). No arbitrary percentage
or ETA is manufactured. A failed or unreadable checkpoint clock is unknown,
with a warning, rather than zero or completed work. These lightweight readers
inspect scalar clocks, not full coordinate histories; they are advisory and do
not replace the integrity checks performed by restart.

`eta_seconds` is an estimate for the **current stage**, as indicated by
`eta_scope="current_stage"`. It uses newly committed work in the current
invocation. Resuming a long saved history therefore does not produce a false
near-zero ETA. Stage transitions reset the estimate; until there is measurable
new work and a known remaining budget, ETA is unknown. It is not a promised
wall time for setup, later stages or analysis.

## Files and terminal streams

The Runner creates two non-scientific artifacts after successful output
initialization/preflight:

- `progress.json`: schema `pyrpmd.progress`, version 1, atomically replaced
  using the existing unique-temporary-file, flush/fsync and rename utility.
- `run.log.jsonl`: append-only structured lifecycle/progress events. Each event
  includes its name, level, attempt `run_id`, timestamp, stage and counters.

Events include `run.started`, throttled `run.progress`, and a terminal
`run.completed`, `run.failed`, `run.interrupted` or `run.partial`. The same events
go to stderr. **Stdout remains the command's result JSON**, so existing result
pipelines keep working:

```bash
pyrpmd run job.inp --output runs/job > result.json 2> terminal.log
pyrpmd status runs/job --watch
```

`terminal.log` may also contain existing Python warnings and `--debug`
tracebacks; `runs/job/run.log.jsonl` contains only the structured run events.
`--debug` retains its existing error-traceback behavior and is not required for
progress. Logging errors emit one best-effort `progress.warning` and do not
replace a scientific exception or alter the numerical calculation.

`resume` appends a new attempt with a fresh `run_id`. `extend` reports its
**authorized effective budget**, not the original shorter input budget. Rate
extensions combine the continued umbrella clocks with the new generation's
parents and children, without double-counting the original result. Explicit
subspace postprocessing updates these two advisory files at the run root but
does not overwrite the source PMF, rates, inputs or checkpoints.

## MPI, multiprocessing and stopped jobs

Only the main process on MPI world rank 0 owns the reporter, writes the two
files and emits run events. Local spawned/forked workers and non-root MPI ranks
continue writing their existing task checkpoints; they do not print duplicate
progress messages or contend on `progress.json`. The heartbeat thread performs
filesystem reads only: it never makes MPI calls, avoiding a new MPI thread-level
requirement. MPI runs still need their existing shared-output-filesystem setup.

A caught exception records `failed`; KeyboardInterrupt/SystemExit records
`interrupted`. No new signal handlers are installed. A machine failure,
SIGKILL or other uncatchable termination can leave the last snapshot marked
`running`. Status adds `heartbeat_age_seconds` and marks it stale after the
larger of 30 seconds or three heartbeat intervals, withholding ETA. A stale
heartbeat **does not prove a remote process is dead**: check the job scheduler
and filesystem. Never start a concurrent `resume` merely because a heartbeat
is stale.

Neither `completed`, 100%, nor a fresh heartbeat establishes physical
convergence. Continue checking the task's saved diagnostics, warnings and
[convergence criteria](../convergence.md).

## 中文速览

用 `pyrpmd status RUN_DIR` 读取 JSON；用 `pyrpmd status RUN_DIR --watch`
在另一个终端刷新阶段、已提交步数、预算百分比和当前阶段 ETA。
`--interval 5` 调整查询间隔，`--json` 强制逐行 JSON 输出。
Ctrl-C 只退出查询，不会停止计算。

进度文件是原子更新的 `progress.json`，结构化日志是 `run.log.jsonl`。
日志由主进程／MPI rank 0 输出到 stderr，stdout 仍保留最终结果 JSON。
统计只反映已保存的 checkpoint，未知或自适应预算显示 `null`，不会伪造百分比。
恢复或扩展计算会使用新的日志批次；ETA 不把恢复前已完成的历史当作新完成的工作。
心跳过期仅表示状态可能陈旧，不能据此判断远程进程已退出；计算完成也不等于科学收敛。
