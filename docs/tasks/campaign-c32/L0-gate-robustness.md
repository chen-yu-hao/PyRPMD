---
orphan: true
---

# L0 门禁、冻结环境与跨工作流鲁棒性

| 字段 | 内容 |
| --- | --- |
| CPU | Phase 0（T0–T4）整机 0–31；Phase 1 起 0–1（2 核） |
| 预算 | Phase 1 92 核时（计划 60）；Phase 0 整机 4 小时另计 |
| 建议分支 | `agent/c32-l0-gate` |
| 记录 | `docs/validation/c32-l0-gate-<日期>/README.md`；Phase 2 另写 `docs/validation/c32-summary-<日期>/README.md` |
| 复核 | L0 复核 L6；L6 复核 L0 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L0 ...` |

## 目标

1. 在 T0–T4 建立并冻结全体车道共用的环境，证明它通过全部测试，并预先构建所有外部模型，
   使其他车道在 Phase 1 不需要安装或下载任何东西。
2. 在安静的整机上测一次扩展性（G16）：其他车道开始后就不再有干净的整机。
3. 在 Phase 1 用 2 个核做跨工作流的工程鲁棒性矩阵：中断续算、块长与后端变化、负载下测试套件。
4. Phase 2 汇总全部车道，写战役总记录。

## Phase 0（T0–T4，整机）

按顺序执行；每一步的命令、输出摘要和耗时写进记录。T+4 h 时未完成的步骤停止并报告，
其他车道照常开始（它们的外部模型缺失时会把相应子测试标为阻塞）。

### 0.1 冻结与机器清单（≤ 15 min）

- 按 [README](README.md#共享环境) 建立 `$C32_SRC`、`$C32_VENV`、`$C32_DATA`，运行
  `tools/campaign_launch.py check --machine` 和 `start`。`check --machine` 报告 SMT 兄弟核时，
  先改 `lanes.json` 的 `cpus`（一次提交，写明原因），再 `start`。
- 记录 `machine.json`：`lscpu -e`、`lscpu`、`numactl -H`（如有）、`free -g`、`df -h $C32_DATA`、内核版本、
  `gcc --version`、`gfortran --version`、`pip freeze`、`pyrpmd info`、扩展 `.so` 的 SHA256、冻结提交。
- 把共享环境变量写进 `$C32_ROOT/env.sh`，供所有车道 `source`。

### 0.2 可选依赖（≤ 20 min）

逐个安装，单独记录成功或失败；任何一项失败都不阻止其他步骤：

```bash
$C32_VENV/bin/python -m pip install "pymbar>=4" matplotlib       # L3 MBAR 交叉检查、全部图表
$C32_VENV/bin/python -m pip install mpi4py                        # 仅当系统有 MPI（mpiexec --version）
$C32_VENV/bin/python -m pip install openmm                        # L5 的 q-TIP4P/F 经典输运（可选）
$C32_VENV/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch   # L4 的 torch 引擎（可选）
```

安装完成后再运行一次 `pyrpmd info` 并更新 `machine.json`。**T+4 h 之后共享环境不再变化。**
如果安装改变了已记录的扩展 SHA256（不应发生），重新 `start` 之前停止并报告。

### 0.3 三种构建的测试（并行，约 60 min）

| 构建 | 环境 | 命令 | 判据 |
| --- | --- | --- | --- |
| 默认（OpenMP） | `$C32_VENV` | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q`（含 campaign） | 0 failed；跳过项逐条归类（可选依赖、平台） |
| NumPy 路径 | 新 worktree `$C32_ROOT/src-nonative` + 新 venv，`PYRPMD_NO_NATIVE=1 pip install -e "$C32_ROOT/src-nonative[test]"` | `python -m pytest -q -m "not campaign"` | 0 failed；`native='required'` 的用例应明确报错并被测试覆盖 |
| FMA / 无 OpenMP | 新 worktree `$C32_ROOT/src-fma` + 新 venv，`CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native" pip install -e ...`（记录实际编译选项）；再一个 `src-noomp` 用 `PYRPMD_NO_OPENMP=1` | `python -m pytest -q -m "not campaign"` | 0 failed（G7） |

NumPy 与 FMA 构建使用同一冻结提交的**另一个** worktree（`git worktree add --detach`），
通过 `--alternate-build --kind test` 启动，避免污染共享构建。三组测试的原始 pytest 汇总行原样写进记录。
有 MPI 时再运行 `docs/development.md` 列出的 `mpiexec -n 4 python tests/mpi_integration.py`
和 `tests/mpi_workflow_smoke.py`。

### 0.4 外部模型（与 0.3 并行，≤ 40 min）

| 模型 | 命令 | 使用车道 |
| --- | --- | --- |
| BKMP2 H₃ | `python benchmarks/h3/build_pes.py`；`python benchmarks/h3/pes.py --check --output $C32_DATA/L0/h3-pes-check.json`；`python benchmarks/h3/numerics.py --output $C32_DATA/L0/h3-numerics.json` | L1、L2 |
| OGRe 二维模型 | `python benchmarks/fes_methods/ogre_benchmark.py build --output $C32_DATA/shared/ogre` | L3 |
| Ni–H EAM | `python benchmarks/materials/ni_h_eam.py --download --model $C32_DATA/shared/NiAlH_jea.eam.alloy --output $C32_DATA/L0/ni-h-pilot --quick` | L5 |

把每个共享库/模型的绝对路径与 SHA256 写进 `$C32_ROOT/env.sh`（例如 `C32_BKMP2`），并设为只读。
需要 GNU Fortran 的模型在缺少 `gfortran` 时标为阻塞，不要换用其他源码。

### 0.5 全部示例冒烟（与 0.3 并行，≤ 30 min）

对 `docs/manual/examples.md` 中不需要外部模型的每个输入执行 `pyrpmd validate`（加 `--check-pes`）→
`run` → `analyze` → `resume`，Python 示例直接运行。记录每个示例的墙钟时间、返回码和 `summary.json` 的关键字段。
这一步的目的是让其他车道不在第一小时撞上基础故障；任何失败立即写进 FINDINGS 并通知受影响的车道。

### 0.6 整机扩展性基准（T+2.5 h 起，机器上只运行这一项，≤ 75 min）

只测，不调优；每个配置重复 3 次，报告中位数和范围，写明命令、线程数、编译器（G16）。

1. 内置原生势吞吐量：`examples/benchmark_builtin_native.py --label c32 --backends auto native python --output ...`。
2. Python/原生边界开销：`benchmarks/efficiency/native_ensembles.py --output ...`。
3. 进程池强扩展：同一个固定的 BKMP2 dRPMD 任务（例如 `benchmarks/h3/drpmd.py --trajectories 512 --parents 16`），
   `--workers` = 1、2、4、8、16、32；报告墙钟、并行效率和逐轨迹结果是否与 `workers=1` 逐位一致。
4. OpenMP 线程扩展：原生 PIMD（`examples/ensembles/pimd_native.inp` 放大到 N=512 与 N=4096 的内置势）
   线程数 1、2、4、8、16、32；同时检查不同线程数下轨迹是否逐位一致（不一致时如实报告，不判失败）。
5. 有 MPI 时：`mpiexec -n 4/8/16/32` 运行 `examples/drpmd/mpi.py` 的 G×B 布局，报告扩展与结果一致性。

## Phase 1（2 核，T4–T50）

### 1.1 中断续算矩阵（约 30 核时）

对每个工作流选一个代表输入（下表），先跑一次不中断的参考，再跑 3 次被 `SIGKILL` 中断的副本
（中断时刻由固定种子在运行时长的 20%–80% 之间抽取），用 `pyrpmd resume` 续完，然后比较全部结果数组与
`summary.json`（时间戳、主机名、耗时、传播计数等执行元数据除外）。

| 工作流 | 输入 | 额外的续算变化 |
| --- | --- | --- |
| `fes` umbrella | `examples/manual/quickstart.inp`、`examples/fes/harmonic2d.inp` | 续算时改 `--block-steps 128` |
| `fes` REUS | `examples/fes/replica_exchange_2d.inp` | — |
| `fes` adaptive | `examples/fes/adaptive_harmonic.inp` | — |
| MetaD / WT-MetaD | `examples/metadynamics/standard.inp`、`well_tempered_2d.inp` | 续算时改 `--loop-backend` |
| OPES | `examples/opes/harmonic_2d.inp` | — |
| `rate` | `examples/rate/pilot.inp`；`examples/rate/research.inp`（若可在 15 min 内完成） | 分别在 PMF、母链、子轨迹阶段中断 |
| `drpmd` | `examples/drpmd/free_scattering.inp` 放大到 64 条碰撞 | 改 `--block-steps` |
| state-selected | `examples/state_selected/ne_rpmd.inp` | — |
| `pimd`（python / native） | `examples/materials/pimd.inp`、`examples/ensembles/pimd_native.inp`、`pimd_native_npt.inp` | — |
| `diffusion` | `examples/materials/diffusion.inp` | — |
| `site_rate`、`surface_rate` | `examples/materials/site_rate.inp`、`examples/surface/diffusion.inp` | — |
| `pcet` | `examples/pcet/harmonic_scan.inp` | — |
| `nonadiabatic` | `examples/nonadiabatic/exchange.inp`、`population.inp` | — |
| `path_sampling` | `examples/sampling/path_sampling_retis.inp`、`path_sampling_infinite_retis.inp` | — |

判据：每个中断副本续算后与参考**逐位一致**（G5）；改块长或改循环后端的续算与参考逐位一致（R5），
或在该工作流声明为 E1 时在声明的容差内一致。工作流明确拒绝续算（例如 PLUMED）时，记录拒绝信息是否清楚。

### 1.2 后端一致性（约 10 核时）

对 1.1 中声明 E0 的输入，比较 `loop_backend` = `python`、`native`、`auto` 以及 NumPy 构建（`--alternate-build`）
的最终结果：E0 要求逐位一致，E1 按 `docs/tasks/native-path/README.md` 的 R1 比较短时程逐帧差异。
长时程统计等价由各科学车道负责（L1 速率链、L3 历史偏置、L4 PILE-L）。

### 1.3 负载下的测试套件（约 15 核时）

在 Phase 1 的 T+8 h、T+24 h、T+40 h 各运行一次 `python -m pytest -q -m "not campaign"`（`--slots 2`），
此时其他 30 个核满载。任何失败都重跑该单个测试 20 次，记录失败率；失败率大于 0 的测试按 T21 的经验
分析是否依赖时序，写进 FINDINGS，不在战役中修改。

### 1.4 监控（每小时，轻量）

每小时追加一行 `$C32_DATA/L0/monitor.jsonl`：`status --json`、`uptime` 负载、`free`、`df`、
`ps` 中不属于任何 ledger 记录的高 CPU 进程。负载持续超过 32 或某条车道超预算时，通知该车道。

## Phase 2（T50–T60）

1. 写本车道记录：0.1–0.6 与 1.1–1.4 的结果、FINDINGS。
2. 复核 L6（C13）。
3. 按 [README](README.md#集成者l0在-phase-2-的工作) 汇总全部车道，写 `docs/validation/c32-summary-<日期>/README.md`。

## 验收

- 三种构建的测试汇总行、MPI 结果和跳过分类都在记录中；共享环境的 `machine.json` 可以让第三方重建同一环境。
- 1.1 的矩阵每一格都有结论（一致 / 不一致 / 工作流拒绝 / 阻塞），不一致的格子有最小复现。
- 扩展性表给出原始数字和硬件，不外推到其他机器。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L0 agent，负责"门禁、冻结环境与跨工作流鲁棒性"，
也是本战役的集成者。这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L0-gate-robustness.md。

要求：
- T0 立即建立共享环境并运行 tools/campaign_launch.py check --machine 与 start；把 env.sh 路径告诉其他车道。
- Phase 0（T0–T4）整机归你：按卡片 0.1–0.6 的顺序执行，0.6 扩展性基准时机器上不能有其他计算。
- T4 之后你只有 CPU 0–1（2 核，预算 92 核时）；所有计算用 $C32_LAUNCH run --lane L0 启动。
- 原始 pytest 汇总行、命令和硬件原样记录；不一致或失败如实报告，不修代码，写进 FINDINGS.md。
- 每小时追加 monitor.jsonl；T+24 h、T+40 h 汇总各车道 PROGRESS.md。
- Phase 2：在分支 agent/c32-l0-gate 提交记录并开 PR（G11 + 判据汇总表）；复核 L6；
  写战役总记录 docs/validation/c32-summary-<日期>/README.md。不合并任何 PR。
```
