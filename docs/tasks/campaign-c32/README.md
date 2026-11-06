---
orphan: true
---

# C32 验证战役：32 核、60 小时、按工作流分车道的多 agent 测试计划

更新日期：2026-11-06。基线：`main@320bc29`（#125 合并后）。卡片中的路径、命令和参考数值以此提交为准；
执行时以 T0 冻结的提交为准，两者不一致时以代码为准并在记录中说明。

本目录把"PyRPMD 现在最应该做的测试"拆成 8 条彼此独立的车道，每条车道交给一个 agent，
在**同一台 32 核机器上并行**运行，**60 小时内**（2.5 天）完成全部计算、分析和记录。
每条车道有固定的 CPU、核时预算、预注册的判据和独立的 PR。

**用法：** 交给 agent 时，同时提供本页（规则 C1–C14 与共享环境）和对应车道卡片；
每张卡片末尾有可以直接复制的派发提示词。车道卡片不修改 `src/`、`native/`，
所以本战役不受 G2/G4 的文件所有权约束，但 `docs/tasks/README.md` 的 G3、G5、G9–G14 继续有效。

## 为什么是这些测试

1. **传播路径刚刚换过。** NP0–NP4（#119–#122）把动量刷新、PILE-L/TRPMD 采样块和历史偏置描述符移进 C++。
   单元测试证明了短时程 E0/E1 等价；规则 R1 要求的**长时程统计等价**（PMF、FES、κ(t)、扩散系数）
   还没有在真实工作流上做过。
2. **旗舰算例只差少量计算。** H + H₂/BKMP2 的三个未完成条件（伞形采样精度、P256 动态珠子收敛、
   母链时间对半等价）按已有记录的实测成本合计约 50 worker·h，32 核机器可以一次完成，
   还能独立重跑一遍已登记的协议（文献验证清单第 4 步：由他人从归档输入重现）。
3. **多个模型基准停在无法判定自身门槛的样本量。** 例如 Richardson–Thoss P=8 的置信区间因符号问题被扣留；
   16 个理想玻色子只有 3 个种子，95% 半宽 0.49，远大于论文的 0.5% 差异；Tully I 的 128 条步长子集
   无法分辨几个百分点的差别；OPES Müller 只跑了论文 2 亿步预算的 2%。这些都是解析或廉价模型，
   在 32 核上加量 10–50 倍只需几十核时。
4. **缺少跨工作流的一致性检验。** 同一模型、同一 P 下，dRPMD 与 Bennett–Chandler、
   `surface_rate` 与直接计数跃迁、RETIS 与约束速率应当给出同一个速率；这类检验比任何单个工作流的
   自洽诊断更能发现归一化和测度错误，目前一项都没有做过。
5. **工程鲁棒性需要跨工作流的矩阵。** 中断续算逐位一致（G5）、改块长不改结果（R5）、
   负载下测试套件不出现间歇失败（T21 的教训），都只在个别工作流上验证过。
6. **性能声明需要同机数字（G16）。** 32 核机器正好用来测进程池、OpenMP 与 MPI 的强扩展。

## 机器与预算假设

| 项目 | 假设 | 不满足时 |
| --- | --- | --- |
| CPU | 32 个物理核，逻辑 CPU 0–31 各对应一个物理核 | L0 在 T0 用 `lscpu -e` 检查；有 SMT 时改写 `lanes.json` 的 `cpus`，只用每个物理核的一个逻辑 CPU；只有 16 个物理核时所有生产预算减半 |
| 内存 | ≥ 64 GiB | 每条车道的常驻内存按 `核数 × 2 GiB` 计划；超出时减少并发而不是用交换区 |
| 磁盘 | ≥ 300 GiB 本地 scratch（`$C32_DATA`） | 原始数据不进 git；每条车道在 Phase 2 前把超过 1 GB 的中间文件压缩或删除，只保留 SHA256 清单 |
| 网络 | T0–T4 可以下载 BKMP2 源码、OGRe 模型、Ni–H EAM、可选 Python 包 | 下载失败的外部模型对应子测试标为"阻塞"，不得用其他模型替代 |
| 时间 | 60 小时墙钟；生产截止 T+50 h | 截止由启动器强制，未完成的计算按"未完成"报告 |

总容量 32 核 × 46 h（Phase 1）= 1472 核时，全部分给 8 条车道；各车道的"计划量"合计约 1200 核时，
其余约 20% 留给预注册的扩展、失败重跑和 pilot。Phase 0 的 4 小时整机归 L0。

## 时间线

| 阶段 | 墙钟 | 内容 |
| --- | --- | --- |
| Phase 0 | T0–T4 | **L0 独占整机**：冻结提交、构建三种扩展、全量测试、构建外部模型、整机扩展性基准、全部示例冒烟。其他车道只读卡片、读代码、写预注册和驱动脚本，**不跑任何计算**（启动器会拒绝） |
| Phase 1a | T4–T7 | 各车道 pilot（≤ 本车道预算的 10%），只用来测吞吐量和定规模；pilot 数据不进入主估计。按 pilot 结果定稿预注册并提交 |
| Phase 1b | T7–T50 | 生产。每个生产任务必须带已提交的预注册文件；所有任务可续算；T+50 h 启动器发 SIGTERM，120 s 后 SIGKILL |
| Phase 2 | T50–T60 | 只允许 `analysis`/`test` 任务：分析、交叉复核、写验证记录、开 PR。L0 汇总并写战役总记录 |

T+24 h 和 T+40 h 各有一次检查点：每条车道在自己的 `$C32_DATA/<lane>/PROGRESS.md` 写一段进度
（已完成的子测试、已用核时、预计能否在截止前完成、需要放弃的子测试）。L0 汇总 `status` 输出。

## 车道总览

| 车道 | 卡片 | CPU | 核数 | 预算/计划（核时） | 覆盖的工作流与入口 | 分支 | 复核方 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | [门禁、冻结环境与鲁棒性](L0-gate-robustness.md) | 0–1（Phase 0 整机） | 2 | 92 / 60 | 全量测试、全部示例、中断续算矩阵、后端等价、扩展性基准 | `agent/c32-l0-gate` | L6 |
| L1 | [气相速率主链：H + H₂/BKMP2](L1-rate-h3.md) | 2–11 | 10 | 460 / 400 | `rate`、`extend`、`study`、`report`，H₃ 量子 MC 与 P256 协议 | `agent/c32-l1-rate` | L2 |
| L2 | [碰撞、散射与跨路线速率](L2-collision-cross-route.md) | 12–15 | 4 | 184 / 150 | `drpmd`、state-selected、`network`、`study` | `agent/c32-l2-collision` | L1 |
| L3 | [自由能与增强采样](L3-fes-enhanced-sampling.md) | 16–19 | 4 | 184 / 150 | `fes`（umbrella、adaptive、REUS、MetaD/WT、OPES、ABF/eABF、VES、steered）、`[tempering]` | `agent/c32-l3-fes` | L7 |
| L4 | [平衡 PIMD 与量子统计](L4-pimd-equilibrium.md) | 20–23 | 4 | 184 / 150 | `pimd`（python/native/torch 引擎）、`[path_integral]`、热浴与压浴、凝聚相估计量、TRPMD/PA-CMD | `agent/c32-l4-pimd` | L5 |
| L5 | [材料、表面与输运](L5-materials-transport.md) | 24–26 | 3 | 138 / 110 | `diffusion`、`site_rate`、`surface_rate`、电子摩擦、`export`、可选 q-TIP4P/F | `agent/c32-l5-materials` | L4 |
| L6 | [非绝热动力学](L6-nonadiabatic.md) | 27–29 | 3 | 138 / 110 | `nonadiabatic`、`nonadiabatic-rate`、NRPMD/MASH/RPSH/FSSH、HEOM 参考 | `agent/c32-l6-nonadiabatic` | L0 |
| L7 | [隧穿、路径采样与 PCET](L7-tunnelling-paths-pcet.md) | 30–31 | 2 | 92 / 70 | `instanton`、`vtst`、低温估计、`path_sampling`、committor、`pcet` | `agent/c32-l7-tunnelling` | L3 |

机器可读的同一份计划在 [`lanes.json`](lanes.json)；`tools/campaign_launch.py check --plan` 会检查
CPU 不重叠、预算不超过容量、每条车道恰好被另一条车道复核。

**少于 8 个 agent 时：** 合并车道而不是合并 CPU，即一个 agent 依次领取两张卡、
分别在两组 CPU 上用各自的 `--lane` 启动任务。建议合并 L6+L7、L4+L5、L2+L3；L0 和 L1 不与其他车道合并。

## 共享环境

L0 在 T0 建立，所有车道只读使用。路径可以改，但变量名固定，卡片中的命令都依赖它们：

| 变量 | 含义 |
| --- | --- |
| `C32_SRC` | 冻结提交的独立 git worktree（共享、只读）；PyRPMD 以 editable 方式从这里安装 |
| `C32_VENV` | 共享虚拟环境；T0 之后**任何人都不得再 `pip install`** |
| `C32_DATA` | 原始数据根目录；每条车道只写 `$C32_DATA/<lane>/` |
| `C32_PLAN` | `$C32_SRC/docs/tasks/campaign-c32/lanes.json` |
| `C32_BKMP2` | Phase 0 构建的 BKMP2 共享库绝对路径 |
| `C32_LAUNCH` | `$C32_VENV/bin/python $C32_SRC/tools/campaign_launch.py` |

L0 的建立步骤（其余车道不执行）：

```bash
export C32_ROOT=/scratch/c32
export C32_SRC=$C32_ROOT/src C32_VENV=$C32_ROOT/venv C32_DATA=$C32_ROOT/data
git -C ~/PyRPMD fetch origin main
git -C ~/PyRPMD worktree add --detach "$C32_SRC" origin/main
python3 -m venv "$C32_VENV"
"$C32_VENV/bin/python" -m pip install --upgrade pip
"$C32_VENV/bin/python" -m pip install -e "$C32_SRC[test,analysis,ase]"
# 可选依赖逐个安装，失败只影响对应子测试（见 L0 卡片 0.2）
"$C32_VENV/bin/python" "$C32_SRC/tools/campaign_launch.py" check --plan "$C32_SRC/docs/tasks/campaign-c32/lanes.json" --machine
"$C32_VENV/bin/python" "$C32_SRC/tools/campaign_launch.py" start --plan "$C32_SRC/docs/tasks/campaign-c32/lanes.json" \
  --root "$C32_DATA" --frozen-commit "$(git -C "$C32_SRC" rev-parse HEAD)"
```

每条车道在自己的 worktree 中工作（预注册、驱动脚本、记录都提交在这里），运行时导入共享安装的 PyRPMD：

```bash
git -C "$C32_SRC" worktree add -b agent/c32-l3-fes "$C32_ROOT/lanes/L3" HEAD
cd "$C32_ROOT/lanes/L3"
export PYTHONPATH="$PWD"          # 让 benchmarks.* 可导入；pyrpmd 仍来自 $C32_SRC
# 不要把车道 worktree 的 src/ 放进 PYTHONPATH：那里没有编译扩展，启动器会拒绝
$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L3 --name ogre-dw-wt-s4101 \
  --kind production --prereg benchmarks/campaign_c32/L3/prereg-ogre.json -- \
  python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method well_tempered \
  --seed 4101 --output "$C32_DATA/L3/ogre"
```

启动器做的事：把任务钉在本车道的 CPU 上；把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 等设为 `--threads`；
拒绝超过本车道核数的并发槽位（`--slots` = 进程数 × 线程数；`--wait` 排队）；检查导入的 PyRPMD
仍是冻结提交且干净；生产任务要求预注册文件已提交、车道 worktree 干净；把命令、CPU 时间、峰值内存、
返回码、源码身份和预注册 SHA256 写进 `$C32_DATA/<lane>/ledger.jsonl`；到截止时间停止任务。
`$C32_LAUNCH status --plan "$C32_PLAN" --root "$C32_DATA"` 汇总每条车道已用核时。

## 通用规则（C1–C14）

- **C1 冻结。** 全部计算使用 T0 冻结的同一提交和同一构建。战役期间不修改 `$C32_SRC`，不 rebase，
  不 `pip install`。车道 worktree 只新增 `benchmarks/campaign_c32/<lane>/`（驱动、输入、预注册、分析脚本）
  和本车道的验证记录目录。
- **C2 只用启动器。** 所有计算都通过 `tools/campaign_launch.py run` 启动，`--slots` 如实填写
  （例如 `--workers 4` 的单线程任务填 `--slots 4`）。不在启动器之外运行超过 1 分钟的计算；
  读取结果、画图等轻量操作除外。
- **C3 先预注册，后生产。** 每个子测试在生产前提交一份预注册（格式见下文）：可观测量、参考值及来源、
  估计量与不确定度单位、种子、预算、判据与等价边界、停止与扩展规则。看到生产结果后不得修改判据；
  预注册的扩展只能由预注册中写明的条件触发。
- **C4 Pilot 只定规模。** Pilot 用独立的种子和输出目录，不超过本车道预算的 10%，
  只用于测吞吐量和检查运行；不进入任何主估计，也不能用来挑选参数以接近参考值。
- **C5 不改代码，记录缺陷。** 发现缺陷时，在 `benchmarks/campaign_c32/<lane>/FINDINGS.md` 写最小复现、
  影响范围和受影响的子测试，标记该子测试"阻塞"并继续下一个；不在战役中修 `src/` 或已登记的基准工具。
  修复在战役结束后按 `docs/tasks/README.md` 的规则另开卡片。
- **C6 已登记协议不改。** `docs/validation/` 中已登记的协议（例如 H₃ 量子收敛的 `protocol.json`）
  只能按原样执行或明确登记新版本；源码哈希校验失败时停下并记录原因，不绕过校验。
- **C7 诚实边界。** 默认 `physical_converged=false`；失败、未判定、被扣留的估计全部保留并单独计数；
  不丢种子、不挑时间段；短运行称"试点"，模型对照称"实现验证"，与论文一致不等于"复现"（G10）。
- **C8 不确定度单位。** 置信区间以独立链/独立种子为单位；同一母链的子轨迹、同一 MetaD 的时间段、
  共享 walker 都是相关样本。比较两个估计时用两者的联合区间（Welch 或配对），并在预注册中写清。
- **C9 截止即停。** T+50 h 之后不再启动 `pilot`/`production`/`benchmark`；被截止停止的任务从最后的
  原子 checkpoint 可续算，但本战役内**不续算**，结果按"未完成"报告并给出已完成的比例。
- **C10 资源。** 每条车道常驻内存不超过 `核数 × 2 GiB`；输出只写 `$C32_DATA/<lane>/`；
  不使用 GPU、付费资源或其他机器。
- **C11 记录。** 记录放在 `docs/validation/c32-<lane>-<YYYY-MM-DD>/README.md`（可附 JSON/CSV/小 PNG，
  合计 ≤ 1 MB，G9）。每个子测试一节：目的、预注册链接、实际运行（命令、种子、规模、核时）、结果表、
  判据通过/失败、局限。附 `ledger-summary.json`（本车道 ledger 的汇总）和原始数据的 SHA256 清单
  （`python tools/data_manifest.py --root $C32_DATA/<lane>`）。
- **C12 一条车道一个 PR。** 分支名见卡片；PR 只含 `benchmarks/campaign_c32/<lane>/` 与本车道的验证记录；
  PR 描述按 G11，并附"判据汇总表"（子测试 × 通过/失败/阻塞/未完成）。不改 README、手册、参考文献（G3）。
- **C13 交叉复核。** Phase 2 中，每条车道从另一条车道的原始数据（只读）独立重跑其**分析**脚本，
  比较结果是否逐位或在声明的数值容差内一致，并检查判据是否与预注册一致。复核结果写进被复核车道 PR 的评论，
  也写进复核方自己记录的"复核"一节。复核方见总览表。
- **C14 停止并报告。** 环境无法构建、共享环境被他人修改、预算明显不足以完成任何主要子测试、
  需要修改 `src/` 才能继续时，停止该子测试并在 `PROGRESS.md` 中报告，不自行扩大范围（G14）。

## 预注册格式

每个子测试一份 JSON（或带同样字段的 Markdown），提交在 `benchmarks/campaign_c32/<lane>/prereg-<test>.json`：

```json
{
  "schema": "pyrpmd.c32-prereg.v1",
  "lane": "L3",
  "test": "L3-a OGRe double-well equal-budget comparison",
  "question": "在相同力评估预算下，各采样方法的 FES 误差是否与预期一致？",
  "observables": ["RMSE of F over F <= 25 kJ/mol", "barrier error", "coverage fraction"],
  "reference": {"kind": "exact bare potential, P=1", "source": "benchmarks/fes_methods/ogre_models.c"},
  "estimator_and_uncertainty": "per-seed estimate; Student-t 95% CI over 8 independent seeds",
  "seeds": [4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108],
  "budget": {"force_evaluations_per_seed": 2.0e7, "core_hours": 40},
  "gates": [{"name": "coverage", "rule": "coverage >= 0.95 for every seed"}],
  "equivalence_margins": {},
  "stopping_and_extension": "fixed budget; no extension",
  "pilot": {"directory": "$C32_DATA/L3/pilot-ogre", "used_for": "throughput only"},
  "committed_before_production": true
}
```

## 派发提示词（通用部分）

每张卡片末尾有完整的提示词。通用部分如下，各卡片只追加本车道的具体要求：

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 {车道} agent，负责"{标题}"。
这是测试与验证任务，不是开发任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：
1. docs/tasks/README.md 的 G3、G5、G9–G14
2. docs/tasks/campaign-c32/README.md（规则 C1–C14、共享环境、时间线）
3. docs/tasks/campaign-c32/{卡片文件}

执行要求：
- 环境变量 C32_SRC、C32_VENV、C32_DATA、C32_PLAN、C32_LAUNCH 由 L0 提供；只用这个环境，不 pip install。
- 你的 CPU 是 {CPU}（{核数} 核），预算 {预算} 核时；所有计算用 $C32_LAUNCH run --lane {车道} 启动。
- Phase 0（T0–T4）只读代码、写预注册和驱动；T4 后先 pilot，再提交预注册，再生产；T+50 h 停止生产。
- 判据先写后跑；结果不符合预期时如实报告，不调参、不挑种子、不改判据。
- 发现缺陷写进 FINDINGS.md 并继续下一个子测试。
- 在 T+24 h 和 T+40 h 更新 $C32_DATA/{车道}/PROGRESS.md。
- 最后在分支 {分支} 上提交验证记录和驱动脚本，开一个 PR（G11 + 判据汇总表），并完成对 {被复核车道} 的交叉复核。
```

## 集成者（L0）在 Phase 2 的工作

```text
你负责汇总 C32 战役。按 L1…L7 的顺序审查各车道 PR：
- 核对 ledger-summary.json 与 $C32_DATA/<lane>/ledger.jsonl 一致，生产任务都带预注册且在截止前启动；
- 核对每个判据的通过/失败与预注册一致，失败项没有被删除或改写；
- 核对记录目录 ≤ 1 MB、没有原始轨迹或编码载荷（G9、G13）；
- 汇总写进 docs/validation/c32-summary-<日期>/README.md：每条车道的判据汇总、核时、FINDINGS 列表，
  以及需要另开任务卡的缺陷与后续计算；
- 不合并任何 PR，不修改其他车道的文件；合并由维护者决定。
```

## 不在本战役范围

- **H + CH₄ 的 P128 生产速率：** 已有估计 7085–7854 CPU h，远超本机 60 小时容量（`benchmarks/hch4`）。
- **量子 q-TIP4P/F 输运与红外：** P=32 实测约 0.54 s/步，8 ns 需要约 12,045 墙钟小时。
  L5 只做经典 P=1 的 PyRPMD 生产输运（需要 OpenMM）。
- **GPU 与多节点性能：** 本机没有 GPU（T43、T19 续）；MPI 只做单机正确性和扩展性。
- **新方法和代码修复：** 本战役只测试冻结提交；修复在战役结束后按任务卡规则另行分派。
