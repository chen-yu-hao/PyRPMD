---
orphan: true
---

# L2 碰撞、散射与跨路线速率一致性

| 字段 | 内容 |
| --- | --- |
| CPU | 12–15（4 核） |
| 预算 | 184 核时（计划 150） |
| 工作流 | `drpmd`（热碰撞）、state-selected NE-RPMD/QCT、`network`、`study`、`rate`（只作为对照路线） |
| 外部模型 | `$C32_BKMP2`；缺失时只做 L2-c、L2-d |
| 建议分支 | `agent/c32-l2-collision` |
| 记录 | `docs/validation/c32-l2-collision-<日期>/README.md` |
| 复核 | L2 复核 L1；L1 复核 L2 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L2 ...` |

## 背景

同一个势能面、温度、珠子数和质量下，直接碰撞（dRPMD）与 Bennett–Chandler（约束母链 + 子轨迹）
是同一个 RPMD 速率的两种估计：前者对热通量能量分布、碰撞参数和取向积分，后者用 QTST × κ。
`bmax`、入口距离和 κ 平台都收敛时，两者必须一致。目前唯一的同条件比较是
[1500 K 记录](../../validation/h3-bkmp2-2026-10-29/README.md)：dRPMD 1024 条碰撞只有 16 条反应，
k=8.50551×10⁻¹²（cluster 95% 区间 [4.2528×10⁻¹², 1.2758×10⁻¹¹]）；BC 的 MBAR 次级速率 1.19182×10⁻¹¹ 的区间固定了 κ。
两个区间都太宽，无法说明两条路线是否一致。

## 子测试

### L2-a dRPMD 与 Bennett–Chandler，1500 K、P=16（约 100 核时）

两条路线使用同一份 BKMP2、`masses_amu`、温度和珠子数（与 `benchmarks/h3/rate.inp` 相同），各自独立的种子。

1. **dRPMD**（`benchmarks/h3/drpmd.py`）：pilot（≤ 8 核时）测每条碰撞耗时和反应概率，
   然后按"反应事件 ≥ 400"定碰撞数（按记录的 1.6% 反应概率约需 25,000 条），
   母链数 ≥ 64（母链 cluster bootstrap 才有意义）。主运行 `--bmax-bohr 3.0 --separation-bohr 10`。
2. **收敛对照**（预注册为主运行之外的独立阶段，各约 30% 主运行的碰撞数）：`--bmax-bohr 4.0`；
   `--separation-bohr 14`；`--timestep-fs` 减半。报告最外壳层（0.8–1.0 bmax）的反应数——非零就说明 bmax 不够。
3. **Bennett–Chandler**：以 `benchmarks/h3/rate.inp` 为模板写 `bc_1500k_p16.inp`，
   把 `[sampling] production`、`replicas`、`[recrossing] parent_replicas`、`parent_samples`、`child_pairs`
   放大到 pilot 估计的 k 半宽 ≤ 10%；做 2 次独立运行。
4. **比较**：预注册判据——ln k_dRPMD − ln k_BC 的 95% 区间（dRPMD 用母链 cluster bootstrap，BC 用其联合
   PMF + κ 区间，两路线独立）**包含 0** 判为"一致"；区间整体落在 ±0.15 内判为"等价"；
   半宽 > 0.25 判为"未分辨"。另报告两者与 `references.json` 中 `h_h2_bkmp2_qm_1500k`（精确量子）的差，
   只作背景，不作判据。

示例命令（pilot）：

```bash
$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L2 --name drpmd-1500k-pilot \
  --kind pilot --slots 4 -- \
  python benchmarks/h3/drpmd.py --output "$C32_DATA/L2/pilot-drpmd" --trajectories 1024 \
  --parents 16 --workers 4 --beads 16 --temperature 1500 --seed 902101
```

### L2-b 第二个温度：1000 K（剩余预算，约 30 核时）

只有 L2-a 结论为"一致"或"等价"且剩余预算足够时才做；反应概率更低，dRPMD 需要更多碰撞。
预注册时先写出碰撞数目标；预算不足以达到半宽 ≤ 20% 时不做，记录原因。

### L2-c state-selected NE-RPMD / QCT 独立重跑（约 25 核时）

[态选择复现记录](../../../benchmarks/h3/STATE_SELECTED_REPRODUCTION.md) 的正式案例在新硬件上用**同样的种子与
id offset** 在新目录重跑一个子集：`--v 0 --etot .70 1.10`（NE-RPMD，P=32）与同能量的 `--method qct`，
每个 `--samples 4096`。比较：

- 逐轨迹账本（初态、结果标签）是否与归档 `benchmarks/h3/reproduction_results/` 逐位一致；
- 反应概率与 Wilson 区间是否一致；
- 不一致时定位首个不同的轨迹 id，分开报告"初态生成不同"和"传播结果不同"。

`--workers` 只改变调度，不改变随机身份；用 `--workers 4 --slots 4`。

### L2-d `network` 与 `study` 的组合一致性（解析模型，约 10 核时）

1. `examples/network/two_entrances.py`：把采样放大 8 倍，对速率矩阵的每个元素另用一个独立的单通道
   `rate` 运行（同模型、同 TS、同设置）估计，比较两者的 95% 区间是否相交；检查对称/等价路径的计数
   是否与输入声明一致（不要用 2 倍因子凑数）。
2. `pyrpmd study examples/studies/drpmd.json --plan` 后完整运行：确认计划列出的案例全部完成、
   `--resume` 不重复已完成案例、数值等价检查（`bmax`、R0、珠子数、样本数）在自由散射模型上给出预期的
   "零速率"结论，日志中没有被吞掉的失败案例。

## 不做

- 不修改分类器、碰撞参数默认值或已登记的态选择协议；不把敏感性案例合并进主样本。

## 交付物

- `benchmarks/campaign_c32/L2/`：输入、预注册、比较脚本；验证记录与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L2 agent，负责"碰撞、散射与跨路线速率一致性"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L2-collision-cross-route.md；benchmarks/h3/README.md；
benchmarks/h3/STATE_SELECTED_REPRODUCTION.md。

要求：
- 你的 CPU 是 12–15（4 核），预算 184 核时；所有计算用 $C32_LAUNCH run --lane L2 启动。
- 核心问题是"同条件下 dRPMD 与 Bennett–Chandler 是否给出同一个速率"。两条路线的 PES、温度、P、质量完全相同，
  种子独立；判据（一致 / 等价 / 未分辨）在看到任何生产结果之前提交。
- bmax、入口距离、步长的对照是独立登记的阶段，不与主样本合并。
- state-selected 重跑使用原种子与 id offset，在新目录运行，逐轨迹比较归档。
- 结果不一致时如实报告并定位，不调参、不挑种子。
- T+24 h、T+40 h 更新 $C32_DATA/L2/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l2-collision 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L1。
```
