---
orphan: true
---

# L1 气相速率主链：H + H₂ / BKMP2

| 字段 | 内容 |
| --- | --- |
| CPU | 2–11（10 核） |
| 预算 | 460 核时（计划 400） |
| 工作流 | `rate`（PMF → QTST → 约束母链 → 子轨迹）、`extend`、`resume`、`study`、`report`；已登记的 H₃ 量子 MC 与 P256 协议 |
| 外部模型 | L0 构建的 BKMP2（`$C32_BKMP2`）；缺失时整条车道阻塞，改做 L1-e 的模型部分并报告 |
| 建议分支 | `agent/c32-l1-rate` |
| 记录 | `docs/validation/c32-l1-rate-<日期>/README.md` |
| 复核 | L1 复核 L2；L2 复核 L1 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L1 ...` |

## 背景

当前证据（[文献验证清单](../../literature-validation.md)、
[量子收敛记录](../../validation/literature-convergence-2026-11-01/h3-quantum/README.md)）：

- 300 K、P=128 的四对子轨迹扩展给出 κ=0.77121022（95% CI [0.75333936, 0.78908108]），
  k=1.96785637×10⁻¹⁶ cm³ molecule⁻¹ s⁻¹，十一项固定 P 判据全部通过。
- **未完成 1：伞形采样精度。** 独立 MC 伞形 QTST 在 100,000 sweep 前缀为 2.587396804×10⁻¹⁶，
  主对数标准误 0.0508737 > 登记上限 0.03；登记的 300,000 前缀已经开始，结果未知。
  按每 sweep 0.046 s 的实测，从零跑到 300k 约 8 × 305,000 × 0.046 s ≈ 31 worker·h。
- **未完成 2：动态 P256。** 设计要求 7680 万物理步，按 P128 实测线性外推 10.44–15.95 worker·h；
  母链约 3,472 worker·s。
- **未完成 3：母链时间对半等价。** 补充判据区间 [−0.02310, +0.03626] 超出固定边界 ±0.03。
- **未完成 4：生产 MD 伞形采样。** 上面的 QTST 来自 MC 采样器；公开 `pyrpmd run` 的 MD 伞形采样
  （Andersen 几何时钟，NP0–NP2 合并后的路径）在 P=128 上的精度从未建立。这一项检验用户实际使用的工作流。

## 子测试（按顺序；括号内为 CPU 分配）

### L1-a 已登记 MC 伞形采样的 300k 前缀（2 核，约 31 核时）

两种模式，在 Phase 0 根据维护者是否提供归档决定，写进预注册：

- **R 模式（续登记阶段）：** 维护者提供 `pyrpmd-h3-quantum-umbrella-stage-100000-2026-11-01.tar.gz`
  （SHA256 见 [300k 工具记录](../../validation/literature-convergence-2026-11-01/h3-quantum/umbrella-isolated300k-tool/README.md)）
  时，按该记录的接纳、恢复和调度流程执行 300k 阶段，不改协议、种子和分析。
- **I 模式（独立重跑）：** 没有归档时，在新目录从零执行已登记协议：

```bash
$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L1 --name mc-umbrella-30k \
  --kind production --slots 2 --prereg benchmarks/campaign_c32/L1/prereg-mc-umbrella.json -- \
  python benchmarks/h3/quantum_sampling_campaign.py --library "$C32_BKMP2" \
  --output "$C32_DATA/L1/mc-umbrella" \
  --protocol docs/validation/literature-convergence-2026-11-01/h3-quantum/protocol.json \
  --production-stage 30000 --workers 2
```

  然后依次 `--production-stage 100000`、`300000`（协议规定的前缀）。I 模式额外回答一个可复现性问题：
  30k 与 100k 前缀的 QTST（记录值 2.80072503×10⁻¹⁶ 与 2.587396804×10⁻¹⁶）是否在新硬件上逐位或在舍入级重现；
  不一致时报告首个分歧的 sweep 和统计上是否等价。

判据（登记值，不改）：300k 主对数标准误 ≤ 0.03；其余七项登记判据。源码或调度文件的 SHA256 校验失败时
按 C6 停止，记录哪一个文件不同，并把这 2 核转给 L1-c。

### L1-b 动态 P256 与母链时间对半（2 核，约 20 核时）

1. 运行 `benchmarks/h3/p256_parent_campaign.py preflight` 与 `p256_child_campaign.py preflight`（只读、零 PES 调用）。
2. 预检要求的历史母链归档存在时，按 [P256 设计](../../validation/literature-convergence-2026-11-01/h3-quantum/p256-fixed-eight-chain-design/README.md)
   与 [子轨迹启动记录](../../validation/literature-convergence-2026-11-01/h3-quantum/p256-child-launch/README.md)
   执行全部八条链、两个步长、200 fs 视界的子轨迹，再运行冻结的分析器，给出 P128/P256 的配对比较。
3. 归档不存在时**不改工具**：记录预检的拒绝信息，把 P256 标为阻塞。剩余 2 核用于母链时间对半问题的
   新登记阶段：先写一份新协议（新 id、新目录），预注册"把每条母链的采样时间加倍、保持种子与分析器不变"，
   判据仍是原来的 ±0.03 配对区间；在看到任何新结果之前提交。

### L1-c 公开 `rate` 工作流的 MD 伞形采样，300 K、P=128（6 核，约 250 核时）

这是本车道最重要的新测试。以 `benchmarks/h3/rate.inp` 为模板，在 `benchmarks/campaign_c32/L1/` 中写
`h3_rate_300k_p128.inp`：`temperature = "300 K"`、`beads = 128`，反应定义（片段、两个等价出口、
`symmetry_factor = 1.0`、`rinf`）不变，窗口覆盖 ξ=0 到超过 1；`kappa`、窗口数、平衡与生产时长、
母链数和子轨迹对数由 pilot 定，写进预注册。

1. **Pilot（≤ 25 核时）：** 一个靠近 ξ=1 的窗口、一个渐近窗口各跑 20 ps，测每步耗时和自相关时间；
   一次 20 个母构型的小规模子轨迹测每 fs 耗时。按测得的成本分配：伞形 ≈ 70%、母链与子轨迹 ≈ 30%。
2. **生产：** 4 次完全独立的运行（不同 `seed`、独立输出目录，每次 `--slots` 不超过 6，可以排队），
   每次都包含完整的 PMF、母链和子轨迹。运行中至少一次用 `pyrpmd extend` 扩展生产步数，
   至少一次在母链阶段中断后 `resume`，检查扩展/续算语义在真实体系上成立。
3. **分析：** 每次运行的 `k_QTST`、`κ`、`rate_cm3_s` 与其联合不确定度；4 次运行之间的离散度
   与运行内 bootstrap 误差是否相容（运行内误差是否低估）；UI 与 MBAR（`rate.pmf_method`）的势垒差。

预注册判据（数值在预注册中固定，下面是建议）：

| 判据 | 规则 |
| --- | --- |
| QTST 与 MC 参考 | ln k_QTST(MD) − ln k_QTST(MC, L1-a 最终前缀) 的 Welch 95% 区间落在 ±0.05 内 |
| κ 与登记的 P128 κ | κ(MD) − 0.77121022 的 95% 区间落在 ±0.03 内 |
| 运行内误差的可靠性 | 4 次运行的样本标准差不超过运行内报告标准误中位数的 2 倍 |
| 速率 | 报告 `rate_cm3_s` 与 `references.json` 的 `h_h2_bkmp2_rpmd_300k` 的差和区间；**不作为通过条件** |

成本超出时减少窗口内生产时长或母链数，不降低 P，并在预注册中写明。

### L1-d 温度与同位素序列（L1-a/L1-b 结束后的 4 核，约 100 核时）

在 600 K 与 1000 K 运行 H + H₂，在 1000 K 运行 D + D₂（`masses_amu` 改为氘），
参考为 `benchmarks/published_rates/references.json` 中 `h_h2_bkmp2_qm_600k`、`h_h2_bkmp2_qm_1000k`、
`d_d2_bkmp2_qm_1000k`（精确量子速率；RPMD 与精确量子的差异是物理近似，不是判据）。

- 珠子数不凭经验填写：每个温度用 P 与 2P 两个 pilot 比较 QTST，差异小于预注册阈值的较小 P 用于生产；
  这个选择本身记录为一项结果。
- 用 `pyrpmd study` 组织这些案例（每案例独立种子），用
  `pyrpmd report RUN... --reference CSV --fit-arrhenius --convergence beads` 出图；
  Arrhenius 拟合只作描述，KIE 用 `--kie`。

### L1-e 后处理与报告工具（轻量，≤ 5 核时）

在 L1-c 的输出上运行 `pyrpmd analyze --recompute`、`pyrpmd rate-postprocess`、`pyrpmd report`，
检查重新分析不传播新轨迹、结果与原 summary 一致；把 `report` 生成的图放进记录（PNG ≤ 200 KB/张）。

## 不做

- 不调整已登记协议的判据、种子、前缀或分析器；不把 MC 与 MD 的样本合并成一个估计。
- 不跑 H + CH₄、OH + H₂、F + H₂（成本或模型未就绪，见 README "不在本战役范围"）。

## 交付物

- `benchmarks/campaign_c32/L1/`：输入、预注册、分析脚本；验证记录与判据汇总表。
- 记录中单独一节"可复现性"：I 模式下与归档数值的比较结果。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L1 agent，负责"气相速率主链：H + H₂/BKMP2"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具（benchmarks/h3/ 下的脚本按原样使用）。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L1-rate-h3.md；docs/validation/literature-convergence-2026-11-01/h3-quantum/README.md。

要求：
- 你的 CPU 是 2–11（10 核），预算 460 核时；所有计算用 $C32_LAUNCH run --lane L1 启动，--slots 如实填写。
  已登记 H₃ 工具最多 2 个 worker，分配见卡片。
- Phase 0 内：确认 $C32_BKMP2 可用；询问维护者（或检查 $C32_DATA/shared）是否有 100k 伞形归档与 P256 母链归档，
  据此选择 L1-a 的 R/I 模式；写 L1-c 的输入和全部预注册。
- 已登记协议只能原样执行；哈希校验失败时停下记录，不绕过（C6）。
- L1-c 先 pilot（≤ 25 核时），再按 pilot 定规模、提交预注册，再跑 4 次独立生产。
- 结果与参考不一致时如实报告；不降低 P、不挑种子、不改判据。
- T+24 h、T+40 h 更新 $C32_DATA/L1/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l1-rate 提交记录与驱动并开 PR（G11 + 判据汇总表）；Phase 2 复核 L2。
```
