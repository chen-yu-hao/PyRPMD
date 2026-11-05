---
orphan: true
---

# NP5 速率主链接入：伞形/PMF/FES、父链、thermalize、REUS、TRPMD 子轨迹

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 2 |
| 前置 | NP0、NP1、NP2 已合并 |
| 等价性等级 | 伞形/FES/REUS/thermalize（无 fragment）与无偏置 TRPMD 子轨迹 E0；父链、带 fragment 的 thermalize、带偏置的 PILE-L E1（R10） |
| 建议分支 | `agent/np5-rate-chain` |

## 背景

速率计算的主链是：Andersen 热化 → 伞形采样/PMF → 约束父链 → 成对子轨迹。前三段目前都在每次
Andersen 碰撞处（默认 10 步）返回 Python：

| 位置 | 现状 |
| --- | --- |
| `workflows/biased_sampling.py:53-125`（`_fixed_step`） | 融合分支块长含 `collision_clock.steps_until_collision` |
| `workflows/biased_sampling.py:205-206`（`run_biased_sampling`） | `fused` 只在 Andersen 且固定偏置时为真；PILE-L 走逐步 `sampler.step` |
| `workflows/rate_recrossing.py:126-160`（`_parent_task`） | 碰撞后 `project_momenta`，再按碰撞切块 |
| `dynamics.py:806-813`（`thermalize`） | 块长 `min(remaining, block_steps, interval - step % interval)` |
| `workflows/replica_exchange.py:246-262`（`_advance_window`） | 同上 |
| `workflows/rate_recrossing.py:295-303`（`_child_task` 的 TRPMD 分支） | 逐步 `trpmd_step` |

dRPMD 初始条件采样（`workflows/drpmd_sampling.py:382,393`）经由 `thermalize`，改 `thermalize` 后自动受益。

## 目标

用 NP1 的 `refresh` 和 NP2 的 `propagate_sampling_block` 改写上述位置，使块只在 checkpoint、
观测缓冲上限、4096 步、预抽预算、交换/阶段边界处结束。

## 范围

**要做：**

1. **伞形/PMF/FES（`_fixed_step`）。** 融合分支改为：用 `plan_refresh` 覆盖块内所有碰撞，块长
   `min(block_steps, 4096, total − elapsed, 距 checkpoint 步数)` 再受预算截短；提交时按
   `refreshes_completed` 恢复 RNG 与 `collision_clock`。保留"首步失败则碰撞回滚"的语义。
   `run_biased_sampling` 的 `fused` 判定扩展到 PILE-L（`FESSampler` 无 `thermostat_spec` 时）→
   `propagate_sampling_block`。`thermostat_spec`（组合算符）、NHC（`runtime.chain`）保持原路径。
2. **父链（`_parent_task`）。** 使用 `refresh(projection="centroid_constraint")`；`accept_parent`、
   checkpoint 时机不变。属于 E1，按 R10 写契约标记。
3. **`thermalize`。** Andersen 融合分支用 `refresh`，有 fragment 时 `projection="fragments"`（E1，R10）。
   `pile-l` 分支（带 fragment）保持逐步路径。
4. **REUS（`_advance_window`）。** Andersen 分支用 `refresh`；块不跨越 `stop`（交换边界）。
   非 Andersen 分支在 NP2 可用时用 `propagate_sampling_block`。
5. **TRPMD 子轨迹。** `_child_task` 的 TRPMD 分支在 NP2 可用时用
   `propagate_sampling_block(thermostat="trpmd", cv=surface)`，每块累加分子。
6. 所有改动的 `loop_backend="python"` 与无扩展路径结果与原路径一致（R4）。
7. 在 `docs/native-propagation.md` 的"Workflow/event semantics"中更新相应段落。

**不做：**

- 不改 NP1/NP2/NP4 的核心模块（如需改，写"需要协调"）。
- 不改 `_history_step`（归 NP7）、materials（归 NP6）、`_child_task` 的 RPMD 分支（NP0 已完成）。
- 不改 checkpoint 中的科学字段与物理指纹；R10 契约标记是唯一新增的执行元数据。

## 文件所有权

- **独占：** `biased_sampling.py` 的 `_fixed_step` 与 `run_biased_sampling` 中的 `fused` 判定；
  `workflows/sampling.py` 的 `FESSampler`；`rate_recrossing.py` 的 `_parent_task` 与 `_child_task` 的 TRPMD 分支；
  `dynamics.py` 的 `thermalize`；`replica_exchange.py` 的 `_advance_window`；新测试文件。
- **共享注册点：** `docs/native-propagation.md`。

## 交付物

- 接入改动与测试。
- `docs/validation/np5-rate-chain-<日期>/README.md`：H + H₂（BKMP2，能构建时）的短速率计算
  （窗口数、父链样本、子轨迹数在记录中写明）替换前后的 PMF、κ(t)、速率与墙钟时间，以及每 1 万步的原生调用次数。
  无法构建 BKMP2 时用内置势并标注"未实测真实体系"。

## 验收标准

- **E0 部分：** 伞形/FES（固定与 geometric 调度、1D/2D 伞形、flux-metric 偏置、有/无固定原子）、REUS、
  无 fragment 的 thermalize、无偏置 TRPMD 子轨迹：输出数组、checkpoint 的 `q`/`p`/RNG、最终
  `rng.bit_generator.state` 与替换前逐位一致。
- **E1 部分：** 父链与带 fragment 的 thermalize、带偏置 PILE-L：短时程逐帧在容差内；
  长时程 PMF/κ(t) 在误差棒内一致；RNG 逐位一致。
- **续算：** 每条链路"中断续算 = 不中断"逐位一致；续算时改变 `block_steps` 结果不变；
  没有契约标记的旧 checkpoint 续算后与旧代码继续运行逐位一致（R10）。
- **R8：** 伞形采样每 1 万步原生调用次数 ≤ `ceil(1e4 / min(block_steps, 4096)) + checkpoint 次数 + 预算截断次数`，
  `python_steps == 0`；父链、thermalize、REUS 同样给出上界测试。
- 现有 `tests/test_fes_block_sampling.py`、`tests/test_biased_sampling_interrupt.py`、
  `tests/test_batched_recrossing.py`、`tests/test_drpmd*.py` 全部通过且未被放宽。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests -k "fes or umbrella or biased or recross or replica or thermal or drpmd"
python benchmarks/h3/build_pes.py && python benchmarks/h3/native_propagation.py --output runs/np5/report.json
python -m pytest -q -m "not campaign"
```
