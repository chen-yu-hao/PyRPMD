---
orphan: true
---

# NP1 C++：块内预抽 Andersen 动量刷新

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无（与第四轮 T27 的文件重叠见本轮 README） |
| 等价性等级 | 无投影时 E0；刷新后有约束/fragment 投影时 E1 |
| 建议分支 | `agent/np1-momentum-refresh` |

## 背景

速率主链（伞形/PMF/FES、再穿越父链、`Dynamics.thermalize`、REUS）的块都在 Andersen 碰撞处截断，
默认每 10 步返回一次 Python，原因只有一个：动量重采样用 NumPy `Generator`，C++ 块
（`native/propagation.h:369` 的 `advance`）没有恒温器。

旧路径中的抽样都与状态无关，可以预先按原顺序生成：

- 动量：`Dynamics.sample_momenta()`（`dynamics.py:515-531`），即 `rng.normal(size=shape) * sqrt(m*P/β)`，
  固定原子置零。带 bond constraints 时依赖 `q`，**不在本卡范围**。
- 调度：`workflows/sampling.py:52-116` 的 `AndersenSchedule`。`fixed` 不抽随机数；
  `geometric` 在每次刷新后调用 `refreshed(step, rng)` 抽一次 `rng.geometric`。

刷新之后有两类依赖状态的投影：再穿越父链的质心切向动量投影
（`workflows/rate_constraints.py:51-56` 的 `project_momenta`），以及 `thermalize` 的 fragment COM 投影
（`dynamics.py:721-731`）。C++ 已有 fragment 投影（`propagation.h:37` 的 `project`）
和质心 RATTLE 的法向量计算，可以复用。

## 目标

`propagate_block` 新增可选参数，让一个原生块内按给定步序号执行多次动量刷新，
从而让块只在 checkpoint、观测缓冲上限、4096 步上限或工作流事件处结束。
本卡只提供能力和 Python 规划器，**不接入工作流**（接入归 NP5/NP6/NP7）。

## 范围

**要做：**

1. **C++（`native/propagation.h`）。** `advance` 的 `options` 接受可选键 `"refresh"`：
   - `offsets`：严格递增的 int64 数组，取值 `0..steps-1`，表示"在块内第 k 步之前刷新"；
   - `momenta`：float64 `[R, P, N, 3]`，Python 已按原函数生成并缩放；
   - `projection`：`"none"`、`"centroid_constraint"`（与 `project_momenta` 相同的数学，使用当前约束 CV 的质心法向量）
     或 `"fragments"`（复用现有 fragment 投影，并按 `_fix_fragments` 的语义同时投影 `q` 与 `p`）。
   刷新只改 `p`（fragment 投影还改 `q`）。无投影或仅约束投影时 `q` 不变，端点力缓存继续有效；
   fragment 投影改变 `q` 时按现有规则重新求力。返回结果新增 `refreshes_completed`：
   第 r 次刷新被提交，当且仅当偏移 `offsets[r]` 处的那一步成功完成（与旧工作流"碰撞后首步失败则回滚碰撞"一致）。
   在模块初始化处（`native/cayley.cpp:365` 附近，共享注册点）追加 `MOMENTUM_REFRESH_API = 1`。
2. **Python 规划器（新 `src/pyrpmd/momentum_refresh.py`）。**
   - `plan_refresh(dynamics, schedule, start_step, max_steps, *, budget=MAX_PREDRAWN_BYTES)`：
     从 `start_step` 起，按旧顺序循环"到期 → `dynamics.sample_momenta()` → `schedule.refreshed(step, rng)`"，
     生成偏移与动量数组；超出内存预算时截短块长。默认**逐次调用原函数**生成，保证逐位一致；
     若要做向量化抽样优化，必须加 R2 回归测试并保留逐次调用的回退。
   - 规划期间不得修改调用方可见的 `schedule` 状态；返回"提交函数"。提交时按 `refreshes_completed`
     把 RNG 和 `schedule.next_collision_step` 精确恢复到"恰好执行了这么多次刷新"的状态
     （先恢复到块起点再重抽，或使用逐次 RNG 快照）。
   - 导出 `MAX_PREDRAWN_BYTES = 64 * 1024**2`，供其他卡复用（R6）。
3. **`block_propagation.propagate_block` 新参数 `refresh=None`。** 接收规划结果；原生路径传给 `advance`；
   Python 路径在相同偏移处执行相同刷新，投影调用与旧工作流相同的 Python 函数（R4）。
   `refresh` 与 `outcome`、`record_previous`、组合算符执行器（`plan.active`）同时出现时，
   在修改状态前报错或在 `auto` 下回退（写明 `fallback_reason`）。旧扩展缺少 `MOMENTUM_REFRESH_API` 时同样处理（R3）。
4. 在 `docs/native-propagation.md` 追加小节，写明参数契约、提交语义和内存预算。

**不做：**

- 不改任何工作流文件（`biased_sampling.py`、`rate_recrossing.py`、`dynamics.py` 的 `thermalize`、
  `replica_exchange.py`、`materials.py`）。
- 不支持 bond constraints 的动量刷新（依赖 `q` 的投影抽样）。
- 不改 `native/thermal.*`，不引入 C++ 随机数。
- 不新增输入键；如确需新增执行参数，必须确认它被 `scientific_config` 豁免并在 PR 中说明。

## 文件所有权

- **独占：** `native/propagation.h`、`src/pyrpmd/block_propagation.py`（`bias_descriptors()` 与新增
  `_check_bias_snapshot()` 归 NP4；NP4 还会在 `propagate_block` 中插入一行快照校验调用，后合并者 rebase）、
  新 `src/pyrpmd/momentum_refresh.py`、新测试文件。
- **共享注册点：** `native/cayley.cpp`（只追加能力常量）、`docs/native-propagation.md`（追加小节）。

## 交付物

- C++ 能力、规划器、`propagate_block(refresh=…)` 与测试。
- `native/README.md` 中"Fused production blocks"一节追加一段说明（此文件归本卡，只追加）。

## 验收标准

- **E0（无投影）：** 内置谐振/Morse 势，P ∈ {1, 2, 8, 16}，固定与 geometric 调度，`interval` ∈ {1, 3, 10}，
  有/无伞形偏置与 flux-metric 偏置、有/无固定原子：一次带 `refresh` 的长块，与"在每次碰撞处切块、
  块间用 Python 调 `sample_momenta()`"的参考调度相比，`q`、`p`、观测值、能量、
  `rng.bit_generator.state`、`next_collision_step` 全部逐位一致；`native`、`python`、`auto` 三种后端彼此逐位一致。
- **E1（有投影）：** 约束投影与 fragment 投影下，随机抽样与 RNG 状态逐位一致；`q`/`p` 与 Python 参考的差异
  不超过现有原生 RATTLE 对比测试的容差，PR 写明数值。
- **失败语义：** 在第 0 步、刷新所在步、两次刷新之间注入 PES 失败，`refreshes_completed`、RNG、
  `next_collision_step`、有效前缀与参考调度一致；首步失败时状态与 RNG 完全不变。
- **预算：** 超出 `MAX_PREDRAWN_BYTES` 时块被截短，结果不变。
- **兼容：** 伪造缺少 `MOMENTUM_REFRESH_API` 的扩展时，`auto` 回退并记录原因，`native` 在修改状态前报错。
- OpenMP 开/关编译、`PYRPMD_NO_NATIVE=1` 路径、开启 FMA 的构建都通过。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
OMP_NUM_THREADS=1 python -m pytest -q tests -k "native or fused or block"
PYRPMD_NO_OPENMP=1 python -m pip install -e . --no-build-isolation && python -m pytest -q tests -k "refresh"
python -m pytest -q -m "not campaign"
```
