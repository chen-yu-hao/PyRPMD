---
orphan: true
---

# NP3 盆地停止事件与 committor/TIS 射击

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | NP1（`native/propagation.h` 的所有权在 NP1 合并后移交本卡） |
| 等价性等级 | E1（需 R10 契约标记） |
| 建议分支 | `agent/np3-basin-events` |

## 背景

committor 与 TIS/RETIS 的射击段都是"无偏 NVE，每步算一次标量序参量，进入盆地 A 或 B 就停"：

- `committor.py:246-266`（`CommittorSampler.shoot`）：逐步 `dynamics.step(state)` + `_value` + `_basin`。
  动量由 `_momenta()`（`committor.py:239-244`）在射击开始时抽一次，与状态无关。
- `transition_interface.py:322-331`（`_propagate_to_basin`）：逐步推进并保存每一帧的 `q`/`p`。
- `workflows/path_sampling.py:429-436`（FFS 的 `forward`）：`dynamics="basin-andersen"` 时只在盆地内以概率抽随机数，
  **抽样依赖状态，不能预抽**；普通 NVE 模式可以使用本卡能力。

C++ 的 `advance` 已有 dRPMD 结果事件（`propagation.h:141-315`）：在检查步上判定、带持续性状态机、
在阈值附近的舍入带内把判定交回 Python（`pending_check`）。盆地停止是它的一个更简单的特例。

## 目标

`advance` 新增"标量 CV 盆地停止"事件；committor 射击与 TIS 的 `_propagate_to_basin` 在原生 PES 和
原生标量 CV 可用时，一次调用推进到盆地或预算上限，只在阈值舍入带内交回 Python。

## 范围

**要做：**

1. **C++（`native/propagation.h`）。** `options["basin"] = {"cv": 标量 CV 描述符, "lower": a, "upper": b,
   "roundoff": 相对带宽}`。每个已接受步后求 CV：`value ≤ a` 或 `value ≥ b` 即成功停止并返回盆地标签；
   落在 `[a, b]` 边界的舍入带内时停止并返回 `pending_check`，由 Python 用原 `_basin` 判定后决定继续还是结束。
   与 `outcome` 互斥。追加 `BASIN_EVENT_API = 1`（`native/cayley.cpp` 共享注册点）。
2. **Python。** `block_propagation.propagate_block` 增加 `basin=` 参数（本卡在 NP1 合并后拥有该文件中这部分改动），
   旧扩展回退规则同 R3。
3. **committor。** `shoot` 在原生可用时循环调用 `propagate_block(..., basin=...)`，块长受 `max_steps`、
   4096 与观测预算约束；返回的 `CommittorShot` 字段（盆地、步数、末值、末态）与旧路径语义相同；
   失败时 RNG 回滚语义不变。坐标是可调用对象或自定义 CV 时保持旧路径。
4. **TIS/RETIS。** `_propagate_to_basin` 在原生可用时使用 `record_states=True` 的块，按块收集帧，
   受 256 MiB 观测预算约束自动分块；`max_path_steps` 超限时抛出同样的 `PathGenerationError`。
   `validate_path` 的逐步重放保持不变（它是校验器，不是性能路径）。
5. FFS 只在非 `basin-andersen` 模式下接入；`basin-andersen` 在 PR 中写明为何不接入。
6. R10：committor/TIS 的 checkpoint 写入契约标记；旧 checkpoint 续算使用旧路径。
7. 在 `docs/native-propagation.md` 追加小节；`docs/committor.md` 追加一段执行说明。

**不做：**

- 不改盆地判定的数学与阈值，不引入新的随机数。
- 不改 RETIS 的交换、接受/拒绝逻辑。

## 文件所有权

- **独占：** `native/propagation.h`（NP1 合并后）、`block_propagation.py` 中 `basin` 相关改动、
  `committor.py` 的 `shoot`、`transition_interface.py` 的 `_propagate_to_basin`、
  `workflows/path_sampling.py` 的 FFS `forward`（仅 NVE 模式）、新测试文件。
- **共享注册点：** `native/cayley.cpp`（能力常量）、`docs/native-propagation.md`、`docs/committor.md`（追加段落）。

## 验收标准

- **短时程（E1）：** 双势阱内置势与 BKMP2（能构建时），P ∈ {1, 4, 16}，射击 200 步内逐帧 `q`/`p`
  与旧路径差异不超过现有融合/非融合对比测试容差；动量抽样与 RNG 状态逐位一致。
- **判定一致：** 构造恰好落在阈值附近的轨迹，证明舍入带内由 Python 判定，盆地标签和停止步与旧路径一致。
- **统计（长时程）：** 双势阱上 ≥ 2000 次射击的 committor 估计与旧路径在 95% 区间内一致，记录随机种子。
- **R8：** 每次射击的原生调用次数 ≤ `ceil(步数 / 4096) + pending_check 次数 + 1`，`python_steps == 0`。
- **R10：** 旧 checkpoint 续算逐位延续旧结果。
- 回退：可调用坐标、自定义 CV、缺少 `BASIN_EVENT_API`、`basin-andersen` 都保持旧行为。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests -k "committor or tis or retis or ffs or path_sampling or native"
python -m pytest -q -m "not campaign"
```
