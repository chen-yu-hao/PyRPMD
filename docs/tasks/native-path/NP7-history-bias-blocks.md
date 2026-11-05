---
orphan: true
---

# NP7 元动力学按 pace 分块

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | NP1、NP4 已合并（PILE-L 支持需要 NP2） |
| 等价性等级 | E1（需 R10 契约标记） |
| 建议分支 | `agent/np7-metad-blocks` |

## 背景

`workflows/biased_sampling.py:127` 的 `_history_step` 对所有历史相关偏置逐步执行：
"冻结历史积分一步 → 观测 → 到期时提交偏置更新"，每步都有完整的 Python 事务（状态、RNG、
统计、热浴、偏置快照）。元动力学只在 `production % pace == 0`（`_metadynamics_base.py:274-276`）时沉积，
两次沉积之间偏置是固定的。NP4 提供了 MetaD/网格 MetaD/墙偏置的冻结快照描述符，NP1 提供了块内 Andersen 刷新。

## 目标

串行（单 walker）standard、well-tempered 与网格元动力学，在原生 PES、原生 CV 和 NP4 描述符可用时，
每个沉积间隔只返回 Python 一次；沉积、观测数组、checkpoint 语义不变。

## 范围

**要做：**

1. 新增 `_history_block`（在 `biased_sampling.py`），适用条件：`update` 是串行 `MetadynamicsBias` 或
   `GridMetadynamicsBias`（非 shared/multiple walkers）；力偏置与 `active_bias` 返回的墙偏置都有 NP4 描述符；
   恒温器为 Andersen（NP1）或无 `thermostat_spec` 的 PILE-L（NP2）；无 NHC（`runtime.chain`）。
   其余情况继续用 `_history_step`。
2. **块边界：** 下一次 `due` 的步（块在该步结束，随后在 Python 中按原顺序完成观测与沉积）、
   `freeze` 边界、`neq`（墙偏置在平衡段的开关、`reset_after_equilibration`）、checkpoint、4096 与预算上限。
3. **观测：** 用观测掩码在块内收集记录步的 CV 值；`sample_bias`（`physical_energy(value)`）在块后由 Python
   用同一冻结偏置计算。记录步能量若使用 `energy_bias` 而非力偏置，用 `record_states=True` 保留记录帧并在块后按原表达式计算，
   保证记录能量与逐步路径同等级。
4. **失败：** 有效前缀提交；沉积只在块尾进行，失败块不沉积；RNG 按 NP1/NP2 的提交规则恢复。
5. R10：新运行写契约标记；旧 checkpoint 续算继续使用 `_history_step`。
6. 在 `docs/enhanced-sampling.md` 或 `docs/grid-metadynamics.md` 追加执行说明，`docs/native-propagation.md` 追加小节。

**不做：**

- 不改 OPES、ABF、VES、steered、shared/async/MPI walkers（在 PR"未完成事项"中写明后续所需）。
- 不改 hill 高度、宽度、well-tempered 公式与沉积时刻。

## 文件所有权

- **独占：** `biased_sampling.py` 的 `_history_step` 与新增 `_history_block`；`workflows/_metadynamics_base.py`
  中与调度相关的部分；新测试文件。
- **共享注册点：** `docs/native-propagation.md`、`docs/enhanced-sampling.md` 或 `docs/grid-metadynamics.md`（追加段落）。

## 交付物

- 分块调度与测试。
- `docs/validation/np7-metad-blocks-<日期>/README.md`：内置双势阱与（能构建时）H₃ 上，替换前后 FES 对照、
  每 1 万步原生调用次数与墙钟时间。

## 验收标准

- **短时程（E1）：** 固定 hill 历史下 200 步内逐帧 `q`/`p` 在容差内；第一次沉积的 hill 中心、高度与旧路径在容差内；
  RNG 逐位一致。
- **长时程统计：** 双势阱 standard 与 well-tempered MetaD 的重构 FES 与旧路径在误差棒内一致，记录种子与块平均误差。
- **续算：** 中断续算逐位一致；改变块长续算一致；旧 checkpoint 续算与旧代码逐位一致（R10）。
- **R8：** 每 1 万步原生调用次数 ≤ `ceil(1e4 / pace) + checkpoint 次数 + 阶段边界数 + 预算截断次数`，`python_steps == 0`。
- 不满足条件的配置（OPES、ABF、NHC、shared walkers、自定义偏置）行为完全不变，有测试覆盖。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests -k "metadynamics or metad or grid or biased or opes"
python -m pytest -q -m "not campaign"
```
