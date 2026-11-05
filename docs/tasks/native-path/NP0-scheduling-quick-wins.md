---
orphan: true
---

# NP0 纯 Python 调度：再穿越子轨迹与 materials 分块

| 字段 | 内容 |
| --- | --- |
| 规模 | S |
| 波次 | 1 |
| 前置 | 无 |
| 等价性等级 | E0 |
| 建议分支 | `agent/np0-scheduling` |

## 背景

两处链路的计算已经全部可以在 C++ 中完成，但调度方式让它们频繁返回 Python 或多做 PES 调用。
本卡只改 Python 调度，不改 `native/`。

1. **再穿越子轨迹**（`workflows/rate_recrossing.py:286-293`，`_child_task` 的 RPMD 分支）
   调用 `dyn.propagate(state, count, observe=surface, sample_interval=1, record_centroids=False)`，
   块长 `min(dyn.block_steps, nsteps - completed)`。`propagate()` 每步 2 次 PES 调用，
   不做端点力复用；`propagate_block()` 默认复用端点力，L 步只需 L+1 次 PES 批调用，
   文档声明对确定性提供者结果逐位一致（`docs/native-propagation.md`、#116）。
2. **materials/diffusion 的 Andersen 路径**（`workflows/materials.py:676-679`）每步调用
   `dynamics.propagate(state, 1, record_centroids=False)`，每步都是一次独立的原生块调用加 Python
   事务开销（状态拷贝、RNG 快照、`propagation_stats` 深拷贝，`materials.py:660-663`）。

## 目标

- 子轨迹每次原生调用推进 `min(dyn.block_steps, 4096, 剩余步数)` 步，使用端点力复用。
- materials 的 Andersen 路径在可融合时，按"下一次 Andersen 刷新 / 下一个记录点 /
  checkpoint / `neq` 切换 / 块上限"中最近者分块推进。
- 两处结果与替换前逐位一致（E0）。

## 范围

**要做：**

1. `_child_task` 的 RPMD 分支改用 `dyn.propagate_block(state, count, cv=surface, ...)`，
   从 `block.values[:, 0]` 累加分子。保留与 `propagate()` 相同的失败可见性：非有限能量/力/CV
   必须像以前一样让该子轨迹失败（用 `check_energy=True` 或等价手段，并用现有失败注入测试证明）。
   `dyn.can_fuse(observe=surface)` 为假时保持原 Python 循环不变。
2. materials 旧引擎（`_sample_replica` 中的逐步循环，`materials.py:651-700`）在以下条件**全部**满足时分块：
   采样阶段为 Andersen 或生产阶段为 NVE；无 `piston`；`monitor is None`（主动学习需要逐步观测）；
   无 `friction_provider`；无 bond constraints；`dynamics.can_fuse()` 为真；
   原生绑定对当前体系可用（周期体系目前 `_native_plan` 不传 cell，读代码确认后，
   周期体系保持逐步路径并在 PR 中写明）。块边界处执行原有的 Andersen 刷新、
   `observe()`、`append_record`、checkpoint；失败回滚语义与原逐步事务一致
   （已提交的步保留，失败步回滚，RNG 与 `propagation_stats` 一并恢复）。
3. 在 `docs/native-propagation.md` 追加一个小节，说明子轨迹的块调度、materials 分块条件，
   以及"dRPMD/state-selected 可把 `dynamics.block_steps` 设为 4096 以减少返回"的建议
   （执行参数，不进指纹）。

**不做：**

- 不改 `native/`、`block_propagation.py`、`dynamics.py`。
- 不改父链（`_parent_task`）、TRPMD 子轨迹分支、PILE-L 路径（分别归 NP5、NP6）。
- 不改 `block_steps` 的默认值。

## 文件所有权

- **独占：** `workflows/rate_recrossing.py` 中 `_child_task` 的 RPMD 分支；
  `workflows/materials.py` 中 `_sample_replica` 的逐步循环；新增测试文件。
- **共享注册点：** `docs/native-propagation.md`（追加小节）。

## 交付物

- 上述两处调度改动与测试。
- `docs/validation/np0-scheduling-<日期>/README.md`：替换前后每 1 万步的原生调用次数、PES 调用次数、
  墙钟时间（内置谐振势；能构建时加 BKMP2），写明硬件。

## 验收标准

- 子轨迹：P ∈ {1, 2, 8, 16}，内置 Morse/谐振势（能构建时加 BKMP2），`numerator`、`denominator`、
  `energy_drift`、`endpoint_energy_error_hartree_per_bead` 与替换前**逐位一致**；
  `reuse_forces=True` 与 `False` 两种调度彼此逐位一致；PES 调用次数从 2L 降到不超过 L+1（每块）。
- materials：Andersen NVT 与 diffusion（NVT→NVE）在 P ∈ {1, 8}、`andersen_interval` ∈ {1, 7, 10}、
  `interval` ∈ {1, 5} 下，所有输出数组、checkpoint 中的 `q`/`p`/`rng`、最终 `rng.bit_generator.state`
  与替换前逐位一致；中断后续算与不中断逐位一致；块长改变后续算结果不变。
- R8：计数测试断言 materials 每 N 步的原生调用次数不超过
  `ceil(N / min(andersen_interval, interval)) + checkpoint 次数 + 1`，且 `python_steps == 0`。
- 不满足分块条件的配置（piston、monitor、friction、PILE-L、周期 cell）行为完全不变，有测试覆盖。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests/test_fused_propagation.py tests/test_native_propagation.py tests/test_batched_recrossing.py
python -m pytest -q tests -k "recross or materials"
python -m pytest -q -m "not campaign"
```
