---
orphan: true
---

# NP6 materials/diffusion 接入与原生系综 opt-in

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | NP0、NP1、NP2 已合并 |
| 等价性等级 | 旧引擎接入 E0；`task="diffusion"` + `sampling.engine="native"` 为 E2 |
| 建议分支 | `agent/np6-materials-route` |

## 背景

NP0 之后，materials 旧引擎（`workflows/materials.py` 的 `_sample_replica`）在 Andersen 下按"下一次刷新 /
记录点 / checkpoint"分块，块长仍受 `andersen_interval`（默认 10）限制；PILE-L 采样
（`materials.py:668-674`）仍是逐步 `sample_step`；NPT 活塞（`materials.py:663-667`）逐步 Python。

周期体系在旧引擎里无法进入 `advance`：`block_propagation._native_plan` 不传 cell，C ABI v1 也没有 cell 参数。
周期体系只能走 `NativeEnsemble`（`native/thermal.hpp`，`NativePeriodicPES`）。目前
`sampling.engine="native"` 只接受 `task="pimd"`（`workflows/native_pimd.py:62-63`），
文件头还写明"not a diffusion/rate propagator"。`NativeEnsemble` 的随机流和 middle splitting 与旧引擎不同，
所以这条路只能是 E2 opt-in。

## 目标

1. 旧引擎（非周期、原生绑定可用）：Andersen 用 NP1 跨越碰撞，PILE-L 用 NP2，均保持 E0。
2. 新增 E2 opt-in：`task="diffusion"` 且 `sampling.engine="native"` 时，平衡段用 `NativeEnsemble` 恒温器，
   生产段用无恒温器的 NVE，全程不逐步返回 Python；输出 schema 与旧引擎相同，使 `_analyze` 不需改动。

## 范围

**要做：**

1. **旧引擎 Andersen。** 在 NP0 的分块条件下，用 `plan_refresh` 覆盖块内全部刷新；块只在记录点
   （需要 `observe()` 的 Python 观测）、checkpoint、`neq` 切换、预算上限处结束。
   若记录点过密成为主要边界，可用 `record_states=True` 在块内保留记录帧，块后在 Python 中按原顺序计算
   `observe()`——前提是 `observe()` 只依赖记录帧的状态与提供者，且结果与逐步计算逐位一致（加测试证明，
   否则保持在记录点结束块）。
2. **旧引擎 PILE-L。** 满足 NP2 资格时用 `propagate_sampling_block`；diffusion 在 `neq` 处的动量重采样
   （`materials.py:682-684`）仍在块边界执行。
3. **E2：原生 diffusion。** 扩展 `resolve_native_pimd`/`_sample` 支持 `task="diffusion"`：
   - 平衡段：现有原生恒温器；生产段：确认 `thermal.hpp` 中 `Thermostat::None` 与 `advance` 的 NVE 使用同一 Cayley 映射，
     或用 `set_state` 把状态交给无恒温器的引擎；写明选择及理由。
   - 产出与旧引擎相同的数组（`sample_steps`、`positions`、`velocities`、`observables`、`radii_gyration_squared` 等）
     和 checkpoint 日志语义；改写文件头那句说明。
   - 物理指纹包含 `engine="native"`（沿用 `select_engine` 现有约定），不影响 `legacy` 与省略 engine 的输入。
4. NPT 活塞不在本卡范围；在文档中说明 NPT 的原生替代是 `task="pimd"` + `sampling.engine="native"` + 原生压浴（E2）。
5. 在 `docs/native-pimd-workflow.md` 和 `docs/materials.md` 各追加一节。

**不做：**

- 不改 `native/thermal.*` 的数学或随机流；若必须改，停止并写"需要协调"。
- 不改 Torch 引擎、主动学习 `monitor`、电子摩擦路径。

## 文件所有权

- **独占：** `workflows/materials.py` 的 `_sample_replica` 循环；`workflows/native_pimd.py`；新测试文件。
- **共享注册点：** `workflows/pimd.py` 的路由（只在确有必要时追加）、`docs/native-pimd-workflow.md`、
  `docs/materials.md`、`docs/native-propagation.md`（追加小节）。

## 交付物

- 接入改动、E2 路由、示例输入 `examples/ensembles/diffusion_native.inp`（数值对照用，写明不是材料验证）。
- `docs/validation/np6-materials-<日期>/README.md`：旧引擎替换前后每 1 万步原生调用次数与墙钟时间；
  E2 路由与旧引擎在同一体系（谐振子或理想气体的解析结果，以及内置周期势的对照）上的扩散系数/动能等统计对照。

## 验收标准

- **E0（旧引擎）：** Andersen NVT、diffusion（NVT→NVE）、PILE-L NVT，P ∈ {1, 8, 16}，`andersen_interval` ∈ {1, 7, 10}：
  全部输出数组、checkpoint `q`/`p`/`rng`、最终 RNG 状态与替换前逐位一致；中断续算一致；改变块长续算一致。
- **E2：** 原生 diffusion 的谐振子/自由粒子统计与解析值一致（误差棒内）；中断续算逐位一致；
  `legacy` 与省略 engine 的输入指纹和结果完全不变；`sampling.engine="native"` 遇到不支持的组合
  （bond constraints、trajectory batching、PILE 以外未实现的配置）在任何副作用之前报错。
- **R8：** 旧引擎每 1 万步原生调用次数 ≤ `ceil(1e4 / 记录间隔) + checkpoint 次数 + 预算截断次数`，`python_steps == 0`；
  E2 路由每个 checkpoint 块一次调用。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q tests -k "materials or diffusion or native_pimd or pimd"
python -m pytest -q -m "not campaign"
```
