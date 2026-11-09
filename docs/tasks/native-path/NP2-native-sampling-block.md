---
orphan: true
---

# NP2 C++：PILE-L/TRPMD 随机采样块

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无（与第四轮 T27 的文件重叠见本轮 README） |
| 等价性等级 | 无偏置 E0；有偏置 E1 |
| 建议分支 | `agent/np2-sampling-block` |

## 背景

`Dynamics.sample_step`（`dynamics.py:840` 起）在 BCOCB/BAOAB 的 PILE-L 与 TRPMD
（`trpmd_step`，`dynamics.py:827-838`，`gamma_centroid=0`）中，每步由 Python 编排：

1. `_kick(q, p, bias)`（半步力，含 Python 偏置）；
2. 满足 `fused_sampler` 条件（`dynamics.py:956-967`：`loop_backend="auto"`、未覆写方法、`CayleyIntegrator`、
   P ≤ 8 或 2 的幂、无 fragment）时，抽噪声 `rng.normal(size=p.shape) * scale`、固定原子置零
   （`dynamics.py:992-993`），调用 C++ `sample_free`（半步自由 + OU + 半步自由）；
3. 再 `_kick`，检查有限性。每步原子提交，失败时恢复 RNG（`dynamics.py:1002-1003`）。

噪声与状态无关，可以按原顺序预抽。自由传播和 OU 已在 C++（`sample_free`），只差把力、偏置和
步循环也放进 C++。受益链路：PILE-L 伞形采样（`sampling.py:194-202`）、TRPMD 子轨迹
（`rate_recrossing.py:296-300`）、materials 的 PILE-L（`materials.py:668-674`）。接入由 NP5/NP6 完成。

## 目标

提供一个原生随机采样块入口：一次调用推进最多 4096 步（受 R6 预算约束），块内无 Python 回调，
与当前 `sample_step` 的 `fused_sampler` 路径逐位一致（无偏置时）。

## 范围

**要做：**

1. **C++（新 `native/sampling_block.h`）。** 新入口 `advance_sampling`：输入 `q`、`p`、步数、
   预抽噪声 `[steps, P, N, 3]`（已缩放、固定原子已置零）、`decay[P]`、`amplitude[P]`、固定原子掩码、
   原生 PES 绑定、偏置/CV 描述符（复用 `cv_core.hpp` 的现有类型，只读使用）、观测掩码、
   `record_energy`/`check_energy`/`step_energies`。每步执行与 `sample_step` 相同的
   "kick → `sample_free` 同一内核 → kick"，复用 `Cayley` 上下文中已有的实现，**不复制一份自由传播代码**。
   端点力复用：确定性提供者默认开启，与 `propagate_block` 的规则一致。返回有效前缀
   （已完成步数、错误信息），失败步不提交。
   在 `native/cayley.cpp` 中只追加 `#include`、`methods[]` 一行和能力常量 `SAMPLING_BLOCK_API = 1`。
2. **Python（新 `src/pyrpmd/sampling_block.py`）。**
   `propagate_sampling_block(dynamics, state, steps, *, thermostat="pile-l"|"trpmd", method=None,
   gamma_centroid=None, friction_scale=1.0, bias=None, cv=None, observe=None, record_energy=True,
   check_energy=False, record_step_energies=False, reuse_forces=None, raise_on_error=True)`，返回与
   `BlockResult` 相同字段的结果。
   - 资格判定与 `sample_step` 的 `fused_sampler` 条件完全一致，并要求原生 PES 绑定、受支持的偏置/CV；
     不满足时逐步调用原 `dynamics.sample_step` / `trpmd_step`（R4，按定义 E0），记录 `fallback_reason`。
   - 噪声按原顺序**逐步调用**生成：每步 `rng.normal(size=p.shape) * scale`，固定原子置零。
     向量化抽样只能作为可选优化，且必须有 R2 回归测试与回退。
   - 失败提交：恢复 RNG 到块起点后重抽"已完成步数"次，或保存逐步快照。
   - 内存预算复用 NP1 的 `MAX_PREDRAWN_BYTES`；NP1 未合并时先在本模块定义同名常量，
     在 PR"需要协调"中注明合并后改为导入。
3. **`dynamics.py` 唯一允许的改动：** 把 `sample_step` 中 PILE 系数计算（`dynamics.py:968-989`
   的 `decay`/`amplitude`/`scale`/`fixed` 及缓存）抽成私有函数供新模块调用，不改数学和缓存键。
4. 在 `docs/native-propagation.md` 追加小节。

**不做：**

- 不接入工作流（归 NP5、NP6）。
- 不支持 fragment 投影、bond constraints、组合算符执行器（`plan.active` / `thermostat_spec`）、
  非 2 的幂且 P > 8 的 NumPy FFT 路径（`dynamics.py:1037` 起）——这些保持原逐步路径。
- 不改 `native/thermal.*`，不引入 C++ 随机数。

## 文件所有权

- **独占：** 新 `native/sampling_block.h`、新 `src/pyrpmd/sampling_block.py`、
  `dynamics.py` 中被抽出的 PILE 系数私有函数、新测试文件。
- **共享注册点：** `native/cayley.cpp`（`#include`、`methods[]`、能力常量各一行）、
  `docs/native-propagation.md`（追加小节）。

## 交付物

- 原生入口、Python 包装与测试；`native/README.md` 追加一段说明（只追加）。
- `docs/validation/np2-sampling-block-<日期>/README.md`：逐步路径与块路径在 P ∈ {4, 8, 32}、N ∈ {3, 64}
  下每步耗时与原生调用次数（内置谐振势；能构建时加 BKMP2），写明硬件。

## 验收标准

- **E0（无偏置）：** PILE-L（BCOCB、BAOAB）与 TRPMD（`friction_scale` ∈ {0.5, 1}），P ∈ {1, 2, 4, 8, 16, 32}，
  有/无固定原子：一次 N 步块与 N 次 `sample_step`/`trpmd_step` 相比，`q`、`p`、时钟、能量、
  `rng.bit_generator.state` 逐位一致；`reuse_forces` 开/关彼此逐位一致。
- **E1（有偏置）：** 伞形偏置、flux-metric 偏置下，RNG 逐位一致；`q`/`p` 差异不超过现有融合/非融合
  偏置对比测试的容差，PR 写明数值。
- **失败语义：** 在第 0 步与中间步注入 PES 失败，有效前缀、RNG 状态与逐步参考一致；首步失败时状态和 RNG 不变。
- **回退：** P=12（非 2 的幂且 > 8）、`loop_backend="python"`、覆写方法、缺少 `SAMPLING_BLOCK_API`
  的扩展都走原逐步路径并记录原因；`native` 后端下不支持的组合在修改状态前报错。
- **预算：** N=1000、P=32 时块长被预算截短，结果不变。
- OpenMP 开/关编译、`PYRPMD_NO_NATIVE=1`、开启 FMA 的构建都通过。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
OMP_NUM_THREADS=1 python -m pytest -q tests -k "pile or trpmd or sampling or native"
python -m pytest -q -m "not campaign"
PYRPMD_NO_NATIVE=1 python -m pip install -e . && python -m pytest -q tests -k "pile or trpmd"
```
